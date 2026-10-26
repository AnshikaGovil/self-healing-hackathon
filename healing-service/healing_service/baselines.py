"""Baselines held on behalf of thin clients.

Why the service keeps these
---------------------------
Two of the seven similarity signals - structure and geometry - need a
*baseline*: a fingerprint of the element taken while the locator still
worked. Without one they return None, are dropped from the weighted mean,
and 0.18 of the total weight is permanently unavailable. A cold heal is
then scored on whatever is left, which is why correct repairs land in the
review band and get refused.

The full toolkit solves this with a memory file next to the bot. A thin
client cannot: ``bis.py`` is a single file an RPA developer copies beside
their script, and giving it a state file, a merge and a cross-process lock
would make it something else. So the state lives here, on the machine that
already holds the model, the policy and the JavaScript.

That it is shared is the point, not a compromise. Baselines are keyed by
locator and page, not by bot, so the first process to walk a page warms
every process that walks it afterwards - including ones on other hosts.

What is stored is an element fingerprint, the same shape already crossing
the wire on every adjudication. No page, no DOM, no typed values.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from bis_core.descriptors import ElementDescriptor, own_value, page_scope

log = logging.getLogger("bis.service.baselines")

# Beside the toolkit's healing memory in the project layout, and under the
# same gitignore rule: this is learned run state, not part of the deliverable.
DEFAULT_PATH = os.path.join("run-memory", "baselines.json")


class BaselineStore:
    """Locator fingerprints, keyed by locator and page, persisted to JSON.

    Deliberately not the toolkit's HealingMemory. That file is co-owned by
    several bot processes and has to merge on every save; this one has a
    single writer - the service - so a lock and an atomic replace are the
    whole concurrency story.
    """

    def __init__(self, path: str = DEFAULT_PATH, max_entries: int = 5000,
                 ttl_days: float = 90.0) -> None:
        self.path = path
        self.max_entries = max_entries
        self.ttl_s = max(0.0, ttl_days) * 86400.0
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        # Where the step after each locator found itself, counted over the
        # runs in which the locator's own selector worked: page scope ->
        # times seen. Kept apart from the fingerprints, which a heal may
        # replace, because this is about the step and not about an element.
        self._next: Dict[str, Dict[str, int]] = {}
        self._dirty = False
        self._load()

    # ----------------------------------------------------------------- keys
    @staticmethod
    def key(locator_key: str, url: str) -> str:
        """The key the toolkit's memory uses, from the same helper."""
        return "%s@%s" % (locator_key or "locator", page_scope(url))

    # ----------------------------------------------------------------- read
    def get(self, locator_key: str, url: str) -> Optional[Dict[str, Any]]:
        """The stored fingerprint, or None. Expired entries count as absent."""
        with self._lock:
            entry = self._data.get(self.key(locator_key, url))
            if not entry:
                return None
            if self.ttl_s and time.time() - entry.get("at", 0) > self.ttl_s:
                # A fingerprint from three months ago describes a page that
                # has probably been redesigned since. Geometry ages fastest.
                return None
            return entry.get("baseline") or None

    def is_proven(self, locator_key: str, url: str) -> bool:
        """Did the authored selector ever resolve this locator on its own?

        A provisional entry - recorded from a healed lookup - does not count.
        It is a guess that warmed the ranker, not evidence the step was ever
        written correctly.
        """
        with self._lock:
            entry = self._data.get(self.key(locator_key, url))
            if not entry:
                return False
            if self.ttl_s and time.time() - entry.get("at", 0) > self.ttl_s:
                return False
            return not entry.get("provisional", False)

    # ---------------------------------------------------------------- write
    def record(self, locator_key: str, url: str, raw: Dict[str, Any],
               selector: str = "", provisional: bool = False) -> bool:
        """Store a fingerprint. Returns whether it was accepted.

        Refuses anything that is not on screen. Structure and geometry are
        the signals that make repairs better the longer a fleet runs, and
        seeding them from a hidden or zero-sized element makes them worse
        forever. No baseline beats a false one - the same rule the toolkit's
        _capture_baseline enforces, applied at the other end of the wire
        because a thin client is not the place to put policy.

        ``provisional`` marks a fingerprint that came from a *healed* lookup
        rather than one the authored selector resolved itself. It still warms
        the ranker, which is the whole point of keeping it, but it is not
        proof that the locator was ever right - so it never lets a repair be
        promoted to the fleet. A provisional entry is upgraded the first time
        the authored selector matches on its own. See invariant 26.
        """
        desc = ElementDescriptor.coerce(raw or {})
        # A bot holds the page JavaScript it fetched for as long as it runs,
        # so one from before own_value still sends what a field holds.
        desc.value = own_value(desc.tag, desc.input_type, desc.value)
        rect = desc.rect or {}
        if not desc.visible or not rect.get("width") or not rect.get("height"):
            log.debug("refusing baseline for %s: element is not on screen",
                      locator_key)
            return False
        with self._lock:
            key = self.key(locator_key, url)
            was = self._data.get(key) or {}
            # A healed lookup never replaces a proven one.
            #
            # "Once proven, always proven" used to mean the flag was kept
            # while the fingerprint beside it was overwritten - so a heal
            # inherited a proven status the seed pass had earned for a
            # *different element*. Observed end to end on a live HR form:
            # seeding recorded the Employee Id box correctly, one bad
            # repair bound the search box in the header, that fingerprint
            # was written over the seeded one and kept `provisional: False`,
            # and from then on every run was warmed to the wrong element
            # and repaired to it at high confidence. There was no path back:
            # a repair that binds the wrong element still binds, so nothing
            # ever demoted it.
            #
            # The proven entry is evidence - the authored selector resolved
            # that element. A healed one is a guess that happened to work.
            # When both exist the evidence wins, and staleness is the TTL's
            # job, not this decision's.
            if provisional and not was.get("provisional", True):
                log.debug("keeping the proven baseline for %s rather than "
                          "the healed one", locator_key)
                return False
            still_provisional = provisional and was.get("provisional", True)
            self._data[key] = {
                "locator": locator_key,
                "scope": page_scope(url),
                "selector": selector,
                "at": time.time(),
                "provisional": still_provisional,
                "baseline": desc.to_dict(),
            }
            self._dirty = True
            self._evict()
        self.save()
        return True

    # What follows a step is only evidence once it has been seen more than
    # once, and only the few commonest destinations are kept: a step with a
    # branch after it has several, and all of them are right.
    NEXT_KEPT = 5

    def note_next(self, locator_key: str, url: str, next_url: str) -> None:
        """Count where the next step found itself after this one worked.

        Only from runs in which this locator's own selector resolved - a
        good run - so a repair's runs can be compared against them.
        """
        landed = page_scope(next_url)
        with self._lock:
            seen = self._next.setdefault(self.key(locator_key, url), {})
            seen[landed] = seen.get(landed, 0) + 1
            if len(seen) > self.NEXT_KEPT:
                for rare in sorted(seen, key=seen.get)[
                        :len(seen) - self.NEXT_KEPT]:
                    del seen[rare]
            while len(self._next) > self.max_entries:
                self._next.pop(next(iter(self._next)))
            self._dirty = True
        self.save()

    def next_pages(self, locator_key: str, url: str) -> Dict[str, int]:
        """Where the next step found itself in good runs, with how often."""
        with self._lock:
            return dict(self._next.get(self.key(locator_key, url), {}))

    def _evict(self) -> None:
        """Drop the oldest entries once the store is over its cap.

        Called with the lock held. A fleet that runs for years against many
        sites must not turn this into an unbounded file on the GPU host.
        """
        if len(self._data) <= self.max_entries:
            return
        ordered = sorted(self._data.items(), key=lambda kv: kv[1].get("at", 0))
        for key, _ in ordered[: len(self._data) - self.max_entries]:
            self._data.pop(key, None)

    # ------------------------------------------------------------- lifecycle
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            entries = raw.get("entries") if isinstance(raw, dict) else None
            self._data = entries if isinstance(entries, dict) else {}
            follows = raw.get("next") if isinstance(raw, dict) else None
            self._next = follows if isinstance(follows, dict) else {}
            log.info("loaded %d baselines from %s", len(self._data), self.path)
        except FileNotFoundError:
            self._data = {}
        except (OSError, ValueError) as exc:
            # A corrupt store must not stop the service from adjudicating.
            # Losing the baselines costs accuracy; refusing to start costs
            # every bot in the fleet.
            log.warning("could not read %s (%s) - starting with no baselines",
                        self.path, exc)
            self._data = {}

    def save(self) -> None:
        """Write the store out. Never raises - a cache, not a ledger.

        The whole write happens under the lock. It used to snapshot under
        the lock and then release it, which is only safe if there is one
        writer - and the "single writer" in the class docstring is the
        service, not a thread. FastAPI runs a sync endpoint in a threadpool,
        so two /v1/baseline posts really do land here at once, and with the
        lock dropped their os.replace calls could complete in either order:
        the older snapshot wins, and the newer baseline is gone. Holding it
        serialises a sub-millisecond write of a small file, which is not a
        contention problem worth a lost fingerprint.
        """
        with self._lock:
            if not self._dirty:
                return
            payload = {"version": 1, "entries": dict(self._data),
                       "next": {k: dict(v) for k, v in self._next.items()}}
            try:
                folder = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(folder, exist_ok=True)
                handle = tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=folder, delete=False,
                    suffix=".tmp")
                try:
                    json.dump(payload, handle, indent=1)
                finally:
                    handle.close()
                os.replace(handle.name, self.path)
            except OSError as exc:
                # Left dirty on purpose, so the next record() retries rather
                # than reporting success over a write that never happened.
                self._dirty = True
                log.warning("could not write %s (%s) - baselines are in "
                            "memory only for this run", self.path, exc)
            else:
                self._dirty = False

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"entries": len(self._data)}
