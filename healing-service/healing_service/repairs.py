"""SQLite store of repairs bots have proven, shared across the fleet.

A thin client has no memory of its own, so a repair it proves dies with the
process and every other bot pays the adjudication cost again for the same
break on the same page. This keeps them, under the same identity the
toolkit's HealingMemory uses, together with which automation process and
which call site inside it proved each one.

Deliberately not the toolkit's HealingMemory. That file is co-owned by
several bot processes and has to merge on every save (invariant 14); this
one has a single writer - the service - so WAL plus a lock around writes is
the whole concurrency story.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from bis_core.descriptors import (normalize_selector, page_scope,
                                  selector_syntax)

log = logging.getLogger("bis.service.repairs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repairs (
  id             INTEGER PRIMARY KEY,
  locator_key    TEXT NOT NULL,
  page_scope     TEXT NOT NULL,
  replaced       TEXT NOT NULL,
  syntax         TEXT NOT NULL,
  selector       TEXT NOT NULL,
  confidence     REAL NOT NULL,
  strategy       TEXT NOT NULL,
  rationale      TEXT NOT NULL DEFAULT '',
  proven_count   INTEGER NOT NULL DEFAULT 0,
  failed_count   INTEGER NOT NULL DEFAULT 0,
  mismatch_count INTEGER NOT NULL DEFAULT 0,
  status         TEXT NOT NULL DEFAULT 'active',
  first_seen_at  REAL NOT NULL,
  last_proven_at REAL NOT NULL
);
-- Partial, so one active repair per identity can sit above any number of
-- demoted rows. A plain unique index would make a fresh repair collide
-- with the demoted one it supersedes.
CREATE UNIQUE INDEX IF NOT EXISTS repairs_identity
  ON repairs (locator_key, page_scope, replaced, syntax)
  WHERE status = 'active';
CREATE TABLE IF NOT EXISTS repair_provenance (
  id           INTEGER PRIMARY KEY,
  repair_id    INTEGER NOT NULL,
  process_name TEXT NOT NULL DEFAULT '',
  file         TEXT NOT NULL DEFAULT '',
  function     TEXT NOT NULL DEFAULT '',
  line         INTEGER NOT NULL DEFAULT 0,
  host         TEXT NOT NULL DEFAULT '',
  at           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS provenance_repair
  ON repair_provenance (repair_id);
"""

# Beside the baseline store, under the same gitignore rule: learned run
# state, not part of either deliverable.
DEFAULT_PATH = os.path.join("run-memory", "repairs.db")

# One failure is as likely to be a slow page as a bad repair, and demoting
# on it throws away good knowledge on a timing flake. Two is a pattern.
DEMOTE_AFTER = 2


class RepairStore:
    """Repairs keyed by locator and page, with per-process provenance."""

    DEFAULT_PATH = DEFAULT_PATH

    def __init__(self, path: str = DEFAULT_PATH, max_entries: int = 5000,
                 ttl_days: float = 90.0,
                 provenance_per_repair: int = 20) -> None:
        self.path = path
        self.max_entries = max_entries
        self.ttl_s = max(0.0, ttl_days) * 86400.0
        self.provenance_per_repair = max(1, provenance_per_repair)
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        self._open()

    def _open(self) -> None:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            # A store from before outcome checks has no column for them.
            columns = {r["name"] for r in self._db.execute(
                "PRAGMA table_info(repairs)")}
            if "mismatch_count" not in columns:
                self._db.execute("ALTER TABLE repairs ADD COLUMN "
                                 "mismatch_count INTEGER NOT NULL DEFAULT 0")
            self._db.commit()
        except Exception as exc:
            # A store that will not open must not stop the service starting.
            log.warning("repair store unavailable (%s) - healing will run "
                        "without it", exc)
            self._db = None

    def close(self) -> None:
        """Release the connection. Further calls degrade to no-ops."""
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                finally:
                    self._db = None

    # ----------------------------------------------------------- identity
    @staticmethod
    def identity(locator_key: str, url: str, authored: str):
        """The four fields a repair is looked up by.

        ``replaced`` and ``syntax`` are in the key on purpose. The first
        gives invariant 15 across the fleet - a repair belongs to the
        selector it replaced, so a re-authored locator stops matching. The
        second gives invariants 8 and 23, so a repair learned against a CSS
        step is never served to an XPath one, which a single bot could not
        have got wrong but a fleet can.
        """
        return (locator_key or "locator", page_scope(url),
                normalize_selector(authored) or "",
                selector_syntax(authored))

    # --------------------------------------------------------------- read
    def lookup(self, locator_key: str, url: str,
               authored: str) -> Optional[Dict[str, Any]]:
        """The active repair for this identity, or None.

        Expired rows count as absent: a repair proven three months ago
        describes a page that has probably been redesigned since.
        """
        key, scope, replaced, syntax = self.identity(locator_key, url,
                                                     authored)
        try:
            with self._lock:
                if self._db is None:
                    return None
                row = self._db.execute(
                    "SELECT id, selector, confidence, strategy, rationale, "
                    "       last_proven_at "
                    "  FROM repairs "
                    " WHERE locator_key=? AND page_scope=? AND replaced=? "
                    "   AND syntax=? AND status='active'",
                    (key, scope, replaced, syntax)).fetchone()
            if row is None:
                return None
            if self.ttl_s and time.time() - row["last_proven_at"] > self.ttl_s:
                return None
            return {"id": int(row["id"]), "selector": row["selector"],
                    "confidence": float(row["confidence"]),
                    "strategy": row["strategy"],
                    "rationale": row["rationale"]}
        except Exception as exc:
            log.warning("repair lookup failed (%s) - treating as a miss", exc)
            return None

    def by_id(self, repair_id: int) -> Optional[Dict[str, Any]]:
        """One repair by id, active or demoted, or None."""
        try:
            with self._lock:
                if self._db is None:
                    return None
                row = self._db.execute(
                    "SELECT id, selector, status FROM repairs WHERE id=?",
                    (int(repair_id),)).fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            log.warning("repair read failed (%s)", exc)
            return None

    def provenance(self, repair_id: int) -> List[Dict[str, Any]]:
        """Which processes, and which parts of them, proved this repair."""
        try:
            with self._lock:
                if self._db is None:
                    return []
                rows = self._db.execute(
                    "SELECT process_name, file, function, line, host, at "
                    "  FROM repair_provenance WHERE repair_id=? "
                    " ORDER BY at DESC", (int(repair_id),)).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("provenance read failed (%s)", exc)
            return []

    def stats(self) -> Dict[str, int]:
        """Active and demoted row counts, for /v1/health."""
        try:
            with self._lock:
                if self._db is None:
                    return {"active": 0, "demoted": 0}
                rows = self._db.execute(
                    "SELECT status, COUNT(*) n FROM repairs "
                    "GROUP BY status").fetchall()
            out = {"active": 0, "demoted": 0}
            for r in rows:
                out[r["status"]] = int(r["n"])
            return out
        except Exception as exc:
            log.warning("repair stats failed (%s)", exc)
            return {"active": 0, "demoted": 0}

    # -------------------------------------------------------------- write
    def record(self, locator_key: str, url: str, authored: str,
               selector: str, confidence: float, strategy: str,
               rationale: str = "",
               process: Optional[Dict[str, Any]] = None) -> int:
        """Store a proven repair. Returns its id, or 0 if it was not stored.

        An existing active repair for the same identity is updated in place
        rather than duplicated: the page has moved again and this is the
        newer answer to the same question.
        """
        if not selector:
            return 0
        key, scope, replaced, syntax = self.identity(locator_key, url,
                                                     authored)
        now = time.time()
        try:
            with self._lock:
                if self._db is None:
                    return 0
                row = self._db.execute(
                    "SELECT id FROM repairs WHERE locator_key=? AND "
                    "page_scope=? AND replaced=? AND syntax=? AND "
                    "status='active'",
                    (key, scope, replaced, syntax)).fetchone()
                if row is None:
                    cur = self._db.execute(
                        "INSERT INTO repairs (locator_key, page_scope, "
                        " replaced, syntax, selector, confidence, strategy, "
                        " rationale, proven_count, failed_count, status, "
                        " first_seen_at, last_proven_at) "
                        "VALUES (?,?,?,?,?,?,?,?,1,0,'active',?,?)",
                        (key, scope, replaced, syntax, selector,
                         float(confidence), strategy, rationale, now, now))
                    repair_id = int(cur.lastrowid)
                else:
                    repair_id = int(row["id"])
                    # A different selector is a different repair: the
                    # mismatches counted against the old one are not its.
                    self._db.execute(
                        "UPDATE repairs SET mismatch_count = CASE WHEN "
                        " selector=? THEN mismatch_count ELSE 0 END, "
                        " selector=?, confidence=?, "
                        " strategy=?, rationale=?, "
                        " proven_count=proven_count+1, failed_count=0, "
                        " last_proven_at=? WHERE id=?",
                        (selector, selector, float(confidence), strategy,
                         rationale, now, repair_id))
                self._add_provenance(repair_id, process, now)
                self._prune()
                self._db.commit()
            return repair_id
        except Exception as exc:
            log.warning("could not record repair for %s (%s)",
                        locator_key, exc)
            return 0

    def reinforce(self, repair_id: int,
                  process: Optional[Dict[str, Any]] = None,
                  confidence: Optional[float] = None,
                  strategy: str = "", rationale: str = "") -> bool:
        """Note that a stored repair worked again.

        ``confidence`` and ``strategy`` replace the stored ones when given -
        a reviewer approving a repair the store served is the case: it is
        then known right, and the next bot should not have to ask.
        """
        now = time.time()
        try:
            with self._lock:
                if self._db is None:
                    return False
                if confidence is not None:
                    cur = self._db.execute(
                        "UPDATE repairs SET proven_count=proven_count+1, "
                        " failed_count=0, last_proven_at=?, confidence=?, "
                        " strategy=?, rationale=? "
                        " WHERE id=? AND status='active'",
                        (now, float(confidence), strategy, rationale,
                         int(repair_id)))
                else:
                    cur = self._db.execute(
                        "UPDATE repairs SET proven_count=proven_count+1, "
                        " failed_count=0, last_proven_at=? "
                        " WHERE id=? AND status='active'",
                        (now, int(repair_id)))
                if not cur.rowcount:
                    # No row matched, so there is nothing to commit - but
                    # SQLite opened a deferred transaction for the UPDATE
                    # regardless, and it would otherwise sit open on the
                    # connection every request shares until some later
                    # write happened to commit it for us.
                    self._db.rollback()
                    return False
                self._add_provenance(int(repair_id), process, now)
                self._db.commit()
            return True
        except Exception as exc:
            log.warning("could not reinforce repair %s (%s)", repair_id, exc)
            return False

    def demote(self, repair_id: int) -> bool:
        """Count a failure. Returns whether that demoted the repair."""
        try:
            with self._lock:
                if self._db is None:
                    return False
                self._db.execute(
                    "UPDATE repairs SET failed_count=failed_count+1 "
                    " WHERE id=? AND status='active'", (int(repair_id),))
                row = self._db.execute(
                    "SELECT failed_count FROM repairs WHERE id=? AND "
                    "status='active'", (int(repair_id),)).fetchone()
                if row is None:
                    self._db.commit()
                    return False
                demoted = int(row["failed_count"]) >= DEMOTE_AFTER
                if demoted:
                    self._db.execute(
                        "UPDATE repairs SET status='demoted' WHERE id=?",
                        (int(repair_id),))
                self._db.commit()
            return demoted
        except Exception as exc:
            log.warning("could not demote repair %s (%s)", repair_id, exc)
            return False

    def mismatch(self, repair_id: int) -> bool:
        """Count an outcome that went somewhere good runs never did.

        Returns whether that demoted the repair. Counted apart from failed
        binds on purpose: a wrong repair binds every time, and every bind
        resets that count, so strikes kept there could never add up.
        """
        try:
            with self._lock:
                if self._db is None:
                    return False
                self._db.execute(
                    "UPDATE repairs SET mismatch_count=mismatch_count+1 "
                    " WHERE id=? AND status='active'", (int(repair_id),))
                row = self._db.execute(
                    "SELECT mismatch_count FROM repairs WHERE id=? AND "
                    "status='active'", (int(repair_id),)).fetchone()
                demoted = (row is not None
                           and int(row["mismatch_count"]) >= DEMOTE_AFTER)
                if demoted:
                    self._db.execute(
                        "UPDATE repairs SET status='demoted' WHERE id=?",
                        (int(repair_id),))
                self._db.commit()
            return demoted
        except Exception as exc:
            log.warning("could not count a mismatch against repair %s (%s)",
                        repair_id, exc)
            return False

    def retire(self, repair_id: int) -> bool:
        """Demote a repair now, not on a second strike. Returns whether it
        was active. For a person's verdict: a reviewer rejecting a repair
        the store served has looked at it, which no failed bind ever did."""
        try:
            with self._lock:
                if self._db is None:
                    return False
                cur = self._db.execute(
                    "UPDATE repairs SET status='demoted', "
                    " failed_count=failed_count+1 "
                    " WHERE id=? AND status='active'", (int(repair_id),))
                self._db.commit()
            return bool(cur.rowcount)
        except Exception as exc:
            log.warning("could not retire repair %s (%s)", repair_id, exc)
            return False

    # ------------------------------------------------------------ private
    def _add_provenance(self, repair_id: int,
                        process: Optional[Dict[str, Any]],
                        now: float) -> None:
        """Append one proof, then trim to the cap. Caller holds the lock."""
        process = process or {}
        self._db.execute(
            "INSERT INTO repair_provenance (repair_id, process_name, file, "
            " function, line, host, at) VALUES (?,?,?,?,?,?,?)",
            (repair_id, str(process.get("name", "")),
             str(process.get("file", "")), str(process.get("function", "")),
             int(process.get("line", 0) or 0), str(process.get("host", "")),
             now))
        self._db.execute(
            "DELETE FROM repair_provenance WHERE repair_id=? AND id NOT IN ("
            "  SELECT id FROM repair_provenance WHERE repair_id=? "
            "   ORDER BY at DESC, id DESC LIMIT ?)",
            (repair_id, repair_id, self.provenance_per_repair))

    def _prune(self) -> None:
        """Drop expired and over-cap active rows. Caller holds the lock.

        Demoted rows are exempt: they are the record of what stopped
        working, and that is the only audit trail of a bad repair.
        """
        if self.ttl_s:
            self._db.execute(
                "DELETE FROM repairs WHERE status='active' AND "
                "last_proven_at < ?", (time.time() - self.ttl_s,))
        self._db.execute(
            "DELETE FROM repairs WHERE status='active' AND id NOT IN ("
            "  SELECT id FROM repairs WHERE status='active' "
            "   ORDER BY last_proven_at DESC LIMIT ?)", (self.max_entries,))
        self._db.execute(
            "DELETE FROM repair_provenance WHERE repair_id NOT IN "
            "(SELECT id FROM repairs)")
