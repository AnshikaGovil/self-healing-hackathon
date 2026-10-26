"""The adjudication seam.

This is where the system splits across the network.

    on the bot            |  over HTTP  |  on the inference host
    ----------------------+-------------+------------------------------
    harvest the live DOM  |             |
    lexical ranking       |  ---------> |  semantic ranking (embeddings)
    (no model needed)     |             |  AI adjudication (GGUF on GPU)
    bind + validate       |  <--------- |  ranked decision
    apply / escalate      |             |

The split is chosen so that everything needing a GPU is on one side and
everything needing a browser is on the other. The bot ships only element
*fingerprints* for the shortlist it already believes in - roughly 45 of them,
tens of kilobytes - never the page, never a screenshot, never the DOM.

Two implementations satisfy the same contract:

``LocalAdjudicator``   runs in-process. Used by the service, and available to
                      a bot that genuinely has its own accelerator.
``RemoteAdjudicator``  posts to the service and returns its decision. Used by
                      every bot in a fleet, none of which needs a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from . import runtime
from .calibration import DEFAULT_PATH as DEFAULT_CALIBRATION, Calibrator
from .config import HealingConfig
from .descriptors import (ElementDescriptor, LocatorSpec,
                          describe_from_baseline, informative_intent_tokens,
                          selector_tag)
from .similarity import HealTarget, score_candidates

log = logging.getLogger("bis.adjudicator")

# What the fine-tuned checker was trained to judge: one control a bot
# drives. Every positive in its corpus was an input, button, select,
# textarea, link or an element carrying one of these roles. Asked about
# a product card for a `count` step it answered 0.000 for every
# candidate, right one included - out of its distribution, not a
# judgement - and would have held every such repair for review.
_CHECKER_TAGS = {"input", "button", "select", "textarea", "a"}
_CHECKER_ROLES = {"button", "link", "tab", "menuitem", "checkbox", "switch",
                  "textbox", "combobox", "searchbox", "radio", "option"}
_COLLECTION_ACTIONS = {"collect", "count"}


def _is_control(descriptor) -> bool:
    """Is this element the kind the checker learned - a control?

    The selector cannot always say: `.sidebar_categories` names no tag, so
    the step reads as checkable, and its target is a <div>. Live, the
    checker scored every candidate of that sidebar near zero and refused a
    step the AI-first path had healed at 81%. The candidates can say, so
    the checker only picks when the ranker's best guess is a control, and
    only judges an element that is one. Mirrors harvest_real's TARGETS_JS.
    """
    tag = (getattr(descriptor, "tag", "") or "").lower()
    role = (getattr(descriptor, "role", "") or "").lower()
    if tag == "input" and (getattr(descriptor, "input_type", "")
                           or "").lower() == "hidden":
        return False
    return tag in _CHECKER_TAGS or role in _CHECKER_ROLES

PROTOCOL_VERSION = "1"


@dataclass
class AdjudicationRequest:
    """What the bot asks the service to decide."""

    locator: Dict[str, Any]
    action: str
    page: Dict[str, str]
    candidates: List[Dict[str, Any]]
    baseline: Optional[Dict[str, Any]] = None
    shortlist_size: int = 8
    budget_s: float = 25.0
    protocol: str = PROTOCOL_VERSION
    # The memory the question is asked against - a demo session's, or ""
    # for the shared one. The service fills it in; a client never does.
    scope: str = ""

    @classmethod
    def build(cls, spec: LocatorSpec, action: str, url: str, title: str,
              candidates: List[ElementDescriptor],
              baseline: Optional[ElementDescriptor],
              shortlist_size: int, budget_s: float) -> "AdjudicationRequest":
        return cls(
            locator={
                "key": spec.key,
                "selector": spec.selector,
                "description": spec.description,
                "tag": spec.tag,
                "role": spec.role,
                "expected_text": spec.expected_text,
                "multiple": spec.multiple,
            },
            action=action,
            page={"url": url, "title": title or ""},
            # `selectors` is stripped: only the bot binds selectors, so the
            # service has no use for them and they are the bulk of the payload.
            candidates=[_strip(c.to_dict()) for c in candidates],
            baseline=_strip(baseline.to_dict()) if baseline else None,
            shortlist_size=shortlist_size,
            budget_s=budget_s,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strip(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw.pop("selectors", None)
    return raw


def _from_odds(verdict) -> bool:
    """Is this verdict's confidence the AI's token probability?

    Then it is a probability and is used as one. The number a reply writes
    is not: the fine-tune learned it as a label, and it separates right
    picks from wrong ones at AUC 0.59 against 0.93 for the odds.
    """
    return getattr(verdict, "source", "") == "token"


def _described(spec: LocatorSpec) -> bool:
    """Does the step carry a real description, not none or a placeholder?"""
    desc = spec.description or ""
    return bool(desc) and bool(informative_intent_tokens(desc))


def _group(spec: LocatorSpec) -> str:
    """How the models were told about the step: the calibration group."""
    if _described(spec):
        return "described"
    return "placeholder" if (spec.description or "").strip() else \
        "undescribed"


def _ai_line(model):
    """How a candidate is written for this AI: the line its weights learned.

    Every fine-tune so far read compact(); one trained on prompt version 3
    reads compact_v2(). The checker reads compact() whatever the AI does.
    """
    if getattr(model, "prompt_version", 2) >= 3:
        return ElementDescriptor.compact_v2
    return ElementDescriptor.compact


def _ai_baseline(model, request) -> Dict[str, Any]:
    """The LAST KNOWN GOOD line for an AI that was trained to read one."""
    if getattr(model, "prompt_version", 2) < 3 or not request.baseline:
        return {}
    return {"baseline": ElementDescriptor.coerce(
        request.baseline).compact_v2(None)}


def _checker_sentence(spec: LocatorSpec) -> str:
    """What the checker is told the step is, phrased as it was trained.

    A real description, or the broken selector when there is none - and a
    placeholder counts as none. "the field" names nothing and hides the
    selector, the one thing the checker could have read: on 210 paired
    cases it was right 47.1% of the time shown the placeholder and 83.3%
    shown the selector. Built the way build_real_dataset.intent builds it
    - description or bare selector as the key, then the role - so the
    checker is served the sentence it learned from.
    """
    desc = spec.description if _described(spec) else ""
    bare = spec.selector
    for prefix in ("xpath=", "css="):         # normalised; trained without
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    return LocatorSpec(key=desc or bare, selector=spec.selector,
                       description=desc, role=spec.role).intent_sentence()


def _shown_spec(spec: LocatorSpec, baseline: Optional[ElementDescriptor],
                action: str, config) -> Tuple[LocatorSpec, str]:
    """The step as the AI and the checker are shown it, and its new words.

    An undescribed step - none, or a placeholder - is described from its
    baseline when that baseline is a single control (descriptors.
    describe_from_baseline, HealingConfig.auto_describe). Everything else
    is shown as written. Returns (spec, the description added or "").
    """
    if (not getattr(config, "auto_describe", True) or baseline is None
            or _described(spec) or action in _COLLECTION_ACTIONS
            or spec.multiple or not _is_control(baseline)):
        return spec, ""
    words = describe_from_baseline(baseline)
    if not words:
        return spec, ""
    return dataclasses.replace(spec, description=words), words


def spec_from_payload(raw: Dict[str, Any]) -> LocatorSpec:
    return LocatorSpec(
        key=raw.get("key", "locator"),
        selector=raw.get("selector", ""),
        description=raw.get("description", "") or "",
        tag=raw.get("tag"),
        role=raw.get("role"),
        expected_text=raw.get("expected_text"),
        multiple=bool(raw.get("multiple", False)),
    )


@dataclass
class ShortlistEntry:
    """One ranked candidate, as the service saw it."""

    index: int                       # position in the candidate list sent
    score: float
    signals: Dict[str, Optional[float]] = field(default_factory=dict)
    summary: str = ""
    chosen: bool = False


@dataclass
class AdjudicationDecision:
    """What the service decided. Contains no selector - the bot binds that."""

    chosen: int = -1                 # index into the candidates sent, -1 = none
    confidence: float = 0.0
    strategy: str = ""               # heuristic | ai-confirmed | ai-override
    rationale: str = ""
    shortlist: List[ShortlistEntry] = field(default_factory=list)
    ai_consulted: bool = False
    ai_choice: int = -1
    ai_confidence: float = 0.0
    ai_reason: str = ""
    ai_latency_s: float = 0.0
    ai_device: str = ""
    # The fine-tuned checker's calibrated probability that the chosen
    # element is the one the step meant - for a "none", the best any
    # candidate scored - or -1.0 when it was not asked.
    # Not 0.0: that is a real answer ("certainly not this element"), and
    # the first build used it for both, so an operator could not tell a
    # skipped check from a damning one.
    checker: float = -1.0
    # The checker's own pick when it gave the first opinion - an index into
    # the candidates sent, -1 for "none" or when it was not asked. Differs
    # from `chosen` when it and the AI disagreed and the AI's pick went to
    # review, so an operator can see both.
    checker_choice: int = -1
    # How many of the ranker's candidates the checker re-ranked to choose
    # the shortlist (HealingConfig.checker_rerank); 0 when it did not.
    checker_pool: int = 0
    # What the AI and the checker were told the step is, when it had no
    # description of its own and its baseline supplied one ("" otherwise).
    # The ranker is never told; see HealingConfig.auto_describe.
    auto_description: str = ""
    # The calibration group - described, undescribed or placeholder - when
    # `confidence` is the calibrated probability of the AI's pick being
    # right; "" when it is any other number. A target precision bars each
    # group on its own evidence.
    group: str = ""
    duration_s: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AdjudicationDecision":
        entries = [ShortlistEntry(**e) for e in raw.get("shortlist", [])]
        known = {f for f in cls.__dataclass_fields__} - {"shortlist"}
        return cls(shortlist=entries,
                   **{k: v for k, v in raw.items() if k in known})


class LocalAdjudicator:
    """Ranks and adjudicates in-process.

    Owns the two model-backed stages - the sentence encoder and the AI
    adjudicator - and nothing else. It never touches a browser, which is what
    lets it run on a host that has a GPU and no display.
    """

    def __init__(self, config: HealingConfig, model=None, encoder=None):
        self.config = config
        self._model = model
        self._encoder = encoder
        # Asked for at most once per process, and only if the config wants
        # it: the weights are a second download, and a deployment that never
        # enables this must not pay for the attempt on every adjudication.
        self._checker = None
        self._checker_tried = False
        self._calibrator = None
        self._calibrator_tried = False
        self._calibrator_refused = ""
        # Identical requests get identical verdicts, and on CPU a verdict costs
        # ~25 seconds. Workers sharding one job arrive here within seconds of
        # each other holding the same shortlist, so this collapses N of those
        # into one.
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self.cache_hits = 0

    @property
    def checker(self):
        """The fine-tuned checker, or None. Never raises."""
        if self._checker is None and self.config.use_checker \
                and not self._checker_tried:
            self._checker_tried = True
            try:
                self._checker = runtime.make_checker(
                    path=self.config.checker_path,
                    device=self.config.checker_device)
            except Exception as exc:
                log.warning("no checker (%s)", str(exc)[:90])
                self._checker = None
        return self._checker

    @property
    def calibrator(self) -> Optional[Calibrator]:
        """The calibration of the AI's odds, or None. Loaded once.

        Not used for an AI that says it is other weights than the ones it
        was fitted on: a mapping from another model's odds is not a
        probability of this one being right. A model that cannot say -
        one loaded here, unhashed - is taken to be the fitted one, and the
        health report says so. Nor for the same weights asked another
        prompt: those are odds it never saw.
        """
        if not self._calibrator_tried:
            self._calibrator_tried = True
            if getattr(self.config, "use_calibration", True):
                self._calibrator = Calibrator.load(
                    getattr(self.config, "calibration_path", "")
                    or DEFAULT_CALIBRATION)
        cal = self._calibrator
        if cal is None:
            return None
        known = str(getattr(self.model, "weights_sha256", "") or "")
        if known and cal.model_sha256 and known != cal.model_sha256:
            if not self._calibrator_refused:
                log.warning("the calibration was fitted on weights %s, the "
                            "AI is %s - using its raw odds", cal.model_sha256
                            [:12], known[:12])
            self._calibrator_refused = known
            return None
        asked = int(getattr(self.model, "prompt_version", 2) or 2)
        if asked != cal.prompt_version:
            if not self._calibrator_refused:
                log.warning("the calibration was fitted on prompt %d, the "
                            "AI is asked prompt %d - using its raw odds",
                            cal.prompt_version, asked)
            self._calibrator_refused = "prompt %d" % asked
            return None
        return cal

    def _calibrated(self, verdict, shown, action) -> Optional[float]:
        """The calibrated probability that the AI's pick is right, or None."""
        cal = self.calibrator
        if cal is None or not _from_odds(verdict) or verdict.choice < 0:
            return None
        odds = getattr(verdict, "odds", None) or {}
        others = [float(v) for k, v in odds.items()
                  if int(k) != verdict.choice]
        try:
            return cal.probability(
                p=verdict.confidence,
                runner=max(others) if others else max(0.0, 1.0
                                                      - verdict.confidence),
                none=float(odds.get(-1, 0.0)),
                rank=verdict.choice, group=_group(shown),
                collect=action in _COLLECTION_ACTIONS or shown.multiple)
        except Exception as exc:
            log.warning("calibration failed (%s) - using the raw odds",
                        str(exc)[:90])
            return None

    def bar_for(self, decision: "AdjudicationDecision") -> Optional[float]:
        """The bar this decision must clear, or None for the usual one.

        Only set with a target precision (HealingConfig.target_precision).
        A calibrated decision is held to its group's bar for that target -
        and a group with none can only be reviewed. Anything else must
        clear the target itself, on top of the ordinary bar.
        """
        target = float(getattr(self.config, "target_precision", 0.0) or 0.0)
        if target <= 0.0:
            return None
        cal = self.calibrator
        if decision.group and cal is not None:
            bar = cal.bar(target, decision.group)
            return bar if bar is not None else 1.01
        return max(self.config.auto_apply_threshold, target)

    @staticmethod
    def _checkable(spec, action):
        """Is this the kind of step the checker was trained to judge?

        One control, not a collection; and when the authored selector
        names a tag, a control's tag or role. A bare `#id` names neither,
        which is most of what it was trained on, so it is checked.
        """
        if action in _COLLECTION_ACTIONS or getattr(spec, "multiple", False):
            return False
        role = (spec.role or "").lower()
        if role in _CHECKER_ROLES:
            return True
        # The payload rarely carries a tag, so read it from the selector
        # the way the ranker does: `div.card-title` names a div.
        tag = (spec.tag or selector_tag(spec.selector) or "").lower()
        return not tag or tag in _CHECKER_TAGS

    def _check(self, spec, element, remaining, action=""):
        """The checker's verdict on one element, or None.

        Skipped - never an error - outside what it was trained to judge
        (see _checkable), when the description is a placeholder (it
        measured worse than the ranker there: 0.795 against 0.847 gate
        AUC), when the budget cannot fit a call, and when anything at all
        goes wrong. Everything fails open (invariant 11).
        """
        if not self._checkable(spec, action):
            return None
        if not _is_control(element):
            return None                     # not the kind it learned to judge
        desc = spec.description or ""
        if desc and not informative_intent_tokens(desc):
            return None
        checker = self.checker
        if checker is None:
            return None
        expected = getattr(checker, "typical_latency_s", 0.0) or 0.0
        if remaining <= ((expected * 1.25) if expected else 3.0):
            log.info("skipping the checker: %.1fs left", remaining)
            return None
        try:
            return checker.score(spec.intent_sentence(), element.compact(0))
        except Exception as exc:
            log.warning("checker failed (%s)", str(exc)[:90])
            return None

    def recheck(self, request: AdjudicationRequest,
                index: int) -> Optional[float]:
        """The checker's verdict on one candidate a request carries, or None.

        For a repair the fleet store is about to serve. The store answers
        before anything is ranked, so this is the only second look its pick
        gets. Skipped wherever _check skips, and never raises.
        """
        if not self.config.use_checker:
            return None
        try:
            spec = self._shown(request)
            element = ElementDescriptor.coerce(request.candidates[index])
        except Exception as exc:
            log.warning("could not recheck a stored repair (%s)",
                        str(exc)[:90])
            return None
        return self._check(spec, element, request.budget_s, request.action)

    def _shown(self, request: AdjudicationRequest) -> LocatorSpec:
        """The step as the models were shown it for this request."""
        baseline = (ElementDescriptor.coerce(request.baseline)
                    if request.baseline else None)
        return _shown_spec(spec_from_payload(request.locator), baseline,
                           request.action, self.config)[0]

    def _ask_ai(self, spec, request, shortlist, started, decision):
        """The AI's verdict on the shortlist, or (None, why not).

        Asked exactly what the AI-first path asks - the ranker's order and
        the ranker's pick - and never what the checker chose. An AI told
        the first opinion's pick tends to hand it back (invariant 38), and
        a second opinion that only agrees is not a second opinion.
        """
        model = self.model
        if model is None:
            return None, "no AI is configured"
        remaining = request.budget_s - (time.time() - started)
        expected = getattr(model, "typical_latency_s", 0.0) or 0.0
        if remaining <= ((expected * 1.25) if expected else 2.0):
            return None, "there was no time to ask the AI (%.0fs left)" % (
                remaining)
        verdict = self._consult(model, spec, request, shortlist)
        decision.ai_consulted = True
        decision.ai_choice = verdict.choice
        decision.ai_confidence = round(verdict.confidence, 3)
        decision.ai_reason = verdict.reason
        decision.ai_latency_s = round(verdict.latency_s, 2)
        decision.ai_device = verdict.device
        if not verdict.ok or not -1 <= verdict.choice < len(shortlist):
            return None, "the AI gave no usable answer (%s)" % (
                (verdict.error or "out of range").split(":")[0])
        return verdict, ""

    @staticmethod
    def _consult(model, spec, request, shortlist):
        """One AI call, made the same way from either path."""
        line = _ai_line(model)
        kwargs = dict(
            intent=spec.intent_sentence(),
            broken_selector=spec.selector,
            action=request.action,
            page_title=request.page.get("title", ""),
            shortlist=[line(c.descriptor, i)
                       for i, c in enumerate(shortlist)],
            heuristic_pick=0,
            required_text=spec.expected_text,
            **_ai_baseline(model, request),
        )
        if hasattr(model, "explain"):
            # The odds decide; a reason is read only when a person reviews
            # the repair, and costs what a whole reply used to. The service
            # asks for one then (LocalAdjudicator.explain).
            kwargs["explain"] = False
        return model.adjudicate(**kwargs)

    def _rerank(self, spec, request, ranked, started):
        """The shortlist the checker chooses from the ranker's top N.

        Returns (shortlist, probabilities) - the candidates it rates best,
        in the ranker's order, each with its score - or None, and the
        ranker's own top shortlist_size stands: off, a step the checker
        was not trained on, no time, or a failure (invariant 11). Scored a
        shortlist at a time, the size its latency is measured in.
        """
        cfg = self.config
        size = max(1, request.shortlist_size)
        pool = ranked[:max(0, int(getattr(cfg, "checker_rerank", 0) or 0))]
        if not cfg.use_checker or len(pool) <= size:
            return None
        if not self._checkable(spec, request.action) \
                or not _is_control(ranked[0].descriptor):
            return None
        checker = self.checker
        if checker is None or not hasattr(checker, "score_all"):
            return None
        passes = -(-len(pool) // size)
        expected = getattr(checker, "typical_shortlist_latency_s", 0.0) or 0.0
        remaining = request.budget_s - (time.time() - started)
        # The first opinion's estimate of one pass, before any sample.
        if remaining <= passes * ((expected * 1.25) if expected else 8.0):
            log.info("no time for the checker to re-rank %d candidates: "
                     "%.1fs left", len(pool), remaining)
            return None
        sentence, probs = _checker_sentence(spec), []
        try:
            for at in range(0, len(pool), size):
                part = pool[at:at + size]
                got = checker.score_all(sentence, [
                    c.descriptor.compact(k) for k, c in enumerate(part)])
                if not got or len(got) != len(part):
                    return None
                probs.extend(got)
        except Exception as exc:
            log.warning("checker failed re-ranking (%s)", str(exc)[:90])
            return None
        best = sorted(range(len(pool)), key=lambda k: (-probs[k], k))[:size]
        keep = sorted(best)
        return [pool[k] for k in keep], [probs[k] for k in keep]

    def _first_opinion(self, spec, request, shortlist, started, decision,
                       probs=None, lead=None):
        """The checker decides, and asks the AI only when it is unsure.

        Returns (chosen, strategy, confidence, rationale), chosen None for
        "nothing here" - or None when the step is not the checker's to
        decide: off, "gate" mode, outside what it was trained on, no time,
        or it failed. The AI-first path then runs exactly as before, so
        every way out of here fails open (invariant 11). `probs` are the
        shortlist's scores when a re-rank already has them, and `lead` the
        ranker's own best candidate when the shortlist is not its top.
        """
        cfg = self.config
        if not cfg.use_checker or cfg.checker_mode != "first":
            return None
        if not self._checkable(spec, request.action):
            return None
        # The ranker's best guess is the evidence the selector lacks: on
        # the paired cases it was a control in 879 of 1,000 control steps,
        # and a sidebar's is a <div>. See _is_control.
        if not _is_control((lead or shortlist[0]).descriptor):
            return None
        checker = self.checker
        if checker is None or not hasattr(checker, "score_all"):
            return None
        if probs is None:
            expected = getattr(checker, "typical_shortlist_latency_s",
                               0.0) or 0.0
            remaining = request.budget_s - (time.time() - started)
            # Before the first sample, eight rows on a laptop CPU took ~6s.
            if remaining <= ((expected * 1.25) if expected else 8.0):
                log.info("no time for the checker's first opinion: %.1fs "
                         "left", remaining)
                return None
            try:
                probs = checker.score_all(
                    _checker_sentence(spec),
                    [c.descriptor.compact(i)
                     for i, c in enumerate(shortlist)])
            except Exception as exc:
                log.warning("checker failed (%s)", str(exc)[:90])
                probs = None
        if not probs or len(probs) != len(shortlist):
            return None

        bands = cfg.checker_bands["described" if _described(spec)
                                  else "selector"]
        order = sorted(range(len(probs)), key=lambda k: -probs[k])
        top, best = order[0], probs[order[0]]
        runner = probs[order[1]] if len(order) > 1 else 0.0
        mine = top if best >= bands["pick"] else -1
        decision.checker_choice = shortlist[mine].index if mine >= 0 else -1

        # Sure, and alone in being sure: two candidates above the accept
        # bar are a tie (invariant 29) - 8 of 253 paired accepts, 3 of them
        # wrong - and a tie goes to the AI.
        if best >= bands["accept"] and runner < bands["accept"]:
            decision.checker = round(best, 3)
            return (shortlist[top], "checker-sure", best,
                    "the checker is sure (%.2f); the AI was not asked" % best)
        if best < bands["none"]:
            decision.checker = round(best, 3)
            return (None, "checker-none", 0.0,
                    "the checker found no candidate (best %.2f); the AI was "
                    "not asked" % best)

        verdict, why = self._ask_ai(spec, request, shortlist, started,
                                    decision)
        ceiling = cfg.auto_apply_threshold - 0.01

        def review(k):
            # Inside the review band, never above it and never below it:
            # a human sees it, whatever the checker's own number was.
            return min(ceiling, max(cfg.review_threshold, probs[k]))

        if verdict is None:
            # One unsure opinion is not enough to act on alone.
            if mine < 0:
                decision.checker = round(best, 3)
                return (None, "checker-none", 0.0,
                        "the checker leans to none (best %.2f) and %s"
                        % (best, why))
            decision.checker = round(probs[mine], 3)
            return (shortlist[mine], "checker-unconfirmed", review(mine),
                    "the checker is unsure (%.2f) and %s; held for review"
                    % (probs[mine], why))
        theirs = verdict.choice
        if theirs == mine and mine < 0:
            # Both saying "none" was right only 54 of 91 times paired, so
            # it is not trusted further than refusing.
            decision.checker = round(best, 3)
            return (None, "checker+ai-none", 0.0,
                    "neither the checker (best %.2f) nor the AI found the "
                    "element" % best)
        if theirs == mine:
            decision.checker = round(probs[mine], 3)
            sure = max(probs[mine], cfg.first_agree_confidence)
            said = verdict.reason or "no reason given"
            if _from_odds(verdict):
                # Two opinions agreeing are at least as sure as the AI was
                # alone - and the AI's odds are a probability, where 0.90
                # was the measured average over every agreement.
                alone = self._calibrated(verdict, spec, request.action)
                sure = max(sure, verdict.confidence if alone is None
                           else alone)
                said = verdict.reason or "the AI put %.0f%% on it" % (
                    verdict.confidence * 100)
            return (shortlist[mine], "checker+ai-agree", sure,
                    "the checker (%.2f) and the AI agree: %s"
                    % (probs[mine], said))
        # They disagree. When they did on the paired cases the AI was right
        # twice as often (44 to 21 of 100), so a reviewer is shown the AI's
        # pick - or the checker's, when the AI said there was none.
        shown = theirs if theirs >= 0 else mine
        decision.checker = round(probs[shown], 3)
        return (shortlist[shown], "checker+ai-disagree", review(shown),
                "the checker chose %s (%.2f) and the AI %s; held for review"
                % ("[%d]" % mine if mine >= 0 else "none", best,
                   "[%d]" % theirs if theirs >= 0 else "none"))

    def _finish(self, decision, shortlist, started, cache_key, chosen,
                strategy, confidence, rationale):
        """Publish a decision: the choice, its confidence, the shortlist."""
        decision.chosen = chosen.index if chosen is not None else -1
        decision.confidence = round(max(0.0, min(1.0, confidence)), 4)
        decision.strategy = strategy
        decision.rationale = rationale
        # Written as the AI read them: a reason is later asked over these
        # very lines (explain), and must be asked over what it was shown.
        line = _ai_line(self._model)
        decision.shortlist = [
            ShortlistEntry(
                index=c.index,
                score=round(c.score, 4),
                signals={k: (round(v, 3) if v is not None else None)
                         for k, v in c.signals.items()},
                summary=line(c.descriptor, rank),
                chosen=(c is chosen),
            )
            for rank, c in enumerate(shortlist)
        ]
        decision.duration_s = round(time.time() - started, 2)
        self._remember(cache_key, decision)
        return decision

    @staticmethod
    def _cache_key(request: AdjudicationRequest) -> str:
        """Everything that could change the verdict, and nothing else.

        The *whole* locator is hashed, not a chosen few of its fields.
        Every one of them reaches ``spec_from_payload`` and from there the
        ranker: ``role`` and ``tag`` drive tag_signal and expected_tag,
        ``expected_text`` drives reference_text and is stated to the model
        as a hard constraint, ``multiple`` changes what a good repair even
        looks like. Listing fields by hand is how five of them came to be
        left out, and the omission was invisible - two locators differing
        only in ``role`` shared one entry and the cached one picked a
        *different element*. Anything added to LocatorPayload is covered
        from here on without a line changing.
        """
        payload = json.dumps({
            "locator": request.locator,
            "action": request.action,
            "url": (request.page or {}).get("url", ""),
            "candidates": request.candidates,
            # The baseline drives the structure and geometry signals, so two
            # requests that differ only in whether one was available are not
            # the same question. Leaving it out replays a cold verdict at a
            # cold confidence over a warm heal that would have scored higher.
            "baseline": request.baseline,
            # How deep the shortlist is decides which candidates the model
            # is shown and therefore which ones it can choose between.
            "shortlist_size": request.shortlist_size,
            # Each demo session is asked against its own memory, so one
            # session's verdict is never replayed into another.
            "scope": request.scope,
        }, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> Optional["AdjudicationDecision"]:
        ttl = getattr(self.config, "decision_cache_s", 0.0)
        if ttl <= 0:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if not hit or time.time() - hit[0] > ttl:
                return None
            self.cache_hits += 1
            # A copy: a caller that mutates the decision must not poison the
            # next worker to ask the same question.
            return AdjudicationDecision.from_dict(hit[1].to_dict())

    def _remember(self, key: str, decision: "AdjudicationDecision") -> None:
        ttl = getattr(self.config, "decision_cache_s", 0.0)
        if ttl <= 0 or decision.chosen < 0:
            return
        limit = max(1, getattr(self.config, "decision_cache_size", 256))
        # A copy, as _cached hands out copies. The decision returned is the
        # caller's to change - the service turns an optional step's weak
        # repair into "absent" on it - and that must not become the answer
        # every identical request is served for the next five minutes.
        kept = AdjudicationDecision.from_dict(decision.to_dict())
        with self._cache_lock:
            self._cache[key] = (time.time(), kept)
            while len(self._cache) > limit:
                self._cache.pop(next(iter(self._cache)))

    def explain(self, request: AdjudicationRequest,
                decision: "AdjudicationDecision", remaining_s: float) -> str:
        """The AI's reason for the element a decision chose, or "".

        Asked for by the service when a repair goes to review. The odds
        decided it without a reply, and a reply - most of what asking the
        AI costs - is generated only for a decision a person will read.
        Only the AI's own pick is explained: asked about an element it did
        not choose, a model argues for it anyway. Never raises.
        """
        if decision.ai_reason or not decision.ai_consulted:
            return decision.ai_reason
        rank = next((r for r, e in enumerate(decision.shortlist)
                     if e.chosen), -1)
        model = self.model
        if rank < 0 or rank != decision.ai_choice or model is None \
                or not hasattr(model, "explain"):
            return ""
        expected = getattr(model, "typical_explain_s", 0.0) or 0.0
        if remaining_s <= max(2.0, expected * 1.25):
            log.info("no time to write the AI's reason: %.1fs left",
                     remaining_s)
            return ""
        spec = self._shown(request)      # what the AI was asked about
        try:
            reason = model.explain(
                intent=spec.intent_sentence(),
                broken_selector=spec.selector,
                action=request.action,
                page_title=request.page.get("title", ""),
                shortlist=[e.summary for e in decision.shortlist],
                choice=rank,
                heuristic_pick=0,
                required_text=spec.expected_text,
                **_ai_baseline(model, request)) or ""
        except Exception as exc:
            log.warning("the AI's reason failed (%s)", str(exc)[:90])
            return ""
        if reason:
            # So an identical request replayed from the cache - the next
            # worker on the same break - is not asked to pay for it again.
            with self._cache_lock:
                hit = self._cache.get(self._cache_key(request))
                if hit and hit[1].chosen == decision.chosen:
                    hit[1].ai_reason = reason
        return reason

    # ------------------------------------------------------------- lazy deps
    @property
    def encoder(self):
        if self._encoder is None and self.config.use_embeddings:
            self._encoder = runtime.make_encoder(self.config.embedding_model)
        return self._encoder

    @property
    def model(self):
        if self._model is None and self.config.use_ai:
            self._model = runtime.make_model(
                model_path=self.config.model_path,
                gpu_layers=self.config.gpu_layers,
                context=self.config.ai_context,
                max_tokens=self.config.ai_max_tokens,
                reasoning_effort=self.config.reasoning_effort,
                prompt_version=self.config.prompt_version,
            )
        return self._model

    def warmup(self) -> Dict[str, Any]:
        status = {"encoder": False, "ai": False, "device": "none"}
        if self.encoder is not None:
            status["encoder"] = self.encoder.warmup()
        if self.model is not None:
            status["ai"] = self.model.warmup()
            status["device"] = self.model.device
        if self.config.use_checker:
            # Loaded now rather than on the first repair, which would
            # otherwise pay ~20s of loading inside a bot's budget.
            status["checker"] = bool(self.checker and self.checker.warmup())
        return status

    def describe(self) -> Dict[str, Any]:
        out = {"encoder_ready": bool(self.encoder and self.encoder.available)}
        out["ai"] = self.model.describe() if self.model else {"loaded": False}
        if self.config.use_checker:
            out["checker"] = dict(self.checker.describe() if self.checker
                                  else {"loaded": False},
                                  mode=self.config.checker_mode)
        out["cache"] = {"entries": len(self._cache), "hits": self.cache_hits}
        cal = self.calibrator
        target = float(getattr(self.config, "target_precision", 0.0) or 0.0)
        out["calibration"] = {
            "loaded": cal is not None,
            "model_sha256": cal.model_sha256 if cal else "",
            "refused_for": self._calibrator_refused,
            "target_precision": target,
            "bars": ({g: cal.bar(target, g) for g in ("described",
                                                      "undescribed",
                                                      "placeholder")}
                     if cal is not None and target > 0.0 else {}),
        }
        return out

    # -------------------------------------------------------------- decide
    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationDecision:
        """Rank the candidates and pick one. Never raises."""
        started = time.time()
        cache_key = self._cache_key(request)
        cached = self._cached(cache_key)
        if cached is not None:
            log.info("serving an identical adjudication from cache")
            cached.duration_s = round(time.time() - started, 3)
            return cached

        decision = AdjudicationDecision()
        spec = spec_from_payload(request.locator)
        # coerce, not from_dict: the full agent sends serialised descriptors
        # and a thin client sends the harvester's own output. Decoding the
        # second shape with from_dict drops id, type, aria-label, data-* and
        # the accessible name, which is most of what identity and attributes
        # are scored on.
        candidates = [ElementDescriptor.coerce(c) for c in request.candidates]
        if not candidates:
            decision.error = "no candidates supplied"
            decision.duration_s = time.time() - started
            return decision

        baseline = (ElementDescriptor.coerce(request.baseline)
                    if request.baseline else None)
        target = HealTarget(spec=spec, baseline=baseline)
        # The ranker scores the step as it was written. The AI and the
        # checker are shown it described from its baseline when it came
        # with no description (HealingConfig.auto_describe).
        shown, decision.auto_description = _shown_spec(
            spec, baseline, request.action, self.config)

        # The bot has already applied the lexical signals to narrow the field.
        # Re-scoring here is cheap and keeps the service self-contained: it
        # never has to trust a score computed on a machine it does not own.
        semantic = None
        if self.encoder is not None and self.encoder.available:
            semantic = self.encoder.similarities(
                spec.intent_sentence(),
                [c.semantic_sentence() for c in candidates],
            )
        ranked = score_candidates(target, candidates, self.config.weights,
                                  semantic)
        shortlist = ranked[: max(1, request.shortlist_size)]
        probs = None
        reranked = self._rerank(shown, request, ranked, started)
        if reranked is not None:
            shortlist, probs = reranked
            decision.checker_pool = min(len(ranked),
                                        self.config.checker_rerank)

        # ---- the checker's first opinion (invariant 39) ------------------
        first = self._first_opinion(shown, request, shortlist, started,
                                    decision, probs=probs, lead=ranked[0])
        if first is not None:
            return self._finish(decision, shortlist, started, cache_key,
                                *first)

        chosen = shortlist[0]
        strategy = "heuristic"
        confidence = chosen.score * 0.92
        rationale = "top-ranked by similarity (%s)" % chosen.explain()

        # ---- AI adjudication -------------------------------------------
        #
        # budget_s gates whether this pass *starts*. It is not an abort:
        # llama.cpp offers no wall-clock stop through create_chat_completion,
        # so once generation begins it runs to its token limit. That makes
        # the gate the only place the budget is enforceable, and it used to
        # be a bare `remaining > 2` - which on a CPU host, where a call
        # costs 25-32s, let a request start with two seconds left and
        # overrun the caller's budget by more than the budget.
        #
        # So gate on what a call has actually cost on this host. Before the
        # first sample there is nothing to go on and the old constant is
        # the only available answer; after it, the estimate is evidence.
        remaining = request.budget_s - (time.time() - started)
        model = self.model
        expected = getattr(model, "typical_latency_s", 0.0) or 0.0
        # 1.25x: the median is not the worst case, and a call that overruns
        # is worse than one that never ran - the heuristics stand either way.
        needed = (expected * 1.25) if expected else 2.0
        if model is not None and remaining <= needed and expected:
            log.info("skipping the AI pass: %.1fs left, a call on this host "
                     "typically takes %.1fs", remaining, expected)
            rationale = ("AI skipped, %.0fs budget could not fit a %.0fs "
                         "call; %s" % (remaining, expected, rationale))
            strategy = "heuristic-unconfirmed"
            confidence = chosen.score * 0.75
        from_odds = False
        if model is not None and remaining > needed:
            verdict = self._consult(model, shown, request, shortlist)
            decision.ai_consulted = True
            decision.ai_choice = verdict.choice
            decision.ai_confidence = round(verdict.confidence, 3)
            decision.ai_reason = verdict.reason
            decision.ai_latency_s = round(verdict.latency_s, 2)
            decision.ai_device = verdict.device

            if verdict.ok and 0 <= verdict.choice < len(shortlist):
                picked = shortlist[verdict.choice]
                from_odds = _from_odds(verdict)
                w = self.config.ai_weight
                # A model whose confidence never moves contributes a constant
                # to this fusion, not a judgement - at ai_weight 0.35 that is
                # a third of every score, discriminating nothing. Its *choice*
                # still carries information (agreeing with the ranker is real
                # evidence, overriding it is a different claim), so keep the
                # choice and drop the number.
                calibrated = getattr(self.model, "calibrated", True)
                if from_odds:
                    # The probability the model put on this very element,
                    # and the decision's confidence as it stands. The fusion
                    # existed to temper a number that was a habit; over the
                    # odds it only dragged a 99% pick toward the ranker's
                    # cold median of about 0.5, and no bar above that could
                    # ever be reached. On 1,000 held-out cases, picks at
                    # >= 0.95 were 99.2% right (515 applied, 4 wrong).
                    #
                    # Calibrated, it becomes the probability of the pick
                    # being right given the step too (calibration.py): the
                    # raw odds run optimistic below 0.95 and know nothing
                    # of a placeholder or a collection.
                    fused = verdict.confidence
                    calibrated_p = self._calibrated(verdict, shown,
                                                    request.action)
                    if calibrated_p is not None:
                        fused = calibrated_p
                        decision.group = _group(shown)
                elif calibrated:
                    fused = (1.0 - w) * picked.score + w * verdict.confidence
                else:
                    fused = picked.score
                if verdict.choice == 0:
                    # The AI returned the ranker's own first choice. That
                    # used to earn a +0.08 "two methods agree" bonus, on
                    # the reasoning that independent agreement is evidence.
                    # They are not independent: heuristic_pick tells the
                    # model what the ranker chose, and a model with nothing
                    # better to say hands it straight back. Measured over
                    # 179 adjudications on 30 live sites, that is the
                    # *weaker* half - 9.9% precision against 58.6% when the
                    # model moved, and 80% against 100% on the subset where
                    # the right element was genuinely in the shortlist.
                    # So agreement keeps the fused score and nothing more.
                    strategy = "ai-confirmed"
                    confidence = fused
                else:
                    # Moving off the ranker's choice is the model actually
                    # contributing: it had to identify something specific to
                    # do it. That is where the evidence is, so that is where
                    # the bonus goes. The odds already price that in.
                    strategy = "ai-override"
                    confidence = fused if from_odds else min(
                        1.0, fused + self.config.ai_override_bonus)
                chosen = picked
                chosen.llm_bonus = confidence - chosen.score
                chosen.llm_reason = verdict.reason
                said = verdict.reason or "no reason given"
                if from_odds and not verdict.reason:
                    said = "the AI put %.0f%% on this element" % (
                        verdict.confidence * 100)
                    if decision.group:
                        said += ", %.0f%% calibrated" % (confidence * 100)
                rationale = "%s: %s" % (strategy, said)
            elif verdict.ok and verdict.choice < 0:
                # An explicit abstention is information: the model looked and
                # saw nothing suitable. Trust the heuristics less, not more.
                strategy = "heuristic-unconfirmed"
                confidence = chosen.score * 0.75
                rationale = "AI found no suitable candidate; " + rationale
            elif verdict.error:
                rationale = "AI unavailable (%s); %s" % (
                    verdict.error.split(":")[0], rationale)

        decision.chosen = chosen.index
        # A runner-up that scored as well as the winner is ambiguity, and
        # the score has to carry it. Measured against the *chosen* element,
        # not the top-ranked one, so an AI override is judged against what
        # it actually beat. See HealingConfig.ambiguity_discount.
        #
        # Not over the AI's odds. A tie between two near-identical elements
        # is already in the distribution - the model splits its mass - and
        # the ranker's tie is no evidence against a pick the model told
        # apart: the 99.2% at >= 0.95 was measured with no discount at all.
        k = self.config.ambiguity_discount
        if (k > 0.0 and confidence > 0.0 and len(shortlist) > 1
                and not from_odds):
            others = [c.score for c in shortlist if c is not chosen]
            if others:
                margin = max(0.0, chosen.score - max(others))
                rel = max(0.0, min(1.0, margin / chosen.score)) \
                    if chosen.score > 0 else 0.0
                confidence *= (1.0 - k * (1.0 - rel))

        # ---- the fine-tuned checker as a gate -------------------------
        #
        # A gate on auto-apply, and nothing more. It can hold a repair for
        # review; it can never raise a confidence, never change the choice
        # and never refuse outright. In "gate" mode this is all the checker
        # does; in "first" mode it is the safety net for a step that came
        # this way after all - no time for a whole shortlist, or a failed
        # first opinion - and it skips the steps it skips in either mode.
        if self.config.use_checker and confidence > 0.0:
            verdict = self._check(shown, chosen.descriptor, request.budget_s
                                  - (time.time() - started), request.action)
            if verdict is not None:
                decision.checker = round(verdict, 3)
                ceiling = self.config.auto_apply_threshold - 0.01
                if verdict < self.config.checker_floor:
                    # Said even when there is no apply to take away, so a
                    # reviewer looking at a low-confidence repair can see
                    # the checker thought it wrong too.
                    held = confidence > ceiling
                    confidence = min(confidence, ceiling)
                    rationale = ("checker doubts this pick (%.2f)%s; %s"
                                 % (verdict, ", held for review" if held
                                    else "", rationale))

        return self._finish(decision, shortlist, started, cache_key, chosen,
                            strategy, confidence, rationale)


class RemoteAdjudicator:
    """Calls the healing service over HTTP.

    This is what a production bot uses. It has no model, no GPU and no heavy
    dependencies - just the standard library - so rolling the agent out to a
    fleet costs a pip install of Playwright and nothing else.

    Every failure mode returns an empty decision rather than raising, because
    a healing service being down must degrade a bot to "this step failed",
    never to "this bot crashed".
    """

    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._healthy: Optional[bool] = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(self.base_url + "/v1/health",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._healthy = bool(data.get("ready"))
            return data
        except Exception as exc:
            self._healthy = False
            log.warning("healing service unreachable at %s (%s)",
                        self.base_url, exc)
            return {"ready": False, "error": str(exc)}

    def warmup(self) -> Dict[str, Any]:
        info = self.health()
        return {
            "encoder": bool(info.get("encoder_ready")),
            "ai": bool((info.get("ai") or {}).get("loaded")),
            "device": (info.get("ai") or {}).get("device", "remote"),
            "service": self.base_url,
        }

    def describe(self) -> Dict[str, Any]:
        return self.health()

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationDecision:
        payload = json.dumps(request.to_dict()).encode("utf-8")
        started = time.time()
        try:
            req = urllib.request.Request(self.base_url + "/v1/adjudicate",
                                         data=payload, headers=self._headers(),
                                         method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return AdjudicationDecision.from_dict(raw)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            log.error("healing service returned %s: %s", exc.code, body)
            return AdjudicationDecision(
                error="service HTTP %s: %s" % (exc.code, body),
                duration_s=round(time.time() - started, 2))
        except Exception as exc:
            log.error("healing service call failed: %s", exc)
            return AdjudicationDecision(
                error="service unreachable: %s" % exc,
                duration_s=round(time.time() - started, 2))


_ADJUDICATORS: Dict[str, Any] = {}


def build_adjudicator(config: HealingConfig):
    """Pick the adjudicator this process should use, once.

    A service URL in the config means "there is an accelerated host somewhere,
    use it". Without one, this process runs the models itself.

    The result is memoised per target because a local adjudicator owns several
    gigabytes of weights: an agent per browser tab must not mean a copy of the
    model per browser tab.
    """
    key = config.service_url or ("local:" + config.model_path)
    cached = _ADJUDICATORS.get(key)
    if cached is not None:
        return cached
    if config.service_url:
        log.info("using healing service at %s", config.service_url)
        built = RemoteAdjudicator(config.service_url, config.service_api_key,
                                  config.service_timeout_s)
    else:
        log.info("running the adjudicator in-process (%s)", config.model_path)
        built = LocalAdjudicator(config)
    _ADJUDICATORS[key] = built
    return built


def reset_adjudicators() -> None:
    """Drop the cache - for tests, and for reloading swapped-out weights."""
    _ADJUDICATORS.clear()
