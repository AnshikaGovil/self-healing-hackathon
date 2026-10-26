"""Runtime configuration for the healing engine.

Every knob a delivery team is likely to tune in production lives here so the
engine itself stays free of magic numbers.  Values can be overridden from a
JSON file (``config.json``) or per-field in code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, Optional


@dataclass
class ScoringWeights:
    """Relative weight of each similarity signal in the candidate ranker.

    The weights are normalised at scoring time, so they express *relative*
    importance rather than absolute magnitudes.
    """

    identity: float = 0.24     # survival of the exact normalised id/class name
    tag: float = 0.08          # tag / ARIA-role compatibility
    attributes: float = 0.18   # id, name, class, data-* token overlap
    text: float = 0.18         # visible + accessible text similarity
    structure: float = 0.10    # DOM path and ancestor-chain similarity
    geometry: float = 0.08     # on-screen position and size (needs a baseline)
    neighbor: float = 0.10     # surrounding page copy (needs a baseline)
    semantic: float = 0.14     # embedding cosine vs the business intent


@dataclass
class HealingConfig:
    """Top-level configuration object."""

    # This object is the configuration of the *adjudicating* half - the
    # ranker, the policy and the two model-backed stages. It used to carry
    # another fifteen fields describing browser-side behaviour: harvest
    # caches, rename maps, action timeouts, the memory path, the message
    # boxes. Every one of them was read by the agent package, and when that
    # package went so did their only reader, leaving knobs that a delivery
    # team could tune all day without changing anything. Config that does
    # nothing is worse than config that is missing, because it reads as a
    # supported control. They are recoverable from git if that half returns.

    shortlist_size: int = 8
    """Number of top-ranked candidates handed to the LLM adjudicator."""

    # --- Decision policy ----------------------------------------------------
    auto_apply_threshold: float = 0.62
    """Confidence at/above which a repair is applied autonomously."""

    review_threshold: float = 0.42
    """Confidence at/above which a repair is offered to a human reviewer.
    Below this the engine refuses to guess and the step fails loudly."""

    # ------------------------------------------------------------------ policy
    APPLY = "apply"
    REVIEW = "review"
    REFUSE = "refuse"

    # An optional locator carries the opposite risk to a required one.
    # If a required element is refused the bot stops, loudly, and someone
    # looks at it. If an optional one is repaired wrongly the bot clicks
    # something nobody asked it to click, and no exception is ever raised
    # to say so. Absence is also the ordinary case for these - a consent
    # banner is missing far more often than it is renamed - so the bar to
    # act on one is deliberately higher than the auto-apply band.
    optional_min_confidence: float = 0.85

    target_precision: float = 0.0
    """The precision auto-applied repairs are held to, or 0 for none.

    Set (`--target-precision 0.95`), a calibrated AI decision applies when
    it clears its group's bar in the calibration - the one at which the
    95% lower confidence bound on precision reached the target on the
    fitting data - and a group with no such bar goes to review. Anything
    uncalibrated must clear the target itself, as well as
    auto_apply_threshold. 0 keeps auto_apply_threshold alone, as before."""

    def decide_optional(self, confidence: float, proven: bool = True,
                        bar: Optional[float] = None) -> str:
        """The band for a locator the developer marked as optional.

        Layered on top of ``decide``, never around it: a repair must clear
        the normal apply band *and* this stricter floor. Skipping an
        optional step costs a pop-up that stays open; clicking the wrong
        element can cost a great deal more.
        """
        if (self.decide(confidence, proven, bar) == self.APPLY
                and confidence >= self.optional_min_confidence):
            return self.APPLY
        return self.REFUSE

    def decide(self, confidence: float, proven: bool = True,
               bar: Optional[float] = None) -> str:
        """Which band a repair falls into.

        Defined once and used by both the engine (which reports the band) and
        the agent (which enforces it), so the two can never disagree about what
        the thresholds mean.

        ``proven`` is False when the locator has no baseline earned by its own
        selector - it has never been observed working, so this is a first
        guess rather than a repair. Such a heal never auto-applies, at any
        confidence, and no threshold is a substitute for that. Measured over
        3,760 breakages on 82 live sites:

            cold, applied : 863   of which 402 bound the WRONG element (47%)
            correct picks : mean confidence 0.586
            wrong picks   : mean confidence 0.609

        Confidence is *anti-correlated* with correctness here, so raising the
        bar makes it worse, not better - precision falls from 53% at 0.62 to
        32% at 0.85. The only honest answer is to send it to a human.

        ``bar`` replaces auto_apply_threshold for this one decision: the
        bar a target precision sets for its group (see target_precision).
        """
        if not proven:
            return self.REVIEW if confidence >= self.review_threshold else self.REFUSE
        if confidence >= (self.auto_apply_threshold if bar is None else bar):
            return self.APPLY
        if confidence >= self.review_threshold:
            return self.REVIEW
        return self.REFUSE

    # --- Signals ------------------------------------------------------------
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- AI adjudication ----------------------------------------------------
    use_ai: bool = True

    auto_describe: bool = True
    """Describe an undescribed step to the models from its baseline.

    The AI was right on 96.5% of 400 held-out steps with a description and
    89.0% on the same steps without one (p = 1.5e-5). A step whose element
    was fingerprinted while it worked already says what it is - "the Email
    address box" - so the AI and the checker are told that, phrased as
    their training set phrased it (descriptors.describe_from_baseline). The
    ranker never is: it reads the baseline's label already, and invariant 8
    is why extra words in its lexical bag are not free. Only a single
    control is described; a collection is never named after one member.

    The 7.5 points are an upper bound - labels never changed in those test
    pages, and a real upgrade can reword one."""

    use_calibration: bool = True
    calibration_path: str = ""
    """The fitted calibration of the AI's token odds (bis_core/
    calibration.py); empty means the one shipped beside it. It belongs to
    one set of AI weights and is not used for others. Off, or missing, the
    raw odds are the confidence, as they were before it existed."""

    ambiguity_discount: float = 0.15
    """How hard to discount a repair whose runner-up scored just as well.

    A class name is not unique. Twenty elements are the same component, the
    identity signal gives them all 1.0, and without a baseline there is
    nothing left to separate them - which is why class-based locators
    recover 22-25% cold against 62-64% for id-based ones. Finding the right
    one there is an inherent limit. Reporting certainty about it is a bug:
    the winner was returning full confidence while the runner-up scored the
    same, so the bands were reading a coin flip as a decision.

    Measured over 4,664 breakages on 102 live sites, on the warm path:

        k=0.00   1864 applied, 83 bound the wrong element
        k=0.15   1800 applied, 55 bound the wrong element
        k=0.30   1369 applied, 12 bound the wrong element

    0.15 trades 36 correct auto-applies for 28 fewer wrong ones, about
    1.3 to 1. 0.30 costs six correct for every wrong one prevented, which
    is the wrong side of the trade for a step a human can simply approve.
    Set to 0.0 to score as before.
    """

    # What an AI override is worth on top of the fused score. An override
    # is the model declining the ranker's answer and naming another, which
    # measured 58.6% precision against 9.9% for handing the ranker's own
    # choice back. Small on purpose: it lifts a genuine override toward
    # the apply band without letting the AI carry a step on its own.
    ai_override_bonus: float = 0.06

    # The fine-tuned Laya checker returns a calibrated probability that an
    # element is the one the step meant. On 4,000 shortlists from 276
    # sites held out of training it separates a right pick from a wrong
    # one at 0.982 gate AUC (real description) and 0.974 (no description),
    # against 0.77 / 0.80 for the ranker's own score. Off by default:
    # about a gigabyte of weights, and on a laptop CPU ~6s to score a
    # shortlist of eight (0.3s on a GPU) - the deployment's call to make.
    use_checker: bool = False
    checker_path: str = "models/laya-checker"
    checker_device: str = ""
    # Below this the checker thinks the pick more likely wrong than right
    # (its output is calibrated on held-out dev sites, ECE ~0.02), and the
    # repair is held for review instead of applied. The CHOICE stands: a
    # reviewer is shown the same element either way. Read in "gate" mode.
    checker_floor: float = 0.50

    checker_mode: str = "first"
    """What the checker does when it is on: "first" or "gate".

    "first" (invariant 39): the checker scores the whole shortlist and
    decides on its own when it is sure, and the AI is asked only when it
    is not. "gate": the AI decides and the checker can only hold its pick
    for review - the behaviour before "first" was measured.

    Paired on the 1,000 held-out cases the bake-off used, same answers
    from the same fine-tuned AI throughout:

        design                        right  wrong apply  review  AI calls
        AI alone                      86.7%      7.9%       0%     100%
        gate (AI first, checker)      76.7%      3.5%     14.4%    100%
        first, review on disagreeing  81.3%      2.8%     10.6%     62%

    "first" beat "gate" on every column; more right at p < 0.001. Steps
    outside what the checker was trained on - a collection, a selector
    naming a non-control tag, or a shortlist whose best candidate is not
    a control - take the AI-first path in either mode."""

    checker_bands: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "described": {"pick": 0.08, "accept": 0.95, "none": 0.07},
        "selector": {"pick": 0.19, "accept": 0.76, "none": 0.05},
    })
    """Where the checker decides alone, per kind of step description.

    "described" is a step with a real description; "selector" is one with
    none, or with a placeholder, which the checker is shown as the broken
    selector instead (a placeholder names nothing and hides the selector:
    47.1% right as given, 83.3% as the selector, on the 210 paired cases).

        accept  its best candidate is taken without asking the AI when it
                scores at least this and the runner-up does not
        none    nothing is taken, without asking the AI, when its best
                candidate scores below this
        pick    in between the AI is asked, and the checker's own pick
                is its best candidate if it scores at least this, else
                "none" - which is what the AI's answer is compared with

    Chosen on the checker's dev set - never on the cases that measured
    them - so that accepts and nones decided alone were each >= 97%
    right there. They belong to one set of weights (dev AUC 0.9712, the
    checker trained to update 13,033) and must be chosen again when the
    weights change: a different checker is calibrated differently."""

    checker_rerank: int = 0
    """How many of the ranker's best the checker re-ranks; 0 is off.

    Cold, the ranker is the bottleneck: the right element reached its top
    eight 63-75% of the time over live sites, and nothing after it can
    pick what it never sees. On the 2026-10-22 smoke harvest it was in the
    top 48 every time. Set above the shortlist size, the checker scores
    the ranker's top N - reading each candidate on its own, so a score
    does not depend on the list - and the shortlist becomes the ones it
    rates best, in the ranker's order: an AI shown the first opinion's
    answer first hands it back (invariant 38). Only for a step the
    checker would judge, and only when the time a pass has taken here,
    times the passes needed, fits the budget - so on a CPU, at 6-8s for
    eight, it rarely starts. Its effect on the bands is not measured yet:
    they were chosen on the ranker's shortlists."""

    first_agree_confidence: float = 0.90
    """The confidence a repair gets when the checker and the AI pick the
    same element in the checker's unsure band. Measured: 324 of 348 such
    agreements were right (93.1%). Both saying "none" is much weaker - 54
    of 91 (59.3%) - so that refuses rather than applying anything."""

    ai_weight: float = 0.35
    """How much the AI verdict is allowed to move the final confidence.
    The deterministic scorer keeps the remaining (1 - ai_weight) share, so a
    hallucinating model can never single-handedly authorise a bad repair."""

    model_path: str = "models/healing-adjudicator.gguf"
    """Open-weight model file, shipped inside the project. Only ever loaded by
    whichever process actually adjudicates - the service, or a bot running
    standalone."""

    gpu_layers: int = -1
    """Layers to offload to the accelerator. -1 offloads the whole model;
    0 forces CPU. Ignored when the installed runtime has no GPU support."""

    ai_context: int = 8192
    ai_max_tokens: int = 160

    reasoning_effort: str = ""
    """Set this when the model file is a *reasoning* model.

    A reasoning model is trained to think before it answers, and emits that
    thinking as tokens. This decision path constrains the reply to a small
    JSON object, so those tokens have nowhere to go: they either get
    suppressed - wasting the training the weights were selected for - or
    they consume ``ai_max_tokens`` and the object never closes. Where the
    format supports it (gpt-oss reads "Reasoning: low"), asking for the
    least reasoning is what keeps a one-index verdict cheap.

    Empty means say nothing, which is right for an instruct-tuned file."""

    prompt_version: int = 2
    """Which prompt the model file was fine-tuned on (build_prompt).

    2 is every fine-tune so far: candidates as compact() writes them and
    two lines naming the ranker's first choice. 3 drops those lines, writes
    candidates with compact_v2() and shows the baseline as LAST KNOWN
    GOOD. Weights read the prompt they learned, so this follows the file:
    a version 3 fine-tune served on 2 - or the reverse - is answering a
    question it was never taught. The checker is unaffected."""

    decision_cache_s: float = 300.0
    """How long an adjudication may be reused for an identical request.

    A sharded job puts every worker on the same broken locator within seconds
    of each other. The candidate list they harvest is identical, so the verdict
    is identical - and at CPU latency, paying for it once instead of once per
    worker is the difference between a fleet stalling for 25s and stalling for
    25s divided by the number of workers. Set to 0 to disable."""

    decision_cache_size: int = 256

    # --- Remote healing service ---------------------------------------------
    service_url: str = ""
    """Base URL of the healing service, e.g. http://gpu-host:8077.

    Set it and this process becomes a thin client: it harvests the DOM, does
    the model-free lexical ranking, and sends only the surviving shortlist for
    scoring. Leave it empty to run the models in-process."""

    service_api_key: str = ""
    service_timeout_s: float = 90.0

    # --- The ReFlex Agent ---------------------------------------------------
    agent_url: str = ""
    """The agent's planner: an OpenAI-compatible chat server, e.g. a
    llama.cpp server at http://gpu-host:8080 (invariant 51).

    A stock instruct model with tool calling is handed the repair tools and
    decides which to call. Empty, with no agent_model_path either, means no
    agent: every repair takes the pipeline, exactly as before. The key comes
    from BIS_AGENT_KEY, kept off the command line."""

    agent_key: str = ""

    agent_model_path: str = ""
    """A planner file loaded on this host instead of a server. A small card
    rarely holds it beside the AI, so the server is the usual way."""

    agent_max_calls: int = 8
    """Tool calls one repair may make before it falls back to the pipeline."""

    agent_session_s: float = 300.0
    """How long a repair waits for the robot to answer a page tool."""

    agent_max_tokens: int = 256
    """Room for one tool call and a few words before it; the reply is a
    call, not prose."""

    agent_timeout_s: float = 12.0
    """One planner call's limit. One /v1/agent/step can hold several calls,
    and bis.py waits 30 s for the service: a hung planner has to fail inside
    that, so the robot hears "fall back" rather than a timeout."""

    # ------------------------------------------------------------------ util
    @classmethod
    def load(cls, path: str | None = None) -> "HealingConfig":
        """Load config from JSON, falling back to defaults for missing keys."""
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
        weights = ScoringWeights(**raw.pop("weights", {}) or {})
        known = {f.name for f in fields(cls)} - {"weights"}
        clean = {k: v for k, v in raw.items() if k in known}
        return cls(weights=weights, **clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
