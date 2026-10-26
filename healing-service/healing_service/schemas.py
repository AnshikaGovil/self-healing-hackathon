"""Wire schemas for the healing service.

Kept deliberately permissive on the element fingerprints: a bot running an
older agent build must not start failing because the service learned about a
new descriptor field. Everything the service actually needs is required;
everything else is passed through.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LocatorPayload(BaseModel):
    key: str = "locator"
    selector: str = ""
    description: str = ""
    tag: Optional[str] = None
    role: Optional[str] = None
    expected_text: Optional[str] = None
    multiple: bool = False


class PagePayload(BaseModel):
    url: str = ""
    title: str = ""


class ProcessPayload(BaseModel):
    """Which automation process, and which part of it, is asking.

    Derived by the client from its own entry script and call stack, never
    authored by an RPA developer. Invariant 13's rule applies to the wire
    too: anything a team has to be taught is something every team has to be
    taught, and this answers "which process, which part" without asking
    anyone to write a thing.
    """

    name: str = ""
    file: str = ""
    function: str = ""
    line: int = 0
    host: str = ""


class AdjudicateRequest(BaseModel):
    locator: LocatorPayload
    action: str = "interact"
    page: PagePayload = Field(default_factory=PagePayload)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    baseline: Optional[Dict[str, Any]] = None
    use_stored_baseline: bool = False
    """May the service supply a baseline this client did not send?

    Only true for a client with no memory of its own. The full agent keeps
    its own and a `baseline` of None from it means "I have none for this
    locator", which is a fact about its memory, not an invitation to
    substitute the fleet's. The benchmark depends on exactly that: it wipes
    its memory per suite so every row is a cold heal, and a service that
    quietly warmed those requests would make the measurement a lie without
    changing a single line of it."""

    use_stored_repair: bool = False
    """May the service answer from its repair store instead of adjudicating?

    Opt-in for exactly the reason use_stored_baseline is. The benchmark runs
    the full agent and wipes its memory per suite so that every row is a
    cold heal; a service that quietly served it a remembered repair would
    make that measurement a lie without changing a line of it. Only bis.py
    sets this."""

    retry: bool = False
    """Skip the repair store for this request.

    Set by a client whose bind of a stored repair failed, so the AI runs
    instead of the store handing back the answer that just did not work."""

    failed_repair_id: int = 0
    """The repair that failed to bind, to be counted against. Only
    meaningful with retry."""

    process: Optional[ProcessPayload] = None
    """Sent on every adjudicate call, but not read on this path.

    Provenance is filed entirely from ``/v1/baseline`` - a repair is only
    ever recorded there, against the report of what worked. This copy costs
    one stack walk on the client and is carried through for logging and
    whatever future use needs to know who was asking."""

    optional: bool = False
    shortlist_size: int = 8
    budget_s: float = 25.0
    protocol: str = "1"


class BaselineRequest(BaseModel):
    """A fingerprint of an element a step has just used successfully.

    ``descriptor`` is deliberately untyped. It is whatever the client had:
    the harvester's own output from a thin client, a serialised
    ElementDescriptor from the full agent. The service coerces one shape
    into the other rather than making every client learn the difference.
    """

    key: str = "locator"
    selector: str = ""
    replaced: str = ""
    """The authored selector this one repaired.

    Empty means the authored selector worked and this report is a baseline,
    not a repair - which is what the full agent always sends."""

    repair_id: int = 0
    """Set when the client used a stored repair, so it is reinforced rather
    than recorded a second time."""

    confidence: float = 0.0
    """The confidence the service gave the repair this report is about.

    Stored with the repair, because a fleet hit is banded on it. It used to
    be 1.0 for every repair, which only ever meant "the selector bound" -
    not that it bound the right element. 0 means a client that predates
    the field, and such a repair is stored at 1.0 as it always was."""

    approved: bool = False
    """A reviewer approved this repair (bis.configure(review_hook=...)).

    Stored at 1.0: a person looked at it. Approving a stored repair raises
    it there too, so the rest of the fleet is not asked the same question
    again."""

    url: str = ""
    descriptor: Dict[str, Any] = Field(default_factory=dict)
    process: Optional[ProcessPayload] = None
    protocol: str = "1"


class ReviewRequest(BaseModel):
    """A reviewer's verdict on a repair the service held for review.

    Posted by bis.py after its review hook decides. Carries what the models
    were shown - the shortlist lines in their order and which one was
    offered - so the verdict is a label a calibration can be fitted to.
    """

    key: str = "locator"
    selector: str = ""
    """The authored selector that broke."""
    proposal: str = ""
    """The selector the reviewer saw bound to the page."""
    approved: bool = False
    url: str = ""
    title: str = ""
    action: str = "interact"
    description: str = ""
    confidence: float = 0.0
    strategy: str = ""
    rationale: str = ""
    ai_reason: str = ""
    shortlist: List[str] = Field(default_factory=list)
    chosen_rank: int = -1
    repair_id: int = 0
    """Set when the fleet store served the repair. A rejection retires it."""
    process: Optional[ProcessPayload] = None
    protocol: str = "1"


class OutcomeRequest(BaseModel):
    """Where the step after this one found itself, posted by bis.py.

    From a good run - the locator's own selector worked - it teaches the
    service what this step leads to. From a healed run it is checked
    against that, and a repair that leads somewhere else is counted
    against, as a failed bind is.
    """

    key: str = "locator"
    url: str = ""
    """The page this step ran on."""
    selector: str = ""
    """The selector the step was bound by - the repair, when healed."""
    replaced: str = ""
    """The authored selector, when the step was healed."""
    healed: bool = False
    next_url: str = ""
    """The page the next step found its element on."""
    process: Optional[ProcessPayload] = None
    protocol: str = "1"


class OutcomeResponse(BaseModel):
    verdict: str = ""
    """noted (a good run), agreed, mismatch, or unknown (too few good runs
    to compare with)."""
    demoted: bool = False


class ReviewResponse(BaseModel):
    stored: bool = False
    labels: int = 0
    """How many review verdicts are kept."""


class BaselineResponse(BaseModel):
    stored: bool = False
    reason: str = ""
    entries: int = 0
    repair_id: int = 0
    """The repair this report created or reinforced, or 0 if it was a
    baseline only."""


class ShortlistItem(BaseModel):
    index: int
    score: float
    signals: Dict[str, Optional[float]] = Field(default_factory=dict)
    summary: str = ""
    chosen: bool = False


class AdjudicateResponse(BaseModel):
    chosen: int = -1
    confidence: float = 0.0
    # Which band the confidence falls in, per HealingConfig.decide(). The
    # full agent computes this itself from the same policy; a thin client
    # has no config of its own, so the service states it. Without it such a
    # client applies every repair it is handed, including refused ones.
    band: str = "refuse"
    strategy: str = ""
    repair_id: int = 0
    """Set when this verdict came from the repair store. The client echoes
    it back on success and on failure, which is the only way the service
    learns whether what it served actually bound."""

    rationale: str = ""
    shortlist: List[ShortlistItem] = Field(default_factory=list)
    ai_consulted: bool = False
    ai_choice: int = -1
    ai_confidence: float = 0.0
    ai_reason: str = ""
    ai_latency_s: float = 0.0
    ai_device: str = ""
    # The fine-tuned checker's calibrated probability for the chosen
    # element (for a "none", the best any candidate scored), or -1.0 when
    # it was not asked (off, out of scope, no budget, or it failed). 0.0
    # is a real answer. Reported so an operator can see why a repair was
    # decided as it was; no client is required to read it.
    checker: float = -1.0
    # The checker's own pick when it gave the first opinion (an index into
    # the candidates sent), -1 for "none" or when it was not asked. Not
    # `chosen` when it and the AI disagreed and the AI's pick went to
    # review - both are reported so a reviewer can weigh them.
    checker_choice: int = -1
    # How many of the ranker's candidates the checker re-ranked to choose
    # the shortlist (--checker-rerank); 0 when the shortlist is the
    # ranker's own top.
    checker_pool: int = 0
    # What the AI and the checker were told the step is when it had no
    # description of its own: one built from its baseline, phrased as their
    # training set phrased it. "" when the step's own words were used.
    auto_description: str = ""
    # described / undescribed / placeholder when `confidence` is the
    # calibrated probability of the AI's pick being right, "" otherwise. A
    # target precision holds each group to its own bar.
    group: str = ""
    duration_s: float = 0.0
    error: str = ""


class AgentStartRequest(BaseModel):
    """A step the ReFlex Agent is asked to repair (invariant 51).

    The same facts /v1/adjudicate is sent, less the candidates: the agent
    asks for a scan when it wants one, so a repair it settles from memory
    or the presence check never pays for a harvest.
    """

    locator: LocatorPayload
    action: str = "interact"
    page: PagePayload = Field(default_factory=PagePayload)
    baseline: Optional[Dict[str, Any]] = None
    use_stored_baseline: bool = False
    use_stored_repair: bool = False
    optional: bool = False
    review_hook: bool = False
    """Is a person there to decide a repair held for review? An escalation
    of a step never seen working, with nobody to review it, cannot change
    the outcome and is not run."""

    escalated: bool = False
    """The pipeline ran first and found nothing it could apply."""

    budget_s: float = 25.0
    process: Optional[ProcessPayload] = None
    protocol: str = "1"


class AgentResult(BaseModel):
    """What one page tool returned in the robot's browser."""

    id: str = ""
    tool: str = ""
    ok: bool = False
    error: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    """A scan's {candidates, harvest, scanned}, the presence check's
    {present, reason, fragments}, Selector Options' {options}."""


class AgentStepRequest(BaseModel):
    session: str = ""
    results: List[AgentResult] = Field(default_factory=list)
    protocol: str = "1"


class AgentCall(BaseModel):
    """A page tool the robot is asked to run."""

    id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """What to do next: run ``calls``, or the repair is ``done``.

    Done with ``fallback`` means the agent could not finish, and the robot
    heals the step the old way, as if it had never asked. Done with a
    ``decision`` is a verdict shaped as /v1/adjudicate's - an index into
    the latest scan and the policy's band, never a selector.
    """

    session: str = ""
    calls: List[AgentCall] = Field(default_factory=list)
    done: bool = False
    fallback: bool = False
    reason: str = ""
    decision: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    """Every call so far: its tool, the planner's why, a one-line result,
    where it ran and how long it took."""


class JsBundleResponse(BaseModel):
    """The in-page JavaScript a thin client executes on the bot.

    Served rather than embedded so that improving the harvester does not mean
    redeploying every bot in the fleet. ``version`` is a content hash: a client
    caches the bundle against it and re-fetches only when it changes.
    """

    protocol: str = "1"
    version: str = ""
    harvest: str = ""
    probe: str = ""
    describe_one: str = ""
    presence: str = ""
    same: str = ""
    """Is the element a repair bound the one the service chose? Asked of
    the elements the harvest kept in the page. An older client ignores it."""


class HealthResponse(BaseModel):
    ready: bool
    service: str
    version: str
    protocol: str
    encoder_ready: bool = False
    ai: Dict[str, Any] = Field(default_factory=dict)
    checker: Dict[str, Any] = Field(default_factory=dict)
    """Empty when --checker is off. When on, "loaded" false with an "error"
    is the symptom of a host missing the laya/peft packages or the
    weights in models/laya-checker/ - repairs are then decided without it."""

    calibration: Dict[str, Any] = Field(default_factory=dict)
    """Whether the AI's odds are calibrated, for which weights, and - with a
    target precision - each group's bar. "refused_for" names an AI whose
    weights are not the fitted ones; its raw odds are used."""

    uptime_s: float = 0.0
    requests_served: int = 0
    baselines: int = 0
    """How many locator baselines are held for thin clients. Zero on a fresh
    service is expected; zero after a fleet has been running is the symptom
    of clients that are not reporting them."""

    repairs: int = 0
    """How many active fleet repairs are held. Zero on a fresh service is
    expected; zero after a fleet has been running is the symptom of clients
    that are not proving repairs back into the store."""

    reviews: int = 0
    """How many reviewers' verdicts are kept as labels. Zero is normal for
    a fleet with no review hook: then review means not found, as before."""

    agent: Dict[str, Any] = Field(default_factory=dict)
    """The ReFlex Agent: whether a planner is configured, where it is (an
    address or a file name - never its key), and the calls one repair may
    make. Not available means a robot asking for it is healed the old
    way."""


class BackendEntry(BaseModel):
    """One remote AI backend, as a GPU host registers itself - a notebook
    whose tunnel address changes every session, a cloud switched on for a
    demo. Lower priority is asked first; the configured ones are 0, 1..."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    url: str = Field(pattern=r"^https?://\S{3,400}$")
    key: str = Field(default="", max_length=400)
    priority: int = Field(default=10, ge=0, le=1000)
