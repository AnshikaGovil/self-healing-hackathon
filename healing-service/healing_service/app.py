"""FastAPI application for the healing service.

Why this exists
---------------
Locator repair needs two things that do not belong on the same machine: a
browser (which every bot has) and an accelerator (which almost none of them
should have to). Putting the model-backed half behind an API means one GPU
host serves a whole fleet, the weights are provisioned and versioned in one
place, and a bot needs nothing heavier than Playwright to benefit from it.

What actually crosses the wire
------------------------------
Element fingerprints for the shortlist the bot already believes in - tag,
attributes, label text, geometry, ancestor chain. Not the page, not the DOM,
not a screenshot, and not the values the bot is typing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from bis_core.adjudicator import (PROTOCOL_VERSION, AdjudicationDecision,
                                  AdjudicationRequest, LocalAdjudicator)
from bis_core.config import HealingConfig
from bis_core.descriptors import page_scope
from bis_core.js_harvest import (DESCRIBE_ONE_JS, HARVEST_JS, PRESENCE_JS,
                                 PROBE_JS, SAME_JS)
from .baselines import DEFAULT_PATH as DEFAULT_BASELINE_PATH, BaselineStore
from .reflex_agent import DEFAULT_TRACE_PATH, TOOLS, ReflexAgent
from .repairs import DEFAULT_PATH as DEFAULT_REPAIR_PATH, RepairStore
from .reviews import DEFAULT_PATH as DEFAULT_REVIEW_PATH, ReviewStore
from . import sessions
from .inference import parse_backends
from .sessions import SessionMiddleware, SessionStores
from .schemas import (AdjudicateRequest, AdjudicateResponse, AgentResponse,
                      AgentStartRequest, AgentStepRequest, BackendEntry,
                      BaselineRequest, BaselineResponse, HealthResponse,
                      JsBundleResponse, OutcomeRequest, OutcomeResponse,
                      ReviewRequest, ReviewResponse)

log = logging.getLogger("bis.service")

# Long enough for a cloud GPU that sleeps between calls to boot and load the
# weights (30-60 s measured); /v1/health says "waking" meanwhile.
WAKE_TIMEOUT_S = 90.0


class RateLimit:
    """Calls a minute, per caller and overall, over a sliding window.

    The public address sits in a public pull request and a call can wake a
    paid GPU, so the address has to be able to say "later". ``check``
    returns 0 to let a call through, else the seconds until it would be.
    """

    def __init__(self, per_caller: int, total: int, window_s: float = 60.0,
                 clock=time.time):
        self.per_caller, self.total = per_caller, total
        self.window_s, self._clock = window_s, clock
        self._lock = threading.Lock()
        self._all: deque = deque()
        self._by: Dict[str, deque] = {}

    def check(self, caller: str) -> float:
        now = self._clock()
        cut = now - self.window_s
        with self._lock:
            for q in [self._all] + list(self._by.values()):
                while q and q[0] <= cut:
                    q.popleft()
            # Callers seen once and not since would otherwise pile up.
            for name in [k for k, q in self._by.items() if not q]:
                del self._by[name]
            mine = self._by.setdefault(caller, deque())
            for q, limit in ((mine, self.per_caller), (self._all, self.total)):
                if limit and len(q) >= limit:
                    return max(1.0, q[0] + self.window_s - now)
            mine.append(now)
            self._all.append(now)
            return 0.0


def _caller(request: Request) -> str:
    """The caller's address: the first hop a proxy recorded, else the peer.
    A hosted Space sits behind a proxy, so the peer is always the proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or (request.client.host if request.client else "unknown")

def _js_bundle() -> dict:
    """The JS payload plus a content hash clients cache against."""
    body = {"harvest": HARVEST_JS, "probe": PROBE_JS,
            "describe_one": DESCRIBE_ONE_JS, "presence": PRESENCE_JS,
            "same": SAME_JS}
    digest = hashlib.sha256()
    for name in sorted(body):
        digest.update(name.encode("utf-8"))
        digest.update(body[name].encode("utf-8"))
    body["version"] = digest.hexdigest()[:16]
    return body


JS_BUNDLE = _js_bundle()

SERVICE_NAME = "bot-immune-system-healing"
SERVICE_VERSION = "1.1.0"

# Good runs a step's outcome needs behind it before a healed run is judged
# against them. One is an anecdote; a step with a branch after it has more
# than one right answer, and needs the runs to show them.
OUTCOME_SETTLED = 2

# The most candidates one /v1/adjudicate call may carry. The in-page
# harvester caps itself at 900, so this is headroom over what any client of
# ours can send, and a bound on what an unauthenticated caller can make this
# host compute.
MAX_CANDIDATES = 1200


class ServiceState:
    """Process-wide state: one loaded model, shared by every request."""

    def __init__(self, config: HealingConfig, api_key: str = "",
                 baseline_path: str = DEFAULT_BASELINE_PATH,
                 repair_path: str = DEFAULT_REPAIR_PATH,
                 review_path: str = DEFAULT_REVIEW_PATH,
                 agent_trace_path: str = DEFAULT_TRACE_PATH):
        self.config = config
        self.api_key = api_key
        self.adjudicator = LocalAdjudicator(config)
        # Baselines for clients that have no memory of their own. The full
        # agent keeps its own and sends it inline; this is what stops a thin
        # client from scoring every heal, forever, with structure and
        # geometry blank.
        self.shared_baselines = BaselineStore(baseline_path)
        # Repairs the fleet has proven. A thin client has no memory, so
        # without this every bot pays the adjudication cost again for a
        # break another bot already solved.
        self.shared_repairs = RepairStore(repair_path)
        # A demo run that names a session keeps both of the above apart from
        # everyone else's, beside the shared ones.
        self.sessions = SessionStores(os.path.join(
            os.path.dirname(baseline_path), "sessions"))
        # What reviewers decided about repairs held for review - the labels
        # a recalibration on production needs.
        self.reviews = ReviewStore(review_path)
        # Where each ReFlex Agent repair is written down: every call, its
        # why, its result and the decision, one JSON line per repair.
        self.agent_trace_path = agent_trace_path
        # The agent itself (invariant 51). Built whether or not a planner
        # is configured: without one, every repair it is asked for falls
        # back to the pipeline, and the health report says so.
        self.agent = ReflexAgent(self, max_candidates=MAX_CANDIDATES)
        self.started_at = time.time()
        self.requests = 0
        self.warm = {}
        # Remote AI backends: the configured ones (BIS_AI_URL, kept as they
        # were at start) and the ones registered since, by name.
        self.admin_key = ""
        self.env_backends = []
        self.registry: Dict[str, Dict] = {}
        self.backends_lock = threading.Lock()
        self.rate = RateLimit(0, 0)
        # One wake at a time; its result stands for a cooldown, so a GPU
        # that is down is not woken again on every health check.
        self.wake_lock = threading.Lock()
        self.waking = False
        self.wake_ok = False
        self.wake_done_at = 0.0

    @property
    def baselines(self) -> BaselineStore:
        """The baselines of the session this request names, else shared."""
        pair = self._session_stores()
        return pair[0] if pair else self.shared_baselines

    @property
    def repairs(self) -> RepairStore:
        """The repairs of the session this request names, else shared."""
        pair = self._session_stores()
        return pair[1] if pair else self.shared_repairs

    def _session_stores(self):
        sid = sessions.current.get()
        return self.sessions.get(sid) if sid else None

    def backend_entries(self):
        """Every remote backend in the order asked: configured first, then
        registered by priority. ``[(name, url, key, priority)]``.

        A registration under a configured name replaces that entry, so an
        operator can move a backend without restarting the service.
        """
        out = [(n, u, k, i) for i, (n, u, k) in enumerate(self.env_backends)
               if n not in self.registry]
        out += [(n, e["url"], e["key"], e["priority"])
                for n, e in self.registry.items()]
        # Stable: on equal priority, configured before registered, then in
        # the order they registered.
        return sorted(out, key=lambda row: row[3])


def create_app(config: Optional[HealingConfig] = None,
               api_key: str = "", warm: bool = True,
               baseline_path: str = "", repair_path: str = "",
               review_path: str = "",
               agent_trace_path: str = "") -> FastAPI:
    """Build the ASGI app. Loading happens at startup, not on first request."""
    config = config or HealingConfig()
    # Environment wins, so a container can be configured without editing code.
    config.model_path = os.environ.get("BIS_MODEL_PATH", config.model_path)
    config.gpu_layers = int(os.environ.get("BIS_GPU_LAYERS", config.gpu_layers))
    # Off the command line, where a process list would show it.
    config.agent_key = config.agent_key or os.environ.get("BIS_AGENT_KEY", "")
    api_key = api_key or os.environ.get("BIS_API_KEY", "")
    baseline_path = (baseline_path
                     or os.environ.get("BIS_BASELINE_STORE", "")
                     or DEFAULT_BASELINE_PATH)
    repair_path = (repair_path
                   or os.environ.get("BIS_REPAIR_STORE", "")
                   or DEFAULT_REPAIR_PATH)
    review_path = (review_path
                   or os.environ.get("BIS_REVIEW_STORE", "")
                   or DEFAULT_REVIEW_PATH)
    agent_trace_path = (agent_trace_path
                        or os.environ.get("BIS_AGENT_TRACES", "")
                        or DEFAULT_TRACE_PATH)
    # A service that phoned another service would defeat the point.
    config.service_url = ""

    state = ServiceState(config, api_key, baseline_path, repair_path,
                         review_path, agent_trace_path)
    state.admin_key = os.environ.get("BIS_ADMIN_KEY", "").strip()
    state.env_backends = parse_backends(os.environ.get("BIS_AI_URL", ""),
                                        os.environ.get("BIS_AI_KEY", ""))
    state.rate = RateLimit(int(os.environ.get("BIS_RATE_PER_IP") or 120),
                           int(os.environ.get("BIS_RATE_TOTAL") or 600))
    state.sessions.most = int(os.environ.get("BIS_MAX_SESSIONS") or 500)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Startup work, as a lifespan rather than an on_event handler.

        ``@app.on_event("startup")`` is deprecated and warns on current
        FastAPI. Same work, same place in the sequence - loading before the
        first request rather than during it, so a bot's very first repair is
        not also paying to load several gigabytes of weights.
        """
        if state.api_key:
            log.info("API key auth enabled")
        else:
            log.warning("no API key set - this service is open to anyone who "
                        "can reach it (set BIS_API_KEY to require one)")
        if config.use_checker and config.checker_mode == "first":
            log.info("checker enabled (%s on %s) as the first opinion - it "
                     "decides a single-control step when it is sure and "
                     "the AI is asked when it is not", config.checker_path,
                     config.checker_device or "cpu")
        elif config.use_checker:
            log.info("checker enabled (%s) as a gate - a repair it doubts "
                     "is held for review instead of applied",
                     config.checker_path)
        if state.agent.available:
            log.info("ReFlex Agent planner at %s, up to %d tool calls a "
                     "repair", config.agent_url or config.agent_model_path,
                     config.agent_max_calls)
        if warm:
            log.info("loading models ...")
            state.warm = state.adjudicator.warmup()
            log.info("ready: %s", state.warm)
        yield
        state.shared_baselines.save()
        state.shared_repairs.close()
        state.sessions.close()

    app = FastAPI(
        title="Bot Immune System - Healing Service",
        version=SERVICE_VERSION,
        summary="Ranks and adjudicates replacement locators for RPA bots "
                "whose selectors have broken.",
        lifespan=lifespan,
    )
    app.add_middleware(SessionMiddleware)
    app.state.bis = state

    def require_key(x_api_key: str = Header(default="")) -> None:
        """Optional shared-secret auth.

        Unset means open, which is right for a demo on localhost and wrong for
        anything reachable from a network. The startup log says which it is.

        Compared with ``compare_digest`` rather than ``==``: this is the only
        authentication the service has, and a plain comparison returns as
        soon as two bytes differ, which leaks the key one character at a
        time to anyone who can time the response.
        """
        if not state.api_key:
            return
        if not hmac.compare_digest(str(x_api_key), str(state.api_key)):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def limit(request: Request) -> None:
        """Turn a flood away before it reaches the ranker or a paid GPU."""
        wait = state.rate.check(_caller(request))
        if wait:
            raise HTTPException(
                status_code=429, detail="too many requests - try again "
                "shortly", headers={"Retry-After": str(int(math.ceil(wait)))})

    def require_admin(x_admin_key: str = Header(default="")) -> None:
        """The key that may change where AI questions are sent.

        Unset means the endpoint does not exist: on a service nobody set an
        admin key for, the only safe answer to "send the AI's questions
        here" is that there is nowhere to ask.
        """
        if not state.admin_key:
            raise HTTPException(status_code=404, detail="Not Found")
        if not hmac.compare_digest(str(x_admin_key), str(state.admin_key)):
            raise HTTPException(status_code=401,
                                detail="invalid or missing admin key")

    def wake(model, failover) -> None:
        """Wait for a sleeping backend to answer, then connect to it."""
        ok = False
        try:
            ok = (bool(failover.wake(WAKE_TIMEOUT_S))
                  and (model.available or model.load()))
        except Exception as exc:
            log.warning("waking the AI failed: %s", exc)
        finally:
            with state.wake_lock:
                state.waking, state.wake_ok = False, bool(ok)
                state.wake_done_at = time.time()
        log.info("AI wake: %s", "answered" if ok else "no backend answered")

    def readiness():
        """Whether a bot may start now, and why: ``(ready, mode)``.

        A remote AI that sleeps between calls takes longer to wake than a
        bot waits for one answer, so asking for health is what wakes it:
        not ready until a backend answers, then "ai". When none did, ready
        anyway, "ranker-only" - the heuristics stand (invariant 11) and a
        demo is not held up for a GPU that is not coming.
        """
        if not state.config.use_ai:
            return True, "off"
        model = state.adjudicator.model
        if model is None:
            return True, "off"
        if not model.remote_url:
            return model.available, "ai" if model.available else "loading"
        failover = model.remote_client()
        if model.available and failover.awake():
            return True, "ai"
        with state.wake_lock:
            if state.waking:
                return False, "waking"
            if (state.wake_done_at and not state.wake_ok
                    and time.time() - state.wake_done_at
                    < failover.cooldown_s):
                return True, "ranker-only"
            state.waking = True
        threading.Thread(target=wake, args=(model, failover),
                         name="bis-wake", daemon=True).start()
        return False, "waking"

    def push_backends() -> None:
        """Hand the failover every backend, in the order they are asked."""
        model = state.adjudicator.model
        if model is not None:
            model.set_remote_backends([(n, u, k) for n, u, k, _p
                                       in state.backend_entries()])
        with state.wake_lock:
            # A backend that just arrived is worth a wake now, not after
            # the last failed one's cooldown.
            state.wake_done_at = 0.0

    @app.get("/v1/admin/backends", dependencies=[Depends(require_admin)])
    def list_backends() -> JSONResponse:
        """Every backend in order, with its state - never its key."""
        model = state.adjudicator.model
        rows = {}
        if model is not None and model.remote_url:
            rows = {r["name"]: r for r
                    in model.remote_client().status(addresses=False)}
        out = []
        for name, url, _key, priority in state.backend_entries():
            row = rows.get(name, {})
            out.append({"name": name, "url": url, "priority": priority,
                        "registered": name in state.registry,
                        "state": row.get("state", "unknown"),
                        "until": row.get("until", 0.0),
                        "last_ok": row.get("last_ok", 0.0)})
        return JSONResponse({"backends": out})

    @app.post("/v1/admin/backends", dependencies=[Depends(require_admin)])
    def register_backend(entry: BackendEntry) -> JSONResponse:
        """Add or move one backend - a GPU host announcing its address."""
        with state.backends_lock:
            state.registry[entry.name] = {"url": entry.url.rstrip("/"),
                                          "key": entry.key,
                                          "priority": entry.priority}
            push_backends()
            order = [row[0] for row in state.backend_entries()]
        log.info("AI backend %s registered at priority %d", entry.name,
                 entry.priority)
        return JSONResponse({"registered": entry.name, "order": order})

    @app.delete("/v1/admin/backends/{name}",
                dependencies=[Depends(require_admin)])
    def unregister_backend(name: str) -> JSONResponse:
        """Take a registered backend out - a GPU host going away."""
        with state.backends_lock:
            if state.registry.pop(name, None) is None:
                raise HTTPException(status_code=404,
                                    detail="no registered backend by that "
                                    "name")
            push_backends()
            order = [row[0] for row in state.backend_entries()]
        log.info("AI backend %s unregistered", name)
        return JSONResponse({"unregistered": name, "order": order})

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready, mode = readiness()
        info = state.adjudicator.describe()
        ai = dict(info.get("ai", {}), mode=mode)
        return HealthResponse(
            ready=ready,
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            protocol=PROTOCOL_VERSION,
            encoder_ready=bool(info.get("encoder_ready")),
            ai=ai,
            checker=info.get("checker", {}),
            calibration=info.get("calibration", {}),
            uptime_s=round(time.time() - state.started_at, 1),
            requests_served=state.requests,
            baselines=state.baselines.stats()["entries"],
            repairs=state.repairs.stats()["active"],
            reviews=state.reviews.stats()["labels"],
            agent={"available": state.agent.available,
                   "where": (state.config.agent_url or os.path.basename(
                       state.config.agent_model_path)),
                   "max_calls": state.config.agent_max_calls},
        )

    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse({
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "protocol": PROTOCOL_VERSION,
            "endpoints": ["/v1/health", "/v1/js", "/v1/adjudicate",
                          "/v1/baseline", "/v1/review", "/v1/outcome",
                          "/v1/agent/tools", "/v1/agent/start",
                          "/v1/agent/step", "/docs"],
        })

    @app.get("/v1/js", response_model=JsBundleResponse)
    def js_bundle() -> JsBundleResponse:
        """Serve the in-page JavaScript the bot executes.

        This is what keeps a thin client thin. The harvester, the probe and
        the single-element fingerprinter are the parts most likely to need
        improving as sites change, and serving them means that work ships
        centrally instead of as a redeploy to every bot.

        The bot still runs the JavaScript itself and still binds its own
        selectors - the service never sees the page and never returns a
        selector.
        """
        return JsBundleResponse(
            protocol=PROTOCOL_VERSION,
            version=JS_BUNDLE["version"],
            harvest=JS_BUNDLE["harvest"],
            probe=JS_BUNDLE["probe"],
            describe_one=JS_BUNDLE["describe_one"],
            presence=JS_BUNDLE["presence"],
            same=JS_BUNDLE["same"],
        )

    @app.post("/v1/baseline", response_model=BaselineResponse,
              dependencies=[Depends(require_key)])
    def record_baseline(body: BaselineRequest) -> BaselineResponse:
        """Remember what an element looked like while its locator worked.

        Posted by a client after a step succeeds. It is the only thing a
        thin client can contribute that the service cannot work out for
        itself: which of the elements on the page the step actually used.

        The store refuses a fingerprint that is not on screen, so a client
        may post freely and let the policy live in one place.
        """
        if body.protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (body.protocol, PROTOCOL_VERSION))
        # Was this locator ever proven, before this report? Asked *before*
        # recording, because recording is what creates the first one.
        # Proven means the *authored* selector resolved this locator on
        # its own at some point. A fingerprint left behind by an earlier
        # heal does not count: without this the guard below only delays
        # a bad guess by one run - run 1 records the baseline, run 2 sees
        # it and promotes the repair fleet-wide anyway. Measured.
        was_proven = state.baselines.is_proven(body.key, body.url)

        stored = state.baselines.record(body.key, body.url, body.descriptor,
                                        body.selector,
                                        provisional=bool(body.replaced))
        if stored:
            log.info("baseline recorded for %s on %s", body.key,
                     state.baselines.key(body.key, body.url))

        # A report carrying `replaced` says the authored selector broke and
        # this one took its place - that is a repair, and it is worth
        # keeping for every other bot that hits the same break. An empty
        # `replaced` is what the full agent always sends, and stays a
        # baseline.
        repair_id = 0
        process = body.process.model_dump() if body.process else None
        if body.replaced and body.selector and not was_proven:
            # A locator with no baseline was never observed working, so this
            # is not a repair - it is a first guess at a selector that may
            # simply have been wrong when it was deployed. Measured over six
            # sites, a typo'd selector that heals at all binds the WRONG
            # element 40% of the time, and confidence does not separate the
            # two: wrong picks ran a median 0.92 against 0.49 for right ones,
            # so no threshold helps. Applying the guess for this run is the
            # caller's business. Promoting it to fleet memory is not: the
            # store answers before the adjudicator ever runs (invariant 22),
            # so one bad guess would silence every future opinion about it.
            log.warning("cold heal on %s - applying, but not storing a fleet "
                        "repair: this locator has never been seen working, so "
                        "the repair cannot be distinguished from a selector "
                        "that was wrong on the day it shipped", body.key)
        elif body.replaced and body.selector:
            # Stored at the confidence it was decided at, because a fleet
            # hit is banded on that number. 1.0 is kept for a person's
            # approval and for a client too old to report one.
            if body.approved:
                confidence, strategy, why = (1.0, "approved-by-reviewer",
                                             "approved by a reviewer")
            else:
                confidence = (float(body.confidence)
                              if 0.0 < body.confidence <= 1.0 else 1.0)
                strategy, why = ("proven-by-client",
                                 "bound and used by a client")
            if body.repair_id:
                if state.repairs.reinforce(
                        body.repair_id, process,
                        **(dict(confidence=confidence, strategy=strategy,
                                rationale=why) if body.approved else {})):
                    repair_id = body.repair_id
            if not repair_id:
                repair_id = state.repairs.record(
                    body.key, body.url, body.replaced, body.selector,
                    confidence=confidence, strategy=strategy,
                    rationale=why, process=process)
            if repair_id:
                log.info("repair %d recorded for %s", repair_id, body.key)

        return BaselineResponse(
            stored=stored,
            reason="" if stored else "the element was not on screen",
            entries=state.baselines.stats()["entries"],
            repair_id=repair_id,
        )

    @app.post("/v1/review", response_model=ReviewResponse,
              dependencies=[Depends(require_key)])
    def record_review(body: ReviewRequest) -> ReviewResponse:
        """Keep a reviewer's verdict on a repair held for review.

        Posted by a client whose review hook decided one. An approval also
        reaches /v1/baseline, which files the repair at 1.0 for the fleet;
        this keeps the verdict itself - rejections included, which are the
        rarer and more useful half - as a label. Fails open: a verdict
        that cannot be written costs the label, never the request.
        """
        if body.protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (body.protocol, PROTOCOL_VERSION))
        row = body.model_dump()
        row.update(kind="review", at=time.time(), page=page_scope(body.url))
        stored = state.reviews.record(row)
        log.info("review of %s on %s: %s at %.0f%%", body.key,
                 row["page"], "approved" if body.approved else "rejected",
                 body.confidence * 100)
        if not body.approved and body.repair_id:
            # A person looked at the repair the fleet store served and said
            # no. That is stronger than any failed bind: retired at once.
            if state.repairs.retire(body.repair_id):
                log.info("repair %d retired - a reviewer rejected it",
                         body.repair_id)
        return ReviewResponse(stored=stored,
                              labels=state.reviews.stats()["labels"])

    @app.post("/v1/outcome", response_model=OutcomeResponse,
              dependencies=[Depends(require_key)])
    def record_outcome(body: OutcomeRequest) -> OutcomeResponse:
        """Where the step after this one found itself.

        A repair that binds the wrong element still binds, so a failed bind
        never retired one. What the step then leads to is evidence that
        does not depend on binding: from good runs the service learns which
        page the next step finds itself on, and a healed run that lands
        somewhere none of them did is a strike against its repair - two,
        as for failed binds, and it is demoted. It catches a wrong repair
        that navigates, not one that types into the wrong box on the same
        page; for those there is review.
        """
        if body.protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (body.protocol, PROTOCOL_VERSION))
        if not body.healed:
            state.baselines.note_next(body.key, body.url, body.next_url)
            return OutcomeResponse(verdict="noted")
        seen = state.baselines.next_pages(body.key, body.url)
        if sum(seen.values()) < OUTCOME_SETTLED:
            return OutcomeResponse(verdict="unknown")
        landed = page_scope(body.next_url)
        agreed = landed in seen
        demoted = False
        if not agreed:
            stored = state.repairs.lookup(body.key, body.url, body.replaced)
            # Only the repair this run bound: another bot may have filed a
            # newer one since, and that one has done nothing wrong.
            if stored and stored["selector"] == body.selector:
                demoted = state.repairs.mismatch(stored["id"])
            log.warning("%s was healed to %s and the next step found itself "
                        "on %s, where good runs found %s%s", body.key,
                        body.selector, landed, ", ".join(sorted(seen)),
                        " - repair demoted" if demoted else "")
        state.reviews.record({
            "kind": "outcome", "at": time.time(), "key": body.key,
            "page": page_scope(body.url), "url": body.url,
            "selector": body.selector, "replaced": body.replaced,
            "landed": landed, "expected": sorted(seen), "agreed": agreed,
            "process": body.process.model_dump() if body.process else None})
        return OutcomeResponse(verdict="agreed" if agreed else "mismatch",
                               demoted=demoted)

    def _candidate_with(candidates, selector: str) -> int:
        """Index of the first candidate the page proposed this selector for.

        ``selectors`` is a flat, de-duplicated list of strings ordered by
        the request's preferred syntax, and XPath entries keep their
        ``xpath=`` prefix (js_harvest.py). The stored selector is the
        winning proposal a client reported verbatim, so this is an exact
        string match - do not strip the prefix and do not re-order.
        """
        for i, cand in enumerate(candidates):
            if selector in (cand.get("selectors") or []):
                return i
        return -1

    def _can_serve_repairs(candidates) -> bool:
        """Does this request carry what the repair store matches on?

        The store maps a remembered selector back to an index by finding the
        candidate that proposed it, so a request whose candidates carry no
        ``selectors`` can never produce a hit - every lookup would miss, and
        miss silently, which reads exactly like an empty store.

        That is not hypothetical. ``AdjudicationRequest.build`` strips the
        field on purpose (it is the bulk of the payload and only the bot
        binds selectors), so any client using it gets a store that can never
        answer. Only the thin client, which posts the harvester's own output,
        carries them. Today that is also the only client setting
        ``use_stored_repair``, so the two line up - but they line up by
        coincidence, and this says so out loud the moment they stop.
        """
        return any(cand.get("selectors") for cand in candidates)

    def _from_the_store(body, request, stored, at, started):
        """Answer from a repair the fleet proved, banded like any other.

        It used to return apply at the stored 1.0, past decide() and past
        the optional floor, and that 1.0 only ever meant "the selector
        bound" - not that it bound the right element. The stored number is
        now the confidence of the decision that produced the repair, the
        checker (when it is on) looks at the pick again, and decide() says
        what happens. Proven is taken as given: the store only holds a
        repair for a locator that was proven when it was filed (invariant
        26), and the authored selector being broken is exactly why its
        baseline stops being refreshed.
        """
        cfg = state.config
        confidence = float(stored["confidence"])
        rationale = (stored["rationale"]
                     or "a repair another run proved for this step")
        checker = state.adjudicator.recheck(request, at)
        if checker is not None and checker < cfg.checker_floor:
            # A gate, as in "gate" mode: it can hold the pick for review,
            # never raise a confidence and never change the choice.
            confidence = min(confidence, cfg.auto_apply_threshold - 0.01)
            rationale = "checker doubts this pick (%.2f); %s" % (checker,
                                                                rationale)
        # Stored as a probability of being right, but filed with no group:
        # with a target precision it must clear the target itself.
        bar = state.adjudicator.bar_for(AdjudicationDecision())
        band = cfg.decide(confidence, True, bar)
        chosen, repair_id = at, stored["id"]
        if body.optional and cfg.decide_optional(confidence, True,
                                                 bar) != cfg.APPLY:
            chosen, repair_id, band = -1, 0, cfg.REFUSE
            rationale = ("optional locator below the %.0f%% floor; treated "
                         "as absent rather than repaired"
                         % (cfg.optional_min_confidence * 100))
        log.info("served %s from the fleet repair store (candidate %d, "
                 "%.0f%%, %s, no AI)", body.locator.key, at,
                 confidence * 100, band)
        return AdjudicateResponse(
            chosen=chosen, confidence=round(confidence, 4), band=band,
            strategy="fleet-memory", rationale=rationale,
            repair_id=repair_id, ai_consulted=False,
            checker=round(checker, 3) if checker is not None else -1.0,
            duration_s=round(time.time() - started, 3))

    @app.post("/v1/adjudicate", response_model=AdjudicateResponse,
              dependencies=[Depends(require_key), Depends(limit)])
    def adjudicate(body: AdjudicateRequest) -> AdjudicateResponse:
        """Rank the supplied candidates and choose one.

        Returns an *index* into the list the bot sent, never a selector: the
        bot owns selector binding because only it can prove a selector
        resolves on the page in front of it.
        """
        if body.protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (body.protocol, PROTOCOL_VERSION))
        if not body.candidates:
            raise HTTPException(status_code=422, detail="no candidates supplied")
        # The harvester's own ceiling is 900 (js_harvest: opts.maxCandidates),
        # so anything past MAX_CANDIDATES is not a bot that harvested a big
        # page - it is a client that is broken or one that is not ours. Every
        # candidate costs seven similarity signals and, past the pre-rank, an
        # embedding, and this endpoint is open by default. Refuse rather than
        # truncate: silently scoring a third of what was sent would look like
        # a bad repair, not a rejected request.
        if len(body.candidates) > MAX_CANDIDATES:
            raise HTTPException(
                status_code=413,
                detail="%d candidates exceeds the %d this service will score"
                       % (len(body.candidates), MAX_CANDIDATES))

        state.requests += 1
        started = time.time()

        # A retry says the repair we served did not bind. Count that first,
        # so a repair that keeps failing stops being served to the fleet -
        # one bad repair must not follow every bot onto the same page.
        demoted_selector = ""
        if body.retry and body.failed_repair_id:
            failed = state.repairs.by_id(body.failed_repair_id)
            demoted_selector = (failed or {}).get("selector", "")
            if state.repairs.demote(body.failed_repair_id):
                log.info("repair %d demoted after repeated failures",
                         body.failed_repair_id)

        # A client that keeps its own memory sends its baseline inline. One
        # that does not asks for the service's, and gets it looked up under
        # the same key the toolkit would have used. Without this, four of the
        # seven signals are blank on every thin-client heal.
        #
        # The asking is deliberate. A None from the full agent means "my
        # memory has none for this locator", which is not the same as "I have
        # no memory" - substituting the fleet's would overrule a client that
        # manages its own, and would silently warm the benchmark, which wipes
        # its memory per suite precisely so that every row is a cold heal.
        baseline = body.baseline
        if not baseline and body.use_stored_baseline:
            baseline = state.baselines.get(body.locator.key, body.page.url)
            if baseline:
                log.info("using a stored baseline for %s", body.locator.key)
        request = AdjudicationRequest(
            locator=body.locator.model_dump(),
            action=body.action,
            page=body.page.model_dump(),
            candidates=body.candidates,
            baseline=baseline,
            shortlist_size=body.shortlist_size,
            budget_s=body.budget_s,
            scope=sessions.current.get(),
        )

        # Has this locator's own selector ever resolved on this page? If not,
        # the heal is a first guess and never auto-applies - measured over
        # 3,760 breakages on 82 live sites, 47% of cold applies bound the
        # wrong element, and confidence does not separate right from wrong
        # (0.586 vs 0.609). See HealingConfig.decide and invariant 26.
        #
        # Only the store can say, because only it records provenance. This
        # was `bool(baseline) or is_proven(...)`, and the baseline just
        # fetched can be the provisional entry a heal wrote - or bis.py's
        # own fingerprint of its last repair, sent inline - so the second
        # heal of a selector never seen working applied: review at 0.685
        # became apply at 0.729 on the same request, measured. A client
        # that keeps its own memory still vouches for its inline baseline;
        # the service cannot see where that one came from.
        proven = (state.baselines.is_proven(body.locator.key,
                                            body.page.url or "")
                  or (bool(body.baseline) and not body.use_stored_baseline))

        # The repair store, in front of the pipeline rather than inside it.
        # A hit returns here, so the adjudicator - and its decision cache -
        # is never reached, and the answer is still an index into the
        # candidates this bot sent. Invariant 3 is untouched.
        if body.use_stored_repair and not _can_serve_repairs(body.candidates):
            log.warning(
                "%s asked for the fleet repair store but sent candidates "
                "with no 'selectors' - the store cannot map a repair back to "
                "a candidate, so every lookup will miss. Send the harvester's "
                "proposals, or stop setting use_stored_repair.",
                body.locator.key)
        if (body.use_stored_repair and not body.retry
                and _can_serve_repairs(body.candidates)):
            stored = state.repairs.lookup(body.locator.key, body.page.url,
                                          body.locator.selector)
            if stored:
                at = _candidate_with(body.candidates, stored["selector"])
                if at >= 0:
                    return _from_the_store(body, request, stored, at,
                                           started)
                # The page has moved again and nothing carries it. Fall
                # through; the miss is the invalidation.
                log.info("stored repair for %s is no longer on the page",
                         body.locator.key)

        decision = state.adjudicator.adjudicate(request)
        # With a target precision, the bar this decision must clear: its
        # calibration group's, or the target itself (None without one).
        bar = state.adjudicator.bar_for(decision)

        # An optional locator is held to a stricter floor. The bot cannot
        # apply this itself without the policy drifting between fleets, so
        # it declares intent and the service decides - which is also what
        # lets the threshold be retuned without touching a single bot.
        if body.optional and decision.chosen >= 0:
            band = state.config.decide_optional(decision.confidence, proven,
                                                bar)
            if band != state.config.APPLY:
                log.info("optional locator %s: %.0f%% is below the optional "
                         "floor of %.0f%% - reporting absent",
                         body.locator.key, decision.confidence * 100,
                         state.config.optional_min_confidence * 100)
                decision.chosen = -1
                decision.rationale = (
                    "optional locator below the %.0f%% floor; treated as "
                    "absent rather than repaired"
                    % (state.config.optional_min_confidence * 100))

        payload = decision.to_dict()
        payload["band"] = (state.config.decide(decision.confidence, proven,
                                               bar)
                           if decision.chosen >= 0 else state.config.REFUSE)
        if decision.chosen >= 0 and not proven:
            log.info("cold locator %s: %.0f%% held at review - never seen "
                     "working, so this is a guess rather than a repair",
                     body.locator.key, decision.confidence * 100)
        if payload["band"] == state.config.REVIEW:
            # Someone will read this one, so it is the one that gets the
            # AI's reason. The odds decided it without a reply.
            why = state.adjudicator.explain(
                request, decision, body.budget_s - (time.time() - started))
            if why:
                # A replay from the decision cache already has the reason
                # (explain() files it there), but not in its rationale.
                payload["ai_reason"] = why
                if why not in payload["rationale"]:
                    payload["rationale"] = "%s; %s" % (payload["rationale"],
                                                       why)

        log.info("adjudicated %s (%s) -> candidate %d, %.0f%% via %s in %.1fs",
                 body.locator.key, body.action, decision.chosen,
                 decision.confidence * 100, decision.strategy or "none",
                 time.time() - started)

        # The client has already tried every proposal this candidate offers
        # and none resolved. Handing it back would spend its one retry on
        # an answer it has just disproved.
        if demoted_selector and payload.get("chosen", -1) >= 0:
            if _candidate_with(body.candidates,
                               demoted_selector) == payload["chosen"]:
                payload["chosen"] = -1
                payload["band"] = state.config.REFUSE
                payload["rationale"] = (
                    "the only confident answer was the element whose "
                    "selectors this client has just proved do not bind")
        return AdjudicateResponse(**payload)

    # ------------------------------------------------------ ReFlex Agent
    def _speaks(protocol: str) -> None:
        if protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (protocol, PROTOCOL_VERSION))

    @app.get("/v1/agent/tools")
    def agent_tools() -> JSONResponse:
        """The tools the ReFlex Agent is offered (invariant 51).

        Each with the name, description and parameters the planner receives
        every turn, the title the trace shows and where it runs - the
        host, the robot's page, or "end" for one that ends the repair.
        Open, like the JavaScript: it is what is shown.
        """
        return JSONResponse({"protocol": PROTOCOL_VERSION, "tools": TOOLS})

    @app.post("/v1/agent/start", response_model=AgentResponse,
              dependencies=[Depends(require_key), Depends(limit)])
    def agent_start(body: AgentStartRequest) -> AgentResponse:
        """Begin a repair the agent plans: the planner is asked what to do,
        and the response is either the page tools to run or the end."""
        _speaks(body.protocol)
        state.requests += 1
        return AgentResponse(**state.agent.start(body.model_dump()))

    @app.post("/v1/agent/step", response_model=AgentResponse,
              dependencies=[Depends(require_key), Depends(limit)])
    def agent_step(body: AgentStepRequest) -> AgentResponse:
        """Hand the agent what the page tools returned, and carry on."""
        _speaks(body.protocol)
        return AgentResponse(**state.agent.step(body.model_dump()))

    return app


def __getattr__(name: str):
    """Build the default app on first access of ``app``.

    ``app = None`` used to sit here, which meant the one incantation every
    ASGI deployment guide reaches for - ``uvicorn healing_service.app:app``
    - loaded a None and failed with nothing useful to say. Built lazily
    rather than at import so that ``create_app`` stays the way tests and
    ``main()`` make one, and importing this module still costs nothing.
    """
    if name == "app":
        return create_app()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def parse_command_line(argv=None):
    """The service's command line, as (args, HealingConfig).

    Separate from main() so the policy it sets can be tested without
    starting a server.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Bot Immune System healing service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--model", default=None, help="path to the GGUF model file")
    ap.add_argument("--gpu-layers", type=int, default=None,
                    help="-1 offloads everything, 0 forces CPU")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--reasoning", default=None,
                    help="reasoning budget for a reasoning model file "
                         "(gpt-oss reads low/medium/high); omit for an "
                         "instruct-tuned file")
    ap.add_argument("--ai-url", default="", dest="ai_url",
                    help="ask the AI server at this address instead of "
                         "loading the AI here - scripts/kaggle_ai_server.py "
                         "on a cloud GPU, serving the same GGUF. Its key "
                         "comes from BIS_AI_KEY, kept off the command line "
                         "where a process list would show it")
    ap.add_argument("--prompt-version", type=int, choices=(2, 3),
                    default=None, dest="prompt_version",
                    help="the prompt the model file was fine-tuned on: 2 "
                         "for every fine-tune so far, 3 for one trained on "
                         "a build_llm_dataset.py --v3 set (no first-guess "
                         "line, richer candidate lines, the baseline shown)")
    ap.add_argument("--no-ai", action="store_true",
                    help="serve heuristic + semantic ranking only")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--baselines", default="",
                    help="where to keep the baselines thin clients rely on "
                         "(default run-memory/baselines.json)")
    ap.add_argument("--repairs", default="",
                    help="where to keep the repairs the fleet has proven "
                         "(default run-memory/repairs.db)")
    ap.add_argument("--reviews", default="",
                    help="where to keep reviewers' verdicts on repairs held "
                         "for review (default run-memory/reviews.jsonl)")
    ap.add_argument("--checker", action="store_true",
                    help="use the fine-tuned Laya checker (invariant 39). "
                         "About 1 GB of weights. Off unless asked for")
    ap.add_argument("--checker-mode", default="first", dest="checker_mode",
                    choices=("first", "gate"),
                    help="first (default): the checker scores the "
                         "shortlist and decides a single-control step "
                         "when it is sure, asking the AI only when it is "
                         "not - ~40%% fewer AI calls. gate: the AI "
                         "decides and the checker can only hold its pick "
                         "for review")
    ap.add_argument("--checker-device", default="", dest="checker_device",
                    help="cpu (default) or cuda. On cuda a shortlist "
                         "takes ~0.3s against ~6s on a laptop CPU, but it "
                         "needs ~2 GB beside the AI's weights")
    ap.add_argument("--checker-path", default="", dest="checker_path",
                    help="folder holding checker.safetensors and "
                         "checker.json (default models/laya-checker)")
    ap.add_argument("--checker-rerank", type=int, default=None,
                    dest="checker_rerank",
                    help="the checker re-ranks the ranker's top N and the "
                         "shortlist is the ones it rates best, in the "
                         "ranker's order (0, the default, is off). A pass "
                         "per eight candidates, so for --checker-device "
                         "cuda: on a CPU it seldom fits the budget")
    # argparse %-formats help text itself, so a literal percent sign is
    # written %% here and the defaults are joined on rather than %-ed in.
    ap.add_argument("--apply-threshold", type=float, default=None,
                    dest="apply_threshold",
                    help="confidence at or above which a repair applies "
                         "on its own (default "
                         + str(HealingConfig.auto_apply_threshold)
                         + "). 0.95 is where the AI's token odds were "
                         "measured: 99%% of picks right on held-out cases. "
                         "Paths without odds - the heuristics alone, an AI "
                         "server that refuses logprobs - then mostly go to "
                         "review; /v1/health says which under ai.confidence")
    ap.add_argument("--review-threshold", type=float, default=None,
                    dest="review_threshold",
                    help="confidence at or above which a repair is offered "
                         "for review rather than refused (default "
                         + str(HealingConfig.review_threshold) + ")")
    ap.add_argument("--target-precision", type=float, default=None,
                    dest="target_precision",
                    help="hold auto-applied repairs to this precision: a "
                         "calibrated AI decision must clear its group's bar "
                         "- where the 95%% lower bound on precision reached "
                         "the target on held-out steps - and anything else "
                         "the target itself. 0.95 is the one the shipped "
                         "calibration was fitted for")
    ap.add_argument("--no-calibration", action="store_true",
                    dest="no_calibration",
                    help="use the AI's raw token odds as the confidence, "
                         "not the calibrated probability of being right")
    ap.add_argument("--agent-url", default="", dest="agent_url",
                    help="the ReFlex Agent's planner: an OpenAI-compatible "
                         "chat server with tool calling (a llama.cpp "
                         "server, for one). Its key comes from "
                         "BIS_AGENT_KEY. With no planner a robot asking for "
                         "the agent is healed the old way")
    ap.add_argument("--agent-model", default="", dest="agent_model",
                    help="a planner model file loaded on this host instead "
                         "of a server")
    ap.add_argument("--agent-max-calls", type=int, default=None,
                    dest="agent_max_calls",
                    help="tool calls one repair may make before it falls "
                         "back to the pipeline (default "
                         + str(HealingConfig.agent_max_calls) + ")")
    ap.add_argument("--agent-traces", default="", dest="agent_traces",
                    help="where each agent repair is written down (default "
                         "run-memory/agent-traces.jsonl)")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    config = HealingConfig()
    if args.apply_threshold is not None:
        config.auto_apply_threshold = args.apply_threshold
    if args.review_threshold is not None:
        config.review_threshold = args.review_threshold
    if not 0.0 < config.review_threshold <= config.auto_apply_threshold <= 1.0:
        ap.error("need 0 < review threshold (%.2f) <= apply threshold "
                 "(%.2f) <= 1" % (config.review_threshold,
                                  config.auto_apply_threshold))
    if args.target_precision is not None:
        if not 0.0 < args.target_precision < 1.0:
            ap.error("the target precision is a fraction, 0 < target < 1")
        config.target_precision = args.target_precision
    config.use_calibration = not args.no_calibration
    if args.model:
        config.model_path = args.model
    if args.gpu_layers is not None:
        config.gpu_layers = args.gpu_layers
    if args.context is not None:
        config.ai_context = args.context
    config.use_ai = not args.no_ai
    if args.reasoning is not None:
        config.reasoning_effort = args.reasoning
    if args.prompt_version is not None:
        config.prompt_version = args.prompt_version
    config.use_checker = bool(args.checker)
    config.checker_mode = args.checker_mode
    if args.checker_device:
        config.checker_device = args.checker_device
    if args.checker_path:
        config.checker_path = args.checker_path
    if args.checker_rerank is not None:
        if args.checker_rerank < 0:
            ap.error("--checker-rerank is a count of candidates, >= 0")
        config.checker_rerank = args.checker_rerank
    if args.agent_url:
        config.agent_url = args.agent_url
    if args.agent_model:
        config.agent_model_path = args.agent_model
    if args.agent_max_calls is not None:
        if args.agent_max_calls < 1:
            ap.error("--agent-max-calls is a count of tool calls, >= 1")
        config.agent_max_calls = args.agent_max_calls
    return args, config


def main() -> int:
    """Console entry point: python -m healing_service"""
    import uvicorn

    args, config = parse_command_line()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S")

    if args.ai_url:
        # Read by AdjudicatorModel when it loads, so the remote path needs
        # no new field in HealingConfig or in runtime.make_model.
        os.environ["BIS_AI_URL"] = args.ai_url

    application = create_app(config, api_key=args.api_key,
                             warm=not args.no_warm,
                             baseline_path=args.baselines,
                             repair_path=args.repairs,
                             review_path=args.reviews,
                             agent_trace_path=args.agent_traces)
    uvicorn.run(application, host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
