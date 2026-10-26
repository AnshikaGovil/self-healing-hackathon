"""Local inference runtime for the healing adjudicator.

Loads a GGUF model straight off disk with llama.cpp - no daemon, no model
registry, no external service. The weights live in ``models/`` inside the
project, so a deployment is a copy of this directory plus a pip install.

GPU is the intended deployment target: ``gpu_layers = -1`` offloads every
layer, and the service reports the device it *actually* got rather than the
one it hoped for. That honesty matters, because the whole point of putting
this behind an API is that one GPU host serves many bots - if it quietly fell
back to CPU, the fleet would slow down with nothing in the logs to say why.

Thread safety: a llama.cpp context is single-threaded. All generation goes
through one lock, so the service serialises requests rather than corrupting
state under concurrent load.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("bis.inference")

DEFAULT_MODEL_PATH = os.path.join("models", "healing-adjudicator.gguf")

# Constrained decoding: the sampler is restricted to tokens that keep the
# output valid against this schema, so the model physically cannot emit prose,
# a reasoning preamble, or a malformed object. It is what makes a small
# open-weight model dependable in an automated decision path.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["choice", "confidence", "reason"],
}

SYSTEM_PROMPT = (
    "You are the locator adjudicator inside an RPA self-healing agent. "
    "A UI automation step failed because its selector no longer matches the "
    "page. You are given the business intent of the step, the selector that "
    "broke, and a numbered shortlist of real elements harvested from the live "
    "DOM. Choose the single element that a human RPA developer would rebind "
    "the step to.\n"
    "Rules: prefer visible, interactive elements for click and type steps; "
    "prefer elements whose text or aria-label matches the intent; treat class "
    "name version bumps (pd19 -> pd21, v3 -> v4) as the same component; ignore "
    "elements that merely contain the target as a child when a tighter match "
    "exists. If none of the candidates plausibly serves the intent, answer -1. "
    "Reply with JSON only."
)


# The longest shortlist whose answer is a single digit. The odds are read
# one token after `{"choice": `, so index 10 and beyond would need a second
# digit read; a longer shortlist keeps the written answer instead.
TOKEN_READ_MAX = 10

# Asking a remote server for the odds: how many alternatives it reports at
# each token, and how far into the reply - a few tokens past the index is
# all the read needs, and every one it writes costs a sort of the vocabulary
# on the server.
REMOTE_TOP = 20
REMOTE_READ_TOKENS = 8


@dataclass
class Verdict:
    """What the AI adjudicator concluded about a shortlist."""

    choice: int = -1
    confidence: float = 0.0
    reason: str = ""
    latency_s: float = 0.0
    ok: bool = False
    error: str = ""
    device: str = ""
    source: str = ""
    """Where `confidence` came from. "token": the probability the model
    put on the index it picked, read from its next-token odds - a
    probability, comparable with a bar. "written": the number its reply
    wrote, which the fine-tune learned as a label (0.55 plus a ranker
    margin, capped at 0.95) and which separates right from wrong picks
    barely better than chance (AUC 0.59, against 0.93 for the odds)."""
    odds: Optional[Dict[int, float]] = None
    """The whole token read when source is "token" - every index and -1 -
    which the calibration reads the runner-up and "none" from."""

    @property
    def abstained(self) -> bool:
        return not self.ok or self.choice < 0


class RemoteLlama:
    """A llama.cpp server elsewhere, called the way a local Llama is called.

    llama-cpp-python's server speaks the OpenAI chat endpoint and accepts
    the same schema-constrained ``response_format``, so the request is the
    one a local ``Llama`` would get and the same weights give the same
    verdict wherever they run. Used when the accelerator that can hold the
    model is not on this host - a cloud GPU serving the AI while a laptop
    keeps the checker.
    """

    # Under bis.py's 30 s wait for the service, with room for the rest of
    # the request: a remote that answers later turns the service's
    # fail-open "no opinion" into a client timeout (invariant 11).
    def __init__(self, url: str, key: str = "", timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout
        # The server is named explicitly, so no system proxy is consulted.
        self._open = urllib.request.build_opener(
            urllib.request.ProxyHandler({}))

    def _call(self, path: str, payload: Optional[dict] = None,
              timeout: Optional[float] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None             else None
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.url + path, data=data,
                                     headers=headers,
                                     method="POST" if data else "GET")
        with self._open.open(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def info(self, timeout: float = 10.0) -> dict:
        """What the server says about itself: device, where, model file."""
        return self._call("/bis/info", timeout=timeout)

    def create_chat_completion(self, **kwargs) -> dict:
        return self._call("/v1/chat/completions", kwargs)

    def create_completion(self, prompt: str, **kwargs) -> dict:
        return self._call("/v1/completions", dict(kwargs, prompt=prompt))


# "name=url" in BIS_AI_URL, so the health report and the logs can say which
# backend answered without ever printing its address.
_NAMED_URL = re.compile(r"^([A-Za-z0-9_-]{1,32})=(https?://\S+)$")

# Codes that mean "this backend is not there right now": a gateway or the
# backend's own guard answering for it. Any other HTTP error is the backend
# itself answering - about the request, not about being down - so failing
# over would only ask the next backend the same bad question.
_UNAVAILABLE = {401, 403, 408, 425, 429, 502, 503, 504}


def _hide_backends() -> bool:
    """BIS_HIDE_BACKENDS: name the backends in reports, never their
    addresses. A public front door sets it: a GPU's address there invites
    unauthenticated calls that wake it - refused, and still billed."""
    return os.environ.get("BIS_HIDE_BACKENDS", "") not in ("", "0")


def parse_backends(urls: str, keys: str = "") -> List[Tuple[str, str, str]]:
    """``[(name, url, key)]`` from ``BIS_AI_URL`` and ``BIS_AI_KEY``.

    Both are comma lists. An entry may be ``name=url``; unnamed ones are
    ``ai1``, ``ai2``... One key serves every backend; several pair with the
    URLs by position.
    """
    parts = [p.strip() for p in (urls or "").split(",") if p.strip()]
    ks = [k.strip() for k in (keys or "").split(",")]
    out = []
    for i, part in enumerate(parts):
        named = _NAMED_URL.match(part)
        name, url = named.groups() if named else ("ai%d" % (i + 1), part)
        key = ks[i] if len(ks) > 1 and i < len(ks) else ks[0] if ks else ""
        out.append((name, url.rstrip("/"), key))
    return out


class FailoverLlama:
    """Several remote AI servers, tried in order, as one client.

    The public service keeps one address while the GPUs behind it come and
    go - a cloud that sleeps between calls, one switched on for a demo, a
    notebook whose address changes every session. A backend that does not
    answer rests for ``cooldown_s``; one whose guard says its credit cap is
    reached (402) rests for ``capped_s``, since asking again within the hour
    only wakes a GPU to say no. With every backend resting, a call raises at
    once and the service gives no opinion - the ranker's answer stands
    (invariant 11) - instead of waiting on each in turn.
    """

    def __init__(self, backends, clock=time.time, cooldown_s: float = 120.0,
                 capped_s: float = 3600.0, awake_s: float = 240.0,
                 deadline_s: float = 25.0):
        self._clock = clock
        self.cooldown_s = cooldown_s
        self.capped_s = capped_s
        # Under the 300 s a sleeping cloud keeps a container after its last
        # call, so "awake" here means it is still warm there.
        self.awake_s = awake_s
        # bis.py waits 30 s for the service: past this, trying the next
        # backend would answer a client that has already given up.
        self.deadline_s = deadline_s
        self._lock = threading.Lock()
        self._backends: List[Dict[str, Any]] = []
        self.last_backend = ""
        self.set_backends(backends)

    def set_backends(self, backends) -> None:
        """Replace the list, in order. A name kept with the same client keeps
        its rest; a new client under an old name starts afresh."""
        with self._lock:
            old = {b["name"]: b for b in self._backends}
            fresh = []
            for name, client in backends:
                was = old.get(name)
                if was is not None and was["client"] is client:
                    fresh.append(was)
                else:
                    fresh.append({"name": name, "client": client,
                                  "state": "up", "until": 0.0,
                                  "last_ok": 0.0})
            self._backends = fresh

    @property
    def url(self) -> str:
        with self._lock:
            return (getattr(self._backends[0]["client"], "url", "")
                    if self._backends else "")

    def status(self, addresses: bool = True) -> List[Dict[str, Any]]:
        """Each backend's name, state ("up", "cooling", "capped") and when a
        rest ends - with its address unless asked to leave it out."""
        now = self._clock()
        with self._lock:
            out = []
            for b in self._backends:
                state = b["state"] if b["until"] > now else "up"
                row = {"name": b["name"], "state": state,
                       "until": b["until"] if state != "up" else 0.0,
                       "last_ok": b["last_ok"]}
                if addresses:
                    row["url"] = getattr(b["client"], "url", "")
                out.append(row)
            return out

    def awake(self) -> bool:
        """Did any backend answer recently enough to be warm still?"""
        now = self._clock()
        with self._lock:
            return any(b["last_ok"] and now - b["last_ok"] <= self.awake_s
                       for b in self._backends)

    def _rest(self, b: Dict[str, Any], state: str, seconds: float) -> None:
        with self._lock:
            b["state"], b["until"] = state, self._clock() + seconds

    def _ok(self, b: Dict[str, Any]) -> None:
        with self._lock:
            b["state"], b["until"], b["last_ok"] = "up", 0.0, self._clock()
            self.last_backend = b["name"]

    def _call(self, method: str, *args, **kwargs):
        started = time.monotonic()
        now = self._clock()
        with self._lock:
            ready = [b for b in self._backends if b["until"] <= now]
        missed = []
        for b in ready:
            if missed and time.monotonic() - started > self.deadline_s:
                missed.append("deadline")
                break
            try:
                out = getattr(b["client"], method)(*args, **kwargs)
            except urllib.error.HTTPError as exc:
                if exc.code == 402:
                    self._rest(b, "capped", self.capped_s)
                elif exc.code in _UNAVAILABLE:
                    self._rest(b, "cooling", self.cooldown_s)
                else:
                    self._ok(b)
                    raise
                missed.append("%s: HTTP %s" % (b["name"], exc.code))
                continue
            except (urllib.error.URLError, OSError, ValueError) as exc:
                # ValueError: a reply that is not JSON - a proxy's page or a
                # server still starting.
                self._rest(b, "cooling", self.cooldown_s)
                missed.append("%s: %s" % (b["name"], str(exc)[:60]))
                continue
            self._ok(b)
            return out
        raise RuntimeError("no AI backend answered (%s)" % (
            "; ".join(missed) or "every backend is resting"))

    def clients(self) -> List[Tuple[str, Any]]:
        """(name, client) in order - what set_backends was last given."""
        with self._lock:
            return [(b["name"], b["client"]) for b in self._backends]

    def wake(self, timeout_s: float = 90.0) -> str:
        """Ask each backend in order - resting or not - to answer within
        ``timeout_s``, long enough for a sleeping cloud to boot. The name of
        the first that answers, or "" when none did (each is then rested).

        Resting ones are asked too: a cloud that timed out a short call
        while booting is exactly the one worth waiting for here.
        """
        with self._lock:
            order = list(self._backends)
        for b in order:
            try:
                b["client"].info(timeout=timeout_s)
            except urllib.error.HTTPError as exc:
                if exc.code == 402:
                    self._rest(b, "capped", self.capped_s)
                    continue
                if exc.code in _UNAVAILABLE:
                    self._rest(b, "cooling", self.cooldown_s)
                    continue
            except (urllib.error.URLError, OSError, ValueError):
                self._rest(b, "cooling", self.cooldown_s)
                continue
            self._ok(b)
            return b["name"]
        return ""

    def info(self) -> dict:
        out = self._call("info")
        return dict(out, backend=self.last_backend)

    def create_chat_completion(self, **kwargs) -> dict:
        return self._call("create_chat_completion", **kwargs)

    def create_completion(self, prompt: str, **kwargs) -> dict:
        return self._call("create_completion", prompt, **kwargs)


class AdjudicatorModel:
    """A loaded GGUF model, ready to adjudicate locator shortlists.

    With ``BIS_AI_URL`` set (``--ai-url``), the model is not loaded here: the
    same requests go to a llama.cpp server at that address instead.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 gpu_layers: int = -1, context: int = 8192,
                 threads: Optional[int] = None, batch: int = 512,
                 max_tokens: int = 160,
                 reasoning_effort: str = "",
                 prompt_version: int = 2):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.context = context
        self.threads = threads or max(2, (os.cpu_count() or 8) // 2)
        self.batch = batch
        self.max_tokens = max_tokens
        self.reasoning_effort = (reasoning_effort or "").strip()
        # Which prompt these weights were fine-tuned on (build_prompt).
        # A property of the file, so it is the operator's to state: the
        # GGUF does not record it.
        self.prompt_version = int(prompt_version or 2)
        self._llm = None
        self._lock = threading.Lock()
        self._load_error = ""
        self._device = "unloaded"
        self._loaded_at = 0.0
        # What a call on this host has actually cost, so the caller can gate
        # on evidence instead of on a constant. Same weights, same prompt
        # shape and one lock, so a handful of samples describes the next
        # call well; a median rather than a mean because the first call
        # after a load is an outlier that should not move the estimate.
        self._latencies: List[float] = []
        self._confidences: List[float] = []
        # A reason is generated only when someone will read it, and costs
        # what a whole reply used to, so it is timed on its own.
        self._explain_latencies: List[float] = []
        # What the token read needs from this file: None until probed,
        # False when it cannot be done here, else the token ids.
        self._reader = None
        self.remote_url = os.environ.get("BIS_AI_URL", "").strip()
        self.remote_key = os.environ.get("BIS_AI_KEY", "").strip()
        # Built once and kept across failed connects, so a backend's rest
        # survives the retry and a registration can change the list before
        # the first one succeeds.
        self._failover: Optional[FailoverLlama] = None
        self._where = ""
        self._remote_model = ""
        self._remote_sha = ""

    @property
    def typical_latency_s(self) -> float:
        """Median observed adjudication time, or 0.0 before the first one."""
        with self._lock:
            seen = sorted(self._latencies)
        return seen[len(seen) // 2] if seen else 0.0

    @property
    def weights_sha256(self) -> str:
        """The sha256 of the weights answering, when they say; "" if not.

        A remote server states it; a file loaded here is not hashed (2.5 GB
        on every start). The calibration is fitted to one set of weights
        and is not used for another that names itself.
        """
        return self._remote_sha

    @property
    def typical_explain_s(self) -> float:
        """Median time to generate a reason, or 0.0 before the first one."""
        with self._lock:
            seen = sorted(self._explain_latencies)
        return seen[len(seen) // 2] if seen else 0.0

    def _observe(self, latency_s: float, confidence: float = None) -> None:
        with self._lock:
            self._latencies.append(latency_s)
            if len(self._latencies) > 32:
                del self._latencies[0]
            if confidence is not None:
                self._confidences.append(float(confidence))
                if len(self._confidences) > 32:
                    del self._confidences[0]

    # Below this spread, the number the model reports is a habit rather than
    # a judgement. Deliberately small: the question is not "is it well
    # calibrated" but "does it move at all".
    CALIBRATION_EPSILON = 0.02
    CALIBRATION_SAMPLES = 8

    @property
    def calibrated(self) -> bool:
        """Does this model's self-reported confidence carry any information?

        Measured on the shipped 4B file: 0.95 on all five benchmark cases,
        and 0.92-0.95 across a live run. A constant fed into a weighted
        fusion is not evidence - it is an offset. At ai_weight 0.35 it was
        contributing a fixed third of every final score while discriminating
        nothing, which is worse than useless, because the confidence bands
        are then reading a number that cannot move.

        Assumed true until there is enough evidence otherwise, so a model
        that does calibrate is never penalised for a quiet start.
        """
        with self._lock:
            seen = list(self._confidences)
        if len(seen) < self.CALIBRATION_SAMPLES:
            return True
        return (max(seen) - min(seen)) > self.CALIBRATION_EPSILON

    # ------------------------------------------------------------- loading
    @property
    def available(self) -> bool:
        return self._llm is not None

    @property
    def device(self) -> str:
        return self._device

    def _shown_remote_url(self) -> str:
        """The configured remote, or only its backends' names when hidden."""
        if not _hide_backends():
            return self.remote_url
        return ",".join(name for name, _url, _key
                        in parse_backends(self.remote_url))

    def describe(self) -> Dict[str, Any]:
        """Everything an operator needs to know about this worker."""
        return {
            "loaded": self.available,
            "device": self._device,
            # Asking initialises CUDA here; when the model runs elsewhere
            # that costs this host memory for an answer about nothing.
            "gpu_offload_supported": (None if self.remote_url
                                      else _gpu_offload_supported()),
            "gpu_layers_requested": self.gpu_layers,
            "context": self.context,
            "threads": self.threads,
            "model_file": (self._remote_model
                           or os.path.basename(self.model_path)),
            "where": self._where or "this host",
            "remote": bool(self.remote_url),
            # The address and the weights it reported - never the key - so a
            # client can tell which server this service is really using.
            "remote_url": self._shown_remote_url(),
            # Each backend's state, and whether one answered recently
            # enough to be warm - what decides if a demo may start now.
            "backends": (self._failover.status(
                addresses=not _hide_backends()) if self._failover else []),
            "awake": bool(self._failover and self._failover.awake()),
            "sha256": self._remote_sha,
            "model_size_gb": (round(os.path.getsize(self.model_path) / 1e9, 2)
                              if os.path.exists(self.model_path) else None),
            "load_error": self._load_error,
            "loaded_at": self._loaded_at,
            # Whether a confidence bar means a probability on this host:
            # "token" when the odds can be read, "written" when the reply's
            # own number is all there is (a server that refuses logprobs, a
            # file with no chat template), "unknown" before the first
            # adjudication.
            "confidence": ("unknown" if self._reader is None
                           else "token" if self._reader else "written"),
            "prompt_version": self.prompt_version,
        }

    def load(self) -> bool:
        """Load the weights. Safe to call repeatedly; only the first works."""
        if self._llm is not None:
            return True
        with self._lock:
            if self._llm is not None:
                return True
            if self.remote_url:
                return self._load_remote()
            if not os.path.exists(self.model_path):
                self._load_error = "model file not found: %s" % self.model_path
                self._device = "unavailable"
                log.error(self._load_error)
                return False
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                self._load_error = "llama-cpp-python not installed (%s)" % exc
                self._device = "unavailable"
                log.error(self._load_error)
                return False

            started = time.time()
            supported = _gpu_offload_supported()
            layers = self.gpu_layers if supported else 0
            if self.gpu_layers != 0 and not supported:
                # Say it once, loudly. A GPU deployment that silently serves
                # from CPU is the failure mode this service exists to avoid.
                log.warning(
                    "this llama-cpp-python build has no GPU offload compiled "
                    "in - serving on CPU. Install a CUDA/Metal/ROCm build on "
                    "the inference host to use the accelerator.")
            try:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context,
                    n_gpu_layers=layers,
                    n_threads=self.threads,
                    n_batch=self.batch,
                    # A model larger than RAM is still servable: llama.cpp
                    # maps the file and the OS pages it in. It is slow, not
                    # broken, which is the right trade on a host that is
                    # waiting for its accelerator.
                    use_mmap=True,
                    verbose=False,
                )
            except Exception as exc:
                self._load_error = "model load failed: %s" % exc
                self._device = "unavailable"
                log.error(self._load_error)
                return False
            self._device = "gpu" if (supported and layers != 0) else "cpu"
            self._loaded_at = time.time()
            log.info("adjudicator model loaded on %s in %.1fs (%s)",
                     self._device, time.time() - started,
                     os.path.basename(self.model_path))
            if self._token_reader():
                log.info("the AI's confidence is read from its token odds")
            return True

    def remote_client(self) -> FailoverLlama:
        """The remote backends, from BIS_AI_URL / BIS_AI_KEY, made once."""
        if self._failover is None:
            self._failover = FailoverLlama(
                [(name, RemoteLlama(url, key)) for name, url, key
                 in parse_backends(self.remote_url, self.remote_key)])
        return self._failover

    def set_remote_backends(self, entries: List[Tuple[str, str, str]]) -> None:
        """Replace the remote backends at runtime, ``[(name, url, key)]`` in
        order. An entry whose address and key did not change keeps its
        client, and with it its rest and its last answer."""
        failover = self.remote_client()
        current = dict(failover.clients())
        fresh = []
        for name, url, key in entries:
            url = url.rstrip("/")
            had = current.get(name)
            if (had is not None and getattr(had, "url", None) == url
                    and getattr(had, "key", None) == key):
                fresh.append((name, had))
            else:
                fresh.append((name, RemoteLlama(url, key)))
        failover.set_backends(fresh)
        self.remote_url = ",".join("%s=%s" % (name, url.rstrip("/"))
                                   for name, url, _key in entries)

    def _load_remote(self) -> bool:
        """Connect to the remote server instead of loading weights here."""
        client = self.remote_client()
        try:
            info = client.info()
        except Exception as exc:
            self._load_error = ("remote AI unreachable at %s (%s)"
                                % (self.remote_url, exc))
            self._device = "unavailable"
            log.error(self._load_error)
            return False
        self._llm = client
        # Whether this server reports token odds decides what a bar is held
        # against here, and /v1/health says which.
        self._reader = _remote_reader(client)
        self._device = str(info.get("device") or "remote")
        self._where = str(info.get("where") or info.get("backend")
                          or self.remote_url)
        self._remote_model = str(info.get("model_file") or "")
        self._remote_sha = str(info.get("sha256") or "")
        self._loaded_at = time.time()
        log.info("adjudicator model served remotely: %s on %s (%s)",
                 self._remote_model or "model", self._device, self._where)
        return True

    # ---------------------------------------------------------- inference
    def _system(self) -> str:
        """The system prompt, plus a reasoning budget when one is asked for.

        A reasoning model emits its thinking as tokens. The reply here is a
        grammar-constrained JSON object with nowhere to put them, so asking
        for the least reasoning is what stops the token budget going on a
        preamble that gets discarded.
        """
        if not self.reasoning_effort:
            return SYSTEM_PROMPT
        return "Reasoning: %s\n%s" % (self.reasoning_effort, SYSTEM_PROMPT)

    @staticmethod
    def build_prompt(intent: str, broken_selector: str, action: str,
                     page_title: str, shortlist: List[str],
                     heuristic_pick: int = 0,
                     required_text: Optional[str] = None,
                     version: int = 2,
                     baseline: Optional[str] = None) -> str:
        """The question the AI is asked, in the form its weights learned.

        Version 2 is what every fine-tune so far was trained on, and
        build_llm_dataset writes its prompts with this method, so it must
        not move by a character. Version 3 drops the two lines naming the
        ranker's first choice - an AI told it hands it back (invariant
        38) - and, when the service has one, shows the element as it was
        the last time the step worked (`baseline`, one compact_v2 line).
        The candidate lines themselves are the caller's, in the rendering
        the same version was trained on.
        """
        lines = [
            "BUSINESS INTENT: " + intent,
            "STEP ACTION: " + action,
            "BROKEN SELECTOR: " + broken_selector,
        ]
        if version >= 3 and baseline:
            lines.append("LAST KNOWN GOOD: " + baseline)
        lines.append("PAGE: " + (page_title or "-"))
        if required_text:
            # A hard constraint, stated as one: the validation gate on the bot
            # will reject any pick that fails it, so spending prompt on it pays.
            lines.append("HARD CONSTRAINT: the element text must match the "
                         "regular expression /%s/" % required_text)
        lines += ["", "CANDIDATES (harvested from the live DOM):"]
        lines += shortlist
        lines.append("")
        if version < 3:
            lines += [
                "The heuristic ranker's current best guess is index %d."
                % heuristic_pick,
                "Confirm it or choose better.",
            ]
        lines.append('Answer JSON: {"choice": <index or -1>, '
                     '"confidence": <0.0-1.0>, "reason": "<max 25 words>"}')
        return "\n".join(lines)

    def _token_reader(self) -> Optional[Dict[str, Any]]:
        """What the token read needs from this model file, or None.

        Probed once. It needs the file's own chat template, so the prompt
        the odds are read after is the one a reply would be written after,
        and one token each for " ", " -" and the ten digits - true of the
        shipped file's tokenizer, where " -" opens -1 and " " precedes a
        digit. A remote server is probed when it connects instead
        (_remote_reader). Anything missing leaves the written confidence
        in place, which is how it always was.
        """
        if self._reader is not None:
            return self._reader or None
        if isinstance(self._llm, (RemoteLlama, FailoverLlama)):
            # None when nobody answered the probe: asked again next time,
            # rather than settling "no odds" because a GPU was still cold.
            self._reader = _remote_reader(self._llm)
            return self._reader or None
        self._reader = False
        llm = self._llm
        if llm is None or isinstance(llm, RemoteLlama) or not hasattr(
                llm, "eval"):
            return None
        try:
            template = (getattr(llm, "metadata", None) or {}).get(
                "tokenizer.chat_template")
            render = _chat_formatter(llm) if template else None
            if render is None:
                log.info("no chat template to read the AI's odds with - "
                         "its written confidence is used")
                return None

            def one(text):
                ids = llm.tokenize(text.encode("utf-8"), add_bos=False,
                                   special=False)
                return ids[0] if len(ids) == 1 else None

            space, minus = one(" "), one(" -")
            digits = [one(str(d)) for d in range(10)]
            if space is None or minus is None or None in digits:
                log.info("this tokenizer splits the first token of the "
                         "answer - the AI's written confidence is used")
                return None
        except Exception as exc:
            log.warning("cannot read the AI's odds here (%s) - its written "
                        "confidence is used", str(exc)[:120])
            return None
        self._reader = {"render": render, "space": space, "none": minus,
                        "digits": digits}
        return self._reader

    def _read_choice(self, reader: Dict[str, Any], messages: List[dict],
                     n: int) -> Optional[Dict[int, float]]:
        """P(the answer is index k) for each k, and P(-1). Lock held.

        Two forward passes over the prompt the reply would follow, and no
        generation: the odds of " -" against " " straight after
        `{"choice":`, then of each digit after the space. Mass that went
        anywhere else - a newline, an index past the shortlist - is
        renormalised away, exactly as in the probe that measured it: AUC
        0.93 against 0.59 for the written number, on the 1,000 held-out
        cases the model comparison used.

        The context is reset first, so nothing a previous call left in the
        cache can reach the read; a reason generated after it re-uses this
        prompt through llama.cpp's own prefix match.
        """
        if reader.get("remote"):
            return self._read_choice_remote(messages, n)
        llm = self._llm
        rendered = reader["render"](messages=messages)
        text = rendered.prompt + '{"choice":'
        ids = llm.tokenize(text.encode("utf-8"),
                           add_bos=not getattr(rendered, "added_special",
                                               False),
                           special=True)
        llm.reset()
        llm.eval(ids)
        opening = _last_logits(llm)
        if opening is None:
            return self._no_odds_here()
        p_space, p_none = _softmax_at(opening,
                                      [reader["space"], reader["none"]])
        llm.eval([reader["space"]])
        after = _last_logits(llm)
        if after is None:
            return self._no_odds_here()
        digits = _softmax_at(after, reader["digits"][:n])
        mass, total = p_space + p_none, sum(digits)
        if mass <= 0.0 or total <= 0.0:
            return None
        none = p_none / mass
        odds = {k: (1.0 - none) * d / total for k, d in enumerate(digits)}
        odds[-1] = none
        return odds

    def _no_odds_here(self) -> None:
        """Give up the token read for this process. Lock held.

        A blank row is not an answer, and the release that left it will
        leave the next one blank too, so the written answer is used from
        here on - and /v1/health says so - rather than paying two forward
        passes per repair for nothing.
        """
        log.warning("this llama.cpp build left no logits to read the AI's "
                    "odds from - its written confidence is used")
        self._reader = False
        return None

    def _read_choice_remote(self, messages: List[dict],
                            n: int) -> Optional[Dict[int, float]]:
        """The same odds, from a remote server's reply. Lock held.

        A server lends no forward pass, but llama.cpp's reports the top
        alternatives of every token it writes (logprobs - on by default
        there, through logits_all) - the same distribution, read where the
        reply opens its choice: " -" against " ", then the digit after the
        space. The reply is cut a few tokens past the index.

        When the reply wrote " -" the digits were never scored, and need
        not be: " -" outweighed " ", so no single index can outweigh none.
        """
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=REMOTE_READ_TOKENS,
            temperature=0.0, top_p=0.9,
            response_format={"type": "json_object",
                             "schema": DECISION_SCHEMA},
            logprobs=True, top_logprobs=REMOTE_TOP)
        steps = _reply_steps(out)
        text, at = "", None
        for i, (token, _alternatives) in enumerate(steps):
            text += token
            if text.endswith('"choice":'):
                at = i + 1
                break
        if at is None or at >= len(steps):
            return None
        token, alternatives = steps[at]
        p_space = _probability(alternatives.get(" "))
        p_none = _probability(alternatives.get(" -"))
        if p_space + p_none <= 0.0:
            return None
        none = p_none / (p_space + p_none)
        if token == " -":
            return {-1: none}
        if token != " " or at + 1 >= len(steps):
            return None
        digits = [_probability(steps[at + 1][1].get(str(d)))
                  for d in range(n)]
        total = sum(digits)
        if total <= 0.0:
            return None
        odds = {k: (1.0 - none) * d / total for k, d in enumerate(digits)}
        odds[-1] = none
        return odds

    def _messages(self, prompt: str) -> List[dict]:
        return [{"role": "system", "content": self._system()},
                {"role": "user", "content": prompt}]

    def _explain(self, messages: List[dict], choice: int) -> str:
        """The reply's reason, generated with its choice fixed. "" if not.

        Fixed by the schema - `choice` becomes a constant in the grammar -
        so the reason is about the element that was picked. Generated
        freely, the reply names a different index for about one case in
        twenty-four, and a reason about another element is worse than none.
        """
        schema = dict(DECISION_SCHEMA, properties=dict(
            DECISION_SCHEMA["properties"], choice={"const": int(choice)}))
        started = time.time()
        try:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=messages, max_tokens=self.max_tokens,
                    temperature=0.0, top_p=0.9,
                    response_format={"type": "json_object",
                                     "schema": schema})
        except Exception as exc:
            log.warning("could not generate the AI's reason (%s)",
                        str(exc)[:120])
            return ""
        with self._lock:
            self._explain_latencies.append(time.time() - started)
            if len(self._explain_latencies) > 32:
                del self._explain_latencies[0]
        raw = ((out.get("choices") or [{}])[0].get("message")
               or {}).get("content", "")
        parsed = _extract_json(raw or "")
        try:
            if not parsed or int(parsed.get("choice", -2)) != int(choice):
                return ""
        except (TypeError, ValueError):
            return ""
        return str(parsed.get("reason", ""))[:220]

    def explain(self, intent: str, broken_selector: str, action: str,
                page_title: str, shortlist: List[str], choice: int,
                heuristic_pick: int = 0,
                required_text: Optional[str] = None,
                baseline: Optional[str] = None) -> str:
        """Why the AI would pick `choice` from this shortlist. Never raises.

        For a decision someone will read - one held for review. The odds
        that decided it need no reply, and a reply is most of the cost of
        asking the AI, so it is no longer written for every decision.
        """
        if not self.load():
            return ""
        prompt = self.build_prompt(intent, broken_selector, action,
                                   page_title, shortlist, heuristic_pick,
                                   required_text, self.prompt_version,
                                   baseline)
        return self._explain(self._messages(prompt), choice)

    def adjudicate(self, intent: str, broken_selector: str, action: str,
                   page_title: str, shortlist: List[str],
                   heuristic_pick: int = 0,
                   required_text: Optional[str] = None,
                   explain: bool = True,
                   baseline: Optional[str] = None) -> Verdict:
        """Pick an index from the shortlist. Never raises.

        Any failure is reported as an abstention so the caller can fall back to
        the deterministic ranking rather than losing the step.

        Where the odds can be read (see _token_reader) the choice is the
        most likely index and the confidence its probability; ``explain``
        then says whether to also generate a reason, which costs what the
        whole reply used to. Elsewhere the reply is generated and parsed,
        as it always was.
        """
        verdict = Verdict(device=self._device)
        if not self.load():
            verdict.error = self._load_error or "model unavailable"
            return verdict
        # Read the device *after* loading. Before this, the first heal in every
        # process reported "unloaded" - which is exactly the run an operator
        # reads to find out whether the GPU is being used.
        verdict.device = self._device

        prompt = self.build_prompt(intent, broken_selector, action, page_title,
                                   shortlist, heuristic_pick, required_text,
                                   self.prompt_version, baseline)
        messages = self._messages(prompt)
        started = time.time()
        if 0 < len(shortlist) <= TOKEN_READ_MAX:
            try:
                with self._lock:
                    reader = self._token_reader()
                    odds = (self._read_choice(reader, messages,
                                              len(shortlist))
                            if reader else None)
            except Exception as exc:
                log.warning("could not read the AI's odds (%s) - asking for "
                            "its written answer", str(exc)[:120])
                odds = None
            if odds:
                choice = max(odds, key=odds.get)
                verdict.choice = int(choice)
                verdict.confidence = max(0.0, min(1.0, odds[choice]))
                verdict.source = "token"
                verdict.odds = dict(odds)
                verdict.ok = True
                verdict.latency_s = time.time() - started
                self._observe(verdict.latency_s, verdict.confidence)
                if explain:
                    verdict.reason = self._explain(messages, verdict.choice)
                return verdict
        try:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0,      # same page, same verdict
                    top_p=0.9,
                    response_format={"type": "json_object",
                                     "schema": DECISION_SCHEMA},
                )
        except Exception as exc:
            verdict.latency_s = time.time() - started
            verdict.error = "inference failed: %s" % exc
            log.warning("adjudication failed after %.1fs: %s",
                        verdict.latency_s, exc)
            return verdict
        verdict.latency_s = time.time() - started
        self._observe(verdict.latency_s)

        raw = ((out.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        parsed = _extract_json(raw or "")
        if parsed is None:
            verdict.error = "unparseable response: %s" % (raw or "")[:120]
            return verdict
        try:
            verdict.choice = int(parsed.get("choice", -1))
            verdict.confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            verdict.reason = str(parsed.get("reason", ""))[:220]
            verdict.ok = True
            verdict.source = "written"
            # Recorded only once the number is real, so an unparseable reply
            # never looks like a model that answers the same thing every time.
            self._observe(verdict.latency_s, verdict.confidence)
        except (TypeError, ValueError) as exc:
            verdict.error = "bad fields: %s" % exc
        return verdict

    def warmup(self) -> bool:
        """Load the weights and run one token, so the first real request does
        not pay the load cost and misreport steady-state latency."""
        if not self.load():
            return False
        try:
            with self._lock:
                self._llm.create_completion("ok", max_tokens=1, temperature=0.0)
            return True
        except Exception as exc:
            log.warning("warmup generation failed: %s", exc)
            return False


_backends_registered = False


def _register_backends() -> None:
    """Register ggml's dynamic backends so the GPU probe can see them.

    Current llama-cpp-python builds ship every ggml backend as its own
    shared library and register them lazily, inside ``Llama.__init__``.
    Until that happens ``llama_supports_gpu_offload()`` answers False on a
    perfectly good CUDA build - so this service announced CPU, passed
    ``n_gpu_layers=0``, and offloaded nothing, which is exactly the silent
    fallback the split exists to prevent. Registering first is cheap and
    idempotent; builds that link their backends statically have no such
    symbol and need nothing done.
    """
    global _backends_registered
    if _backends_registered:
        return
    try:
        import ctypes
        import pathlib

        import llama_cpp
        from llama_cpp._ggml import ggml_backend_load_all_from_path

        lib = pathlib.Path(llama_cpp.__file__).resolve().parent / "lib"
        ggml_backend_load_all_from_path(
            ctypes.c_char_p(str(lib).encode("utf-8")))
    except Exception:
        # Older or statically linked builds. The probe below is still
        # authoritative either way, so this is not worth a warning.
        pass
    _backends_registered = True


def _chat_formatter(llm):
    """The model file's own chat template, as a formatter, or None.

    The same Jinja formatter llama-cpp-python builds from a GGUF template
    for create_chat_completion, so the odds are read after exactly the
    prompt a reply would follow. None when the file carries no template or
    this build has no such formatter; the written answer is used then.
    """
    template = (getattr(llm, "metadata", None) or {}).get(
        "tokenizer.chat_template")
    if not template:
        return None
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except Exception:
        return None

    def text(token_id):
        if token_id is None or token_id < 0:
            return ""
        try:
            return llm._model.token_get_text(token_id)
        except Exception:
            try:
                return llm.detokenize([token_id]).decode("utf-8", "ignore")
            except Exception:
                return ""

    try:
        return Jinja2ChatFormatter(template=template,
                                   eos_token=text(llm.token_eos()),
                                   bos_token=text(llm.token_bos()))
    except Exception as exc:
        log.warning("the model's chat template would not load (%s)",
                    str(exc)[:120])
        return None


def _remote_reader(client) -> Any:
    """{"remote": True} if this server reports token odds, else False.

    Asked once, with a one-token completion. A server whose model was
    loaded without logits_all refuses logprobs; one that answers without
    them is no help either. Both keep the written answer.
    """
    try:
        got = client.create_completion("ok", max_tokens=1, temperature=0.0,
                                       logprobs=1)
    except urllib.error.HTTPError as exc:
        # The server answered: it does not lend its odds.
        log.info("the remote AI reports no token odds (%s) - its written "
                 "confidence is used", str(exc)[:120])
        return False
    except Exception as exc:
        # Nobody answered - a GPU still waking. Not a verdict on the odds.
        log.info("the remote AI did not answer the odds probe (%s) - asking "
                 "again later", str(exc)[:120])
        return None
    steps = _reply_steps(got)
    if steps and steps[0][1]:
        return {"remote": True}
    log.info("the remote AI reports no token odds - its written confidence "
             "is used")
    return False


def _reply_steps(out) -> List[tuple]:
    """(token, {alternative: logprob}) for each token a reply wrote.

    Read from either shape llama-cpp-python returns: a chat reply's
    `logprobs.content`, or a completion's `tokens` and `top_logprobs`.
    """
    try:
        logprobs = ((out or {}).get("choices") or [{}])[0].get(
            "logprobs") or {}
        content = logprobs.get("content")
        if isinstance(content, list):
            return [(str(c.get("token", "")),
                     {str(t.get("token", "")): t.get("logprob")
                      for t in (c.get("top_logprobs") or [])})
                    for c in content]
        tokens, tops = logprobs.get("tokens"), logprobs.get("top_logprobs")
        if isinstance(tokens, list) and isinstance(tops, list):
            return [(str(t), dict(a or {})) for t, a in zip(tokens, tops)]
    except (AttributeError, TypeError, ValueError):
        pass
    return []


def _probability(logprob) -> float:
    """exp() of a log probability, 0.0 for one that was not reported."""
    import math
    try:
        return math.exp(float(logprob))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _last_logits(llm):
    """The logits after the last token evaluated, or None if there are none.

    Releases disagree about where eval() leaves them when logits_all is off:
    0.3.49 copies the row into scores[0]; 0.3.36, the prebuilt CUDA 12.4
    wheel, keeps none, so scores[0] is the buffer as allocated - zeros, which
    read as " -" tied with " " and made "none" the answer to every shortlist
    of a Kaggle replay. llama.cpp's context holds the row on every release
    (0.3.49's own eval copies it from there), so it is read from there, and
    scores[0] only where there is no context to ask. A row with no spread,
    or with a non-finite value, is what a buffer nothing wrote into looks
    like: no reading at all.
    """
    get = getattr(getattr(llm, "_ctx", None), "get_logits_ith", None)
    if get is not None:
        n_vocab = int(llm.n_vocab())
        pointer = get(-1)
        try:
            import numpy as np
            row = np.ctypeslib.as_array(pointer, shape=(n_vocab,)).copy()
        except ImportError:
            row = [pointer[i] for i in range(n_vocab)]
    else:
        row = llm.scores[0]
    try:
        import numpy as np
        z = np.asarray(row, dtype=np.float64)
        if not np.isfinite(z).all() or float(z.max() - z.min()) == 0.0:
            return None
    except ImportError:
        import math
        if not all(math.isfinite(v) for v in row) or max(row) == min(row):
            return None
    return row


def _softmax_at(row, ids: List[int]) -> List[float]:
    """Softmax over a whole row of logits, read at ``ids``."""
    try:
        import numpy as np
        z = np.asarray(row, dtype=np.float64)
        z = np.exp(z - z.max())
        total = float(z.sum())
        return [float(z[i]) / total for i in ids]
    except ImportError:
        import math
        top = max(row)
        total = sum(math.exp(v - top) for v in row)
        return [math.exp(row[i] - top) / total for i in ids]


def _gpu_offload_supported() -> bool:
    try:
        from llama_cpp import llama_supports_gpu_offload
        _register_backends()
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def _extract_json(raw: str) -> Optional[dict]:
    """Parse JSON out of a response, tolerating fences and stray prose."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, depth = -1, 0
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None
