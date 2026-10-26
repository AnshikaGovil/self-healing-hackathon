"""The public front door on Modal: the healing service on a CPU, per call.

    python -m modal deploy deploy/modal_front.py

The healing service itself - ranking, policy, memory, the AI asked over
HTTPS - in one CPU container that Modal starts on the first call and keeps
for twenty minutes after the last. One container at most, so every call of
a demonstration reaches the same memory; that memory lives as long as the
container, so a demonstration learns and heals in one sitting.

Its minutes go into the same monthly ledger as the judge's (budget.py), and
it refuses everything (402) once the month's total reaches
BIS_FRONT_BUDGET_USD - set above the judge's own cap, so with the GPU
capped the ranker still answers until the total is reached.

The secret ``bis-front`` holds BIS_AI_URL and BIS_AI_KEY (the judge, and
any backend after it), BIS_ADMIN_KEY and BIS_FRONT_BUDGET_USD;
``deploy/bis_cloud.py front deploy`` makes it from deploy/accounts.env.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import budget                                                  # noqa: E402

ROOT = os.path.dirname(HERE)
APP = "bis-front"
LABEL = "bis-healing"
MEMORY = "/root/run-memory"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _skip(path):
    """What the image never takes from the source tree."""
    parts = str(path).replace("\\", "/").split("/")
    return ("__pycache__" in parts or ".venv" in parts
            or str(path).endswith((".pyc", ".gguf")))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.0",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(os.path.join(HERE, "front",
                                                "requirements.txt"))
    .env({"HF_HOME": "/root/.cache/huggingface", "BIS_HIDE_BACKENDS": "1"})
    # The sentence encoder, baked in: the service loads it offline.
    .run_commands("python -c \"from sentence_transformers import "
                  "SentenceTransformer as S; S('%s')\"" % ENCODER)
    .add_local_dir(os.path.join(ROOT, "bis-core"), "/root/app/bis-core",
                   ignore=_skip)
    .add_local_dir(os.path.join(ROOT, "healing-service", "healing_service"),
                   "/root/app/healing-service/healing_service", ignore=_skip)
    .add_local_python_source("budget")
)
secret = modal.Secret.from_name("bis-front")
app = modal.App(APP)


async def _refuse(scope, receive, send):
    """The whole service, past the month's cap: 402 to everything."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return
    body = json.dumps({"detail": "this deployment's spending cap for the "
                       "month is reached"}).encode("utf-8")
    await send({"type": "http.response.start", "status": 402,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class _Ledger:
    """This container's minutes, charged to the month, and the cap."""

    def __init__(self):
        self.book = modal.Dict.from_name("bis-ledger",
                                         create_if_missing=True)
        self.month = budget.month_key()
        self.cap = float(os.environ.get("BIS_FRONT_BUDGET_USD") or 0)
        self.capped = False
        self.lock = threading.Lock()
        # The first write charges the boot the ledger never saw.
        self.last = time.time() - budget.BOOT_S

    def spent(self):
        return float(self.book.get(self.month, 0.0))

    def write(self):
        with self.lock:
            now = time.time()
            try:
                total = budget.charge(self.spent(), now - self.last, "CPU")
                self.book[self.month] = total
            except Exception as exc:
                print("ledger not written (%s)" % exc, flush=True)
                return None
            self.last = now
            return total

    def run(self):
        while True:
            time.sleep(60)
            total = self.write()
            if total is not None and budget.over(total, self.cap):
                self.stop(total)
                return

    def stop(self, total):
        self.capped = True
        print("CAPPED: $%.2f of $%.2f this month - refusing and stopping"
              % (total, self.cap), flush=True)
        threading.Timer(20.0, os._exit, args=(0,)).start()


@app.function(image=image, secrets=[secret], cpu=1.0, memory=2048,
              scaledown_window=1200, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label=LABEL)
def web():
    ledger = _Ledger()
    total = ledger.spent()
    if budget.over(total, ledger.cap):
        ledger.book[ledger.month] = budget.charge(total, budget.BOOT_S + 20.0,
                                                  "CPU")
        ledger.stop(total)
        return _refuse
    print("ledger: $%.2f of $%.2f spent this month" % (total, ledger.cap),
          flush=True)
    threading.Thread(target=ledger.run, name="bis-ledger",
                     daemon=True).start()

    sys.path.insert(0, "/root/app/healing-service")
    os.makedirs(MEMORY, exist_ok=True)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  "
                        "%(levelname)-7s %(name)-16s %(message)s",
                        datefmt="%H:%M:%S")
    from healing_service import create_app
    from healing_service.app import parse_command_line
    _args, config = parse_command_line([])
    service = create_app(
        config, warm=True,
        baseline_path=os.path.join(MEMORY, "baselines.json"),
        repair_path=os.path.join(MEMORY, "repairs.db"),
        review_path=os.path.join(MEMORY, "reviews.jsonl"),
        agent_trace_path=os.path.join(MEMORY, "agent-traces.jsonl"))

    async def gate(scope, receive, send):
        if ledger.capped and scope["type"] == "http":
            await _refuse(scope, receive, send)
            return
        await service(scope, receive, send)

    return gate


@app.function(image=image, timeout=600)
def check():
    """Import the service in the image, as the endpoint will - no call."""
    sys.path.insert(0, "/root/app/healing-service")
    import healing_service
    from sentence_transformers import SentenceTransformer
    os.environ["HF_HUB_OFFLINE"] = "1"
    SentenceTransformer(ENCODER, device="cpu")
    print("service and encoder import:", healing_service.__name__, "ok")
