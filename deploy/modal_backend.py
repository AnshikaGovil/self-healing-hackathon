"""The judge on Modal: started per call, under a spend cap.

    python -m modal run deploy/modal_backend.py::fetch   # weights, on a CPU
    python -m modal deploy deploy/modal_backend.py       # the endpoint

The same llama.cpp server the Kaggle notebook runs
(healing-service/scripts/kaggle_ai_server.py), on a T4 that Modal starts on
the first call and stops five minutes after the last. The front door holds
its address and key (BIS_AI_URL / BIS_AI_KEY); without the key nothing
answers.

A ledger in the modal.Dict ``bis-ledger`` charges every minute a container
runs (deploy/budget.py) against BIS_BUDGET_USD, one ledger per month. At
the cap the server answers 402 to everything - the front door rests a
backend that says so for an hour and asks the next one - and the container
ends itself, so a capped judge costs next to nothing while it waits. Set
the workspace's own budget in Modal's settings too: that is the limit Modal
itself enforces.

The secret ``bis-backend`` holds HF_TOKEN (the private model repo),
BIS_BACKEND_KEY (the key the front door sends) and BIS_BUDGET_USD;
``deploy/bis_cloud.py modal deploy`` makes it from deploy/accounts.env.
``bis_cloud.py modal warm on`` holds one container awake (about $0.70 an
hour) - for the hour of a live demonstration, not for days.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import budget                                                  # noqa: E402

APP = "bis-judge"
GPU = os.environ.get("BIS_MODAL_GPU", "T4")
REPO = "anshikagovil/bis-healing-adjudicator"
FILE = "healing-adjudicator.gguf"
MODELS = "/models"
MODEL = MODELS + "/" + FILE
CAP_FILE = "/tmp/bis-cap"
PORT = 8000
SERVER = os.path.join(os.path.dirname(HERE), "healing-service", "scripts",
                      "kaggle_ai_server.py")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04",
                              add_python="3.11")
    # The prebuilt CUDA 12.4 wheel: compiling llama.cpp in the image would
    # take most of an hour and the same version is what Kaggle runs.
    .pip_install("llama-cpp-python[server]==0.3.36", "huggingface_hub",
                 extra_index_url="https://abetlen.github.io/"
                                 "llama-cpp-python/whl/cu124",
                 extra_options="--only-binary llama-cpp-python")
    .add_local_file(SERVER, "/root/kaggle_ai_server.py")
    .add_local_python_source("budget")
)
volume = modal.Volume.from_name("bis-models", create_if_missing=True)
secret = modal.Secret.from_name("bis-backend")
app = modal.App(APP)


@app.function(image=image, volumes={MODELS: volume}, secrets=[secret],
              timeout=1800)
def fetch():
    """Put the weights in the volume - on a CPU, not on GPU time."""
    if os.path.exists(MODEL):
        print("already in the volume: %s (%.1f GB)"
              % (MODEL, os.path.getsize(MODEL) / 1e9))
        return
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO, FILE, token=os.environ["HF_TOKEN"],
                    local_dir=MODELS)
    volume.commit()
    print("downloaded %s (%.1f GB)" % (MODEL, os.path.getsize(MODEL) / 1e9))


class _Refuse(http.server.BaseHTTPRequestHandler):
    """The whole server of a judge past its cap: 402 to everything."""

    def _refuse(self):
        body = json.dumps({"detail": "this backend's spending cap is "
                           "reached"}).encode("utf-8")
        self.send_response(402)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_DELETE = _refuse

    def log_message(self, *args):
        return


@app.cls(image=image, gpu=GPU, volumes={MODELS: volume}, secrets=[secret],
         scaledown_window=300, max_containers=1, timeout=3600,
         startup_timeout=900)
@modal.concurrent(max_inputs=8)
class Judge:
    """One container: the model server, and the ledger that can stop it."""

    @modal.enter()
    def begin(self):
        self.ledger = modal.Dict.from_name("bis-ledger",
                                           create_if_missing=True)
        self.cap = float(os.environ.get("BIS_BUDGET_USD") or 0)
        self.month = budget.month_key()
        spent = float(self.ledger.get(self.month, 0.0))
        self.capped = budget.over(spent, self.cap)
        if self.capped:
            # Its boot and the seconds until it ends itself bill too.
            self.ledger[self.month] = budget.charge(
                spent, budget.BOOT_S + 20.0, GPU)
            self._stop(spent)
            return
        print("ledger: $%.2f of $%.2f spent this month" % (spent, self.cap),
              flush=True)
        self.lock = threading.Lock()
        # The first write charges the boot the ledger never saw.
        self.last = time.time() - budget.BOOT_S
        if not os.path.exists(MODEL):
            # fetch() was not run first; this pays GPU time for it.
            fetch.local()
        threading.Thread(target=self._charge, name="bis-ledger",
                         daemon=True).start()

    @modal.exit()
    def end(self):
        """The part-minute since the last charge, as the container goes."""
        if not self.capped:
            self._write()

    @modal.web_server(PORT, startup_timeout=300)
    def serve(self):
        if self.capped:
            refuse = http.server.ThreadingHTTPServer(("0.0.0.0", PORT),
                                                     _Refuse)
            threading.Thread(target=refuse.serve_forever,
                             daemon=True).start()
            return
        subprocess.Popen([
            sys.executable, "/root/kaggle_ai_server.py", "--model", MODEL,
            "--platform", "Modal", "--host", "0.0.0.0", "--port", str(PORT),
            "--no-tunnel", "--key-env", "BIS_BACKEND_KEY", "--idle-min", "0",
            "--cap-file", CAP_FILE])

    def _write(self):
        """Add the time since the last charge to the month; the new total,
        or None when the ledger could not be written (that time is then
        charged in full by the next write)."""
        with self.lock:
            now = time.time()
            try:
                spent = budget.charge(float(self.ledger.get(self.month, 0.0)),
                                      now - self.last, GPU)
                self.ledger[self.month] = spent
            except Exception as exc:
                print("ledger not written (%s)" % exc, flush=True)
                return None
            self.last = now
            return spent

    def _charge(self):
        """Every minute, charge this container's minute and check the cap."""
        while True:
            time.sleep(60)
            spent = self._write()
            if spent is not None and budget.over(spent, self.cap):
                self.capped = True
                self._stop(spent)
                return

    def _stop(self, spent):
        """Refuse everything from now on, then end the container: a capped
        judge left to idle would bill its whole scaledown window."""
        with open(CAP_FILE, "w", encoding="utf-8") as handle:
            handle.write("capped at $%.2f of $%.2f\n" % (spent, self.cap))
        print("CAPPED: $%.2f of $%.2f this month - refusing and stopping"
              % (spent, self.cap), flush=True)
        try:
            # Kept awake by a keep-warm switch, an ended container would be
            # replaced at once, and billed again.
            modal.Cls.from_name(APP, "Judge")().update_autoscaler(
                min_containers=0)
        except Exception as exc:
            print("keep-warm not switched off (%s)" % exc, flush=True)
        threading.Timer(20.0, os._exit, args=(0,)).start()
