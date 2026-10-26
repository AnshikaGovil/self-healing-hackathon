"""Serve the healing adjudicator from a cloud GPU, for a host without one.

The AI and the fine-tuned checker do not fit one 4 GB laptop card together:
both on it spilled into system memory and took the machine to 91% (see
POC/UI_Automation/README.md). This puts the AI on a free Kaggle GPU instead.
The healing service on the laptop runs with ``--ai-url`` and asks it every
AI question over HTTPS (``inference.RemoteLlama``); the checker stays local.

In a Kaggle notebook - accelerator GPU T4, internet on, the HF_TOKEN secret
attached - with this file in /kaggle/working:

    !pip -q install "llama-cpp-python[server]==0.3.36" \\
        --only-binary llama-cpp-python \\
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    !python /kaggle/working/kaggle_ai_server.py

It then:

1. downloads the GGUF from the private model repo, with the HF_TOKEN
   secret (read, never printed);
2. loads it fully onto one GPU behind llama.cpp's own OpenAI-compatible
   server, so a request is answered exactly as a local ``Llama`` answers
   it - the same weights and the same grammar-constrained JSON;
3. publishes the server through a Cloudflare quick tunnel and prints the
   address and a key made for this session alone. Without the key,
   nothing answers;
4. stops when ``/bis/stop`` is called - the demonstration calls it when
   its show ends - or after ``--idle-min`` minutes without a request, so a
   forgotten notebook does not hold a GPU.

The address and key go into the client's ``robot/state/remote-ai.json``,
or to the service as ``--ai-url`` and ``BIS_AI_KEY``. The Kaggle session
itself keeps running after the server stops; stop it from the notebook.

``--model PATH --local`` serves a file already on disk at 127.0.0.1 with no
tunnel - the same server, for checking the protocol on one machine.

The same file serves the other GPU platforms behind the public front door,
each with the options its platform needs:

    --platform NAME   named in /bis/info's "where" (default Kaggle)
    --host 0.0.0.0    the address to bind (default 127.0.0.1)
    --key-env NAME    a fixed key from that variable instead of one made for
                      the session - for a platform whose address the front
                      door already holds with its key (Modal)
    --no-tunnel       the platform publishes the port itself
    --idle-min 0      never stop for idleness: a platform that scales to
                      zero does that itself
    --register URL    announce the address and key to the front door at
                      URL (/v1/admin/backends, with BIS_ADMIN_KEY from the
                      environment) as --name at --priority, again every
                      ten minutes, and withdraw them on the way out
    --max-hours H     stop after H hours and then run --stop-cmd - for a
                      machine that is switched on and bills until stopped
    --cap-file PATH   while that file says "capped", answer every request
                      with 402: a spending ledger writes it, and the front
                      door rests a backend that says 402 for an hour
"""
# No ``from __future__ import annotations``: FastAPI reads the handlers'
# annotations when the routes are added, and Request is imported inside
# main(), where a postponed annotation could not find it.
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = "anshikagovil/bis-healing-adjudicator"
FILE = "healing-adjudicator.gguf"
CLOUDFLARED = ("https://github.com/cloudflare/cloudflared/releases/latest/"
               "download/cloudflared-linux-amd64")
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
# The weights alone are ~2.5 GB, so a card holding less than this has not
# taken them and the "GPU" llama.cpp reports is only what it was built for.
MIN_GPU_MIB_IN_USE = 1500
# How often a registered server announces itself again. The front door
# keeps registrations in memory, so one that restarted has forgotten it.
REGISTER_EVERY_S = 600.0


def registration(url, key, name, priority):
    """What --register posts: this server as one backend of the front
    door's failover - its address, its key, where it stands in the order."""
    return {"name": name, "url": (url or "").rstrip("/"), "key": key,
            "priority": int(priority)}


def ready_json(url, key, shared_env):
    """The line a person running the notebook copies: the address, and the
    key when it was made for this session. A key shared with the front
    door (--key-env) outlives the session and the platform keeps every
    printed line, so it is named by its variable instead."""
    shown = "(in $%s)" % shared_env if shared_env else key
    return "REMOTE_AI_JSON " + json.dumps({"url": url, "key": shown})


def stop_due(started, now, max_hours):
    """Has a server given --max-hours run its time? 0 is no limit."""
    return max_hours > 0 and now - started >= max_hours * 3600.0


def idle_due(last, now, idle_min):
    """Has it gone --idle-min minutes without a request? 0 is never."""
    return idle_min > 0 and now - last > idle_min * 60.0


def capped(path):
    """Does the spending ledger's file say this backend's cap is reached?"""
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read(64).strip().lower().startswith("capped")
    except OSError:
        return False


def admin_call(front, path, admin_key, payload=None, method="POST"):
    """One call to the front door's admin API; True when it took it."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        front.rstrip("/") + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 "X-Admin-Key": admin_key})
    try:
        # Long: a free Space that went to sleep takes a while to wake.
        with urllib.request.urlopen(req, timeout=60) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        print("The front door did not take it (%s)." % str(exc)[:120],
              flush=True)
        return False


def tunnel_address(line):
    """The public address in one line of cloudflared's log, or ""."""
    match = TUNNEL_URL.search(line or "")
    return match.group(0) if match else ""


def authorised(header, key):
    """Does an Authorization header carry this session's key?"""
    return hmac.compare_digest((header or "").encode("utf-8"),
                               ("Bearer " + key).encode("utf-8"))


def hf_token():
    """The HF_TOKEN secret, from the environment or Kaggle's secret store."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception:
        return ""


def sha256_of(path):
    """The file's sha256, so the client can prove which weights answer it."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia(query):
    """One value nvidia-smi reports for the first card, or ""."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,"
             "nounits", "-i", "0"], capture_output=True, text=True,
            timeout=20).stdout
        return out.strip().splitlines()[0].strip()
    except Exception:
        return ""


def start_tunnel(port, wait_s=90.0):
    """Start a quick tunnel to the local port; (process, public address)."""
    path = "/tmp/cloudflared"
    if not os.path.exists(path):
        urllib.request.urlretrieve(CLOUDFLARED, path)
        os.chmod(path, 0o755)
    proc = subprocess.Popen(
        [path, "tunnel", "--no-autoupdate", "--url",
         "http://127.0.0.1:%d" % port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found = []
    ready = threading.Event()

    def pump():
        # Read every line for as long as it runs: a full pipe would stall
        # cloudflared, and with it the tunnel.
        for line in proc.stdout:
            if not found and tunnel_address(line):
                found.append(tunnel_address(line))
                ready.set()
        ready.set()

    threading.Thread(target=pump, daemon=True).start()
    ready.wait(wait_s)
    return proc, (found[0] if found else "")


def wait_public(url, key, wait_s=150.0):
    """The server's info through the tunnel, once its new name resolves."""
    deadline = time.time() + wait_s
    req = urllib.request.Request(url + "/bis/info",
                                 headers={"Authorization": "Bearer " + key})
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 402:
                # Up, and its spending cap says no: it answered.
                return {"capped": True}
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def main(argv=None):
    """Download, load, publish, then wait for the stop. The exit code."""
    began = time.time()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--file", default=FILE)
    ap.add_argument("--model", default="",
                    help="a GGUF already on disk - skips the download")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--idle-min", type=float, default=30.0, dest="idle_min",
                    help="stop after this many minutes without a request "
                         "(0 = never)")
    ap.add_argument("--local", action="store_true",
                    help="serve at 127.0.0.1 only, with no tunnel")
    ap.add_argument("--platform", default="Kaggle",
                    help="the platform named in /bis/info (default Kaggle)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to bind (default 127.0.0.1)")
    ap.add_argument("--key-env", default="", dest="key_env",
                    help="take the key from this environment variable "
                         "instead of making one for the session")
    ap.add_argument("--no-tunnel", action="store_true", dest="no_tunnel",
                    help="no Cloudflare tunnel - the platform publishes "
                         "the port")
    ap.add_argument("--register", default="",
                    help="the front door to announce this server to")
    ap.add_argument("--name", default="",
                    help="the backend's name there (default: the platform)")
    ap.add_argument("--priority", type=int, default=10,
                    help="where it stands in the failover (lower first)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    dest="max_hours",
                    help="stop after this many hours (0 = no limit)")
    ap.add_argument("--stop-cmd", default="", dest="stop_cmd",
                    help="a command to run once the server has stopped")
    ap.add_argument("--cap-file", default="", dest="cap_file",
                    help="answer 402 while this file says capped")
    args = ap.parse_args(argv)
    if args.local:
        args.host, args.no_tunnel = "127.0.0.1", True
    name = args.name or args.platform.lower()

    # Both checked before minutes go on a download and a load.
    if args.key_env:
        key = os.environ.get(args.key_env, "")
        if not key:
            print("--key-env %s: that variable is empty." % args.key_env)
            return 2
    else:
        key = secrets.token_urlsafe(24)
    admin_key = os.environ.get("BIS_ADMIN_KEY", "")
    if args.register and not admin_key:
        print("--register needs the front door's admin key in "
              "BIS_ADMIN_KEY.")
        return 2

    # One card is enough for a 4B model; splitting it over Kaggle's two T4s
    # would only add traffic between them. Set before llama.cpp loads.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import llama_cpp
    import uvicorn
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    path = args.model
    if not path:
        token = hf_token()
        if not token:
            print("No HF_TOKEN - attach it to this notebook under Add-ons > "
                  "Secrets, then run this again.")
            return 2
        from huggingface_hub import hf_hub_download
        print("Downloading %s from %s ..." % (args.file, args.repo),
              flush=True)
        path = hf_hub_download(args.repo, args.file, token=token,
                               local_dir="/tmp/bis-model")
    print("Hashing the model file ...", flush=True)
    digest = sha256_of(path)

    started = time.time()
    app = create_app(
        server_settings=ServerSettings(host=args.host, port=args.port,
                                       api_key=key),
        model_settings=[ModelSettings(model=path, n_gpu_layers=-1,
                                      n_ctx=args.context, n_batch=512,
                                      verbose=False)])
    used = nvidia("memory.used")
    on_gpu = (bool(llama_cpp.llama_supports_gpu_offload())
              and used.isdigit() and int(used) >= MIN_GPU_MIB_IN_USE)
    where = "%s %s" % (args.platform, nvidia("name") or "GPU")
    print("Model loaded in %.0f s on %s (%s MiB in use)."
          % (time.time() - started, where if on_gpu else "the CPU",
             used or "?"), flush=True)
    info = {"device": "gpu" if on_gpu else "cpu", "where": where,
            "model_file": os.path.basename(path), "sha256": digest,
            "context": args.context}
    state = {"last": time.time(), "reason": ""}
    stop = threading.Event()

    @app.middleware("http")
    async def seen(request, call_next):
        state["last"] = time.time()
        if capped(args.cap_file):
            # Every path, /bis/info too: a wake that found this backend up
            # would hand the next AI question to a server that refuses it.
            return JSONResponse({"detail": "this backend's spending cap is "
                                 "reached"}, status_code=402)
        return await call_next(request)

    @app.get("/bis/info")
    def bis_info(request: Request):
        if not authorised(request.headers.get("authorization"), key):
            return JSONResponse({"detail": "Invalid API key"},
                                status_code=401)
        return info

    @app.post("/bis/stop")
    def bis_stop(request: Request):
        if not authorised(request.headers.get("authorization"), key):
            return JSONResponse({"detail": "Invalid API key"},
                                status_code=401)
        state["reason"] = "the client asked it to stop"
        stop.set()
        return {"stopping": True}

    server = uvicorn.Server(uvicorn.Config(app, host=args.host,
                                           port=args.port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.2)
    if not server.started:
        print("The server did not start.")
        return 1

    if args.no_tunnel:
        tunnel, url = None, "http://127.0.0.1:%d" % args.port
    else:
        tunnel, url = start_tunnel(args.port)

    def shut_down():
        server.should_exit = True
        if tunnel is not None:
            tunnel.terminate()

    if not url:
        print("The Cloudflare tunnel did not report an address.")
        shut_down()
        return 1
    if wait_public(url, key) is None:
        print("The address %s never answered." % url)
        shut_down()
        return 1

    print("=" * 64)
    print("REMOTE AI READY")
    print(ready_json(url, key, args.key_env))
    print("model  %s  sha256 %s" % (info["model_file"], digest))
    print("on     %s (%s)" % (where, info["device"]))
    print("Stops on /bis/stop%s%s." % (
        ", after %g minutes without a request" % args.idle_min
        if args.idle_min > 0 else "",
        ", or after %g hours" % args.max_hours if args.max_hours > 0
        else ""))
    print("=" * 64, flush=True)

    registered = threading.Event()

    def announce():
        # Again on a timer: the front door keeps registrations in memory,
        # and one that restarted would otherwise never ask this server.
        entry = registration(url, key, name, args.priority)
        while not stop.is_set():
            if admin_call(args.register, "/v1/admin/backends", admin_key,
                          entry):
                if not registered.is_set():
                    print("Registered with %s as %s, priority %d."
                          % (args.register, name, args.priority),
                          flush=True)
                registered.set()
            stop.wait(REGISTER_EVERY_S if registered.is_set() else 30.0)

    if args.register:
        threading.Thread(target=announce, name="bis-register",
                         daemon=True).start()

    state["last"] = time.time()
    try:
        while not stop.wait(5.0):
            now = time.time()
            if idle_due(state["last"], now, args.idle_min):
                state["reason"] = ("no request for %g minutes"
                                   % args.idle_min)
                break
            if stop_due(began, now, args.max_hours):
                state["reason"] = "its %g hours are up" % args.max_hours
                break
            if not thread.is_alive():
                state["reason"] = "the server stopped"
                break
            if tunnel is not None and tunnel.poll() is not None:
                state["reason"] = "the tunnel closed"
                break
    except KeyboardInterrupt:
        state["reason"] = "interrupted"
    stop.set()
    if registered.is_set():
        # First, so the front door stops asking an address about to go.
        admin_call(args.register, "/v1/admin/backends/" + name, admin_key,
                   method="DELETE")
    shut_down()
    thread.join(20)
    print("REMOTE AI STOPPED - %s." % state["reason"], flush=True)
    if args.stop_cmd:
        print("Running the stop command ...", flush=True)
        try:
            subprocess.run(args.stop_cmd, shell=True, timeout=600)
        except Exception as exc:
            print("The stop command failed (%s) - stop the machine by "
                  "hand, or it keeps billing." % exc, flush=True)
    else:
        print("Stop the %s session too, or it keeps its GPU."
              % args.platform, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
