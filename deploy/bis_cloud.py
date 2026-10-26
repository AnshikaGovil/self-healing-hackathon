"""One control script for the public deployment, reading deploy/accounts.env.

    python deploy/bis_cloud.py status            # what is up, what it costs
    python deploy/bis_cloud.py modal deploy      # the judge: key, weights
    python deploy/bis_cloud.py front deploy      # the public front door
    python deploy/bis_cloud.py modal warm on     # hold one T4 awake...
    python deploy/bis_cloud.py modal warm off    # ...or let it sleep again
    python deploy/bis_cloud.py modal off         # the judge refuses (402)
    python deploy/bis_cloud.py modal on          # the judge serves again
    python deploy/bis_cloud.py modal stop        # end a running judge now
    python deploy/bis_cloud.py front stop        # end the front door now
    python deploy/bis_cloud.py register NAME URL KEY [--priority N]
    python deploy/bis_cloud.py unregister NAME
    python deploy/bis_cloud.py kaggle cell       # a Kaggle T4 as a backend
    python deploy/bis_cloud.py lightning cell    # a Lightning T4 likewise

Everything an account is - tokens, keys, budgets - lives in accounts.env
(the template, accounts.env.example, explains each line). A line left empty
means this machine's own login. To run on a colleague's accounts, put
theirs in and run `modal deploy` and `front deploy` again; the public
address is then theirs (Modal names it after the workspace).

Both run on Modal, per call: the judge sleeps five minutes after its last
call and the front door twenty, so between demonstrations neither costs
anything. "warm on" is for the hour of a live demonstration; "modal off"
makes the judge refuse, and the front door then asks the next backend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACCOUNTS = os.path.join(HERE, "accounts.env")
JUDGE_URL = re.compile(r"https://[a-z0-9-]+--bis-judge-judge-serve"
                       r"[a-z0-9.-]*\.modal\.run")
FRONT_URL = re.compile(r"https://[a-z0-9-]+--bis-healing"
                       r"[a-z0-9.-]*\.modal\.run")
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


# ------------------------------------------------------------- the file
def load_accounts(path):
    """``KEY=value`` lines as a dict: comments, blank lines, quotes and an
    ``export`` in front read as in a shell's .env. No file, no settings."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        if line.strip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        out[key] = value
    return out


def set_value(text, key, value):
    """The file's text with ``key`` set to ``value``: its line replaced
    where it is, or one added at the end."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        match = _LINE.match(line)
        if match and match.group(1) == key:
            lines[i] = "%s=%s" % (key, value)
            return "\n".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + "%s=%s\n" % (key, value)


def save_value(key, value, path=ACCOUNTS):
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        text = ""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(set_value(text, key, value))


def modal_env(acc):
    """The environment that points the Modal CLI at the file's account: a
    token pair, else a profile in ~/.modal.toml, else this machine's
    default - and half a token pair is no account."""
    if acc.get("MODAL_TOKEN_ID") and acc.get("MODAL_TOKEN_SECRET"):
        return {"MODAL_TOKEN_ID": acc["MODAL_TOKEN_ID"],
                "MODAL_TOKEN_SECRET": acc["MODAL_TOKEN_SECRET"]}
    if acc.get("MODAL_PROFILE"):
        return {"MODAL_PROFILE": acc["MODAL_PROFILE"]}
    return {}


def ensure_keys(acc):
    """The front door's two keys, made once and kept in the file."""
    for name in ("BIS_BACKEND_KEY", "BIS_ADMIN_KEY"):
        if not acc.get(name):
            acc[name] = secrets.token_urlsafe(24)
            save_value(name, acc[name])
            print("made %s (kept in accounts.env)" % name)
    return acc


def hf_token(acc):
    """The file's HF token, else this machine's HF login."""
    if acc.get("HF_TOKEN"):
        return acc["HF_TOKEN"]
    try:
        from huggingface_hub import get_token
        return get_token() or ""
    except Exception:
        return ""


# ---------------------------------------------------------------- Modal
def run_modal(acc, args, capture=False, timeout=1800):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               BIS_MODAL_GPU=acc.get("MODAL_GPU") or "T4", **modal_env(acc))
    return subprocess.run([sys.executable, "-m", "modal"] + args, env=env,
                          cwd=ROOT, capture_output=capture, text=True,
                          timeout=timeout)


def modal_python(acc, code, timeout=120):
    """Run a few lines of the Modal SDK as the file's account; their
    output, or "" when they failed."""
    env = dict(os.environ, PYTHONUTF8="1", **modal_env(acc))
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=timeout)
    return out.stdout.strip() if out.returncode == 0 else ""


def make_secret(acc, name, values):
    """A Modal secret from ``values``, through a file that is gone again
    before this returns - never on a command line."""
    folder = tempfile.mkdtemp(prefix="bis-secret-")
    path = os.path.join(folder, name + ".env")
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join("%s=%s\n" % kv for kv in values.items()))
        out = run_modal(acc, ["secret", "create", name, "--from-dotenv",
                              path, "--force"], capture=True, timeout=120)
    finally:
        os.remove(path)
        os.rmdir(folder)
    if out.returncode != 0:
        sys.exit("Modal did not take the secret %s (exit %d)."
                 % (name, out.returncode))


def judge_secret(acc, budget=None):
    """What the judge reads: the HF token, its key and its budget."""
    token = hf_token(acc)
    if not token:
        sys.exit("No HF token: set HF_TOKEN in accounts.env, or log in "
                 "with `hf auth login`.")
    budget = acc.get("BIS_BUDGET_USD") or "3.00" if budget is None else budget
    make_secret(acc, "bis-backend", {
        "HF_TOKEN": token, "BIS_BACKEND_KEY": acc["BIS_BACKEND_KEY"],
        "BIS_BUDGET_USD": budget})
    print("secret bis-backend made, judge budget $%s" % budget)


def front_secret(acc):
    """What the front door reads: where the judge is, its key, the admin
    key, and the month's total the front door stops at."""
    if not acc.get("MODAL_URL"):
        sys.exit("No MODAL_URL in accounts.env - run `modal deploy` first.")
    budget = acc.get("BIS_FRONT_BUDGET_USD") or "3.50"
    make_secret(acc, "bis-front", {
        "BIS_AI_URL": "modal=" + acc["MODAL_URL"],
        "BIS_AI_KEY": acc["BIS_BACKEND_KEY"],
        "BIS_ADMIN_KEY": acc["BIS_ADMIN_KEY"],
        "BIS_FRONT_BUDGET_USD": budget})
    print("secret bis-front made, month's total cap $%s" % budget)


def deploy(acc, path, pattern, key, what):
    """``modal deploy`` one app; its address, kept in the file under key."""
    out = run_modal(acc, ["deploy", path], capture=True)
    found = pattern.search(out.stdout + out.stderr)
    if out.returncode != 0 or not found:
        print((out.stdout + out.stderr)[-1500:])
        sys.exit("The Modal deploy of %s failed." % path)
    if acc.get(key) != found.group(0):
        save_value(key, found.group(0))
        acc[key] = found.group(0)
    print("%s deployed: %s" % (what, found.group(0)))


def containers(acc, app_name):
    raw = run_modal(acc, ["container", "list", "--json"], capture=True,
                    timeout=120).stdout
    try:
        return [r["container_id"] for r in json.loads(raw)
                if r.get("app_name") == app_name]
    except (ValueError, KeyError, TypeError):
        return []


def stop_containers(acc, app_name, what):
    ids = containers(acc, app_name)
    for cid in ids:
        run_modal(acc, ["container", "stop", "--yes", cid], capture=True,
                  timeout=120)
    print("stopped %d running %s container(s)" % (len(ids), what))


def cmd_modal(acc, args):
    acc = ensure_keys(acc)
    what = args.what
    if what == "deploy":
        judge_secret(acc)
        print("weights into the volume (on a CPU) ...", flush=True)
        if run_modal(acc, ["run", "deploy/modal_backend.py::fetch"]
                     ).returncode != 0:
            sys.exit("Fetching the weights failed.")
        deploy(acc, "deploy/modal_backend.py", JUDGE_URL, "MODAL_URL",
               "judge")
        print("Now `front deploy`, so the front door asks this judge.")
    elif what == "warm":
        want = 1 if args.state == "on" else 0
        done = modal_python(acc, (
            "import modal\n"
            "modal.Cls.from_name('bis-judge', 'Judge')().update_autoscaler("
            "min_containers=%d)\nprint('ok')" % want))
        print("keep-warm %s%s" % (args.state, "" if done == "ok" else
                                  " - FAILED, check `modal app list`"))
        if want:
            print("One T4 stays awake (about $0.70 an hour) until `modal "
                  "warm off`.")
    elif what in ("on", "off"):
        judge_secret(acc, budget=None if what == "on" else "0")
        deploy(acc, "deploy/modal_backend.py", JUDGE_URL, "MODAL_URL",
               "judge")
        if what == "off":
            stop_containers(acc, "bis-judge", "judge")
            print("The judge refuses every call now; the front door asks "
                  "the next backend. `modal on` serves again.")
        elif acc.get("FRONT_URL"):
            # A judge that said 402 is rested for an hour; registering it
            # again under its own name gives the front door a fresh one.
            code, _body = admin(acc, args, "/v1/admin/backends", {
                "name": "modal", "url": acc["MODAL_URL"],
                "key": acc["BIS_BACKEND_KEY"], "priority": 0})
            print("front door told the judge is back: %s" % (
                "yes" if code == 200 else "no (%s)" % code))
    elif what == "stop":
        stop_containers(acc, "bis-judge", "judge")


# ----------------------------------------------------------- front door
def front(acc, args):
    url = getattr(args, "front", "") or acc.get("FRONT_URL", "")
    if not url:
        sys.exit("No front door yet: run `front deploy`, or pass --front.")
    return url.rstrip("/")


def health(url, timeout=120):
    with urllib.request.urlopen(url + "/v1/health", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cmd_front(acc, args):
    acc = ensure_keys(acc)
    if args.what == "stop":
        stop_containers(acc, "bis-front", "front door")
        print("Its memory went with it; the next call starts a fresh one.")
        return 0
    front_secret(acc)
    print("building the image and checking it imports (on a CPU) ...",
          flush=True)
    if run_modal(acc, ["run", "deploy/modal_front.py::check"]
                 ).returncode != 0:
        sys.exit("The front door's image did not import the service.")
    deploy(acc, "deploy/modal_front.py", FRONT_URL, "FRONT_URL",
           "front door")
    started, deadline = time.time(), time.time() + 600
    while time.time() < deadline:
        try:
            body = health(acc["FRONT_URL"])
        except Exception:
            time.sleep(5)
            continue
        ai = body.get("ai") or {}
        if body.get("ready"):
            print("ready after %.0f s - AI %s on %s" % (
                time.time() - started, ai.get("mode"), ai.get("where")))
            return 0
        time.sleep(3)
    print("not ready after 10 minutes - `status` says why")
    return 1


# ---------------------------------------------------- notebook backends
PUBLIC_REPO = "AnshikaGovil/self-healing-hackathon"
SERVER = ("https://raw.githubusercontent.com/%s/main/healing-service/"
          "scripts/kaggle_ai_server.py")
WHEEL = ('"llama-cpp-python[server]==0.3.36" --only-binary llama-cpp-python'
         ' --extra-index-url https://abetlen.github.io/llama-cpp-python/'
         'whl/cu124')

KAGGLE_CELL = '''\
# Bot Immune System - this notebook's T4 as an AI backend of the public
# front door. Notebook settings: Accelerator "GPU T4 x2", Internet on.
# Add-ons > Secrets: HF_TOKEN (reads the model repo) and BIS_ADMIN_KEY (its
# value is in deploy/accounts.env). The server registers itself, keeps its
# place for up to 11 hours, then withdraws; stop the session afterwards.
import os, subprocess, sys, urllib.request
from kaggle_secrets import UserSecretsClient
store = UserSecretsClient()
os.environ["HF_TOKEN"] = store.get_secret("HF_TOKEN")
os.environ["BIS_ADMIN_KEY"] = store.get_secret("BIS_ADMIN_KEY")
subprocess.run(sys.executable + ' -m pip -q install %(wheel)s',
               shell=True, check=True)
urllib.request.urlretrieve("%(server)s",
                           "/kaggle/working/kaggle_ai_server.py")
cmd = ("python /kaggle/working/kaggle_ai_server.py --register %(front)s "
       "--name kaggle --priority 3 --idle-min 0 --max-hours 11")
subprocess.run(cmd.split())
'''

LIGHTNING_CELL = '''\
# Bot Immune System - this Lightning Studio's T4 as an AI backend of the
# public front door. Paste into the Studio's terminal (machine: T4). It asks
# for the HF token and the admin key (deploy/accounts.env) without showing
# them, registers itself, and after %(hours)g hours withdraws and stops the
# Studio - or stop it from its page sooner.
read -s -p "HF token: " HF_TOKEN; echo; export HF_TOKEN
read -s -p "BIS admin key: " BIS_ADMIN_KEY; echo; export BIS_ADMIN_KEY
pip -q install %(wheel)s
curl -fsSLo kaggle_ai_server.py %(server)s
python kaggle_ai_server.py --platform Lightning --register %(front)s \\
    --name lightning --priority 2 --idle-min 0 --max-hours %(hours)g \\
    --stop-cmd "python -c 'from lightning_sdk import Studio; Studio().stop()'"
'''


def kaggle_cell(front, repo=PUBLIC_REPO):
    """The notebook cell that makes a Kaggle T4 a backend of ``front``."""
    return KAGGLE_CELL % {"front": front, "server": SERVER % repo,
                          "wheel": WHEEL}


def lightning_cell(front, hours=4.0, repo=PUBLIC_REPO):
    """The terminal lines that make a Lightning Studio a backend."""
    return LIGHTNING_CELL % {"front": front, "server": SERVER % repo,
                             "wheel": WHEEL, "hours": float(hours)}


def cmd_cell(acc, args):
    url = front(acc, args)
    if args.cmd == "kaggle":
        print(kaggle_cell(url))
    else:
        print(lightning_cell(url, args.hours or float(
            acc.get("LIGHTNING_HOURS") or 4)))


def admin(acc, args, path, payload=None, method="POST"):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        front(acc, args) + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 "X-Admin-Key": acc.get("BIS_ADMIN_KEY", "")})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read()[:200].decode("utf-8",
                                                            "replace")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, {"detail": str(exc)}


def cmd_register(acc, args):
    code, body = admin(acc, args, "/v1/admin/backends", {
        "name": args.name, "url": args.url, "key": args.key,
        "priority": args.priority})
    print("register %s: %s %s" % (args.name, code, body.get("order")
                                  or body.get("detail")))


def cmd_unregister(acc, args):
    code, body = admin(acc, args, "/v1/admin/backends/" + args.name,
                       method="DELETE")
    print("unregister %s: %s %s" % (args.name, code, body.get("order")
                                    or body.get("detail")))


def cmd_status(acc, args):
    url = getattr(args, "front", "") or acc.get("FRONT_URL", "")
    running = len(containers(acc, "bis-front"))
    print("front door  %s, %d running" % (url or "not deployed", running))
    if url and not running and not getattr(args, "front", ""):
        # Asking its health would wake it - and its GPU - to say so.
        print("            asleep, costing nothing; the next call wakes it")
    elif url:
        try:
            body = health(url.rstrip("/"))
            ai = body.get("ai") or {}
            print("            %s, AI %s on %s" % (
                "ready" if body.get("ready") else "NOT ready",
                ai.get("mode", "?"), ai.get("where") or "-"))
            for b in ai.get("backends") or []:
                print("            - %-10s %s" % (b.get("name"),
                                                  b.get("state")))
        except Exception as exc:
            print("            not answering (%s)" % str(exc)[:80])
    ledger = modal_python(acc, (
        "import json, modal, time\n"
        "d = modal.Dict.from_name('bis-ledger', create_if_missing=True)\n"
        "print(json.dumps(d.get(time.strftime('spent-%Y-%m', "
        "time.gmtime()), 0.0)))"))
    try:
        ledger = "%.2f" % float(ledger)
    except ValueError:
        ledger = "?"
    print("judge       %s, %d running" % (
        acc.get("MODAL_URL") or "not deployed",
        len(containers(acc, "bis-judge"))))
    print("ledger      $%s this month: the judge stops at $%s, the front "
          "door at $%s" % (ledger, acc.get("BIS_BUDGET_USD") or "?",
                           acc.get("BIS_FRONT_BUDGET_USD") or "?"))
    billing = run_modal(acc, ["billing", "summary", "--json"], capture=True,
                        timeout=120).stdout
    try:
        print("            the Modal account has used $%.2f of its $30 this "
              "month" % float(json.loads(billing)["metered_cost"]))
    except (ValueError, KeyError, TypeError):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--front", default="",
                    help="the front door's address (default: FRONT_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    modal = sub.add_parser("modal")
    modal.add_argument("what", choices=["deploy", "warm", "on", "off",
                                        "stop"])
    modal.add_argument("state", nargs="?", choices=["on", "off"])
    door = sub.add_parser("front")
    door.add_argument("what", choices=["deploy", "stop"])
    reg = sub.add_parser("register")
    reg.add_argument("name")
    reg.add_argument("url")
    reg.add_argument("key")
    reg.add_argument("--priority", type=int, default=10)
    unreg = sub.add_parser("unregister")
    unreg.add_argument("name")
    kaggle = sub.add_parser("kaggle")
    kaggle.add_argument("what", choices=["cell"])
    light = sub.add_parser("lightning")
    light.add_argument("what", choices=["cell"])
    light.add_argument("--hours", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.cmd == "modal" and args.what == "warm" and not args.state:
        ap.error("modal warm takes on or off")
    acc = load_accounts(ACCOUNTS)
    return {"status": cmd_status, "modal": cmd_modal, "front": cmd_front,
            "register": cmd_register, "unregister": cmd_unregister,
            "kaggle": cmd_cell, "lightning": cmd_cell}[args.cmd](
                acc, args) or 0


if __name__ == "__main__":
    sys.exit(main())
