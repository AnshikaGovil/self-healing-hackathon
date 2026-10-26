# Setting up Bot Immune System on another machine

Everything here starts from the public repository:

```bash
git clone https://github.com/AnshikaGovil/self-healing-hackathon.git
cd self-healing-hackathon
```

Pick the path that matches what you want to do:

| you want to | follow | accounts needed |
| --- | --- | --- |
| **A.** run the Wikipedia demonstration against the existing public service | Part 1 | none |
| **B.** manage the existing deployment (warm it up, switch it off, add a GPU) from another machine | Parts 2, 3, 4 and 7 | the same Modal workspace, and the Hugging Face login that reads the model |
| **C.** deploy your own copy on new accounts - your own Modal, and optionally Lightning and Kaggle | Parts 2 to 7 | a Modal account, read access to the model weights, optionally Lightning and Kaggle |

What a deployment is:

```
caller (wiki_bot.py, bis.py, curl)
   |
   v
front door: the healing service on a Modal CPU, per call   memory, ranking, policy
   |  asks the first GPU that answers
   +--> 1. the judge: the fine-tuned model on a Modal T4, per call, capped
   +--> 2. a Lightning T4 Studio, when one is pasted in
   +--> 3. a Kaggle T4 notebook, when one is pasted in
   +--> none answers: the ranker alone - never an error
```

Both Modal parts start on the first call and stop by themselves (the judge
five minutes after its last call, the front door twenty), so between
demonstrations nothing is billed. One monthly ledger charges both and
stops them before the month's free credit is gone.

---

## Part 1 - Run the demonstration (no accounts)

You need Python 3.9 or later, Google Chrome and an internet connection.

```bash
pip install selenium
cd Wikipedia_POC
python wiki_bot.py --design old           # 1. learn on Wikipedia's 2010 design
python wiki_bot.py --design new --plain   # 2. the 2022 design, no healing: stops at step 4
python wiki_bot.py --design new           # 3. healing on: 7 of 7, three steps repaired
python wiki_bot.py --design new           # 4. again: the repairs come from memory
```

Selenium fetches the matching Chrome driver itself. The bot uses the public
service unless told otherwise: `--service URL`, or the `BIS_SERVICE_URL`
environment variable, points it at another one - your own after Part 5.
After a quiet spell the first run prints `waiting for the healing service
and its AI` for up to a minute or two while the service and its GPU wake,
then starts by itself. Run steps 1 to 4 in one sitting: the service keeps
a demonstration's memory for twenty minutes after its last call.
`Wikipedia_POC/README.md` explains each step and what to say over it.

---

## Part 2 - Prepare the machine

You need **Python 3.10 to 3.14** (the Modal client's range) and **Git**.

Make a virtual environment in the repository and install the tools the
deployment uses:

```bash
python -m venv .venv
# Windows (PowerShell):    .venv\Scripts\Activate.ps1
# macOS / Linux:           source .venv/bin/activate
pip install modal huggingface_hub selenium
```

Add `pip install lightning-sdk` if you will use Lightning (Part 6).

On Windows, set `PYTHONUTF8=1` before running the Modal CLI yourself, or
its progress output can fail on the console's code page:

```powershell
$env:PYTHONUTF8 = "1"
```

`deploy/bis_cloud.py` sets it for every Modal call it makes.

---

## Part 3 - Accounts and logins

### Modal - runs the front door and the judge

1. **New account:** sign up at [modal.com](https://modal.com). The free
   Starter plan comes with a monthly credit ($30 when this was written).
   In the dashboard's billing settings, set the workspace's own spending
   limit so Modal itself stops at the credit - the ledger in Part 4 stops
   earlier, but this is the limit Modal enforces.
2. **Log in from this machine** - one of:
   * `python -m modal setup` - opens a browser and saves a token in
     `~/.modal.toml`. For an existing workspace (path B), log in as a
     member of that same workspace.
   * a token pair from the dashboard's API tokens page, put in
     `deploy/accounts.env` as `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
     (Part 4). A pair in the file wins over the machine's login.
3. **Several accounts on one machine:** `python -m modal profile list`
   shows them, and `MODAL_PROFILE` in `deploy/accounts.env` picks one.
4. **Check:** `python -m modal profile current` names the workspace, and
   `python -m modal billing summary` says what this month has used.

Modal names every address after the workspace
(`https://<workspace>--bis-healing.<region>.modal.run`), so a new account
gives the public service a new address.

### Hugging Face - read access to the model weights

The judge is a fine-tuned GGUF model (`healing-adjudicator.gguf`, 2.5 GB)
kept in the **private** model repository
`anshikagovil/bis-healing-adjudicator`. The deployment downloads it with a
Hugging Face token that can read that repository:

* **The repository's owner:** log in with `hf auth login`, or put a token
  in `HF_TOKEN` in `deploy/accounts.env`.
* **Anyone else:** ask the owner for a *fine-grained* token with read
  access to that one repository only (huggingface.co > Settings > Access
  Tokens > Create new token > Fine-grained), and put it in `HF_TOKEN` in
  `deploy/accounts.env`. Keep it out of every file you commit.
* **Or host the weights yourself:** copy the file into a model repository
  of your own, change `REPO` at the top of `deploy/modal_backend.py` to
  it, and for Kaggle or Lightning (Part 6) add `--repo <you>/<repo>` to
  the model server's command in the pasted cell.

Without access the judge cannot start; the front door then answers from
the ranker alone, which still heals but holds most repairs for review.

### Lightning AI - optional, a fallback GPU

Sign up at [lightning.ai](https://lightning.ai). Nothing on this machine
needs a Lightning login: the fallback runs inside a Lightning Studio,
from lines you paste into its terminal (Part 6).

### Kaggle - optional, a free fallback GPU

Sign up at [kaggle.com](https://www.kaggle.com) and verify a phone number
in your account settings - Kaggle requires it before a notebook may use a
GPU or the internet. Kaggle gives 30 free GPU hours a week.

---

## Part 4 - The accounts file

Every token, key and budget the deployment uses lives in one file, which
git ignores:

```bash
cp deploy/accounts.env.example deploy/accounts.env
```

Open it; each line is explained in the file. What to fill in:

| line | what to put |
| --- | --- |
| `MODAL_PROFILE` | the profile to use, or empty for this machine's default Modal login |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | a token pair, or empty to use the login above |
| `MODAL_GPU` | `T4` (about $0.70 an hour while running) or `L4` |
| `BIS_BUDGET_USD` | this month's total at which the judge refuses every call - under what is left of the month's credit (`python -m modal billing summary`). With a full $30, `24` |
| `BIS_FRONT_BUDGET_USD` | the total at which the front door stops too, a little higher, so the ranker outlives the GPU. With a full $30, `26` |
| `HF_TOKEN` | the token from Part 3, or empty to use `hf auth login` |
| `LIGHTNING_HOURS` | how long a Lightning fallback runs before it stops itself |
| `BIS_BACKEND_KEY`, `BIS_ADMIN_KEY` | **leave empty** - made on the first deploy and kept here |
| `MODAL_URL`, `FRONT_URL` | **leave empty** - written by the deploy commands |

**Path B, managing an existing deployment:** copy the original
`deploy/accounts.env` from the machine that deployed it, over a private
channel - it holds the deployment's keys - and skip to Part 7. If you
start a fresh file instead, run both deploy commands in Part 5 once: the
addresses stay the same (same workspace), the judge and the front door get
new matching keys, and any Kaggle or Lightning fallback then needs the new
admin key.

---

## Part 5 - Deploy

Run these from the repository root, in the virtual environment.

### 5.1 The judge (GPU)

```bash
python deploy/bis_cloud.py modal deploy
```

It makes the deployment's keys if the file has none, stores the judge's
secret on Modal (the HF token, its key and its budget), builds the GPU
image, downloads the weights into a Modal volume on a CPU - not on GPU
time - and deploys. About 5 to 10 minutes the first time. It ends with

```
judge deployed: https://<workspace>--bis-judge-judge-serve.<region>.modal.run
```

and keeps that address in the file as `MODAL_URL`.

### 5.2 The front door - your public address

```bash
python deploy/bis_cloud.py front deploy
```

It stores the front door's secret (where the judge is, its key, the admin
key, the budget), builds the CPU image, checks that the healing service
imports in it, deploys, and waits until the service reports itself ready:

```
front door deployed: https://<workspace>--bis-healing.<region>.modal.run
ready after 38 s - AI ai on Modal Tesla T4
```

That address is the public API. It is kept in the file as `FRONT_URL`.

### 5.3 Check it

```bash
python deploy/bis_cloud.py status
curl https://<workspace>--bis-healing.<region>.modal.run/v1/health
```

`status` shows both parts, how many containers are running, and what the
ledger and the Modal account have spent this month; it never wakes a
sleeping deployment. `/v1/health` does wake it, and answers `"ready":
true` with `"mode": "ai"` once the judge answers.

Then run Part 1's four commands with `--service` and your address.

### 5.4 Point the repository at your address (new accounts only)

The bot and the documents name the original address. Either pass
`--service <your address>` (or set `BIS_SERVICE_URL`) every time, or
change `PUBLIC_SERVICE` near the top of `Wikipedia_POC/wiki_bot.py` and
the address in `README.md`, `Wikipedia_POC/README.md` and
`deploy/README.md`, then commit the change to your copy of the
repository.

---

## Part 6 - Optional fallback GPUs

Both run the same model server as the judge, publish it through a
Cloudflare tunnel and register it with the front door, which then asks it
whenever the GPUs before it do not answer. Both need `FRONT_URL` in the
accounts file (Part 5.2).

### Kaggle

1. Create a new notebook. In its settings: Accelerator **GPU T4 x2**,
   Internet **on**.
2. Under Add-ons > Secrets, add two secrets: `HF_TOKEN` (the token from
   Part 3) and `BIS_ADMIN_KEY` (its value is in `deploy/accounts.env`).
3. Print the cell, paste it into the notebook and run it:

   ```bash
   python deploy/bis_cloud.py kaggle cell
   ```

4. It installs the server, loads the model (about 2 minutes), registers
   as `kaggle` and keeps its place for up to 11 hours. Stop the session
   from Kaggle when you are done.

### Lightning

1. Create a Studio and switch its machine to a **T4**.
2. Print the lines and paste them into the Studio's terminal:

   ```bash
   python deploy/bis_cloud.py lightning cell --hours 4
   ```

3. They ask for the HF token and the admin key without showing them,
   install the server, register as `lightning`, and stop the Studio after
   the hours given. This path has not been run end to end: if the Studio
   is still running afterwards, stop it from its page - it bills while it
   runs.

`python deploy/bis_cloud.py status` lists every backend the front door
knows and whether it is up.

---

## Part 7 - Day to day

| command | what it does |
| --- | --- |
| `python deploy/bis_cloud.py status` | what is up and what it has cost, without waking anything |
| `python deploy/bis_cloud.py modal warm on` | hold one T4 awake (about $0.70 an hour) - before a live demonstration, so nobody waits for a wake |
| `python deploy/bis_cloud.py modal warm off` | let it sleep again - after the demonstration |
| `python deploy/bis_cloud.py modal off` | the judge refuses every call; the front door asks the next backend, or answers from the ranker |
| `python deploy/bis_cloud.py modal on` | the judge serves again |
| `python deploy/bis_cloud.py modal stop` | end a running judge container now |
| `python deploy/bis_cloud.py front stop` | end the front door now, and its memory with it |
| `python deploy/bis_cloud.py register NAME URL KEY --priority N` | add a backend by hand |
| `python deploy/bis_cloud.py unregister NAME` | take one away |

**Changing a cap:** edit `BIS_BUDGET_USD` or `BIS_FRONT_BUDGET_USD` in the
accounts file, then run `modal on` (the judge reads its budget) and `front
deploy` (the front door reads its own).

**A new month:** the ledger starts again from $0 each calendar month
(UTC), as Modal's credit does. Raise the caps when a new month's credit
arrives.

**Moving to other accounts later:** change the Modal lines (and `HF_TOKEN`
if needed) in the accounts file, empty `MODAL_URL` and `FRONT_URL`, and
run Part 5 again; then Part 5.4 for the new address.

---

## Optional - run the healing service on your own machine

Instead of the Modal front door, the service can run locally - asking the
Modal judge for its AI, or with no AI at all. From the repository root, in
a Python 3.10+ virtual environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy/front/requirements.txt
python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
cd healing-service
```

The last `python` line puts the sentence encoder in the local cache; the
service loads it offline. Then, with the judge as its AI (both values are
in `deploy/accounts.env`):

```bash
# PowerShell:  $env:BIS_AI_URL = "modal=<MODAL_URL>"; $env:BIS_AI_KEY = "<BIS_BACKEND_KEY>"
# bash:        export BIS_AI_URL="modal=<MODAL_URL>" BIS_AI_KEY="<BIS_BACKEND_KEY>"
python -m healing_service --host 127.0.0.1 --port 8077
```

or with no AI - the ranker alone:

```bash
python -m healing_service --host 127.0.0.1 --port 8077 --no-ai
```

and point the bot at it with `--service http://127.0.0.1:8077`.

---

## When something goes wrong

| what you see | do this |
| --- | --- |
| `No HF token` | set `HF_TOKEN` in the accounts file, or run `hf auth login` |
| the weights download fails with 401 or 403 | the token cannot read the model repository - Part 3 |
| `No MODAL_URL in accounts.env` | run `modal deploy` before `front deploy` |
| `No front door yet` | run `front deploy`, or pass `--front <address>` |
| a Modal command fails with a token or workspace error | log in again (`python -m modal setup`), or check `MODAL_PROFILE` and the token pair in the accounts file |
| `/v1/health` says `"mode": "ranker-only"` | no GPU answered: check `status`; the judge may be switched off, capped for the month, or unable to read the weights |
| every call is refused with 402 | the month's ledger reached its cap - raise the cap (Part 7) only if the account still has credit |
| the demonstration stops with `repair held for review` at step 4 | step 1 was not run in this sitting, or the service had no AI - run step 1, then step 3 |
