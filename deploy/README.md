# The public deployment

One public address answers healing requests at any hour:

    https://anshikagovil--bis-healing.us-east.modal.run

It is the healing service itself, on one Modal CPU container that starts on
the first call (about 20-40 s from cold) and stays for twenty minutes after
the last. It keeps the memory - baselines and proven repairs, per demo
session, for as long as the container lives, so a demonstration learns and
heals in one sitting - and asks a GPU somewhere else whenever a repair
needs the AI.

```
caller (bis.py, wiki_bot.py, curl)
   |
   v
front door: Modal CPU, per call, one container    memory, ranking, policy
   |  asks the first GPU that answers
   +--> 1. Modal       T4, wakes per call, sleeps 5 min after the last; capped
   +--> 2. Lightning   T4 Studio, pasted in when needed, stops after N hours
   +--> 3. Kaggle      notebook T4, pasted in when needed, up to 11 hours
   +--> none answer:   the ranker's answer alone - never an error
```

(A free Hugging Face Space was the first choice for the front door, being
always on; Hugging Face now asks for a PRO subscription to run a Docker
Space on its free CPU, so the front door runs on Modal instead.)

A backend that does not answer rests for two minutes; one that says its
spending cap is reached (HTTP 402) rests for an hour. Asked for its health,
the front door wakes a sleeping GPU first and says `ready` once one answers
(or once none can), so a bot that waits for `ready` - `wiki_bot.py` does -
meets a warm GPU.

## What it costs, and the caps

| part | billed | the cap |
| --- | --- | --- |
| front door (Modal CPU) | per second while its container runs: about $0.06 an hour | it stops (402) once the month's ledger reaches `BIS_FRONT_BUDGET_USD` |
| judge (Modal T4) | per second while a container runs: about $0.70 an hour, all in. A demonstration costs a few cents | it refuses every call (402) once the ledger reaches `BIS_BUDGET_USD`, and a capped container ends itself |
| Lightning | per second while the Studio runs | the server stops the Studio after `--hours` |
| Kaggle | free GPU hours (30 a week) | the server withdraws after 11 hours |

Both Modal parts charge one monthly ledger, the judge's cap below the
front door's: with the GPU capped, the ranker still answers until the
month's total is reached. Set both under what is left of the month's free
credit (`python -m modal billing summary` says what is used), and set the
workspace budget in Modal's settings as well - that limit Modal enforces.

`python deploy/bis_cloud.py status` prints what is up and what Modal has
spent this month.

## Switching it on and off

Per call is the default and needs nothing: Modal sleeps when nobody calls.

```bash
python deploy/bis_cloud.py modal warm on    # before a live demo: no wake wait
python deploy/bis_cloud.py modal warm off   # after it
python deploy/bis_cloud.py modal off        # Modal refuses; the next backend answers
python deploy/bis_cloud.py modal on         # back to per call
python deploy/bis_cloud.py modal stop       # end a running judge now
python deploy/bis_cloud.py front stop       # end the front door (and its memory)
```

## Accounts: one file

Every token, key and budget lives in `deploy/accounts.env`, which git
ignores; `accounts.env.example` is the template and explains each line.
An empty line means this machine's own login.

To run on someone else's accounts - a colleague's, when your credits run
out - put their values in the file and deploy again:

```bash
python deploy/bis_cloud.py modal deploy
python deploy/bis_cloud.py front deploy
```

Their keys are needed only on the machine that deploys, and only for the
platforms deployed to; nothing in the code changes. Modal names an address
after its workspace, so on another account the public address is a new
one: `front deploy` prints it and keeps it in `accounts.env`, and
`wiki_bot.py --service URL` (or `BIS_SERVICE_URL`) points the bot at it.

## When Modal's credit runs out: a Kaggle or Lightning GPU

Both run the same model server as the Modal judge
(`healing-service/scripts/kaggle_ai_server.py`), publish it through a
Cloudflare tunnel and register it with the front door, which then asks it
whenever the GPUs before it do not answer. Nothing on the front door
changes.

**Kaggle** (free GPU hours, 30 a week). In a new notebook: Accelerator
"GPU T4 x2", Internet on, and under Add-ons > Secrets two secrets, `HF_TOKEN`
(reads the model repo) and `BIS_ADMIN_KEY` (its value is in
`accounts.env`). Then paste what this prints into a cell and run it:

```bash
python deploy/bis_cloud.py kaggle cell
```

It installs the server, loads the model (about 2 minutes), registers as
`kaggle` and keeps its place for up to 11 hours. Stop the session when you
are done.

**Lightning** (credit). Its CLI login on this machine had lapsed when the
deployment was built (the API answered 403), so this path was not run end
to end. On a T4 Studio, paste what this prints into its terminal; it asks
for the HF token and the admin key without showing them, registers as
`lightning`, and stops the Studio after the hours given:

```bash
python deploy/bis_cloud.py lightning cell --hours 4
```

`status` lists every backend the front door knows and whether it is up.

## Registering a backend by hand

A GPU whose address changes (a notebook's tunnel) announces itself, but a
backend can be added or removed by hand with the admin key from
`accounts.env`:

```bash
python deploy/bis_cloud.py register kaggle https://x.trycloudflare.com KEY --priority 3
python deploy/bis_cloud.py unregister kaggle
```

## Files

| file | what it is |
| --- | --- |
| `bis_cloud.py` | the control script |
| `accounts.env.example` | the accounts template |
| `modal_backend.py` | the judge on Modal: image, volume, ledger, endpoint |
| `modal_front.py` | the front door on Modal: the healing service on a CPU |
| `front/requirements.txt` | the front door's dependencies |
| `budget.py` | what a GPU second costs, and when the cap is reached |
| `tests/test_deploy.py` | offline tests: `python deploy/tests/test_deploy.py` |
