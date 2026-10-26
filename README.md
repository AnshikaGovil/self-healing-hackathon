# Bot Immune System - self-healing element lookup for UI and RPA bots

When a bot's selector stops matching - a vendor release renamed a class,
moved a link into a menu, rebuilt a button - the bot does not stop. It
sends the healing service the elements on its page; the service ranks them
against what the step meant, asks a fine-tuned AI on a GPU, and answers
with the index of the element to use. The bot builds the new selector in
its own browser, checks that it finds exactly that element, and carries
on. A repair that is not sure enough is held for a person, never guessed.

## The public API

    https://anshikagovil--bis-healing.us-east.modal.run

| call | what it does |
| --- | --- |
| `GET /v1/health` | ready or not, and whether the AI is awake. Asking wakes a sleeping GPU |
| `GET /v1/js` | the JavaScript a bot runs in its page to list and describe elements |
| `POST /v1/adjudicate` | which of the page's elements is the one the step meant |
| `POST /v1/baseline` | what a working element looked like, kept for the next repair |
| `GET /docs` | every endpoint, with its request and response schema |

```bash
curl https://anshikagovil--bis-healing.us-east.modal.run/v1/health
```

The service sleeps when nobody uses it. The first call after a quiet spell
takes 20-40 s, and the AI's GPU a little longer: `/v1/health` says
`"ready": false` and `"mode": "waking"` until it answers, then `"ready":
true` and `"mode": "ai"`. If no GPU answers, it is ready anyway in
`"ranker-only"` mode - every call still gets an answer, never an error.

## Run the demonstration

A bot written for Wikipedia's 2010 design, run on its 2022 redesign. You
need Python 3.9+, Google Chrome and the internet:

```bash
pip install selenium
cd Wikipedia_POC
python wiki_bot.py --design old           # 1. learn on the 2010 design
python wiki_bot.py --design new --plain   # 2. the 2022 design, no healing: stops at step 4
python wiki_bot.py --design new           # 3. healing on: 7 of 7, three steps repaired
python wiki_bot.py --design new           # 4. again: the repairs come from memory
```

[Wikipedia_POC/README.md](Wikipedia_POC/README.md) has what each step
shows, what was measured, and what to say over it.

## Use it in your own bot

Copy `rpa-toolkit/bis.py` next to the script - one file, standard library
only - and change the lookups:

```python
from bis import find

# before: driver.find_element(By.ID, "submit_btn").click()
find(driver, By.ID, "submit_btn", describe="the button that submits the form").click()
```

```bash
export BIS_SERVICE_URL=https://anshikagovil--bis-healing.us-east.modal.run
export BIS_SESSION=my-team-demo     # your own memory on the shared service
export BIS_HTTP_TIMEOUT=90          # a repair takes about 20 s here
```

It works with Selenium and Playwright and returns the library's own
element; the module docstring of `bis.py` is the full guide.

## What is in this repository

| folder | what it is |
| --- | --- |
| `rpa-toolkit/` | `bis.py`, the client a bot imports, and its tests |
| `healing-service/` | the service: ranking, policy, memory, the AI over HTTP, the API |
| `bis-core/` | the standard-library core both halves share, and its tests |
| `Wikipedia_POC/` | the demonstration bot |
| `deploy/` | how the public service runs: the front door and the GPU judge on Modal, the spending caps, the fallbacks |

The fine-tuned model weights are not in the repository. A service started
from it without them answers from the ranker alone, or asks any
llama.cpp-compatible server named in `BIS_AI_URL`.

## Tests

All offline - no browser, no service, no model, no network:

```bash
python bis-core/tests/test_similarity.py
python rpa-toolkit/tests/test_bis.py
python deploy/tests/test_deploy.py
```

## How the public service runs

The front door is the healing service itself on a Modal CPU, started per
call. It asks a GPU in order - the judge on a Modal T4, which also starts
per call, then a Lightning or Kaggle GPU when one is registered - and
falls back to the ranker when none answers. A monthly ledger stops both
before their free credit runs out. [deploy/README.md](deploy/README.md) has
the details and the control script.
