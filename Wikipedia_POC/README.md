# Wikipedia POC - self-healing on a real website

A bot written for **Wikipedia's 2010 design** runs on **Wikipedia's 2022
redesign** - the design Wikipedia switched to in 2023. Wikipedia still serves
both, live, at the same address:

| version | URL switch | what the bot meets |
| --- | --- | --- |
| OLD - Vector legacy (2010) | `?useskin=vector` | all 7 steps work |
| NEW - Vector 2022 | `?useskin=vector-2022` | 3 of the 7 steps broken |

What the redesign really changed, and how each break was repaired through
the public healing service (measured 2026-10-26):

| step | selector written for 2010 | on the 2022 design | repaired to | by |
| --- | --- | --- | --- | --- |
| Log in link | `#pt-login a` | still in the page, but hidden in a menu | `#pt-login-2 > a:nth-of-type(1)` | the LLM confirmed the ranker, 73% |
| search box | `input.vector-search-box-input` | class removed | `input[aria-label="Search Wikipedia"]` | the LLM confirmed the ranker, 100% |
| search button | `#searchButton` | gone; a new button with no id | `#searchform > div:nth-of-type(1) > button...` | the LLM confirmed the ranker, 89% |

Title, section headings, last-edited date and the search result use
selectors that survived the redesign, so they pass untouched.

Files: `wiki_bot.py` (the bot) and this README. The bot imports
`../rpa-toolkit/bis.py`, the one-file client. It writes only `memory/`,
which git ignores.

**Each demo keeps its own memory.** Step 1 starts a new *session* on the
healing service - a memory of its own, named by a short id the bot saves
in `memory/session.txt` - and the runs after it heal from what that run
learned, and nothing else. Running step 1 again starts a fresh one, so
there is nothing to reset between demos, and one person's demo never
warms another's.

---

## Run the demo against the public service

Nothing to start: the bot uses the public healing service,

    https://anshikagovil--bis-healing.us-east.modal.run

whose AI runs on a cloud GPU behind it. You need Python 3.9 or later,
Google Chrome and an internet connection. From the repository root:

```bash
pip install selenium
cd Wikipedia_POC
```

Chrome opens while the bot works, ringing each element it uses - blue when
the selector worked, green when it was repaired. When the run ends, Chrome
closes and the terminal shows the result table. Add `--headless` to any
command to run without a window.

The public service sleeps when nobody uses it. The first run after a quiet
spell prints `waiting for the healing service and its AI` for up to a
minute or two while it and its GPU wake, then starts by itself.

**Step 1 - learn on the 2010 design** (about 15 s)

```bash
python wiki_bot.py --design old
```

Shows `memory : session ... - new`, `learning: 6 of 6 elements fingerprinted
on the 2010 design`, then 7 steps `OK` and `LEARNED`. Say: *"This is the bot
as it runs today, on the design it was written for. Every element it uses
is fingerprinted for the healing service."*

**Step 2 - the 2022 redesign, healing OFF** (about 10 s)

```bash
python wiki_bot.py --design new --plain
```

Stops at step 4: `BOT STOPPED at step 4 (log in link): no visible element
matches #pt-login a`. Say: *"Same bot, same article, Wikipedia's new
design. One moved link and the run is dead until an engineer fixes the
script."*

**Step 3 - the 2022 redesign, healing ON** (about 90 s)

```bash
python wiki_bot.py --design new
```

Each broken step takes 25-30 s: the bot first gives the old selector its
normal 4 seconds, then the service decides - about 20 s, most of it
ranking Wikipedia's 900 elements on the service's CPU, 3-4 s of it the AI
on the GPU. Ends with `PASSED: 7 of 7 steps on the new design - 3 repaired
automatically`. Each `HEALED` row shows the new selector, the confidence
and who decided. Talk over the repairs:

- **Log in link** - the old link is still in the page, only hidden in a
  menu, so the old selector finds nothing a person could click. The
  visible copy is `#pt-login-2`; the ranker put it first and the LLM,
  asked over the ranker's shortlist, agreed: 73%, above the 62% a repair
  needs to be applied on its own.
- **Search box** - its class is gone. Its accessible name still says
  "Search Wikipedia": 100%.
- **Search button** - the old button and its id are gone; the 2022 one is
  a new component. 89%.
- The new selectors are built inside the browser from the live page. The
  AI only ever answers with the number of an element that really exists,
  so it cannot invent a selector.

**Step 4 - the same command again** (about 35 s)

```bash
python wiki_bot.py --design new
```

All three repairs now come from **fleet memory - no AI needed**. Each
healed step still shows 8-10 s: 4 of those are the bot giving the old
selector its normal chance, since a new run does not know it is broken,
and the rest is the page's elements going to the service and the stored
repair coming back. Say: *"The first robot paid for the repair; every
robot after it gets it free."*

---

## Or against your own healing service

On a machine with the healing service set up (`healing-service/README.md`),
start it and leave it running; `--checker` adds the fine-tuned Laya checker
as a first opinion, which the public service does not run:

```powershell
cd "E:\Bot Immune System\healing-service"
.venv\Scripts\python.exe -m healing_service --host 127.0.0.1 --port 8077 --gpu-layers -1 --context 4096 --checker --baselines ..\Wikipedia_POC\memory\baselines.json --repairs ..\Wikipedia_POC\memory\repairs.db
```

Ready when it prints `Uvicorn running on http://127.0.0.1:8077`. Then, in a
second terminal, point the bot at it and run steps 1-4 as above:

```powershell
cd "E:\Bot Immune System\Wikipedia_POC"
$env:BIS_SERVICE_URL = "http://127.0.0.1:8077"
```

---

## Before the next demo

Step 3 shows live AI repairs only once in a session; after that they come
from memory. To show the AI again, run step 1 again: it starts a new
session with an empty memory, and steps 2-4 then use that one. Nothing
needs stopping or deleting.

On the public service a session's memory lasts while the service is awake
- twenty minutes after its last call - so run steps 1 to 4 in one sitting.

**Keep to the default article.** On a much longer one the Log in repair
can be refused - tested on `Deep_learning`. The healer reads at most 900
elements a page, keeping the likeliest and preferring elements with an id,
and Wikipedia gives the links in its article text generated ids (`mwCXM`),
so on a long article the Log in link is not among the 900 and no model
ever sees it. The bot then stops with `no safe repair found - refused, not
guessed`. That is the safe outcome - it says it does not know instead of
clicking something else - but it is not the story to open with.

## Options

| option | default | what it does |
| --- | --- | --- |
| `--design old` / `new` | `new` | which Wikipedia design to run on |
| `--plain` | off | plain Selenium, no healing - the control run |
| `--article NAME` | `Robotic_process_automation` | the article to start on - see above before changing it |
| `--query TEXT` | `Machine learning` | what the bot searches for |
| `--pace SECONDS` | `1.0` | how long each step stays highlighted on screen |
| `--headless` | off | no browser window |
| `--service URL` | `BIS_SERVICE_URL`, else the public service | which healing service to use |
| `--session ID` | a new one for `--design old`, then the one in `memory/session.txt` | which of the service's memories to use |
| `--shared` | off | use the service's shared memory instead of a session |

## If you are asked

**"Has the AI seen Wikipedia before?"** Yes - say so. Wikipedia was one of
the 1,558 websites crawled for training data, and it fell in the training
share (the split is by website, so test websites are never trained on).
About 1,600 training cases came from Wikipedia pages, 109 of them from this
article, in today's design. What it never saw is this redesign: the
training breaks were simulated edits to today's selectors, and of the three
elements healed here, the Log in link and the search button were never a
training target at all - the search box was, 9 times, on other Wikipedia
pages. On 1,000 held-out cases from websites kept out of training, the LLM
picks the right element 85.1% of the time.

**"Why is the search button's new selector so long?"** The healer prefers
a selector that matches exactly one element. The button has no id, and its
class is shared with a second, hidden search bar that Wikipedia keeps for
its sticky header - so the repair is anchored on the nearest id,
`#searchform`.

**"What happens when it is not sure?"** It does not guess. An unsure
answer is held for a person (`repair held for review`); no answer is
refused (`no safe repair found`). Both stop the step instead of clicking
the wrong thing. With the GPU out of reach the service still answers, from
the ranker alone - and on this page the Log in repair is then held for
review, measured.

**"Is any of this specific to Wikipedia?"** No. The bot is ordinary
Selenium with its lookups going through `bis.find`, which takes the
selector plus a plain-English description of the element. Nothing in the
bot or the service knows about Wikipedia.

## If something goes wrong

| what you see | do this |
| --- | --- |
| `waiting for the healing service ...` for a while | the service and its GPU are waking; the bot carries on by itself once they are ready (it waits up to 4 minutes) |
| `The healing service ... is not answering` | the bot gave up waiting - check the internet connection, or the address given with `--service`, and run again |
| `service : ready - ranker only, no AI answered` | no GPU answered - the run still works, but expect `repair held for review` at step 4 |
| step 3 heals instantly from `fleet memory` | this session already healed once - run step 1 again for a fresh one |
| `FAILED` with `repair held for review` or `no safe repair found` | step 1 was not run against this service, or more than twenty quiet minutes passed since, or the article was changed - run steps 1 and 3 on the default article |
| `memory : the service's shared memory (no session yet ...)` | step 1 has not been run on this machine - run it first |
| no internet | the POC uses the live Wikipedia site; it needs a connection |
