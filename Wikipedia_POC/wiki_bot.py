"""Wikipedia POC: a bot built for Wikipedia's 2010 design, run on its 2022 one.

Wikipedia still serves both of its desktop designs, live, at the same
address:

    old  Vector legacy (2010)  ?useskin=vector
    new  Vector 2022           ?useskin=vector-2022   (the default since 2023)

The article and its text are identical; the page around them was rebuilt.
That is exactly what a vendor release does to an RPA bot - on a real site
anyone in the room can open.

The bot does an ordinary research job on one article: read the title and
section headings, note when it was last edited, check the Log in link, then
search for the next topic and confirm where it landed. Every selector in
LOCATORS was written against the old design and works there. On the new
one the search box lost its class, the search button was replaced by a new
component, and the Log in link is still in the page but hidden in a menu,
with a visible copy elsewhere - three breaks a vendor release really made.

The old table of contents (#toc) was left out on purpose. Its 2022
replacement is found - the ranker puts the right entry first - but every
entry looks alike, so the policy holds that repair for a person instead of
applying it. Correct behaviour, and the wrong thing to show live.

    python wiki_bot.py --design old            # learn on the 2010 design
    python wiki_bot.py --design new --plain    # 2022 design, healing OFF
    python wiki_bot.py --design new            # 2022 design, healing ON

It uses the public healing service unless told otherwise (--service, or
BIS_SERVICE_URL). Each learning run starts a memory of its own there - a
session, kept in memory/session.txt for the runs after it - so one person's
demo never warms another's. README.md has the exact commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import secrets
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# bis.py is one file, normally copied next to a bot. Here it is imported from
# the toolkit beside this folder, so the POC stays a single script.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rpa-toolkit"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from selenium import webdriver                                  # noqa: E402
from selenium.common.exceptions import WebDriverException      # noqa: E402
from selenium.webdriver.chrome.options import Options           # noqa: E402
from selenium.webdriver.common.by import By                     # noqa: E402

import bis                                                      # noqa: E402

WIKI = "https://en.wikipedia.org/wiki/%s?useskin=%s"
# The healing service anyone can call, its AI on a cloud GPU behind it.
PUBLIC_SERVICE = "https://anshikagovil--bis-healing.us-east.modal.run"
SESSION_FILE = os.path.join(HERE, "memory", "session.txt")
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
DESIGNS = {
    "old": ("vector", "OLD - Wikipedia's 2010 design (Vector legacy)"),
    "new": ("vector-2022", "NEW - Wikipedia's 2022 redesign, live since 2023"),
}

# Written against the 2010 design, the way a recorder or a developer captures
# them. The description is evidence for the healer, not a comment: it names
# the element itself, never its surroundings.
LOCATORS = {
    "article.title": ("#firstHeading",
                      "the article title at the top of the page"),
    "article.headings": ("#mw-content-text .mw-heading2 h2",
                         "the section headings in the article text"),
    "article.edited": ("#footer-info-lastmod",
                       "the line saying when the page was last edited"),
    "account.login": ("#pt-login a",
                      "the Log in link for user accounts"),
    "search.box": ("input.vector-search-box-input",
                   "the search box for searching Wikipedia"),
    "search.button": ("#searchButton",
                      "the button that runs the search"),
}

# (label, locator, what the step does). The last step runs on the page the
# search lands on, so it proves the search really worked.
STEPS = [
    ("article title", "article.title", "read"),
    ("headings", "article.headings", "collect"),
    ("last edited", "article.edited", "read"),
    ("log in link", "account.login", "read"),
    ("search box", "search.box", "type"),
    ("search button", "search.button", "click"),
    ("result title", "article.title", "verify"),
]

# How the service reached a decision, in words for the room.
STRATEGY = {
    "fleet-memory": "fleet memory - no AI needed",
    "checker-sure": "Laya checker, sure on its own",
    "checker+ai-agree": "Laya checker and LLM agree",
    "ai-confirmed": "LLM confirmed the ranker",
    "ai-override": "LLM overruled the ranker",
    "heuristic": "ranker",
    "heuristic-unconfirmed": "ranker",
}


# ------------------------------------------------------------------ output
class Paint(object):
    """ANSI colour when the console can show it, plain text when it cannot."""

    def __init__(self):
        self.on = sys.stdout.isatty()
        if self.on and os.name == "nt":
            os.system("")               # switches on VT processing in conhost

    def __call__(self, text, code):
        return "\033[%sm%s\033[0m" % (code, text) if self.on else text

    def ok(self, t):
        return self(t, "1;34")

    def healed(self, t):
        return self(t, "1;32")

    def bad(self, t):
        return self(t, "1;31")

    def dim(self, t):
        return self(t, "2")

    def bold(self, t):
        return self(t, "1")


P = Paint()


def clip(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


class HealWatch(logging.Handler):
    """Listens to bis for the repairs it makes, so each step can show one."""

    def __init__(self):
        logging.Handler.__init__(self, logging.INFO)
        self.events = []

    def emit(self, record):
        # Four arguments, and a fifth since a reviewer can approve one.
        if str(record.msg).startswith("healed ") and len(record.args) >= 4:
            self.events.append({"authored": record.args[0],
                                "bound": record.args[1],
                                "pct": record.args[2],
                                "strategy": record.args[3]})
        elif (str(record.msg).startswith("declining repair")
              and len(record.args) == 3):
            self.events.append({"declined": record.args[2]})


# ------------------------------------------------------------------ browser
def open_browser(headless):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1440,900")
    else:
        opts.add_argument("--start-maximized")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    return webdriver.Chrome(options=opts)


def highlight(driver, els, colour):
    """Ring what the step is using, so the room can follow the bot."""
    try:
        driver.execute_script(
            "const els = arguments[0];"
            "for (const e of els) {"
            "  e.style.outline = '4px solid ' + arguments[1];"
            "  e.style.outlineOffset = '2px'; }"
            "if (els[0]) els[0].scrollIntoView({block: 'center'});",
            els[:30], colour)
    except WebDriverException:
        pass


def wait_for_new_page(driver, old_url, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if (driver.current_url != old_url and driver.execute_script(
                    "return document.readyState") == "complete"):
                return True
        except WebDriverException:
            pass
        time.sleep(0.25)
    return False


WAIT_S = 4.0     # how long either arm waits for an element before giving up


def plain_lookup(driver, selector, many, timeout=WAIT_S):
    """The bot as it was written: plain Selenium, waiting for a visible match.

    Same rule and same wait as bis, so the two runs differ only in whether
    a repair may be applied.
    """
    end = time.time() + timeout
    while True:
        try:
            shown = [e for e in driver.find_elements(By.CSS_SELECTOR, selector)
                     if e.is_displayed()]
        except WebDriverException:
            shown = []
        if shown:
            return shown if many else shown[:1]
        if time.time() >= end:
            raise LookupError("no visible element matches %s" % selector)
        time.sleep(0.25)


def lookup(driver, key, healing, many):
    selector, describe = LOCATORS[key]
    if not healing:
        return plain_lookup(driver, selector, many)
    if many:
        return bis.find_all(driver, By.CSS_SELECTOR, selector,
                            describe=describe)
    return [bis.find(driver, By.CSS_SELECTOR, selector, describe=describe)]


def section_names(els):
    """Section names, with any numbering and contents-box headings removed."""
    out = []
    for e in els:
        text = re.sub(r"^\d+(\.\d+)*\s*", "", " ".join(e.text.split()))
        if text and text not in ("(Top)", "Contents"):
            out.append(text)
    return out


# ------------------------------------------------------------------ the run
def run_steps(driver, healing, watch, pace, query, data, rows):
    """Run the job. Returns the step that failed, or None."""
    for n, (label, key, kind) in enumerate(STEPS, 1):
        selector = LOCATORS[key][0]
        del watch.events[:]
        started = time.time()
        error = ""
        value = ""
        try:
            els = lookup(driver, key, healing, many=(kind == "collect"))
            if not els or els[0] is None:
                raise LookupError("nothing matched")
            heal = next((e for e in watch.events if "bound" in e), None)
            highlight(driver, els, "#2E9E47" if heal else "#1428A0")

            if kind == "read":
                value = " ".join(els[0].text.split())
                if not value:
                    raise LookupError("the element has no text")
            elif kind == "collect":
                names = section_names(els)
                if len(names) < 3:
                    raise LookupError("only %d section names" % len(names))
                data["sections"] = names
                value = "%d sections: %s" % (len(names), ", ".join(names[:3]))
            elif kind == "type":
                els[0].clear()
                els[0].send_keys(query)
                value = 'typed "%s"' % query
            elif kind == "click":
                time.sleep(pace)
                before = driver.current_url
                els[0].click()
                if not wait_for_new_page(driver, before):
                    raise LookupError("the click did not open a new page")
                value = "opened /%s" % driver.current_url.split(
                    "?")[0].split("/", 3)[-1]
            elif kind == "verify":
                value = " ".join(els[0].text.split())
                if value.lower() != query.lower():
                    raise LookupError('landed on "%s"' % value)
            data[label] = value
        except Exception as exc:
            heal = next((e for e in watch.events if "bound" in e), None)
            declined = next((e for e in watch.events if "declined" in e), None)
            error = str(exc).split("\n")[0]
            # Both are the policy working: an unsure answer goes to a
            # person and no answer is refused, rather than either guessed.
            if declined and declined["declined"] == "review":
                error = "repair held for review, not applied"
            elif declined:
                error = "no safe repair found - refused, not guessed"
        took = time.time() - started
        rows.append({"n": n, "label": label, "selector": selector,
                     "heal": heal, "value": value, "error": error,
                     "secs": took})
        print_row(rows[-1])
        if error:
            return rows[-1]
        if kind != "click":
            time.sleep(pace)
            # Back to the top before the next step. Highlighting scrolls to
            # what a step used, and scrolled down, the 2022 design shows a
            # sticky header full of second copies of its controls - every
            # lookup should see the page as it was learned, not as the last
            # highlight left it.
            try:
                driver.execute_script("window.scrollTo(0, 0);")
            except WebDriverException:
                pass
    return None


ROW = "  %-2s %-14s %-33s "
INDENT = " " * len(ROW % ("", "", ""))


def print_row(r):
    head = ROW % (r["n"], r["label"], r["selector"])
    secs = "%5.1f s" % r["secs"]
    if r["error"]:
        print(head + P.bad("FAILED") + "  " + clip(r["error"], 36).ljust(36)
              + "  " + secs)
    elif r["heal"]:
        h = r["heal"]
        print(head + P.healed("HEALED") + "  " + clip(r["value"], 36).ljust(36)
              + "  " + secs)
        print(INDENT + P.healed("        -> now " + clip(h["bound"], 50)))
        print(INDENT + P.dim("           %.0f%% confidence, %s" % (
            h["pct"], STRATEGY.get(h["strategy"], h["strategy"]))))
    else:
        print(head + P.ok("OK    ") + "  " + clip(r["value"], 36).ljust(36)
              + "  " + secs)


def session_for(args):
    """The service memory this run uses, or "" for the shared one.

    A learning run starts a new session and saves it; the runs after it
    read it back, so they heal from what that run learned - and only from
    that. --session names one outright; --shared uses none.
    """
    if args.shared:
        return ""
    if args.session:
        return args.session
    if args.design == "old":
        sid = secrets.token_hex(6)
        try:
            os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
            with open(SESSION_FILE, "w", encoding="utf-8") as handle:
                handle.write(sid + "\n")
        except OSError as exc:
            print(P.bad("  could not save the session (%s) - pass "
                        "--session %s to the next runs" % (exc, sid)))
        return sid
    try:
        with open(SESSION_FILE, encoding="utf-8") as handle:
            sid = handle.read().strip()
    except OSError:
        return ""
    return sid if SESSION_ID.match(sid) else ""


def wait_for_service(timeout=240):
    """The service may be waking its AI on a cloud GPU; wait, not fail."""
    end = time.time() + timeout
    said = False
    lg = logging.getLogger("bis")
    level = lg.level
    # Every refused connection while the service starts is a warning in the
    # toolkit; here it is the expected state, so it is not printed.
    lg.setLevel(logging.ERROR)
    try:
        while time.time() < end:
            info = bis.health()
            if info and info.get("ready"):
                return info
            if not said:
                print(P.dim("  waiting for the healing service and its AI - "
                            "a cloud GPU can take a minute or two to wake; "
                            "the run starts by itself when it is ready ..."))
                said = True
            time.sleep(2)
    finally:
        lg.setLevel(level)
    return None


def flush_reports(timeout=10):
    """Let the fingerprints bis sends in the background reach the service.

    They are fire-and-forget by design, so a bot never waits on them - but
    this process is about to exit, and the next design's repairs are only
    as good as what this run managed to report.
    """
    for t in threading.enumerate():
        if t.name.startswith("bis-"):
            t.join(timeout)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=sorted(DESIGNS), default="new",
                    help="old = the 2010 design, new = the 2022 redesign")
    ap.add_argument("--plain", action="store_true",
                    help="plain Selenium, no healing (the control run)")
    ap.add_argument("--article", default="Robotic_process_automation",
                    help="the article to start on (default: %(default)s); "
                         "on a long one the healer may never see the Log "
                         "in link - see README.md")
    ap.add_argument("--query", default="Machine learning",
                    help="what to search for (default: %(default)s)")
    ap.add_argument("--pace", type=float, default=1.0,
                    help="seconds to hold each step on screen (0 = fast)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--service", default=os.environ.get(
        "BIS_SERVICE_URL") or PUBLIC_SERVICE,
        help="the healing service (default: BIS_SERVICE_URL, else the "
             "public one)")
    ap.add_argument("--session", default="",
                    help="the service memory to use, by id (default: a new "
                         "one for --design old, kept in memory/session.txt "
                         "for the runs after it)")
    ap.add_argument("--shared", action="store_true",
                    help="use the service's shared memory, no session")
    args = ap.parse_args()
    if args.session and not SESSION_ID.match(args.session):
        ap.error("--session takes letters, digits, - and _ (at most 40)")

    healing = not args.plain
    skin, design_name = DESIGNS[args.design]
    url = WIKI % (args.article, skin)
    watch = HealWatch()
    bis_log = logging.getLogger("bis")
    bis_log.setLevel(logging.INFO)
    bis_log.addHandler(watch)
    # The toolkit's own warnings stay visible: if anything goes wrong in
    # front of an audience, the reason should be on the screen.
    loud = logging.StreamHandler()
    loud.setLevel(logging.WARNING)
    loud.setFormatter(logging.Formatter("  bis: %(message)s"))
    bis_log.addHandler(loud)
    bis_log.propagate = False

    print()
    print(P.bold("  WIKIPEDIA SELF-HEALING POC"))
    print("  design  : %s" % design_name)
    print("  page    : %s" % url)
    if healing:
        session = session_for(args)
        # A repair on the public service takes about 20 s - ranking 900 of
        # Wikipedia's elements on one CPU, then the GPU - against bis.py's
        # default wait of 30 s, so the bot waits longer than that.
        bis.configure(service_url=args.service, find_timeout_s=WAIT_S,
                      session=session, http_timeout_s=90)
        print("  lookups : bis.find - self-healing ON  (service %s)"
              % args.service)
        if session:
            print("  memory  : session %s - %s" % (session, (
                "new, learned into by this run" if args.design == "old"
                and not args.session else "what its learning run recorded")))
        else:
            print("  memory  : the service's shared memory%s" % (
                "" if args.shared or args.design == "old" else
                " (no session yet - run --design old first)"))
        info = wait_for_service()
        if not info:
            print(P.bad("\n  The healing service at %s is not answering.\n"
                        "  Check the address, or start your own - "
                        "README.md.\n" % args.service))
            return 2
        ai = info.get("ai") or {}
        print(P.dim("  service : ready - %s, %s fingerprints and %s proven "
                    "repairs in memory" % (
                        "ranker only, no AI answered"
                        if ai.get("mode") in ("ranker-only", "off") else
                        "AI on %s" % (ai.get("where") or ai.get("device")
                                      or "?"),
                        info.get("baselines", 0), info.get("repairs", 0))))
    else:
        bis.configure(enabled=False)
        print("  lookups : plain Selenium - self-healing OFF")
    print()

    driver = open_browser(args.headless)
    rows, data = [], {}
    started = time.time()
    try:
        driver.get(url)
        time.sleep(max(args.pace, 1.0))
        if healing and args.design == "old":
            # Record what every step on this page looks like while it still
            # works. Seeding first also loads the healing JavaScript before
            # the first lookup, so no step goes unrecorded.
            proven = bis.seed(driver, {k: (By.CSS_SELECTOR,) + LOCATORS[k]
                                       for k in LOCATORS})
            print(P.ok("  learning: %d of %d elements fingerprinted on the "
                       "2010 design" % (sum(proven.values()), len(proven))))
            print()
        print(P.dim(ROW % ("#", "step", "selector written for 2010")
                    + "result"))
        failed = run_steps(driver, healing, watch, args.pace, args.query,
                           data, rows)
        time.sleep(max(args.pace, 1.0))
    finally:
        if healing:
            flush_reports()
        try:
            driver.quit()
        except Exception:
            pass

    total = time.time() - started
    healed = [r for r in rows if r["heal"]]
    print()
    if failed:
        print(P.bad("  BOT STOPPED at step %d (%s): %s" % (
            failed["n"], failed["label"], failed["error"])))
        if not healing:
            print(P.dim("  This is today: one changed element stops the run "
                        "until an engineer fixes the script."))
        return 1

    if "sections" in data:
        print(P.dim("  captured: %s - %d sections: %s ..." % (
            data.get("article title", ""), len(data["sections"]),
            ", ".join(data["sections"][:4]))))
    if "last edited" in data:
        print(P.dim("            %s" % clip(data["last edited"], 70)))
    if args.design == "old" and healing:
        print(P.ok("  LEARNED: %d of %d steps work on the 2010 design, and "
                   "their fingerprints are with the healing service."
                   % (len(rows), len(STEPS))))
    elif healed:
        ai_time = sum(r["secs"] for r in healed)
        from_memory = sum(1 for r in healed
                          if r["heal"]["strategy"] == "fleet-memory")
        print(P.healed("  PASSED: %d of %d steps on the %s design - %d "
                       "repaired automatically, %.1f s of healing, no human."
                       % (len(rows), len(STEPS), args.design, len(healed),
                          ai_time)))
        if from_memory < len(healed):
            print(P.dim("  The repairs are now in fleet memory: run the same "
                        "command again and they apply instantly, no AI."))
    else:
        print(P.ok("  PASSED: %d of %d steps, nothing needed repairing."
                   % (len(rows), len(STEPS))))
    print(P.dim("  total run time %.1f s" % total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
