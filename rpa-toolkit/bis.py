"""bis.py - self-healing element lookup in one function.

Copy this file next to your bot script. There is nothing to install and
nothing to import beyond this one line:

    from bis import find

Then change your element lookups:

    - driver.find_element(By.ID, "submit_btn").click()
    + find(driver, By.ID, "submit_btn").click()

``find`` returns a real element from whichever library you are already
using - a Selenium WebElement or a Playwright ElementHandle - so
``.click()`` and the rest keep working exactly as before. Hand it a
WebDriver or a Playwright Page; it works out which it was given.

    find(page, "#submit_btn").click()          # Playwright, selector only

For the elements that may or may not be on the page - consent banners,
promo pop-ups, survey overlays - say so, and you get ``None`` instead of an
exception:

    el = find(driver, By.ID, "promo-close", optional=True)
    if el:
        el.click()

That replaces the ``try / except: pass`` pattern, and it is safer than it:
an optional element is held to a stricter standard before anything is
clicked, so a missing pop-up can never be "repaired" into clicking some
other button.

Inside web components and frames
--------------------------------
A control inside a web component's shadow root, or inside a frame, is
reached through what holds it, with ``>>>`` between the steps:

    find(driver, "app-shell >>> button.save")   # in app-shell's shadow root
    find(page, "iframe#main >>> #email")        # in that frame's document

A repair for a control that a new release moved into one comes back in the
same form, so nothing else in the script changes. Only open shadow roots,
and frames from the page's own origin, can be reached - the browser lets no
script past the others.

With Selenium an element inside a frame is usable only while the driver is
in that frame, so ``find`` leaves the driver there, as ``switch_to.frame``
would, and the next ``find`` brings it back out before it looks. A script
that switches frames itself is never moved.

It gets better the more you run it
---------------------------------
Every time a lookup succeeds, this file takes a small fingerprint of the
element your step used and sends it to the healing service. Nothing is
written to your machine and nothing is asked of you.

That record is what makes a later repair confident. Judging a replacement
by name alone is guesswork; judging it against the shape, position and
label of the element that used to work is evidence, and the difference is
routinely a repair applied rather than a repair refused. Because the
service keeps them, the first bot to walk a page warms every bot that
walks it afterwards.

It also notes where each step led - the page the next step found its
element on - so a repair that sends the bot somewhere a working run never
went is caught and retired, even though it found an element to click.

Set ``BIS_BASELINES=0`` if you would rather send nothing. Healing still
works; it is just colder.

Repairs a person should decide
------------------------------
The service applies a repair on its own only when it is confident enough.
Below that bar it holds the repair for review, and by default this file
treats that as "not found" - the step fails, as it would have without
healing. To decide those yourself - a prompt for an attended bot, a ticket
or a queue for an unattended one - hand it a function:

    import bis
    bis.configure(review_hook=lambda r: input(
        "Use %s for %s? [y/N] " % (r["summary"], r["selector"])) == "y")

It is called with the repair already bound to the page - ``r["element"]``
is the element, so it can be highlighted or screenshotted - along with
``proposal``, ``confidence``, ``reason`` and ``url``. Its answer is final:
True uses the element, anything else (an exception included) is "not
found". An approved repair is shared with the rest of the fleet, so the
next bot is not asked again, and every verdict is sent back as a label for
tuning the bar. ``bis.configure(review_hook=None)`` removes it.

AI agent mode
-------------
The healing service can also run a repair as an AI agent: a language model
handed the repair's tools - scan the page, rank the candidates, ask the
fine-tuned judge, compare with the element that last worked, ask a person -
that decides, one call at a time, which to use. This file runs the tools
that need the browser when the agent asks for them, and binds its decision
exactly as it binds any other:

    import bis
    bis.configure(agent="always")           # or BIS_AGENT=always

"escalate" asks the agent only when the ordinary repair found nothing it
could apply; "off", the default, never asks it. An agent that cannot finish
- no model on the service, a reply it cannot use, out of time - hands the
step back to the ordinary repair, so turning it on can cost time but never
a step. The agent never writes a selector and never decides on its own that
a repair is applied: the service's policy does, as for any repair. To watch
it work:

    bis.configure(on_agent_step=lambda s: print(s["title"], s["result"]))

is called once per tool call, with the tool's ``title``, the agent's
``why`` for calling it and a one-line ``result``.

Configuration
-------------
One environment variable, normally set by the orchestrator:

    BIS_SERVICE_URL=http://healing-host:8077

Optionally ``BIS_API_KEY``. With no service configured this module does
nothing but call your own driver for you, so a script written against it
behaves exactly like the script you had before.

On a service shared by many people - a public demonstration - set
``BIS_SESSION`` (or ``configure(session=...)``) to a short id of letters,
digits, ``-`` and ``_``: what this run learns is kept under it, apart from
everyone else's. Unset, the service's shared memory is used.

Why this file is so small
-------------------------
Everything that decides *which* element you meant runs on the healing
service: the DOM harvesting JavaScript is downloaded from it, and the
ranking, the policy, the memory and the AI adjudication all happen there.
This file only does the things that must happen on the machine holding the
browser - try a selector, run the JavaScript it is given, bind the result
to a real element, and report which element that was.

That is the point. When the healing logic improves, the service is
upgraded and every bot benefits without being touched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

__version__ = "1.0.0"
PROTOCOL = "1"

log = logging.getLogger("bis")

try:
    from selenium.common.exceptions import (NoSuchElementException,
                                            WebDriverException)
    from selenium.webdriver.common.by import By
except ImportError:                                   # pragma: no cover
    # Importable without Selenium so the file can be linted or unit tested
    # anywhere. Any real call still needs a driver, and therefore Selenium.
    By = None

    class NoSuchElementException(Exception):
        pass

    class WebDriverException(Exception):
        pass


# --------------------------------------------------------------- settings
class _Settings(object):
    """Everything tunable, and every default readable from the environment."""

    def __init__(self):
        self.service_url = os.environ.get("BIS_SERVICE_URL", "").rstrip("/")
        self.api_key = os.environ.get("BIS_API_KEY", "")
        # Which of the service's memories this run learns into; "" is the
        # shared one. Sent with every call, so a repair is looked up in the
        # same memory its baseline was recorded in.
        self.session = os.environ.get("BIS_SESSION", "").strip()
        self.http_timeout_s = float(os.environ.get("BIS_HTTP_TIMEOUT", "30"))
        self.find_timeout_s = float(os.environ.get("BIS_FIND_TIMEOUT", "6"))
        # An optional element is not worth waiting six seconds for. It is
        # usually absent rather than broken, and absence is the common case.
        self.optional_timeout_s = float(
            os.environ.get("BIS_OPTIONAL_TIMEOUT", "1.5"))
        # How long to keep trying an authored selector this process has
        # already had to repair once. Still tried, so a page that reverts
        # is picked up immediately; not waited on for the full timeout,
        # because RPA work is loop-shaped and the same dead address comes
        # round once per row. See _probe_wait.
        self.known_broken_probe_s = float(
            os.environ.get("BIS_KNOWN_BROKEN_PROBE", "1"))
        self.enabled = os.environ.get("BIS_ENABLED", "1") != "0"
        # Report what a working element looked like, so a future repair on
        # this locator can be scored on structure and geometry as well as
        # names. Costs one JavaScript call and one background POST per
        # locator per run; set BIS_BASELINES=0 to send nothing.
        self.baselines = os.environ.get("BIS_BASELINES", "1") != "0"
        # How many elements to fingerprint from the page. This used to be a
        # literal in _heal() while HealingConfig carried a max_candidates
        # nobody read - the knob and the value had drifted into different
        # halves of the system. The value lives here, on the side that
        # actually harvests. Raising it costs harvest time on the bot and
        # ranking time on the service, in that order.
        self.max_candidates = int(os.environ.get("BIS_MAX_CANDIDATES", "900"))
        # Who decides a repair the service held for review. None means
        # nobody, and review counts as not found - the bar is the bar. Set
        # from code only (configure), since it is a function.
        self.review_hook = None
        # Whether the service's AI agent repairs a step: "off" (never),
        # "escalate" (when the ordinary repair found nothing it could
        # apply) or "always". Off unless asked for - a bot that does not
        # ask behaves exactly as it always did.
        self.agent = _agent_mode(os.environ.get("BIS_AGENT", "off"),
                                 "off")
        # Told about each tool call the agent makes, as it makes it - for
        # an overlay, a log, a console. Set from code only.
        self.on_agent_step = None


_AGENT_MODES = ("off", "escalate", "always")


def _agent_mode(asked, current):
    """``asked`` when it is an agent mode, else ``current``, said aloud."""
    mode = str(asked or "").strip().lower()
    if mode in _AGENT_MODES:
        return mode
    log.warning("%r is not an agent mode (off, escalate, always) - keeping "
                "%r", asked, current)
    return current


_S = _Settings()
_KEEP = object()            # configure()'s "leave this setting alone"
# _lock guards the JS bundle - and is held across the download, which is a
# blocking HTTP call. Nothing on the path of a step that is *working* may
# wait on it, so the baseline bookkeeping below has its own.
_lock = threading.Lock()
_bundle = {"version": "", "harvest": "", "probe": "", "describe_one": "",
           "presence": "", "same": ""}
_baseline_lock = threading.Lock()

# Authored selectors this process has already repaired at least once.
# Not a cache of repairs - only a note that the full find timeout is no
# longer worth paying on this selector. See _probe_wait.
_repaired_here = set()
_prefetching = []

# Baselines this process has already fingerprinted, so a locator used on
# fifty rows of a table is measured once. Nothing here is written to disk and
# nothing survives the run - the service keeps the durable copy, because a
# file next to the bot would need merging, locking and a place to live, and
# this is one file an RPA developer copies by hand.
_seen_baselines = set()
_baselines = {}

# The step each browser found last, so the next step can say where it led,
# and the moves this run has already reported. Per run, like the above.
_previous = {}
_reported_outcomes = set()
_step_lock = threading.Lock()


def configure(service_url=None, api_key=None, find_timeout_s=None,
              optional_timeout_s=None, http_timeout_s=None, enabled=None,
              baselines=None, max_candidates=None,
              known_broken_probe_s=None, review_hook=_KEEP, agent=None,
              on_agent_step=_KEEP, session=None):
    """Override any setting from code, for the rare bot that needs to.

    Normally the environment variables are enough and nothing calls this.

    ``review_hook`` is the exception: a function, so only code can set it.
    It decides the repairs the service holds for review - see the module
    docstring. ``review_hook=None`` removes it. ``on_agent_step`` is the
    other: it is told about each tool call the AI agent makes.
    ``on_agent_step=None`` removes it.

    ``agent`` is "off", "escalate" or "always" - see "AI agent mode" in the
    module docstring. ``session`` names the service memory this run uses
    ("" for the shared one) - see "Configuration".
    """
    if session is not None:
        _S.session = str(session).strip()
    if review_hook is not _KEEP:
        _S.review_hook = review_hook
    if on_agent_step is not _KEEP:
        _S.on_agent_step = on_agent_step
    if agent is not None:
        _S.agent = _agent_mode(agent, _S.agent)
    if service_url is not None:
        _S.service_url = service_url.rstrip("/")
    if api_key is not None:
        _S.api_key = api_key
    if find_timeout_s is not None:
        _S.find_timeout_s = float(find_timeout_s)
    if optional_timeout_s is not None:
        _S.optional_timeout_s = float(optional_timeout_s)
    if known_broken_probe_s is not None:
        _S.known_broken_probe_s = float(known_broken_probe_s)
    if http_timeout_s is not None:
        _S.http_timeout_s = float(http_timeout_s)
    if enabled is not None:
        _S.enabled = bool(enabled)
    if baselines is not None:
        _S.baselines = bool(baselines)
    if max_candidates is not None:
        _S.max_candidates = int(max_candidates)


# ------------------------------------------------------------------- http
def _call(path, payload=None):
    """GET or POST the healing service. Returns None instead of raising.

    Nothing in this module may take a bot down because the service is
    unreachable. A failed call means no healing, not a failed step.
    """
    if not _S.service_url:
        return None
    url = _S.service_url + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if _S.api_key:
        headers["X-API-Key"] = _S.api_key
    if _S.session:
        headers["X-BIS-Session"] = _S.session
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=_S.http_timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("healing service unavailable (%s) - continuing without "
                    "it", exc)
        return None


def health():
    """What the service says about itself, or None if it is not reachable."""
    return _call("/v1/health")


def _js():
    """The harvesting JavaScript, downloaded once and cached per version."""
    with _lock:
        if _bundle["harvest"]:
            return _bundle
        got = _call("/v1/js")
        if not got:
            return None
        if got.get("protocol") != PROTOCOL:
            log.warning("healing service speaks protocol %s, this client "
                        "speaks %s - not healing",
                        got.get("protocol"), PROTOCOL)
            return None
        _bundle.update({"version": got.get("version", ""),
                        "harvest": got.get("harvest", ""),
                        "probe": got.get("probe", ""),
                        "describe_one": got.get("describe_one", ""),
                        "presence": got.get("presence", ""),
                        "same": got.get("same", "")})
        log.info("healing JavaScript loaded, version %s", _bundle["version"])
        return _bundle


# -------------------------------------------------------------- selectors
_BY_CSS = {"id": "#%s", "class name": ".%s", "name": "[name='%s']",
           "tag name": "%s", "css selector": "%s"}


def _is_xpath(raw):
    """Is this bare string XPath rather than CSS?

    Has to agree with ``bis_core.descriptors.selector_syntax``, because the
    prefix it produces is what decides three separate things: whether this
    file hands the selector to querySelectorAll or to document.evaluate,
    which language the harvester proposes repairs in (``preferSyntax``),
    and which language a stored fleet repair is served for.

    It used to test ``startswith('/')`` alone, which missed the two forms
    RPA developers paste most often after ``//``: a parenthesised
    expression, ``(//div)[1]``, and a relative one, ``./span``. Both were
    sent to querySelectorAll, where they raise a syntax error that
    _lookup_indexed catches - so a perfectly good XPath spent the full
    find timeout matching nothing and was then "healed", in the wrong
    language, from a selector that had never been broken.
    """
    body = str(raw or "").strip()
    return body.startswith(("//", "/", "./", "(/", "(//"))


def _selector(by, value):
    """Turn a Selenium (by, value) pair into one selector string.

    XPath keeps an ``xpath=`` prefix, which is how the rest of the system
    tells the two languages apart, and how a repair comes back in the
    language the step was written in. Each step of a chain keeps its own,
    as bis_core's normalize_selector gives it.
    """
    if value is None:                       # find(driver, "#submit")
        raw = str(by)
        steps = _chain(raw)
        if len(steps) > 1:
            return (" %s " % _CHAIN).join(_selector(s, None) for s in steps)
        if raw.startswith("xpath=") or not _is_xpath(raw):
            return raw
        return "xpath=" + raw
    by = str(by)
    if by == "link text":
        return "xpath=//a[normalize-space()='%s']" % value
    if by == "partial link text":
        return "xpath=//a[contains(normalize-space(),'%s')]" % value
    raw = "xpath=" + value if by == "xpath" else _BY_CSS.get(by, "%s") % value
    return _selector(raw, None) if len(_chain(raw)) > 1 else raw


# A selector may step past the page's own document. Each ">>>" steps into
# what the element before it holds - an open shadow root, or a frame's
# document - and the next step is asked there: "app-shell >>> button.save"
# is the button inside <app-shell>'s shadow root, "iframe#main >>> #email"
# the box inside that frame. The service's harvest proposes these for an
# element it found there, and a script may be written with them.
_CHAIN = ">>>"


def _chain(selector):
    """A selector's steps across shadow roots and frames, outermost first.

    Split only outside quotes and brackets, as the page and bis_core's
    chain_steps split it - an attribute value may hold ">>>", and a test
    holds the readings to one answer.
    """
    body = str(selector or "")
    steps, buf, depth, quote, i = [], [], 0, "", 0
    while i < len(body):
        ch = body[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif depth == 0 and body.startswith(_CHAIN, i):
            steps.append("".join(buf))
            buf = []
            i += len(_CHAIN)
            continue
        else:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth = max(0, depth - 1)
            buf.append(ch)
        i += 1
    steps.append("".join(buf))
    return [s for s in (x.strip() for x in steps) if s]


def _last_step(selector):
    """The step of a chain that names the element; the rest is the way."""
    steps = _chain(selector)
    return steps[-1] if steps else ""


def _uses_id(selector):
    """Does this selector identify its element by id?

    A '#' inside an attribute value is not an id selector, so brackets and
    quoted strings are removed before looking. Kept deliberately small - this
    file has no package behind it, and the full parser lives in bis-core where
    the agent can reach it. A chain is asked about its last step: an id on a
    host on the way says nothing about how the element itself is named.
    """
    sel = _last_step(selector)
    if sel.startswith("xpath="):
        return "@id" in sel.lower()
    return "#" in re.sub(r"\[[^\]]*\]|'[^']*'|\"[^\"]*\"", " ", sel)


_CSS_TAG = re.compile(r"^([a-zA-Z][\w\-]*)")
_XPATH_TAG = re.compile(r"^(?:[\w\-]+::)?([a-zA-Z][\w\-]*)")


def _steps(body, separators):
    """A selector's steps, split only outside quotes and brackets."""
    steps, buf, depth, quote = [], [], 0, ""
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in separators:
            if buf:
                steps.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        steps.append("".join(buf))
    return [s for s in (x.strip() for x in steps) if s]


def _selector_tag(selector):
    """The tag a selector's last step names, or "" - the harvest's hint.

    The harvest keeps 900 elements and a third of real pages have more, so
    what the broken selector was after is what keeps its kind in the cut.
    The same reading as bis_core's selector_tag, which this file cannot
    import; a test holds the two to the same answer for every shape.
    """
    body = _last_step(selector)
    low = body.lower()
    if not body or low.startswith("text="):
        return ""
    xpath = low.startswith("xpath=") or (not low.startswith("css=")
                                         and _is_xpath(body))
    for prefix in ("xpath=", "css="):
        if low.startswith(prefix):
            body = body[len(prefix):]
            break
    steps = _steps(body, "/" if xpath else " >+~\t\n")
    if not steps:
        return ""
    if xpath:
        match = _XPATH_TAG.match(steps[-1].lstrip("./"))
        tag = match.group(1).lower() if match else ""
        return "" if tag in ("and", "or", "not", "text") else tag
    match = _CSS_TAG.match(steps[-1])
    return match.group(1).lower() if match else ""


def _split(selector):
    """Selector string back into the (by, value) Selenium wants."""
    if selector.startswith("xpath="):
        return ("xpath", selector[6:])
    return ("css selector", selector)


# ----------------------------------------------------------------- drivers
# Four operations is the whole surface this file needs from a browser: find
# by selector, ask whether an element is visible, run a function in the page,
# and say where the page is. Everything else - clicking, typing, reading text
# - happens on the element this returns, in whichever library the script was
# already written against.


class _Selenium(object):
    """A Selenium WebDriver.

    A chain is followed a step at a time: into a host's shadow root through
    WebDriver's own shadow_root, which leaves the driver where it is, and
    into a frame with switch_to.frame - which does not, see _framed.
    """

    native = "selenium"

    def __init__(self, driver):
        self.driver = driver

    def query(self, selector):
        # Every lookup starts where the step began.
        _frames_out(self.driver)
        steps = _chain(selector)
        if len(steps) < 2:
            by, val = _split(selector)
            return self.driver.find_elements(by, val)
        return self._through([self.driver], steps)

    def _through(self, scopes, steps):
        """What the rest of a chain names, asked of these scopes."""
        by, val = _split(steps[0])
        found = []
        for scope in scopes:
            try:
                found.extend(scope.find_elements(by, val))
            except Exception:
                pass        # WebDriver will not search a shadow root by XPath
        if len(steps) == 1:
            return found
        roots = []
        for el in found:
            if _is_frame(el):
                # A driver is in one document at a time: into the frame,
                # and straight out again unless the rest of the chain is
                # found there.
                if _frame_in(self.driver, el):
                    got = self._through([self.driver], steps[1:])
                    if got:
                        return got
                    _frame_out_one(self.driver)
                continue
            root = _shadow_root(self.driver, el)
            if root is not None:
                roots.append(root)
        return self._through(roots, steps[1:]) if roots else []

    @staticmethod
    def visible(el):
        try:
            return el.is_displayed()
        except Exception:
            return False

    def evaluate(self, expression, arg):
        # The service's JavaScript is written for the page the step began
        # on, so it runs there; the driver then goes back into the frame the
        # last lookup found its element in, which the step still holds.
        path = _frames_out(self.driver)
        try:
            # Playwright calls the function for us; Selenium has to be told.
            return self.driver.execute_script(
                "return (%s)(arguments[0]);" % expression, arg)
        finally:
            _frames_back(self.driver, path)

    def where(self):
        return {"url": self.driver.current_url or "",
                "title": self.driver.title or ""}

    def home(self):
        """Out of any frame a lookup of this file took the driver into."""
        _frames_out(self.driver)


# The frames this file took each Selenium driver into. A WebDriver element
# inside a frame is usable only while the driver is in that frame, so a
# lookup that ends in one leaves the driver there, as switch_to.frame
# would, and the next lookup brings it back out. Each frame entered is
# marked, and the driver is only brought out while it is still in the one
# this file left it in: a script that has switched frames itself since has
# said where it wants to be. Keyed by the driver, per process.
_framed = {}


def _is_frame(el):
    try:
        return (el.tag_name or "").lower() in ("iframe", "frame")
    except Exception:
        return False


def _shadow_root(driver, el):
    """An element's open shadow root as WebDriver searches one, or None."""
    try:
        return el.shadow_root                      # Selenium 4.1 and later
    except AttributeError:
        pass
    except Exception:
        return None               # none, or a closed one nobody can reach
    try:
        return driver.execute_script("return arguments[0].shadowRoot;", el)
    except Exception:
        return None


def _frame_in(driver, el):
    """Take the driver into the frame `el` owns, and mark where it is."""
    try:
        driver.switch_to.frame(el)
    except Exception:
        return False
    rec = _framed.get(id(driver))
    if rec is None:
        rec = _framed[id(driver)] = {
            "path": [], "mark": "bis-%x-%x" % (id(driver),
                                               int(time.time() * 1e6))}
    rec["path"].append(el)
    try:
        driver.execute_script("window.__bisEntered = arguments[0];",
                              rec["mark"])
    except Exception:
        pass
    return True


def _frame_out_one(driver):
    """Back out of the last frame _frame_in took the driver into."""
    try:
        driver.switch_to.parent_frame()
    except Exception:
        pass
    rec = _framed.get(id(driver))
    if rec and rec["path"]:
        rec["path"].pop()
        if not rec["path"]:
            _framed.pop(id(driver), None)


def _frames_out(driver):
    """Bring a driver back out of the frames this file took it into.

    Returns them, outermost first, for _frames_back - or nothing, if the
    driver is not where it was left.
    """
    rec = _framed.pop(id(driver), None)
    if not rec or not rec["path"]:
        return []
    try:
        here = driver.execute_script("return window.__bisEntered || '';")
    except Exception:
        here = None             # the frame has gone: out regardless
    if here is not None and here != rec["mark"]:
        return []
    for _ in rec["path"]:
        try:
            driver.switch_to.parent_frame()
        except Exception:
            break
    return list(rec["path"])


def _frames_back(driver, path):
    """Into the frames _frames_out brought the driver out of, again."""
    for el in path:
        if not _frame_in(driver, el):
            break


class _Playwright(object):
    """A Playwright Page.

    Playwright understands ``xpath=`` prefixes natively, so the selector
    goes across untouched - the same string the rest of the system uses to
    tell the two languages apart. A chain is followed a step at a time and
    nothing is switched: a handle carries its own frame.
    """

    native = "playwright"

    def __init__(self, page):
        self.page = page

    def query(self, selector):
        steps = _chain(selector)
        if len(steps) < 2:
            return self.page.query_selector_all(selector)
        return self._through([("document", self.page)], steps)

    def _through(self, scopes, steps):
        """What the rest of a chain names, asked of these scopes.

        Asked in the page, as the page's own resolver asks it - inside a
        host's shadow root alone, and in a frame's document - rather than
        through Playwright's selectors, which pierce every shadow root
        under the scope and would find more than the page does.
        """
        step = _selector(steps[0], None)
        found = []
        for kind, scope in scopes:
            try:
                found.extend(_pw_elements(scope.evaluate_handle(
                    _PW_IN_SHADOW if kind == "shadow" else _PW_IN_DOCUMENT,
                    step)))
            except Exception:
                pass
        if len(steps) == 1:
            return found
        held = []
        for el in found:
            try:
                frame = el.content_frame()
            except Exception:
                frame = None
            held.append(("document", frame) if frame is not None
                        else ("shadow", el))
        return self._through(held, steps[1:]) if held else []

    @staticmethod
    def visible(el):
        try:
            return el.is_visible()
        except Exception:
            return False

    def evaluate(self, expression, arg):
        return self.page.evaluate(expression, arg)

    def where(self):
        return {"url": self.page.url or "", "title": self.page.title() or ""}

    def home(self):
        """Nothing to come back out of: a Playwright page is never moved."""


# One step of a chain, asked in a Playwright page or frame, or of a host.
_PW_IN_DOCUMENT = r"""(s) => {
  if (s.startsWith('xpath=')) {
    const it = document.evaluate(s.slice(6), document, null,
                                 XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
    const out = [];
    for (let i = 0; i < it.snapshotLength; i++) out.push(it.snapshotItem(i));
    return out;
  }
  return Array.from(document.querySelectorAll(s.replace(/^css=/, '')));
}"""
_PW_IN_SHADOW = r"""(h, s) => (h.shadowRoot && !s.startsWith('xpath='))
    ? Array.from(h.shadowRoot.querySelectorAll(s.replace(/^css=/, '')))
    : []"""


def _pw_elements(handle):
    """A Playwright handle on an array of elements, as element handles."""
    try:
        return [h for h in (p.as_element() for p in
                            handle.get_properties().values())
                if h is not None]
    finally:
        try:
            handle.dispose()
        except Exception:
            pass


def _adapt(driver):
    """Wrap whatever we were handed. Playwright pages expose query_selector_all;
    Selenium drivers do not."""
    if isinstance(driver, (_Selenium, _Playwright)):
        return driver
    if hasattr(driver, "query_selector_all"):
        return _Playwright(driver)
    return _Selenium(driver)


def _lookup_indexed(page, selector, timeout_s, want_all=False):
    """``_lookup``, plus where in the match set the answer came from.

    The index is what the baseline fingerprinter needs. A selector matching
    a desktop copy and a mobile copy of the same component resolves to
    whichever is on screen, and recording the other one as the baseline
    would measure every future repair against an element the step has never
    touched.
    """
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        try:
            found = page.query(selector)
            shown = [(i, e) for i, e in enumerate(found) if page.visible(e)]
            if shown:
                if want_all:
                    return [e for _, e in shown], shown[0][0]
                return shown[0][1], shown[0][0]
        except Exception:
            pass                            # invalid selector, or mid-navigation
        if time.time() >= deadline:
            return None, -1
        time.sleep(0.25)


def _lookup(page, selector, timeout_s, want_all=False):
    """Try a selector for up to ``timeout_s``. Returns element(s) or None.

    Visibility matters: a matched-but-hidden element is the single most
    common cause of a bot that clicks nothing and reports success.
    """
    return _lookup_indexed(page, selector, timeout_s, want_all)[0]


# ---------------------------------------------------------------- healing
def _evaluate(page, expression, arg):
    """Run one of the service's JavaScript functions in the page."""
    return _adapt(page).evaluate(expression, arg)


def _home(page):
    """Back out of any frame a lookup took the driver into.

    For a step that is handing nothing back: a frame is only worth staying
    in for the element found there.
    """
    home = getattr(page, "home", None)
    if home is not None:
        try:
            home()
        except Exception:
            pass


def _family_present(page, selector):
    """Is anything of this element's component family on the page?

    Only asked for optional elements, and only to answer the question the
    ranker cannot: was this renamed, or is it simply not here? A consent
    banner is absent far more often than it is renamed, and harvesting nine
    hundred elements to conclude "nothing matched" is a slow way to find
    that out.

    Returns True when it cannot tell. Wasting one heal is cheap; wrongly
    concluding "absent" would stop repairing a step for good.
    """
    bundle = _js()
    if not bundle or not bundle.get("presence"):
        return True
    try:
        got = page.evaluate(bundle["presence"], {"selector": selector})
    except Exception:
        return True
    if not isinstance(got, dict):
        return True
    if not got.get("present", True):
        log.info("%s: nothing of its family on the page (%s) - absent rather "
                 "than broken", selector, ", ".join(got.get("fragments", [])))
        return False
    return True


# --------------------------------------------------------------- baselines
#
# Six of the seven similarity signals get stronger, or only work at all, when
# there is a record of the element the step used *last time it worked*.
# Structure and geometry have nothing to compare against without one and
# return "no opinion"; tag and text fall back to whatever can be read out of
# the selector string, which for a bare "#firstName" is nothing at all.
# Signals with no evidence are dropped from the weighted mean - correctly,
# since scoring them zero would punish a repair for being the first - so a
# cold heal is judged on a fraction of the weight. That is how a right answer
# lands in the review band and gets refused.
#
# So: after a step succeeds, fingerprint what it used and tell the service.
# The service keys it by locator and page and hands it back on the next
# adjudication for that locator, including to a different bot on a different
# host. This file stays stateless; the durable copy lives where the model and
# the policy already live.


def _prefetch_js():
    """Start downloading the JavaScript, off the calling thread, once.

    Nothing on a *working* step may wait on the healing service. _js() is a
    blocking GET, and a host that black-holes rather than refuses would turn
    the first successful lookup in a run into a thirty-second stall - a bot
    made slower by a feature that exists to make it more reliable.

    So the capture path never fetches: it uses the bundle if it is already
    there and asks for it in the background if it is not. The locator that
    triggered the download is skipped and picked up by the next one, and a
    fetch that failed is retried rather than remembered as attempted.
    """
    with _baseline_lock:
        if _bundle["harvest"] or _prefetching:
            return
        _prefetching.append(True)

    def _go():
        try:
            _js()
        finally:
            with _baseline_lock:
                del _prefetching[:]

    threading.Thread(target=_go, daemon=True, name="bis-js").start()


def _fingerprint(page, key, selector, index=0):
    """Fingerprint the element this step just used. Returns it, or None.

    ``index`` is the match the step actually resolved to. Passing it
    matters: a selector matching several nodes must record the one that was
    used rather than whichever happens to be first, or the structure and
    geometry signals are seeded from a different element and every later
    repair on this locator is measured against the wrong thing.

    The JavaScript has to run on this thread. Neither driver is safe to
    drive from two at once, and the page belongs to the caller.
    """
    # Read without _lock on purpose: that lock is held across the download,
    # and a single dict lookup needs no protection from an update() that
    # only ever replaces whole values.
    describe = _bundle.get("describe_one")
    if not describe:
        _prefetch_js()
        return None
    try:
        raw = page.evaluate(describe, {"selector": selector, "index": index})
    except Exception as exc:
        log.debug("could not fingerprint %s (%s)", key, exc)
        return None
    return raw or None


def _probe_wait(selector, wait):
    """How long to wait on an authored selector before trying to repair.

    The full wait, unless this process has already had to repair this
    exact selector. RPA work is overwhelmingly loop-shaped - a queue of
    invoices, a spreadsheet of joiners - so a selector the vendor renamed
    is waited on once per row. Measured on a three-row onboarding queue
    with three stale addresses: 54 seconds of a 229-second run, spent
    waiting for addresses the process already knew were gone.

    It is still tried, briefly, rather than skipped. The authored selector
    is the developer's stated intent and it must win the moment it works
    again - after a rollback, or on a screen where the old name survives.
    Skipping it outright would make a repair permanent for the life of
    the process, which is a much worse trade than one extra second.
    """
    with _lock:
        seen = selector in _repaired_here
    return min(wait, _S.known_broken_probe_s) if seen else wait


def _report_baseline(page, key, selector, index=0, replaced="",
                     repair_id=0, confidence=0.0, approved=False):
    """Record a working element, once per locator per run.

    ``replaced`` promotes the report from a baseline to a repair: it names
    the authored selector that broke, which is what lets the service file
    it where the next bot to hit the same break will find it.

    ``confidence`` is what the service said the repair was worth, and the
    fleet store keeps it, because the next bot's repair is banded on it.
    ``approved`` says a reviewer accepted it rather than the policy.

    Fire and forget: a bot must never wait on the healing service to type
    into a box that already worked. Whether the fingerprint is good enough
    to keep is the service's decision, not this file's - an off-screen
    element is refused there, so the same rule applies to every client.
    """
    if not (_S.enabled and _S.baselines and _S.service_url):
        return
    key = key or selector
    url = _url(page)
    # `replaced` is part of the scope, not just the payload. A repair for a
    # locator whose baseline was already reported this run is new
    # information, and keying it the same way would drop it silently.
    scope = (key, _origin(url), replaced)
    with _baseline_lock:
        if scope in _seen_baselines:
            return

    raw = _fingerprint(page, key, selector, index)
    if not raw:
        # Deliberately not marked as seen. The usual reason to be here is a
        # bundle that had not arrived yet, and this locator gets another
        # chance next time the step runs.
        return
    with _baseline_lock:
        _seen_baselines.add(scope)
        _baselines[scope] = raw

    # Captured here, on the caller's own thread, and closed over below.
    # _process_context() walks the stack to find the first frame outside
    # this file - called from inside the daemon thread instead, that walk
    # would find threading.Thread.run() rather than the RPA developer's
    # step, and every report would file under the wrong call site.
    ctx = _process_context()

    def _send():
        _call("/v1/baseline", {"key": key, "selector": selector, "url": url,
                               "descriptor": raw, "replaced": replaced,
                               "repair_id": int(repair_id or 0),
                               "confidence": float(confidence or 0.0),
                               "approved": bool(approved),
                               "process": ctx,
                               "protocol": PROTOCOL})

    threading.Thread(target=_send, daemon=True, name="bis-baseline").start()


def _browser(page):
    """Which browser a step ran in, so two tabs keep two chains."""
    return id(getattr(page, "driver", None) or getattr(page, "page", None)
              or page)


def _step_done(page, key, selector, replaced=""):
    """Note a step that found its element, and report where the last led.

    The page this step found its element on is where the previous step
    led. From a run whose selectors worked, that teaches the service what
    each step leads to; after a repair it is checked against that
    (/v1/outcome). A repair that binds the wrong element still binds, so
    no failed bind ever caught one - but one that leads somewhere a good
    run never went is caught here. Each move is reported once per run,
    and the report is fire and forget, held to the same switch as the
    baselines.
    """
    if not (_S.enabled and _S.baselines and _S.service_url):
        return
    url = _url(page)
    with _step_lock:
        before = _previous.pop(_browser(page), None)
        _previous[_browser(page)] = {"key": key, "url": url,
                                     "selector": selector,
                                     "replaced": replaced}
        if not before:
            return
        move = (before["key"], _origin(before["url"]), _origin(url),
                before["selector"], before["replaced"])
        if move in _reported_outcomes:
            return
        _reported_outcomes.add(move)
    payload = {"key": before["key"], "url": before["url"],
               "selector": before["selector"],
               "replaced": before["replaced"],
               "healed": bool(before["replaced"]), "next_url": url,
               "process": _process_context(), "protocol": PROTOCOL}
    threading.Thread(target=lambda: _call("/v1/outcome", payload),
                     daemon=True, name="bis-outcome").start()


def _step_failed(page):
    """A step found nothing: what comes next is not where the last one led."""
    with _step_lock:
        _previous.pop(_browser(page), None)


def _url(page):
    """Where the page is, or "" - asking must never fail a step."""
    try:
        return page.where().get("url", "") or ""
    except Exception:
        return ""


def _origin(url):
    """Coarse page identity, used only to key this run's own fingerprints.

    The service derives the real key; this one only has to be stable within
    a run, so the query string goes and nothing else is parsed.
    """
    return (url or "").split("?")[0].split("#")[0]


_PROCESS_NAME = None


def _process_context():
    """Which automation process, and which part of it, is calling.

    Derived, never authored. An RPA team should not have to learn a new
    field to get an audit trail, so the process name comes from the entry
    script and the call site from the first frame outside this file. That
    frame is the caller's own code, which is exactly the "which part"
    the store is asked to answer.

    Only ever called on the heal and report paths. The path where the
    authored selector resolves must stay free.
    """
    global _PROCESS_NAME
    if _PROCESS_NAME is None:
        entry = os.path.basename(sys.argv[0] or "") or "unknown"
        _PROCESS_NAME = os.path.splitext(entry)[0]
    ctx = {"name": _PROCESS_NAME, "file": "", "function": "", "line": 0,
           "host": ""}
    try:
        ctx["host"] = socket.gethostname()
    except Exception:
        pass
    try:
        here = os.path.abspath(__file__)
        frame = sys._getframe(1)
        while frame is not None:
            path = frame.f_code.co_filename
            if os.path.abspath(path) != here:
                ctx["file"] = os.path.basename(path)
                ctx["function"] = frame.f_code.co_name
                ctx["line"] = frame.f_lineno
                break
            frame = frame.f_back
    except Exception:
        # Context is a nicety; a missing frame must never fail a step.
        pass
    return ctx


def _is_the_chosen(page, bundle, harvest, index, bound, proposal="",
                   collect=False):
    """Did a proposal bind the element the service chose? True when unsure.

    Resolving and being visible was all a repair used to be checked for,
    and a selector the page proposed for one element can still resolve to
    another: one mined from classes the element shares, or a page that
    moved after the harvest. The step then clicks the wrong thing, with
    nothing raised - the worst failure this system can produce. So the
    page is asked, against the elements the harvest kept (SAME_JS).

    It cannot always tell - a re-render replaces the node, an older service
    serves no such function - and then the bind stands as it always did.
    """
    same = (bundle or {}).get("same")
    if not same or not harvest:
        return True
    arg = {"harvest": harvest, "index": int(index)}
    if len(_chain(proposal)) > 1:
        # An element inside a frame cannot be handed to the page's
        # JavaScript from outside the frame, so the page is given the
        # proposal and resolves it itself, as the bot did.
        arg.update(selector=proposal, all=bool(collect))
    else:
        arg["els"] = list(bound)
    try:
        got = page.evaluate(same, arg)
    except Exception:
        return True
    if not isinstance(got, dict) or not got.get("known"):
        return True
    return bool(got.get("same"))


def _reviewed(key, selector, describe, proposal, el, verdict, where, action):
    """Hand a repair held for review to the review hook. True if approved.

    Anything but a yes is a no - False, None, or a hook that raised: a bot
    must never act on a repair nobody accepted, and never crash because
    the reviewer's code did. The verdict goes back to the service either
    way, because a person deciding a real break is the one label this
    system can get from production.
    """
    shortlist = verdict.get("shortlist") or []
    rank = next((r for r, e in enumerate(shortlist) if e.get("chosen")), -1)
    info = {
        "selector": selector, "description": describe or "",
        "proposal": proposal, "element": el,
        "confidence": float(verdict.get("confidence", 0.0) or 0.0),
        "strategy": verdict.get("strategy", ""),
        "rationale": verdict.get("rationale", ""),
        "reason": verdict.get("ai_reason", ""),
        "summary": shortlist[rank].get("summary", "") if rank >= 0 else "",
        "url": (where or {}).get("url", ""),
        "title": (where or {}).get("title", ""),
    }
    try:
        approved = bool(_S.review_hook(info))
    except Exception as exc:
        log.warning("the review hook failed (%s) - treating the repair for "
                    "%s as rejected", exc, selector)
        approved = False
    log.info("repair for %s -> %s %s on review (%.0f%%)", selector, proposal,
             "approved" if approved else "rejected",
             100.0 * info["confidence"])
    _report_review(key, info, action, approved,
                   [e.get("summary", "") for e in shortlist], rank,
                   int(verdict.get("repair_id", 0) or 0))
    return approved


def _report_review(key, info, action, approved, shortlist, rank,
                   repair_id=0):
    """Tell the service what the reviewer decided. Fire and forget.

    Held to the same switch as baselines: BIS_BASELINES=0 is a team saying
    it would rather send nothing it does not have to, and a label is that.
    """
    if not (_S.enabled and _S.baselines and _S.service_url):
        return
    ctx = _process_context()          # on the caller's thread; see above
    payload = {"key": key, "selector": info["selector"],
               "proposal": info["proposal"], "approved": bool(approved),
               "url": info["url"], "title": info["title"], "action": action,
               "description": info["description"],
               "confidence": info["confidence"],
               "strategy": info["strategy"], "rationale": info["rationale"],
               "ai_reason": info["reason"], "shortlist": shortlist,
               "chosen_rank": rank, "repair_id": repair_id,
               "process": ctx, "protocol": PROTOCOL}
    threading.Thread(target=lambda: _call("/v1/review", payload),
                     daemon=True, name="bis-review").start()


def _harvest(page, bundle, selector):
    """Fingerprint the page's candidates for a repair of ``selector``.

    Returns (candidates, harvest id), or ([], "") when the page could not
    be read. The id names the elements the page kept for these candidates,
    so a bound selector can be checked against the one the service chose.
    """
    try:
        harvested = page.evaluate(bundle["harvest"], {
            "maxCandidates": _S.max_candidates,
            # What the broken selector was after, so that kind of element
            # - and anything still carrying its name - survives the cut on
            # a page with more than maxCandidates of them.
            "tagHint": _selector_tag(selector),
            "selector": selector,
            "scope": "",
            "needText": False,
            "preferSyntax": ("xpath" if _last_step(selector).startswith(
                "xpath=") else "css"),
        })
    except Exception as exc:
        log.warning("could not harvest the page (%s)", exc)
        return [], ""
    if not isinstance(harvested, dict):
        return [], ""
    return (harvested.get("candidates") or [],
            harvested.get("harvest") or "")


def _heal(page, selector, action, optional, describe):
    """Repair a step whose selector found nothing, and bind the repair here.

    The ordinary repair is a fixed pipeline (_pipeline_heal); the AI agent
    plans one tool call at a time (_agent_heal). Which runs is the agent
    mode: "off" the pipeline alone, "always" the agent with the pipeline
    behind it, "escalate" the pipeline with the agent behind it. Whichever
    answers, the element is bound the same way (_bind_verdict).

    An optional element gets one cheap question first: if nothing of its
    family is on the page it was never here, so there is nothing to repair
    and no reason to pay for a harvest. With the agent always on, the
    question is the agent's own - its Presence Check.
    """
    if _S.agent == "always":
        got = _agent_heal(page, selector, action, optional, describe)
        if got is not None:
            return got
    if optional and not _family_present(page, selector):
        return None, ""
    el, bound = _pipeline_heal(page, selector, action, optional, describe)
    if el is None and _S.agent == "escalate":
        got = _agent_heal(page, selector, action, optional, describe,
                          escalated=True)
        if got is not None:
            return got
    return el, bound


def _pipeline_heal(page, selector, action, optional, describe, retry_of=0,
                   scan=None):
    """Ask the service which element the step meant, then bind it here.

    The service is sent element fingerprints and answers with an *index*
    into the list this function sent it. It never sends back a selector,
    because only the machine with the browser can prove one resolves.

    ``retry_of`` names a fleet repair that has just failed to bind, and
    ``scan`` the (candidates, harvest) it was judged against - the agent's
    last scan, reused so the page is not harvested twice.
    """
    bundle = _js()
    if not bundle:
        return None, ""
    candidates, harvest = (scan if scan is not None
                           else _harvest(page, bundle, selector))
    if not candidates:
        return None, ""

    try:
        where = page.where()
    except Exception:
        where = {"url": "", "title": ""}

    # If this run already saw the locator working, send that fingerprint
    # rather than making the service look one up - it is newer, and it is
    # certainly this page. When there is none, the service falls back to
    # whatever the fleet recorded for the same locator on the same page.
    #
    # Two scopes can hold that fingerprint now that `replaced` is part of
    # the baseline key: a plain resolution, recorded before anything broke,
    # and a previous heal of this exact selector, recorded after. The
    # second is checked first because it is the newer of the two whenever
    # both exist - it reflects the element the locator resolved to *after*
    # the last repair, not the one it pointed at before the page changed.
    key = describe or selector
    origin = _origin(where.get("url", ""))
    with _baseline_lock:
        baseline = (_baselines.get((key, origin, selector))
                    or _baselines.get((key, origin, "")))

    timeout = _S.optional_timeout_s if optional else _S.find_timeout_s

    # A stored repair the fleet believes in can still be stale on this page
    # - the release that broke it may have moved again since. One retry
    # asks the service to answer without that repair, sending the exact
    # candidates it just judged rather than harvesting a second time: those
    # are what the stale repair was compared against, and a fresh harvest
    # would both cost another round of JavaScript and risk the page having
    # moved underneath the comparison.
    ctx = {"key": key, "selector": selector, "describe": describe,
           "action": action, "where": where, "timeout": timeout}
    attempt = 0
    while attempt < 2:
        payload = {
            "locator": {"key": key, "selector": selector,
                        "description": describe},
            "action": action,
            "page": where,
            "candidates": candidates,
            "baseline": baseline,
            # This client has no memory, so it asks for the fleet's. A
            # client that keeps its own must not set this: an absent
            # baseline is then a fact about that client's memory, not a gap
            # to be filled.
            "use_stored_baseline": True,
            # This client keeps no repairs of its own either, so it asks
            # for the fleet's. Without it every bot pays the full
            # adjudication cost again for a break another bot already
            # solved.
            "use_stored_repair": True,
            "process": _process_context(),
            "optional": bool(optional),
            # Optional locators get the same budget as anything else. They
            # used to get a smaller one, to keep absent pop-ups cheap - but
            # the presence gate now settles absence before this point, so
            # everything arriving here is a real repair candidate. Starving
            # it of budget only cost the AI pass, which dropped a genuine
            # 96% repair to a heuristic 78% and straight through the
            # optional floor.
            "budget_s": 25.0,
            "protocol": PROTOCOL,
        }
        if retry_of:
            payload["retry"] = True
            payload["failed_repair_id"] = retry_of
        verdict = _call("/v1/adjudicate", payload)
        attempt += 1
        if not verdict:
            return None, ""
        el, proposal, unbound = _bind_verdict(page, bundle, harvest,
                                              candidates, verdict, ctx)
        if el is not None:
            return el, proposal
        if not unbound:
            return None, ""
        repair_id = int(verdict.get("repair_id", 0) or 0)
        if repair_id and not retry_of:
            # A stored repair that no longer binds. Tell the service so it
            # stops handing it to the rest of the fleet, and let the AI
            # answer this one properly. Exactly one retry - a second miss
            # is a normal "not found", never a loop.
            log.info("the stored repair for %s no longer binds; "
                     "re-healing", selector)
            retry_of = repair_id
            continue
        log.info("the chosen element produced no selector that resolves "
                 "here")
        return None, ""
    return None, ""


def _bind_verdict(page, bundle, harvest, candidates, verdict, ctx):
    """Bind the element a verdict chose, here in the browser.

    Shared by the pipeline and the AI agent, so a decision is bound one
    way whichever made it. Returns (element, proposal, unbound): the
    element and the selector that bound it, or None and "" - ``unbound``
    True when the verdict was to be used and no proposal bound the element
    chosen, which is what a stale fleet repair looks like.
    """
    key, selector, describe = ctx["key"], ctx["selector"], ctx["describe"]
    action, where, timeout = ctx["action"], ctx["where"], ctx["timeout"]

    # The band is policy, and policy is the service's to define - but
    # it is this client's to *enforce*. Applying a repair the service
    # placed in the review or refuse band is how a bot ends up
    # confidently clicking the wrong element, so anything short of
    # "apply" counts as not found - unless a person decides it. A
    # review hook (configure(review_hook=...)) is shown the repair
    # bound to the page, and only its yes applies one.
    band = verdict.get("band", "refuse")
    reviewing = band == "review" and _S.review_hook is not None
    if band != "apply" and not reviewing:
        log.info("declining repair for %s: %.0f%% falls in the %s band",
                 selector, 100.0 * verdict.get("confidence", 0.0), band)
        return None, "", False

    chosen = verdict.get("chosen", -1)
    if chosen is None or chosen < 0 or chosen >= len(candidates):
        log.info("no confident repair for %s (%s)", selector,
                 verdict.get("rationale", "no candidate met the "
                                          "threshold"))
        return None, "", False

    # Bind here, in the browser, against the proposals the page itself
    # made.
    #
    # Kept in the shape the caller wrote, not just the language.
    # Someone who located an element without an id has usually learned
    # that this app's ids are generated and do not survive a deploy,
    # and an id-based repair would override that. Order is otherwise
    # untouched: the page proposes in descending durability and that
    # stays the primary key.
    proposals = candidates[chosen].get("selectors") or []
    if not _uses_id(selector):
        proposals = ([p for p in proposals if not _uses_id(p)]
                     + [p for p in proposals if _uses_id(p)])
    # How many elements a proposal actually matches is evidence, and
    # which way it cuts depends on what the step is for. Ask the page
    # once and order on the answer; durability still breaks the ties.
    counted = [(p, _breadth(page, p)) for p in proposals]
    if action == "collect":
        # A collection step scrapes a table. The page proposes in
        # descending durability, and the most durable proposal for one
        # row is usually the one pinned *to* that row - an id, or an
        # nth-of-type path. Binding it satisfies the loop and returns a
        # single element, so the step reports success and scrapes one
        # row out of twenty. That is the worst failure this system can
        # produce: a bot that is confidently wrong and says nothing.
        proposals = ([p for p, n in counted if n > 1]
                     + [p for p, n in counted if n <= 1])
    else:
        # The mirror image, and the same failure. A step that wants one
        # element can be handed a proposal matching several - mined
        # from two shared classes, say - and binding takes the first
        # match, which need not be the element the adjudicator chose.
        # Measured on a live HR form: the repair for the Employee Id
        # box came back as a two-class XPath matching five inputs, the
        # first of which was an unrelated search box in the header. The
        # step reported success and the payroll number was typed into
        # the wrong field. A proposal matching five elements is not an
        # address for this one, so anything unique goes first.
        proposals = ([p for p, n in counted if n == 1]
                     + [p for p, n in counted if n != 1])
    for proposal in proposals:
        if action == "collect":
            found, _ = _lookup_indexed(page, proposal,
                                       min(timeout, 2.0), want_all=True)
            el = found[0] if found else None
        else:
            el = _lookup(page, proposal, min(timeout, 2.0))
            found = [el] if el is not None else []
        if el is None:
            continue
        if not _is_the_chosen(page, bundle, harvest, chosen, found,
                              proposal, action == "collect"):
            log.info("%s resolves, but not to the element the service "
                     "chose - trying the next proposal", proposal)
            continue
        if reviewing and not _reviewed(key, selector, describe, proposal,
                                       el, verdict, where, action):
            return None, "", False
        log.info("healed %s -> %s (%.0f%% via %s%s)", selector, proposal,
                 100.0 * verdict.get("confidence", 0.0),
                 verdict.get("strategy", "unknown"),
                 ", approved on review" if reviewing else "")
        with _lock:
            _repaired_here.add(selector)
        # The repair is what the step is about to use, so it is the
        # baseline from here on. Recorded against the *authored* key,
        # because that is the key the next run will present when the
        # same selector fails again - the developer has not edited
        # their script.
        _report_baseline(page, key, proposal, replaced=selector,
                         repair_id=int(verdict.get("repair_id", 0) or 0),
                         confidence=verdict.get("confidence", 0.0),
                         approved=reviewing)
        return el, proposal, False
    return None, "", True


def _breadth(page, selector):
    """How many elements a selector matches here; 0 when it cannot run."""
    try:
        return len(page.query(selector))
    except Exception:
        return 0


# How many times one repair may come back to the browser. The service ends
# a repair at its tool-call limit and its budget; this is the client's own
# bound, so a service that never says "done" cannot hold a step forever.
_AGENT_ROUNDS = 12


def _agent_heal(page, selector, action, optional, describe, escalated=False):
    """Have the service's AI agent repair the step, running the page tools
    it asks for here, in the browser.

    Returns what the pipeline returns - (element, bound selector), or
    (None, "") when the agent decided there is nothing to apply - or None
    when it fell back: no agent on the service, a reply it could not use,
    out of time. Then the pipeline heals the step, as if the agent had
    never been asked.
    """
    bundle = _js()
    if not bundle:
        return None
    try:
        where = page.where()
    except Exception:
        where = {"url": "", "title": ""}
    key = describe or selector
    origin = _origin(where.get("url", ""))
    with _baseline_lock:
        baseline = (_baselines.get((key, origin, selector))
                    or _baselines.get((key, origin, "")))
    ctx = {"key": key, "selector": selector, "describe": describe,
           "action": action, "where": where,
           "timeout": _S.optional_timeout_s if optional
           else _S.find_timeout_s}
    reply = _call("/v1/agent/start", {
        "locator": {"key": key, "selector": selector,
                    "description": describe},
        "action": action, "page": where, "baseline": baseline,
        # As for /v1/adjudicate: this client has no memory of its own.
        "use_stored_baseline": True, "use_stored_repair": True,
        "optional": bool(optional),
        # Whether a person can decide a repair held for review - without
        # one, an escalation of a step never seen working changes nothing.
        "review_hook": _S.review_hook is not None,
        "escalated": bool(escalated), "budget_s": 25.0,
        "process": _process_context(), "protocol": PROTOCOL})
    scan, told = ([], ""), 0
    for _ in range(_AGENT_ROUNDS):
        if not isinstance(reply, dict):
            # Unreachable, or a service from before the agent: heal the
            # old way (invariant 11).
            return None
        told = _tell(reply.get("trace") or [], told)
        if reply.get("done"):
            if reply.get("fallback"):
                log.info("the AI agent handed %s back (%s)", selector,
                         reply.get("reason") or "no reason given")
                return None
            verdict = reply.get("decision") or {}
            el, proposal, unbound = _bind_verdict(page, bundle, scan[1],
                                                  scan[0], verdict, ctx)
            if el is not None:
                return el, proposal
            repair_id = int(verdict.get("repair_id", 0) or 0)
            if unbound and repair_id:
                # The fleet's repair the agent chose no longer binds: the
                # pipeline's retry, over the same scan, demotes it and asks
                # the AI without it.
                log.info("the stored repair the AI agent chose for %s no "
                         "longer binds; re-healing", selector)
                return _pipeline_heal(page, selector, action, optional,
                                      describe, retry_of=repair_id,
                                      scan=scan)
            return None, ""
        results = []
        for call in reply.get("calls") or []:
            result, scanned = _page_tool(page, bundle, selector, call, scan)
            if scanned is not None:
                scan = scanned
            results.append(result)
        reply = _call("/v1/agent/step", {"session": reply.get("session", ""),
                                         "results": results,
                                         "protocol": PROTOCOL})
    log.info("the AI agent did not finish %s in %d rounds", selector,
             _AGENT_ROUNDS)
    return None


def _page_tool(page, bundle, selector, call, scan):
    """Run one page tool the AI agent asked for. Returns (result, the new
    scan or None). Never raises: a tool that fails is an answer the agent
    reads, not a failed step."""
    call = call if isinstance(call, dict) else {}
    tool, args = str(call.get("tool") or ""), call.get("args") or {}
    out = {"id": str(call.get("id") or ""), "tool": tool, "ok": False,
           "error": "", "data": {}}
    try:
        if tool in ("page_scanner", "wait_and_rescan"):
            if tool == "wait_and_rescan":
                # A page still drawing itself. Bounded: the wait is spent
                # out of the step's own budget.
                time.sleep(min(5.0, max(0.0, float(args.get("seconds", 2)
                                                   or 0))))
            found, harvest = _harvest(page, bundle, selector)
            out.update(ok=True, data={"candidates": found, "harvest": harvest,
                                      "scanned": len(found)})
            return out, (found, harvest)
        if tool == "presence_check":
            got = (page.evaluate(bundle["presence"], {"selector": selector})
                   if bundle.get("presence") else None)
            if not isinstance(got, dict):
                got = {"present": True, "reason": "the page could not tell"}
            out.update(ok=True, data={
                "present": bool(got.get("present", True)),
                "reason": str(got.get("reason") or ""),
                "fragments": [str(f) for f in got.get("fragments") or []][:8]})
            return out, None
        if tool == "selector_options":
            index, found = int(args.get("index", -1)), scan[0]
            if not 0 <= index < len(found):
                out["error"] = "[%d] is not in the latest scan" % index
                return out, None
            out.update(ok=True, data={"options": [
                {"selector": p, "matches": _breadth(page, p)}
                for p in (found[index].get("selectors") or [])[:12]]})
            return out, None
        out["error"] = "this robot does not run %s" % (tool or "that")
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out, None


def _tell(trace, told):
    """Hand on_agent_step the tool calls it has not been told about yet.
    Returns how many it has been told about. A hook that raises is logged
    and passed over: an overlay must never fail a step."""
    hook = _S.on_agent_step
    for entry in trace[told:]:
        if not isinstance(entry, dict):
            continue
        log.info("AI agent: %s - %s", entry.get("title") or entry.get("tool"),
                 entry.get("result", ""))
        if hook is not None:
            try:
                hook(entry)
            except Exception as exc:
                log.warning("on_agent_step failed (%s) - carrying on", exc)
    return max(told, len(trace))


# ------------------------------------------------------------------- API
def seed(driver, locators, timeout=None):
    """Resolve every locator once on a working page, and record what it found.

    A repair is scored against a *baseline* - a fingerprint of the element
    taken while the locator still worked. Every lookup here already records
    one on success, so a step that has run even once is repaired warm. The
    gap is the step that has never run: a locator that was already broken,
    or simply wrong, the first time the bot executed. It has no baseline,
    it never gets one, and it stays on the weak path for the life of the
    process.

    Measured over 102 live sites, that is the difference between 43% and
    87% of breaks recovered. It is the largest single number in this
    system, and it costs no hardware - only running this once against an
    environment where the selectors still work.

    ``locators`` is the table the script already keeps, in any of the shapes
    an RPA script naturally writes it:

        {"catalogue.card": (By.CSS_SELECTOR, "article.product_pod")}
        {"catalogue.card": (By.CSS_SELECTOR, "article.product_pod", "a card")}
        {"catalogue.card": "article.product_pod"}

    Returns ``{key: bool}``. A False is not a healing failure - healing is
    deliberately not attempted here, because a repair recorded during
    seeding would be a guess dressed as evidence. It means the selector
    matched nothing on a page where it should have, which is a locator that
    was wrong when it was written. Printing that list at seed time is worth
    more than discovering it at three in the morning.
    """
    page = _adapt(driver)
    wait = timeout if timeout is not None else _S.find_timeout_s

    # Fetch the bundle before the loop rather than letting the first
    # fingerprint trigger it. _report_baseline deliberately never waits for
    # the download - a host that black-holes would otherwise turn the first
    # good lookup of a *running* bot into a stall - and it relies on the
    # locator getting another chance the next time the step runs. Seeding
    # runs each locator exactly once, so there is no next time: without
    # this, the first locator in the table silently records nothing. Found
    # by seeding a ten-step table and getting nine baselines back.
    _js()

    out = {}
    for key, spec in dict(locators).items():
        describe = ""
        if isinstance(spec, (tuple, list)):
            by, value = spec[0], spec[1] if len(spec) > 1 else None
            if len(spec) > 2:
                describe = spec[2] or ""
        else:
            by, value = spec, None
        selector = _selector(by, value)
        el, index = _lookup_indexed(page, selector, wait)
        if el is None:
            out[key] = False
            log.warning("seed: %s did not resolve (%s) - this locator has "
                        "never been observed working, so a later break "
                        "cannot be told from a selector that was wrong on "
                        "the day it shipped", key, selector)
            continue
        # Filed under the key find() and _heal() look it up by. The table
        # key is not one of them: find() is never told it, so a baseline
        # filed under "catalogue.card" is one no repair can ever find.
        _report_baseline(page, describe or selector, selector, index)
        out[key] = True
    _home(page)
    proven = sum(1 for v in out.values() if v)
    log.info("seed: %d of %d locators proven", proven, len(out))
    return out


def find(driver, by, value=None, optional=False, describe="", timeout=None):
    """Find one element, repairing the selector if the page has changed.

    ``find(driver, By.ID, "submit")`` behaves like ``driver.find_element``
    until the selector stops matching. Then it asks the healing service
    which element the step meant, binds that, and carries on.

    Pass ``optional=True`` for an element that may legitimately be absent.
    You get ``None`` rather than an exception, and the repair is held to a
    stricter standard so a missing element is never clicked as some other
    one.

    Raises ``NoSuchElementException`` when a required element cannot be
    found or repaired - the same exception the plain Selenium call raises,
    so existing error handling still works.
    """
    page = _adapt(driver)
    selector = _selector(by, value)
    wait = timeout if timeout is not None else (
        _S.optional_timeout_s if optional else _S.find_timeout_s)

    el, index = _lookup_indexed(page, selector, _probe_wait(selector, wait))
    if el is not None:
        # The step worked. That makes this element the evidence a future
        # repair on this locator is measured against, and it is the only
        # moment it can be collected - once the selector breaks, nothing
        # can say which element it used to mean.
        _report_baseline(page, describe or selector, selector, index)
        _step_done(page, describe or selector, selector)
        return el

    if _S.enabled:
        action = "check" if optional else "interact"
        bound = ""
        try:
            # An optional element absent from the page comes back as not
            # found without a harvest - _heal asks the page first.
            el, bound = _heal(page, selector, action, optional, describe)
        except Exception as exc:                       # never take a bot down
            log.warning("healing failed for %s (%s)", selector, exc)
            el = None
        if el is not None:
            _step_done(page, describe or selector, bound, replaced=selector)
            return el

    _home(page)
    if optional:
        # An absent pop-up is not a broken flow: the chain carries on.
        return None
    _step_failed(page)
    raise NoSuchElementException(
        "no element for %s, and it could not be repaired" % selector)


def find_all(driver, by, value=None, describe="", timeout=None):
    """Find every matching element. Returns ``[]`` rather than raising.

    Collection steps are the ones that quietly scrape nothing when a class
    name changes, so this heals too - but an empty list is a legitimate
    result, and is never treated as a failure worth an exception.
    """
    page = _adapt(driver)
    selector = _selector(by, value)
    wait = timeout if timeout is not None else _S.find_timeout_s

    found, index = _lookup_indexed(page, selector,
                                   _probe_wait(selector, wait),
                                   want_all=True)
    if found:
        _report_baseline(page, describe or selector, selector, index)
        _step_done(page, describe or selector, selector)
        return found

    if _S.enabled:
        try:
            el, bound = _heal(page, selector, "collect", False, describe)
        except Exception as exc:
            log.warning("healing failed for %s (%s)", selector, exc)
            el, bound = None, ""
        if el is not None:
            _step_done(page, describe or selector, bound, replaced=selector)
            # One match proves the repair; the caller asked for the set, so
            # re-query with the selector that was just *bound*.
            #
            # This used to re-query with `selector` - the authored one, the
            # one that had already failed - so the query returned nothing
            # and the `or [el]` fallback handed back a single element. Every
            # healed collection step silently scraped one row instead of all
            # of them, which for the steps this exists to protect (scrape a
            # price table, read every order id) is worse than failing: the
            # bot reports success and the data is quietly a third of itself.
            # Caught by the end-to-end check, which scraped 1 of 3 products.
            if bound:
                try:
                    every = page.query(bound)
                    shown = [e for e in every if page.visible(e)]
                    if shown:
                        return shown
                except Exception:
                    pass
            return [el]
    _home(page)
    return []
