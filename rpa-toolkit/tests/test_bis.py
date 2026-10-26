"""Offline unit tests for the thin client.

``bis.py`` is the one file an RPA developer copies next to their script, so
these run with no browser, no driver binary, no healing service and no
network - the same rule the other suites follow.

The selector-language tests are checked against ``bis_core`` rather than
against a list of expected strings. The two implementations have to agree,
and asserting that directly is what catches a drift that a fixed expectation
would sail past.

    python rpa-toolkit/tests/test_bis.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLKIT = os.path.dirname(_HERE)
sys.path.insert(0, _TOOLKIT)
sys.path.insert(0, os.path.join(os.path.dirname(_TOOLKIT), "bis-core"))

import bis                                                    # noqa: E402
from bis_core.descriptors import normalize_selector           # noqa: E402


# Every shape an RPA developer plausibly pastes into find(). The XPath ones
# past the first two are the point: a parenthesised expression and a relative
# one are what a team migrating from another platform writes, and both were
# being handed to querySelectorAll.
_SELECTORS = [
    "//div",
    "/html/body/div[2]",
    "(//div)[1]",
    "(//a[@class='x'])[last()]",
    "./span",
    ".//span",
    "xpath=//p",
    "#submit",
    "a.card",
    "[data-testid='go']",
    "div > span",
    "input[name='q']",
]


def test_the_thin_client_and_the_core_agree_on_selector_language():
    """One selector must not be CSS on the bot and XPath on the service.

    The prefix decides three separate things: whether this file hands the
    selector to querySelectorAll or to document.evaluate, which language the
    harvester proposes repairs in, and which language a stored fleet repair
    is served for. bis.py used to test startswith('/') alone, so '(//div)[1]'
    and './span' went to querySelectorAll, raised a syntax error that
    _lookup_indexed swallows, burned the whole find timeout matching nothing,
    and were then "healed" - in the wrong language - from a selector that had
    never been broken.
    """
    for raw in _SELECTORS:
        assert bis._selector(raw, None) == normalize_selector(raw), (
            "%r: bis.py says %r, bis-core says %r"
            % (raw, bis._selector(raw, None), normalize_selector(raw)))


def test_bare_xpath_gets_its_prefix_and_css_does_not():
    """The concrete half of the agreement above, stated outright."""
    assert bis._selector("(//div)[1]", None) == "xpath=(//div)[1]"
    assert bis._selector("./span", None) == "xpath=./span"
    assert bis._selector("//button", None) == "xpath=//button"
    assert bis._selector("#submit", None) == "#submit"
    assert bis._selector("a.card", None) == "a.card"
    # Already prefixed: left exactly as it was, never doubled.
    assert bis._selector("xpath=//p", None) == "xpath=//p"


def test_a_selenium_pair_still_maps_the_way_it_always_did():
    """The (by, value) form is untouched by the bare-string fix."""
    assert bis._selector("xpath", "//a") == "xpath=//a"
    assert bis._selector("id", "submit") == "#submit"
    assert bis._selector("name", "q") == "[name='q']"
    assert bis._selector("css selector", ".card") == ".card"
    assert bis._selector("link text", "Next") == \
        "xpath=//a[normalize-space()='Next']"


def test_a_prefixed_selector_round_trips_through_split():
    """Whatever _selector produces, _split has to be able to take back."""
    for raw in _SELECTORS:
        by, value = bis._split(bis._selector(raw, None))
        assert by in ("xpath", "css selector")
        assert value and not value.startswith("xpath=")


def test_the_candidate_ceiling_is_configurable():
    """It was a literal in _heal while HealingConfig carried a max_candidates
    that nothing read - the knob and the value in different halves."""
    before = bis._S.max_candidates
    try:
        assert before == 900
        bis.configure(max_candidates=120)
        assert bis._S.max_candidates == 120
    finally:
        bis.configure(max_candidates=before)


def test_no_service_configured_means_no_calls_and_no_exceptions():
    """A script written against this file behaves like the one it replaced.

    With nothing configured there is no service to reach, and reaching for
    one must never be how a bot dies.
    """
    before = bis._S.service_url
    try:
        bis.configure(service_url="")
        assert bis._call("/v1/health") is None
        assert bis.health() is None
    finally:
        bis.configure(service_url=before)


def test_an_unreachable_service_degrades_instead_of_raising():
    """Invariant 10, at the client end of the wire."""
    before = bis._S.service_url
    timeout = bis._S.http_timeout_s
    try:
        # Loopback, so there is no DNS lookup and no packet leaves the host:
        # port 9 is discard and nothing is listening, which is refused
        # immediately. A hostname here - even a reserved one - would put a
        # resolver call in a suite that is meant to touch no network at all.
        bis.configure(service_url="http://127.0.0.1:9", http_timeout_s=0.5)
        assert bis._call("/v1/health") is None
        assert bis._call("/v1/adjudicate", {"x": 1}) is None
    finally:
        bis.configure(service_url=before, http_timeout_s=timeout)


def test_the_client_sends_its_session_with_every_call():
    """A public service keeps each demo run's memory under the session it
    names, so the session goes with every call - a GET as much as a POST -
    and with none set, no header goes at all."""
    import io
    import json
    sent = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        sent.append(dict(req.header_items()))
        return Reply(json.dumps({"ready": True}).encode("utf-8"))

    saved = (bis.urllib.request.urlopen, bis._S.service_url, bis._S.session)
    bis.urllib.request.urlopen = urlopen
    try:
        bis.configure(service_url="http://svc.test", session="demo-1")
        bis._call("/v1/health")
        bis._call("/v1/adjudicate", {"x": 1})
        bis.configure(session="")
        bis._call("/v1/health")
    finally:
        (bis.urllib.request.urlopen, bis._S.service_url,
         bis._S.session) = saved
    assert [h.get("X-bis-session") for h in sent] == [
        "demo-1", "demo-1", None], sent


def test_seed_reports_which_locators_have_never_been_seen_working():
    """bis.seed() proves locators before they break, and names the ones
    that never resolved - those were wrong when they were written, and
    saying so at seed time beats discovering it at three in the morning."""
    import bis
    calls = []

    class _Page:
        def query(self, selector):
            calls.append(selector)
            return [] if "never-existed" in selector else ["el"]

        def visible(self, el):
            return True

        def where(self):
            return {"url": "https://example.test/app", "title": "t"}

        def evaluate(self, expr, arg):
            return None

    original = bis._adapt, bis._report_baseline
    bis._adapt = lambda driver: _Page()
    bis._report_baseline = lambda *a, **k: None
    try:
        out = bis.seed(object(), {
            "good": ("css selector", ".exists"),
            "bad": ("css selector", ".never-existed"),
        }, timeout=0)
    finally:
        bis._adapt, bis._report_baseline = original
    assert out == {"good": True, "bad": False}, out


def test_seeding_records_a_baseline_for_the_first_locator_too():
    """The capture path never waits for the JavaScript bundle to download.

    That rule protects a *running* bot: a service that black-holes must not
    turn the first good lookup of a run into a stall. It works because the
    locator gets another chance the next time the step runs. Seeding runs
    each locator exactly once, so there is no next time - and the first
    entry in the table recorded nothing at all. Ten locators seeded, nine
    baselines stored, and the missing one was always whichever came first.
    """
    import bis
    served = {"protocol": bis.PROTOCOL, "version": "t",
              "harvest": "h", "probe": "p", "describe_one": "d",
              "presence": "pr"}

    class _Page:
        def query(self, selector):
            return ["el"]

        def visible(self, el):
            return True

        def where(self):
            return {"url": "https://example.test/app", "title": "t"}

        def evaluate(self, expr, arg):
            return {"tag": "input", "id": "x", "visible": True}

    def _call(path, payload=None):
        return served if path == "/v1/js" else {}

    saved = (bis._adapt, bis._call, dict(bis._bundle),
             set(bis._seen_baselines), dict(bis._baselines),
             bis._S.service_url)
    bis._adapt = lambda driver: _Page()
    bis._call = _call
    bis._bundle.update({"harvest": "", "describe_one": ""})
    bis._seen_baselines.clear()
    bis._baselines.clear()
    bis._S.service_url = "http://svc.test"
    try:
        bis.seed(object(), {
            "first": ("css selector", ".a", "the first control"),
            "second": ("css selector", ".b", "the second control"),
        }, timeout=0)
        keys = {scope[0] for scope in bis._baselines}
    finally:
        (bis._adapt, bis._call, bundle, seen, base,
         bis._S.service_url) = saved
        bis._bundle.clear(); bis._bundle.update(bundle)
        bis._seen_baselines.clear(); bis._seen_baselines.update(seen)
        bis._baselines.clear(); bis._baselines.update(base)
    assert keys == {"the first control", "the second control"}, keys


def test_seed_and_find_file_a_locator_under_the_same_key():
    """A seeded baseline the step can never look up again is no baseline.

    seed() filed a locator with no description under its table key
    ("catalogue.card"), while find() and _heal() file and look it up under
    the selector ("article.product_pod"). Every undescribed locator a team
    seeded stayed on the cold path - 43% of breaks recovered against 87%
    warm - with nothing to say the seed pass had been wasted.
    """
    import bis
    keys = []

    class _Page:
        def query(self, selector):
            return ["el"]

        def visible(self, el):
            return True

        def where(self):
            return {"url": "https://example.test/app", "title": "t"}

        def evaluate(self, expr, arg):
            return None

    original = bis._adapt, bis._report_baseline
    bis._adapt = lambda driver: _Page()
    bis._report_baseline = lambda page, key, *a, **k: keys.append(key)
    try:
        for describe in ("", "a book card"):
            del keys[:]
            spec = ("css selector", "article.product_pod")
            if describe:
                spec += (describe,)
            bis.seed(object(), {"catalogue.card": spec}, timeout=0)
            bis.find(object(), "css selector", "article.product_pod",
                     describe=describe, timeout=0)
            assert len(keys) == 2, keys
            assert keys[0] == keys[1], (
                "seed filed %r, find looks up %r" % (keys[0], keys[1]))
            assert keys[0] == (describe or "article.product_pod"), keys
    finally:
        bis._adapt, bis._report_baseline = original


def test_a_selector_already_repaired_is_not_waited_on_again():
    """RPA work is loop-shaped, and a dead address comes round every row.

    Measured on a three-joiner onboarding queue with three stale
    addresses: 54 seconds of a 229-second run went on waiting the full
    find timeout for addresses the process had already repaired once. The
    selector is still tried - briefly - so a page that reverts is picked
    up at once, and the developer's stated intent still wins whenever it
    works.
    """
    import bis
    waits = []

    class _Page:
        def query(self, selector):
            return []

        def visible(self, el):
            return True

        def where(self):
            return {"url": "https://example.test/app", "title": "t"}

        def evaluate(self, expr, arg):
            return None

    def _spy(page, selector, wait, want_all=False):
        waits.append(wait)
        return ([], 0) if want_all else (None, 0)

    saved = (bis._adapt, bis._lookup_indexed, set(bis._repaired_here),
             bis._S.enabled)
    bis._adapt = lambda driver: _Page()
    bis._lookup_indexed = _spy
    bis._S.enabled = False          # no healing: only the wait matters here
    bis._repaired_here.clear()
    try:
        for _ in range(2):
            try:
                bis.find(object(), "css selector", ".gone")
            except bis.NoSuchElementException:
                pass
        first_pass = list(waits)

        # Now say this selector has been repaired once already.
        bis._repaired_here.add(".gone")
        waits[:] = []
        try:
            bis.find(object(), "css selector", ".gone")
        except bis.NoSuchElementException:
            pass
        after = list(waits)
    finally:
        (bis._adapt, bis._lookup_indexed, known, bis._S.enabled) = saved
        bis._repaired_here.clear()
        bis._repaired_here.update(known)

    assert first_pass == [bis._S.find_timeout_s] * 2, first_pass
    assert after == [bis._S.known_broken_probe_s], after
    assert bis._S.known_broken_probe_s < bis._S.find_timeout_s


# ------------------------------------------------- a heal, end to end
#
# A page whose authored selector is gone, a service that answers with a
# verdict, and nothing else - enough to drive _heal() the way a real step
# does. The JavaScript is replaced by names, so what reaches the page is
# visible without a browser.


class _Element(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<%s>" % self.name


class _HealPage(object):
    """matches maps a selector to the elements it resolves to."""

    def __init__(self, matches, candidates, same=None):
        self.matches = matches
        self.candidates = candidates
        self.same = same
        self.evaluated = []

    def query(self, selector):
        return list(self.matches.get(selector, []))

    def visible(self, el):
        return True

    def where(self):
        return {"url": "https://example.test/app", "title": "Sign in"}

    def evaluate(self, expr, arg):
        self.evaluated.append((expr, arg))
        if expr == "HARVEST":
            return {"harvest": "h-1", "candidates": self.candidates}
        if expr == "SAME":
            return self.same(arg) if self.same else None
        if expr == "DESCRIBE":
            return {"tag": "input", "id": "x", "visible": True}
        return None


_BUNDLE = {"protocol": "1", "version": "t", "harvest": "HARVEST",
           "probe": "PROBE", "describe_one": "DESCRIBE",
           "presence": "PRESENCE", "same": "SAME"}


class _Synchronous(object):
    """threading.Thread, run where it is started, so a test sees the POST."""

    def __init__(self, target=None, daemon=None, name=None):
        self.target = target

    def start(self):
        self.target()


import contextlib  # noqa: E402


@contextlib.contextmanager
def _service(verdict, review_hook=None):
    """bis talking to a fake service that answers every heal with `verdict`.

    Yields the list of (path, payload) POSTs the client makes; every
    module-level memory bis keeps is restored afterwards.
    """
    import bis
    posts = []

    def _call(path, payload=None):
        if path == "/v1/js":
            return dict(_BUNDLE)
        posts.append((path, payload))
        if path == "/v1/adjudicate":
            return dict(verdict)
        return {}

    memory = (bis._seen_baselines, bis._baselines, bis._repaired_here,
              getattr(bis, "_previous", {}),
              getattr(bis, "_reported_outcomes", set()))
    kept = [type(m)(m) for m in memory]
    saved = (bis._call, dict(bis._bundle), bis._S.service_url,
             getattr(bis._S, "review_hook", None), bis.threading.Thread,
             bis._S.find_timeout_s, bis._adapt)
    adapt = bis._adapt
    # A fake page already speaks the adapter's four operations.
    bis._adapt = lambda d: d if isinstance(d, _HealPage) else adapt(d)
    bis._call = _call
    bis._bundle.update({"harvest": "", "describe_one": ""})
    for m in memory:
        m.clear()
    bis._S.service_url = "http://svc.test"
    bis._S.review_hook = review_hook
    bis._S.find_timeout_s = 0
    bis.threading.Thread = _Synchronous
    try:
        yield posts
    finally:
        (bis._call, bundle, bis._S.service_url, bis._S.review_hook,
         bis.threading.Thread, bis._S.find_timeout_s, bis._adapt) = saved
        bis._bundle.clear(); bis._bundle.update(bundle)
        for m, k in zip(memory, kept):
            m.clear()
            m.update(k)


def _heal_with(page, verdict, review_hook=None, action="interact",
               selector="#legacyEmail", describe=""):
    """One _heal() against `page`, the service answering `verdict`.

    Returns (element, bound selector, the POSTs the client made).
    """
    import bis
    with _service(verdict, review_hook) as posts:
        el, bound = bis._heal(page, selector, action, False, describe)
    return el, bound, posts


def _login_page(same=None):
    """Two candidates; the service will choose the email box, index 0."""
    email, password = _Element("email"), _Element("password")
    candidates = [{"tag": "input", "id": "email", "selectors": ["#email"]},
                  {"tag": "input", "id": "password",
                   "selectors": ["#password"]}]
    page = _HealPage({"#email": [email], "#password": [password]},
                     candidates, same)
    return page, email, password


def _verdict(band="apply", confidence=0.97, chosen=0, **extra):
    v = {"chosen": chosen, "confidence": confidence, "band": band,
         "strategy": "ai-confirmed", "rationale": "the email box",
         "repair_id": 0,
         "shortlist": [{"index": 0, "score": 0.9, "summary":
                        '[0] <input id="email">', "chosen": True}]}
    v.update(extra)
    return v


# The selector-language list above, plus the shapes where reading the tag
# off the last step goes wrong: an ancestor naming a different tag, a
# combinator or a slash inside quotes, a predicate, a parenthesised XPath.
_TAGGED = _SELECTORS + [
    "input#firstName", "div.card > a.title", "form#login input[type='email']",
    "//div[@class='card']//a[@class='title']", "xpath=//input[@name='q']",
    "//*[@id='x']", "button[aria-label='Go > now']",
    "//a[contains(text(),'a/b')]", "text=Login", "ul li:nth-of-type(2)",
    "//div/text()", "descendant::span", "(//button)[2]",
    "//input[@id='a' and @type='text']", "css=select.country",
    "a.card ~ span.badge", "div+p",
]


def test_the_harvest_hint_names_the_tag_the_core_reads():
    """The harvest keeps 900 elements, and 35% of real pages have more. The
    tag the broken selector names is what keeps its kind in the cut, so
    bis.py reads it the way bis_core's selector_tag does."""
    from bis_core.descriptors import selector_tag
    for raw in _TAGGED:
        sel = bis._selector(raw, None)
        want = selector_tag(normalize_selector(raw)) or ""
        assert bis._selector_tag(sel) == want, (
            "%r: bis.py reads %r, bis-core %r"
            % (raw, bis._selector_tag(sel), want))


def test_the_harvest_is_told_what_the_broken_selector_named():
    """It used to be sent an empty tagHint, so the cut kept no preference."""
    page, _, _ = _login_page()
    _heal_with(page, _verdict(), selector="input#legacyEmail")
    opts = [arg for expr, arg in page.evaluated if expr == "HARVEST"]
    assert opts and opts[0]["tagHint"] == "input", opts
    assert opts[0]["selector"] == "input#legacyEmail", opts


def _identity(chosen):
    """SAME_JS, as the page would answer it: is `chosen` among the bound?"""
    return lambda arg: {"known": True, "same": chosen in arg["els"]}


def test_a_proposal_that_binds_another_element_is_not_taken():
    """A repair must bind the element the service chose, not merely one its
    selector matches. Here the first proposal - a shared class - resolves
    to the password box; it is passed over for one that is the email box,
    and the page is asked about exactly the harvest the index refers to."""
    email, password = _Element("email"), _Element("password")
    candidates = [{"tag": "input", "id": "email",
                   "selectors": ["input.form-control", "#email"]},
                  {"tag": "input", "id": "password",
                   "selectors": ["#password"]}]
    matches = {"input.form-control": [password], "#email": [email],
               "#password": [password]}
    page = _HealPage(matches, candidates, same=_identity(email))
    el, bound, _ = _heal_with(page, _verdict(chosen=0))
    assert el is email and bound == "#email", (el, bound)
    asked = [arg for expr, arg in page.evaluated if expr == "SAME"]
    assert asked and asked[0]["harvest"] == "h-1", asked
    assert asked[0]["index"] == 0 and asked[0]["els"] == [password], asked

    # A page that cannot tell - no record of that harvest, a node a
    # re-render replaced - is no evidence either way: bound as before.
    unsure = _HealPage(matches, candidates,
                       same=lambda arg: {"known": False})
    el, bound, _ = _heal_with(unsure, _verdict(chosen=0))
    assert bound == "input.form-control", bound


def test_a_collection_binds_a_set_that_holds_the_element_chosen():
    """For a set, the chosen row has to be one of the rows matched."""
    rows = [_Element("row-%d" % k) for k in range(3)]
    other = [_Element("promo-%d" % k) for k in range(2)]
    candidates = [{"tag": "div", "classes": ["row"],
                   "selectors": ["div.promo", "div.row", "#row-0"]}]
    page = _HealPage({"div.promo": other, "div.row": rows,
                      "#row-0": rows[:1]}, candidates, same=_identity(rows[0]))
    el, bound, _ = _heal_with(page, _verdict(chosen=0), action="collect",
                              selector="div.row-v1")
    assert bound == "div.row" and el is rows[0], (el, bound)


def test_without_a_review_hook_a_repair_held_for_review_is_not_found():
    """What it always was: review is short of apply, so the step fails."""
    page, _, _ = _login_page()
    el, bound, posts = _heal_with(page, _verdict(band="review",
                                                 confidence=0.70))
    assert el is None and bound == "", (el, bound)
    assert [path for path, _ in posts] == ["/v1/adjudicate"], posts


def test_a_review_hook_decides_a_repair_held_for_review():
    """A higher bar sends more repairs to review, and review used to be a
    failed step and nothing else. The hook gives it somewhere to go: it
    sees the element the repair bound and says yes or no. Its verdict is
    reported either way - the one label this system gets from production
    - and an approved repair is reported as approved, so the fleet store
    keeps it at 1.0 and the next bot is not asked again.
    """
    seen = []

    def approve(info):
        seen.append(info)
        return True

    page, email, _ = _login_page()
    el, bound, posts = _heal_with(
        page, _verdict(band="review", confidence=0.70,
                       ai_reason="The input 'Email' matches."),
        review_hook=approve, describe="the Email box")
    assert el is email and bound == "#email", (el, bound)
    assert len(seen) == 1, seen
    info = seen[0]
    assert info["element"] is email and info["proposal"] == "#email", info
    assert info["selector"] == "#legacyEmail", info
    assert info["description"] == "the Email box", info
    assert abs(info["confidence"] - 0.70) < 1e-9, info
    assert info["reason"] == "The input 'Email' matches.", info
    assert info["summary"] == '[0] <input id="email">', info
    assert info["url"] == "https://example.test/app", info
    reviews = [p for path, p in posts if path == "/v1/review"]
    assert len(reviews) == 1 and reviews[0]["approved"] is True, reviews
    assert reviews[0]["proposal"] == "#email", reviews[0]
    assert reviews[0]["shortlist"] == ['[0] <input id="email">'], reviews[0]
    assert reviews[0]["chosen_rank"] == 0, reviews[0]
    reports = [p for path, p in posts if path == "/v1/baseline"]
    assert len(reports) == 1 and reports[0]["approved"] is True, reports


def test_anything_but_an_approval_is_a_rejection():
    """False, None, or a hook that raises: not found, and reported as a
    rejection. A bot never acts on a repair nobody accepted, and never
    crashes because the reviewer's code did."""
    def boom(info):
        raise RuntimeError("the review queue is down")

    for hook in (lambda info: False, lambda info: None, boom):
        page, _, _ = _login_page()
        el, bound, posts = _heal_with(page, _verdict(band="review"),
                                      review_hook=hook)
        assert el is None and bound == "", (el, bound)
        reviews = [p for path, p in posts if path == "/v1/review"]
        assert len(reviews) == 1 and reviews[0]["approved"] is False
        assert not [p for path, p in posts if path == "/v1/baseline"]

    # An applied repair never asks.
    asked = []
    page, email, _ = _login_page()
    el, _, _ = _heal_with(page, _verdict(band="apply"),
                          review_hook=lambda info: asked.append(info))
    assert el is email and asked == []


def test_a_review_hook_is_set_and_cleared_through_configure():
    before = getattr(bis._S, "review_hook", None)
    try:
        hook = lambda info: True               # noqa: E731
        bis.configure(review_hook=hook)
        assert bis._S.review_hook is hook
        bis.configure(find_timeout_s=bis._S.find_timeout_s)
        assert bis._S.review_hook is hook, "an unrelated call cleared it"
        bis.configure(review_hook=None)
        assert bis._S.review_hook is None
    finally:
        bis._S.review_hook = before


class _Journey(_HealPage):
    """A page whose address the test moves, as a bot's clicks would."""

    def __init__(self, matches, candidates=(), same=None):
        super().__init__(matches, list(candidates), same)
        self.url = "https://example.test/cart"

    def where(self):
        return {"url": self.url, "title": "t"}


def _outcomes(posts):
    return [p for path, p in posts if path == "/v1/outcome"]


def test_each_step_reports_where_the_next_one_found_itself():
    """The page a step finds its element on is where the previous step led.
    From a run whose selectors worked, that is what the service learns
    each step leads to - once per transition per run, not once per row."""
    import bis
    page = _Journey({"#checkout": ["el"], "#pay": ["el"], "#done": ["el"]})
    with _service(_verdict()) as posts:
        bis.find(page, "#checkout", timeout=0)
        page.url = "https://example.test/pay?step=2"
        bis.find(page, "#pay", timeout=0)
        page.url = "https://example.test/done"
        bis.find(page, "#done", timeout=0)
        page.url = "https://example.test/cart"
        bis.find(page, "#checkout", timeout=0)
        page.url = "https://example.test/pay?step=3"
        bis.find(page, "#pay", timeout=0)    # the same move again
    got = _outcomes(posts)
    assert [(o["key"], o["url"], o["next_url"]) for o in got] == [
        ("#checkout", "https://example.test/cart",
         "https://example.test/pay?step=2"),
        ("#pay", "https://example.test/pay?step=2",
         "https://example.test/done"),
        ("#done", "https://example.test/done",
         "https://example.test/cart")], got
    assert not any(o["healed"] for o in got), got


def test_a_healed_step_reports_where_its_repair_led():
    """So a repair that binds the wrong element - and so never fails to
    bind - is caught when it leads somewhere a good run never went."""
    import bis
    email = _Element("email")
    page = _Journey({"#email": [email], "#next": ["el"]},
                    candidates=[{"tag": "input", "id": "email",
                                 "selectors": ["#email"]}])
    with _service(_verdict()) as posts:
        assert bis.find(page, "#legacyEmail", timeout=0) is email
        page.url = "https://example.test/welcome"
        bis.find(page, "#next", timeout=0)
    got = _outcomes(posts)
    assert len(got) == 1, got
    assert got[0]["healed"] is True and got[0]["selector"] == "#email", got
    assert got[0]["replaced"] == "#legacyEmail", got
    assert got[0]["next_url"] == "https://example.test/welcome", got


def test_a_step_that_fails_breaks_the_chain():
    """After a failure the next page found is not where the last success
    led - the bot handled an exception, retried, moved on."""
    import bis
    page = _Journey({"#first": ["el"], "#third": ["el"]})
    with _service(_verdict(band="refuse")) as posts:
        bis.find(page, "#first", timeout=0)
        try:
            bis.find(page, "#second", timeout=0)
        except bis.NoSuchElementException:
            pass
        page.url = "https://example.test/elsewhere"
        bis.find(page, "#third", timeout=0)
    assert _outcomes(posts) == [], _outcomes(posts)


def test_a_repair_reports_the_confidence_it_was_given():
    """The fleet store keeps the confidence a repair was decided at, so the
    client has to say what it was told - it used to be stored as 1.0."""
    page, email, _ = _login_page()
    el, bound, posts = _heal_with(page, _verdict(confidence=0.83))
    assert el is email and bound == "#email"
    reports = [p for path, p in posts if path == "/v1/baseline"]
    assert len(reports) == 1, posts
    assert reports[0]["replaced"] == "#legacyEmail"
    assert abs(reports[0].get("confidence", 0.0) - 0.83) < 1e-9, reports[0]
    assert reports[0].get("approved") is False, reports[0]


# ------------------------------------------------ shadow roots and frames
# 'app-shell >>> button.save' is the button inside <app-shell>'s open shadow
# root, and 'iframe#main >>> #email' the box inside that frame's document.
# The harvest proposes them for elements it found there; either adapter
# resolves one a step at a time.

_CHAINS = [
    "app-shell >>> button.save",
    "app-shell>>>button.save",
    "iframe#main >>> //input[@id='u']",
    "//iframe[@id='m'] >>> #email",
    "x-a >>> x-b >>> [data-testid='go']",
    "a[title='x >>> y'] >>> b",
]


def test_a_chain_is_read_the_same_on_both_sides():
    """The page splits a chain, this file resolves it a step at a time and
    the core reads its last step for the ranker and the stores - three
    readings that must agree on where the steps are and what language each
    one is in, or a repair is made in one shape and looked up in another."""
    from bis_core.descriptors import chain_steps, selector_tag
    for raw in _CHAINS:
        sel = bis._selector(raw, None)
        assert sel == normalize_selector(raw), (raw, sel,
                                                normalize_selector(raw))
        assert bis._chain(sel) == chain_steps(sel), (raw, bis._chain(sel))
        assert bis._selector_tag(sel) == (selector_tag(sel) or ""), raw
    assert bis._chain("a[title='x >>> y'] >>> b") == ["a[title='x >>> y']",
                                                      "b"]
    assert bis._selector("//iframe >>> //input", None) == \
        "xpath=//iframe >>> xpath=//input"
    # The shape of the element itself decides, not the host's.
    assert not bis._uses_id("app-shell#shell >>> button.save")
    assert bis._uses_id("app-shell >>> #save")


class _NoShadow(Exception):
    """What Selenium raises for an element without an open shadow root."""


class _Node(object):
    """A Selenium element: what it holds, if anything."""

    def __init__(self, name, tag="div", shadow=None, frame=None, shown=True):
        self.name, self.tag_name = name, tag
        self._shadow, self.frame, self.shown = shadow, frame, shown

    @property
    def shadow_root(self):
        if self._shadow is None:
            raise _NoShadow(self.name)
        return self._shadow

    def is_displayed(self):
        return self.shown

    def __repr__(self):
        return "<%s>" % self.name


class _Doc(object):
    """A document or a shadow root, answering by selector. WebDriver will
    not search a shadow root by XPath."""

    def __init__(self, found, shadow=False):
        self.found, self.shadow = found, shadow

    def find_elements(self, by, value):
        if self.shadow and by == "xpath":
            raise ValueError("invalid locator")
        return list(self.found.get(value, []))


class _Driver(object):
    """A WebDriver over nested documents, tracking which one it is in."""

    def __init__(self, top):
        self.top = top
        self.where = [top]
        self.switch_to = self
        self.ran = []
        self.marks = {}
        self.current_url, self.title = "https://example.test/app", "App"

    def find_elements(self, by, value):
        return self.where[-1].find_elements(by, value)

    def frame(self, el):
        self.where.append(el.frame)

    def parent_frame(self):
        if len(self.where) > 1:
            self.where.pop()

    def default_content(self):
        del self.where[1:]

    def execute_script(self, script, *args):
        doc = self.where[-1]
        if "window.__bisEntered = " in script:
            self.marks[id(doc)] = args[0]
            return None
        if "window.__bisEntered" in script:
            return self.marks.get(id(doc), "")
        self.ran.append((doc, script))
        return {"ran": True}


def _framed():
    """A page whose sign-in form now sits inside a frame."""
    email = _Node("email", "input")
    inner = _Doc({"#email": [email]})
    frame = _Node("main", "iframe", frame=inner)
    top = _Doc({"iframe#main": [frame], "#nav": [_Node("nav", "a")]})
    return _Driver(top), top, inner, email


def test_selenium_steps_into_a_shadow_root_without_moving_the_driver():
    """A shadow root is part of the page's document, so WebDriver searches
    it from the host's shadow_root and the driver stays where it is."""
    save = _Node("save", "button")
    shell = _Node("shell", "app-shell",
                  shadow=_Doc({"button.save": [save]}, shadow=True))
    driver = _Driver(_Doc({"app-shell": [shell], "#plain": [save]}))
    page = bis._Selenium(driver)
    assert page.query("app-shell >>> button.save") == [save]
    assert page.query("app-shell >>> button.gone") == []
    # XPath cannot see into a shadow root: nothing, rather than an error.
    assert page.query("app-shell >>> xpath=//button") == []
    assert page.query("#plain") == [save]
    assert driver.where == [driver.top]


def test_selenium_goes_into_a_frame_for_an_element_inside_it():
    """A WebDriver element inside a frame can only be used while the driver
    is in that frame, so a lookup that ends in one leaves the driver there,
    as switch_to.frame would. The page's JavaScript still runs where the
    step began, and the driver goes back in after, so the element found
    stays usable. The next lookup starts where the step began again."""
    driver, top, inner, email = _framed()
    page = bis._Selenium(driver)
    assert page.query("iframe#main >>> #email") == [email]
    assert driver.where == [top, inner], driver.where
    page.evaluate("HARVEST", {})
    assert driver.ran[-1][0] is top, "the JavaScript ran inside the frame"
    assert driver.where == [top, inner], "it did not go back into the frame"
    # find() makes a new adapter every call; the driver's frames are kept.
    again = bis._Selenium(driver)
    assert [n.name for n in again.query("#nav")] == ["nav"]
    assert driver.where == [top], driver.where
    # A frame that does not hold the rest of the chain is left at once.
    assert again.query("iframe#main >>> #gone") == []
    assert driver.where == [top], driver.where


def test_a_driver_the_script_moved_itself_is_left_where_it_is():
    """Once the script has switched frames itself it has said where it
    wants to be. Only a driver still in the frame this file left it in is
    brought back out."""
    driver, top, inner, email = _framed()
    other = _Doc({"#x": [_Node("x")]})
    page = bis._Selenium(driver)
    assert page.query("iframe#main >>> #email") == [email]
    driver.default_content()
    driver.where.append(other)                 # the script's own frame
    assert [n.name for n in bis._Selenium(driver).query("#x")] == ["x"]
    assert driver.where == [top, other], driver.where


class _PwArray(object):
    """A Playwright JSHandle holding an array of elements."""

    def __init__(self, items):
        self.items = items

    def get_properties(self):
        return {str(k): _PwProp(v) for k, v in enumerate(self.items)}

    def dispose(self):
        pass


class _PwProp(object):
    def __init__(self, el):
        self.el = el

    def as_element(self):
        return self.el


class _PwEl(object):
    """A Playwright ElementHandle: a frame's owner, or a shadow host."""

    def __init__(self, name, frame=None, shadow=None):
        self.name, self.frame, self.shadow = name, frame, shadow

    def content_frame(self):
        return self.frame

    def evaluate_handle(self, expression, arg=None):
        return _PwArray(list((self.shadow or {}).get(arg, [])))

    def is_visible(self):
        return True

    def __repr__(self):
        return "<%s>" % self.name


class _PwDoc(object):
    """A Playwright Page or Frame, answering by selector."""

    def __init__(self, found):
        self.found = found
        self.url = "https://example.test/app"

    def query_selector_all(self, selector):
        return list(self.found.get(selector, []))

    def evaluate_handle(self, expression, arg=None):
        return _PwArray(list(self.found.get(arg, [])))


def test_playwright_steps_into_frames_and_shadow_roots():
    """A Playwright handle carries its own frame, so nothing is switched:
    the next step is asked of the frame an <iframe> owns, or of the shadow
    root a host holds - that root alone, as the page reads a chain."""
    save = _PwEl("save")
    card = _PwEl("card", shadow={"button.save": [save]})
    inner = _PwDoc({"user-card": [card]})
    owner = _PwEl("main", frame=inner)
    page = bis._Playwright(_PwDoc({"iframe#main": [owner]}))
    assert page.query("iframe#main >>> user-card >>> button.save") == [save]
    assert page.query("iframe#main >>> user-card >>> button.gone") == []
    assert page.query("iframe#gone >>> user-card") == []


def test_a_repair_behind_a_boundary_is_checked_by_its_selector():
    """The bind check compares what a proposal bound with the element the
    service chose. An element behind a frame cannot be handed to the page's
    JavaScript from outside the frame, so for a chain the page is given the
    proposal and resolves it itself, as the bot did."""
    email = _Element("email")
    chain = "iframe#main >>> #email"
    candidates = [{"tag": "input", "id": "email", "selectors": [chain]}]
    page = _HealPage({chain: [email]}, candidates,
                     same=lambda arg: {"known": True,
                                       "same": arg.get("selector") == chain})
    el, bound, _ = _heal_with(page, _verdict(chosen=0))
    assert el is email and bound == chain, (el, bound)
    asked = [arg for expr, arg in page.evaluated if expr == "SAME"]
    assert asked and asked[0]["selector"] == chain, asked
    assert "els" not in asked[0] and asked[0]["all"] is False, asked


def test_the_harvest_is_told_about_the_step_that_names_the_element():
    """A broken chain's last step is what the harvest's cut and its
    language preference are about."""
    page, _, _ = _login_page()
    _heal_with(page, _verdict(),
               selector="iframe#main >>> xpath=//input[@id='legacy']")
    opts = [arg for expr, arg in page.evaluated if expr == "HARVEST"]
    assert opts and opts[0]["tagHint"] == "input", opts
    assert opts[0]["preferSyntax"] == "xpath", opts


def test_a_failed_lookup_brings_the_driver_back_out():
    """A lookup that went into a frame and found nothing usable there - the
    element is in it, but not on screen - must not leave the script's
    driver in the frame."""
    driver, top, inner, email = _framed()
    email.shown = False
    saved = bis._S.enabled
    bis._S.enabled = False
    try:
        assert bis.find(driver, "iframe#main >>> #email", optional=True,
                        timeout=0) is None
    finally:
        bis._S.enabled = saved
    assert driver.where == [top], driver.where


# ------------------------------------------------------------ agent mode
#
# The ReFlex Agent plans a repair on the service (invariant 51); bis.py
# runs the page tools it asks for and binds its decision exactly as it
# binds the pipeline's. The service here is scripted: each reply below is
# what /v1/agent/start or /v1/agent/step answers, in order.


@contextlib.contextmanager
def _agent_service(script, verdict=None, review_hook=None, mode="always",
                   on_step=None):
    """bis talking to a fake service whose agent answers from `script` -
    and whose /v1/adjudicate answers `verdict` - in agent mode `mode`."""
    import bis
    replies = list(script)
    with _service(verdict or _verdict(), review_hook) as posts:
        inner = bis._call

        def _call(path, payload=None):
            if path in ("/v1/agent/start", "/v1/agent/step"):
                posts.append((path, payload))
                return dict(replies.pop(0)) if replies else None
            return inner(path, payload)
        saved = (bis._S.agent, bis._S.on_agent_step)
        bis._call = _call
        bis._S.agent, bis._S.on_agent_step = mode, on_step
        try:
            yield posts
        finally:
            bis._S.agent, bis._S.on_agent_step = saved


def _agent_heal_with(page, script, on_step=None, mode="always", verdict=None,
                     review_hook=None):
    """One _heal() of "#legacyEmail" against `page`, in agent mode."""
    import bis
    with _agent_service(script, verdict, review_hook, mode,
                        on_step) as posts:
        el, bound = bis._heal(page, "#legacyEmail", "interact", False, "")
    return el, bound, posts


def _scan_call(n=1):
    return {"session": "s1", "done": False, "calls": [
        {"id": "c%d" % n, "tool": "page_scanner", "args": {}}]}


def test_the_agent_is_off_by_default():
    assert bis._S.agent == "off"
    page, email, _ = _login_page()
    el, bound, posts = _heal_with(page, _verdict())
    assert el is email, el
    assert not [p for p, _ in posts if "/agent/" in p], posts


def test_the_agent_runs_page_tools_when_asked_and_its_decision_binds():
    page, email, _ = _login_page()
    seen = []
    script = [_scan_call(1),
              {"session": "s1", "done": False, "calls": [
                  {"id": "c2", "tool": "selector_options",
                   "args": {"index": 0}}],
               "trace": [{"n": 1, "tool": "page_scanner"}]},
              {"session": "s1", "calls": [], "done": True,
               "decision": _verdict(chosen=0), "trace": [
                   {"n": 1, "tool": "page_scanner"},
                   {"n": 2, "tool": "selector_options"},
                   {"n": 3, "tool": "commit_repair"}]}]
    el, bound, posts = _agent_heal_with(page, script, on_step=seen.append)
    assert el is email and bound == "#email", (el, bound)
    sent = [b for p, b in posts if p == "/v1/agent/step"]
    assert sent[0]["session"] == "s1", sent
    assert sent[0]["results"][0]["id"] == "c1", sent
    assert sent[0]["results"][0]["data"]["harvest"] == "h-1", sent
    assert len(sent[0]["results"][0]["data"]["candidates"]) == 2, sent
    assert sent[1]["results"][0]["data"]["options"] == [
        {"selector": "#email", "matches": 1}], sent
    assert [t["tool"] for t in seen] == ["page_scanner", "selector_options",
                                         "commit_repair"], seen
    start = [b for p, b in posts if p == "/v1/agent/start"][0]
    assert start["locator"]["selector"] == "#legacyEmail", start
    assert start["use_stored_repair"] is True and start["escalated"] is False
    # Bound as the pipeline binds: the repair becomes the step's baseline.
    reports = [b for p, b in posts if p == "/v1/baseline"]
    assert reports and reports[0]["replaced"] == "#legacyEmail", reports


def test_an_agent_that_falls_back_leaves_the_pipeline_to_heal():
    page, email, _ = _login_page()
    script = [{"session": "s1", "done": True, "fallback": True,
               "reason": "no planner"}]
    el, bound, posts = _agent_heal_with(page, script)
    assert el is email, el
    assert [p for p, _ in posts][:2] == ["/v1/agent/start",
                                         "/v1/adjudicate"], posts


def test_a_decision_outside_the_candidates_is_not_found():
    page, _, _ = _login_page()
    script = [_scan_call(1),
              {"session": "s1", "done": True, "decision": _verdict(chosen=7)}]
    el, bound, posts = _agent_heal_with(page, script)
    assert el is None and bound == "", (el, bound)
    assert "/v1/adjudicate" not in [p for p, _ in posts], posts


def test_a_fleet_repair_the_agent_chose_that_no_longer_binds_is_retried():
    page, _, _ = _login_page()
    page.matches["#email"] = []                 # the stored repair is stale
    script = [_scan_call(1),
              {"session": "s1", "done": True,
               "decision": _verdict(chosen=0, repair_id=5)}]
    _, _, posts = _agent_heal_with(page, script)
    retried = [b for p, b in posts if p == "/v1/adjudicate"]
    assert retried and retried[0]["failed_repair_id"] == 5, retried
    assert retried[0]["retry"] is True, retried
    assert [e for e, _ in page.evaluated].count("HARVEST") == 1, (
        "the retry harvested the page again instead of reusing the scan")


def test_escalate_mode_asks_the_agent_only_after_the_pipeline_declines():
    page, _, _ = _login_page()
    _, _, posts = _agent_heal_with(page, [{"session": "s1", "done": True,
                                           "fallback": True}],
                                   mode="escalate",
                                   verdict=_verdict(band="review"))
    order = [p for p, _ in posts if p in ("/v1/adjudicate",
                                          "/v1/agent/start")]
    assert order == ["/v1/adjudicate", "/v1/agent/start"], order
    start = [b for p, b in posts if p == "/v1/agent/start"][0]
    assert start["escalated"] is True and start["review_hook"] is False

    # A pipeline that heals the step never calls the agent.
    page, email, _ = _login_page()
    el, _, posts = _agent_heal_with(page, [], mode="escalate")
    assert el is email and "/v1/agent/start" not in [p for p, _ in posts]


def test_in_always_mode_the_agent_decides_whether_an_optional_element_is_there():
    """find() asks the page about an optional element before healing it.
    With the agent always on that question is the agent's - its Presence
    Check - so an absent banner is settled without a scan."""
    page, _, _ = _login_page()
    script = [{"session": "s1", "done": False, "calls": [
                  {"id": "c1", "tool": "presence_check", "args": {}}]},
              {"session": "s1", "done": True,
               "decision": _verdict(band="refuse", chosen=-1)}]
    with _agent_service(script) as posts:
        got = bis.find(page, "#promo-close", optional=True, timeout=0)
    assert got is None, got
    asked = [e for e, _ in page.evaluated]
    assert asked.count("PRESENCE") == 1 and "HARVEST" not in asked, asked
    assert [p for p, _ in posts][:2] == ["/v1/agent/start",
                                         "/v1/agent/step"], posts


def test_a_failing_trace_hook_never_fails_the_step():
    page, email, _ = _login_page()

    def boom(entry):
        raise RuntimeError("the overlay is gone")
    script = [_scan_call(1),
              {"session": "s1", "done": True, "decision": _verdict(chosen=0),
               "trace": [{"n": 1, "tool": "page_scanner"}]}]
    el, _, _ = _agent_heal_with(page, script, on_step=boom)
    assert el is email, el


def test_the_agent_mode_is_set_through_configure():
    before = (bis._S.agent, bis._S.on_agent_step)
    try:
        hook = lambda entry: None               # noqa: E731
        bis.configure(agent="always", on_agent_step=hook)
        assert (bis._S.agent, bis._S.on_agent_step) == ("always", hook)
        bis.configure(agent="sometimes")        # not a mode: kept as it was
        assert bis._S.agent == "always"
        bis.configure(find_timeout_s=bis._S.find_timeout_s)
        assert bis._S.on_agent_step is hook, "an unrelated call cleared it"
        bis.configure(on_agent_step=None)
        assert bis._S.on_agent_step is None
    finally:
        bis._S.agent, bis._S.on_agent_step = before


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failures += 1
            print("  FAIL  %s  %s" % (name, exc))
        except Exception as exc:
            failures += 1
            print("  ERROR %s  %s: %s" % (name, type(exc).__name__, exc))
    print("\n%s" % ("all tests passed" if not failures
                    else "%d test(s) failed" % failures))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
