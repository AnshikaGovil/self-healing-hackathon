"""Offline unit tests for the deterministic ranker.

These run without a browser, without the healing service and without
the sentence encoder, because the whole point of the heuristic layer is
that it holds up when none of those are available.

    python -m pytest tests -q       (or)      python tests/test_similarity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bis_core.config import ScoringWeights                # noqa: E402
from bis_core.descriptors import (ElementDescriptor,      # noqa: E402
                                  LocatorSpec, collapse_digits,
                                  tokenize)
from bis_core.descriptors import (intent_tokens,           # noqa: E402
                                  normalize_selector,
                                  selector_syntax,
                                  selector_tag,
                                  selector_uses_id)
from bis_core.similarity import (HealTarget,              # noqa: E402
                                 _path_similarity,
                                 geometry_signal,
                                 identity_signal,
                                 neighbor_text_signal,
                                 score_candidates,
                                 selector_hints,
                                 text_signal)


def el(tag="div", classes=(), text="", eid="", role="", aria="",
       visible=True, interactive=False, rect=None, **kw):
    return ElementDescriptor(
        tag=tag, classes=list(classes), text=text, element_id=eid, role=role,
        aria_label=aria, visible=visible, interactive=interactive,
        rect=rect or {"x": 10, "y": 10, "width": 120, "height": 40}, **kw)


def target(selector, description="", **kw):
    return HealTarget(spec=LocatorSpec(key="t", selector=selector,
                                       description=description, **kw))


def test_digit_collapse_makes_version_bumps_identical():
    assert collapse_digits("pd19-product-card__item") == \
           collapse_digits("pd21-product-card__item")
    assert collapse_digits("nv00-gnb-v3__utility") == \
           collapse_digits("nv00-gnb-v4__utility")


def test_tokenizer_handles_bem_and_camel_case():
    assert "product" in tokenize("pd21-product-card__item")
    assert "search" in tokenize("gnbSearchKeyword")


def test_selector_hints_are_mined_from_css():
    _ids, toks, tag, text = selector_hints("a.pd19-product-card__title")
    assert tag == "a"
    assert "product" in toks and "title" in toks
    assert text is None


def test_identity_signal_scores_a_version_bump_as_exact():
    t = target(".pd19-product-card__item")
    assert identity_signal(t, el(classes=["pd21-product-card__item"])) == 1.0
    assert identity_signal(t, el(classes=["cookie-bar__close"])) < 0.35


def test_identity_signal_reports_no_evidence_without_identifiers():
    assert identity_signal(target("text=Accept"), el(classes=["x"])) is None


def test_ranker_picks_the_renamed_component_over_its_siblings():
    """The regression this signal was added for.

    Every candidate here lives in the same redesigned component, so token
    overlap alone leaves them within a hair of each other; only the surviving
    BEM element name separates the right one.
    """
    t = target(".pd19-product-card__item",
               "every smartphone product card tile in the listing grid")
    cands = [
        el(classes=["pd21-product-card__options-wrap"], text="Colour : Mint"),
        el(classes=["pd21-product-card__name-wrap"], text="Galaxy F70 Pro 5G"),
        el(classes=["pd21-product-card__item", "js-pfv2-product-card"],
           text="New Galaxy F70 Pro 5G"),
        el(classes=["pd21-product-card__rating-wrap"], text="Ratings 4.9"),
    ]
    ranked = score_candidates(t, cands, ScoringWeights(), None)
    assert ranked[0].descriptor.classes[0] == "pd21-product-card__item"


def test_ranker_finds_the_renamed_nav_button():
    t = target("button.nv00-gnb-v3__utility-search",
               "the magnifier icon in the global nav that opens site search",
               tag="button", role="button")
    cands = [
        el(tag="button", classes=["nv00-gnb-v4__utility-search"], text="Search",
           interactive=True),
        el(tag="a", classes=["nv00-gnb-v4__logo"], aria="Samsung"),
        el(tag="button", classes=["cookie-bar__close"], text="close",
           interactive=True),
    ]
    ranked = score_candidates(t, cands, ScoringWeights(), None)
    assert ranked[0].descriptor.classes == ["nv00-gnb-v4__utility-search"]


def test_hidden_duplicates_lose_to_the_visible_one():
    """samsung.com renders the nav twice - mobile and desktop. Only one is
    on screen, and clicking the other would hang the process."""
    t = target("button.nv00-gnb-v3__utility-search", "site search button",
               tag="button", role="button")
    hidden = el(tag="button", classes=["nv00-gnb-v4__utility-search"],
                text="Search", visible=False, interactive=True)
    shown = el(tag="button", classes=["nv00-gnb-v4__utility-search"],
               text="Search", visible=True, interactive=True)
    ranked = score_candidates(t, [hidden, shown], ScoringWeights(), None)
    assert ranked[0].descriptor.visible is True


def test_missing_signals_are_dropped_not_scored_as_zero():
    """A cold heal has no baseline, so structure and geometry have nothing to
    say. They must be excluded from the mean rather than dragging it down."""
    t = target(".pd19-product-card__item", "a product card")
    cand = el(classes=["pd21-product-card__item"], text="Galaxy")
    ranked = score_candidates(t, [cand], ScoringWeights(), None)
    signals = ranked[0].signals
    assert signals["structure"] is None and signals["geometry"] is None
    assert signals["identity"] == 1.0
    assert ranked[0].score > 0.5


def test_expected_text_drives_the_text_signal():
    t = target(".cookie-bar__accept-button", "the accept button",
               tag="a", role="button", expected_text="^Accept$")
    accept = el(tag="a", classes=["cta", "cta--contained"], text="Accept",
                interactive=True)
    close = el(tag="button", classes=["cookie-bar__close"], text="close",
               interactive=True)
    ranked = score_candidates(t, [accept, close], ScoringWeights(), None)
    assert ranked[0].descriptor.text == "Accept"


# ---------------------------------------------------------------- XPath
# An RPA team standardised on XPath must get the same quality of repair as one
# using CSS. Before these, every XPath locator healed blind: none of the CSS
# patterns matched //div[@class='...'], so the identity signal - the
# highest-weighted one - reported "no evidence" on every single repair.


def test_bare_xpath_is_given_its_prefix():
    assert normalize_selector("//button[@id='go']") == "xpath=//button[@id='go']"
    assert normalize_selector("(//a)[2]").startswith("xpath=")
    assert normalize_selector(".card") == ".card"
    assert selector_syntax("//div") == "xpath"
    assert selector_syntax("div.card") == "css"


def test_xpath_attribute_predicates_are_mined():
    ids, toks, tag, _text = selector_hints("//div[@class='pd19-product-card__item']")
    assert tag == "div"
    assert ids == ["pd#-product-card__item"]
    assert "product" in toks and "card" in toks


def test_xpath_and_css_yield_the_same_identity_score():
    """The same intent written two ways must score identically."""
    css = target(".pd19-product-card__item")
    xp = target("//div[@class='pd19-product-card__item']")
    cand = el(classes=["pd21-product-card__item"])
    assert identity_signal(css, cand) == identity_signal(xp, cand) == 1.0


def test_xpath_contains_and_starts_with_are_mined():
    for sel in ("//a[contains(@class,'pd19-product-card__title')]",
                "//a[starts-with(@class,'pd19-product-card__title')]"):
        ids, _toks, tag, _text = selector_hints(sel)
        assert tag == "a"
        assert ids == ["pd#-product-card__title"]


def test_xpath_text_predicates_become_a_text_hint():
    for sel in ("//button[text()='Accept']",
                "//button[normalize-space()='Accept']",
                "//button[contains(text(),'Accept')]",
                "//button[.='Accept']"):
        _ids, _toks, _tag, text = selector_hints(sel)
        assert text == "Accept", sel


def test_identifiers_come_from_the_last_step_only():
    """//div[card]//a[title] selects the anchor. Scoring a candidate against
    the ancestor's class name would be measuring the wrong element."""
    sel = ("//div[@class='pd19-product-card__item']"
           "//a[@class='pd19-product-card__title']")
    ids, toks, tag, _text = selector_hints(sel)
    assert tag == "a"
    assert ids == ["pd#-product-card__title"]
    # Ancestor tokens are deliberately excluded: a candidate cannot carry its
    # ancestors' words, so including them only inflates the denominator of the
    # attribute signal and pushes good candidates below the apply threshold.
    assert "item" not in toks


def test_css_descendant_paths_target_the_last_compound():
    _ids, _toks, tag, _text = selector_hints("div.card > a.name")
    assert tag == "a"


def test_xpath_quotes_do_not_break_mining():
    _ids, _toks, tag, text = selector_hints('//button[@aria-label="Search"]')
    assert tag == "button"
    _ids2, toks2, _t, _x = selector_hints('//input[@name="search"]')
    assert "search" in toks2


def test_the_same_intent_scores_the_same_in_either_language():
    """A CSS repository and an XPath repository must rank identically.

    Regression: XPath forms used to tokenise the attribute *name* ("class")
    and to absorb ancestor tokens, both of which the CSS form never did. The
    cookie-consent locator ended up at 0.61 in XPath and 0.68 in CSS - either
    side of the auto-apply threshold, so the same repair applied in one
    language and was refused in the other.
    """
    css = target(".cookie-bar__accept-button", "the accept button",
                 tag="a", role="button", expected_text="^Accept$")
    xp = target("//div[contains(@class,'cookie-bar')]"
                "//a[contains(@class,'cookie-bar__accept-button')]",
                "the accept button", tag="a", role="button",
                expected_text="^Accept$")
    assert sorted(set(css.selector_tokens)) == sorted(set(xp.selector_tokens))
    assert css.target_identifiers() == xp.target_identifiers()

    cands = [
        el(tag="a", classes=["cta", "cta--contained", "cta--emphasis"],
           text="Accept", interactive=True),
        el(tag="button", classes=["cookie-bar__close"], text="close",
           interactive=True),
        el(classes=["cookie-bar__manage"], text="Accept"),
    ]
    a = score_candidates(css, cands, ScoringWeights(), None)
    b = score_candidates(xp, cands, ScoringWeights(), None)
    assert a[0].descriptor.text == b[0].descriptor.text == "Accept"
    assert abs(a[0].score - b[0].score) < 0.02


# ------------------------------------------------- description as evidence
# The description reaches the ranker twice - whole, for the semantic signal and
# the AI prompt, and tokenised, in the lexical bag. Only the second use needs
# filtering, and getting that wrong is expensive: the unfiltered descriptions
# were costing the top candidate ~0.13 of final score against a 0.62 threshold.


def test_english_function_words_stay_out_of_the_lexical_bag():
    toks = intent_tokens("the product title link inside every smartphone card")
    for junk in ("the", "inside", "every"):
        assert junk not in toks
    for real in ("product", "title", "link", "smartphone", "card"):
        assert real in toks


def test_filtering_recovers_score_a_prose_description_would_have_cost():
    """The defect, stated as an invariant.

    Note what this does *not* claim: filtering cannot rescue a genuinely
    verbose description, because thirteen content words really are thirteen
    entries in the denominator. That is the author's problem and the linter
    warns about it (W102). What is fixed here is that function words - which
    no DOM element could ever carry - no longer cost anything.
    """
    from bis_core.similarity import attribute_signal
    from bis_core.descriptors import tokenize

    cand = el(tag="a", classes=["pd21-product-card__name"],
              text="Galaxy F70 Pro", interactive=True)
    spec = LocatorSpec(key="k", selector="a.pd19-product-card__title",
                       description="the product title link inside every card",
                       tag="a", role="link")
    filtered = HealTarget(spec=spec)

    unfiltered = HealTarget(spec=spec)
    unfiltered.selector_tokens = (
        selector_hints(spec.selector)[1] + tokenize(spec.description))

    assert attribute_signal(filtered, cand) > attribute_signal(unfiltered, cand)


def test_the_cookie_banner_ranking_survives_the_description_change():
    """Kept because dropping description tokens outright inverts this pair.

    The component was renamed wholesale, so the *correct* element carries none
    of the target's identity tokens while the close button carries two of four
    ('cookie', 'bar'). The attribute signal is actively misleading here, which
    is why the fix filters function words rather than removing the description
    from the lexical bag altogether.
    """
    t = target(".cookie-bar__accept-button",
               "the Accept button that dismisses the cookie consent banner",
               tag="a", role="button", expected_text="^Accept$")
    accept = el(tag="a", classes=["cta", "cta--contained", "cta--emphasis"],
                text="Accept", interactive=True)
    close = el(tag="button", classes=["cookie-bar__close"], text="close",
               interactive=True)
    manage = el(classes=["cookie-bar__manage"], text="Accept")
    ranked = score_candidates(t, [accept, close, manage], ScoringWeights(), None)
    assert ranked[0].descriptor is accept


# ------------------------------------------------------- derived role / tag


def test_selector_tag_reads_the_last_step():
    assert selector_tag("div.card > a.name") == "a"
    assert selector_tag("//div[@class='c']//a[@id='n']") == "a"
    assert selector_tag(".card") is None


def test_role_is_derived_from_the_tag_when_omitted():
    cases = [("a.buy", "link"), ("input#q", "textbox"),
             ("//select[@id='x']", "dropdown"), ("img.hero", "image"),
             ("//h2[@class='t']", "text")]
    for sel, expected in cases:
        assert LocatorSpec(key="k", selector=sel).role == expected, sel
    assert LocatorSpec(key="k", selector=".x", tag="h2").role == "text"


def test_an_explicit_role_is_never_overridden():
    assert LocatorSpec(key="k", selector="a.x", role="text").role == "text"


def test_an_unmappable_tag_leaves_the_role_unset():
    """A div could be anything. Better one signal fewer than a guess."""
    assert LocatorSpec(key="k", selector=".card").role is None
    assert LocatorSpec(key="k", selector="div.card").role is None


def test_a_derived_role_scores_like_an_explicit_one():
    cand = el(tag="a", classes=["pd21-product-card__name"], text="Galaxy",
              interactive=True)
    explicit = HealTarget(spec=LocatorSpec(
        key="k", selector="a.pd19-product-card__title", role="link"))
    derived = HealTarget(spec=LocatorSpec(
        key="k", selector="a.pd19-product-card__title"))
    w = ScoringWeights()
    assert (score_candidates(explicit, [cand], w, None)[0].score
            == score_candidates(derived, [cand], w, None)[0].score)


def test_the_dropdown_role_recognises_a_select():
    """Without _ROLE_TAGS['dropdown'] the derivation would be a regression."""
    t = target("//select[@id='year']", "the year dropdown")
    assert t.spec.role == "dropdown"
    cand = el(tag="select", classes=["year"], interactive=True)
    from bis_core.similarity import tag_signal
    assert tag_signal(t, cand) > 0.8


def test_the_core_imports_nothing_outside_the_standard_library():
    """What keeps this installable on an RPA machine that has nothing.

    Every bot in the fleet installs bis-core. The moment something here
    imports torch, or numpy, or a GGUF runtime, that cost lands on every
    automation VM to no purpose - the models run on the healing service.
    The accelerator half plugs in through ``bis_core.runtime`` instead, so
    there is never a reason for a third-party import to appear here.
    """
    import ast
    import os

    core = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bis_core")
    allowed = set(sys.stdlib_module_names) | {"bis_core"}
    offenders = []

    for name in sorted(os.listdir(core)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(core, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import - inside the package.
                mods = [] if node.level else [node.module or ""]
            else:
                continue
            for mod in mods:
                root = mod.split(".")[0]
                if root and root not in allowed:
                    offenders.append("%s: %s" % (name, mod))

    assert not offenders, "bis-core reached outside stdlib: %s" % offenders


# ------------------------------------------- what a baseline unlocks
#
# Six of the seven signals compare a candidate against the element the step
# used while its locator still worked. These pin the three that were either
# newly built or found to be measuring the wrong thing.


def test_neighbour_text_scores_the_copy_around_the_element():
    """The one piece of evidence a wholesale rename cannot touch.

    A design-system bump rewrites every class on the page and leaves the words
    beside the button where they were. The harvester has always collected that
    text; until this signal existed nothing scored it.
    """
    base = el(tag="input", eid="firstName",
              neighbor_text="First Name Enter your legal first name")
    t = HealTarget(spec=LocatorSpec(key="t", selector="#hr-v2-firstName"),
                   baseline=base)

    same = el(tag="input", eid="firstName",
              neighbor_text="First Name Enter your legal first name")
    assert neighbor_text_signal(t, same) == 1.0

    unrelated = el(tag="input", neighbor_text="Payment card expiry month")
    assert neighbor_text_signal(t, unrelated) < 0.3

    # Long words carry more evidence than short ones: sharing "legal" and
    # "first" says far more than sharing "your".
    long_words = el(tag="input", neighbor_text="legal first name")
    short_words = el(tag="input", neighbor_text="your")
    assert neighbor_text_signal(t, long_words) > \
        neighbor_text_signal(t, short_words)


def test_neighbour_text_says_nothing_without_a_baseline():
    """Invariant 1: no evidence reports None, never zero.

    A cold heal has no recorded surroundings to compare against. Scoring that
    as zero would punish a repair for being the first one.
    """
    cold = target("#hr-v2-firstName")
    assert neighbor_text_signal(cold, el(neighbor_text="anything")) is None

    blank = HealTarget(spec=LocatorSpec(key="t", selector="#x"),
                       baseline=el(neighbor_text=""))
    assert neighbor_text_signal(blank, el(neighbor_text="something")) is None


def test_geometry_survives_a_scroll_and_a_different_screen():
    """rect is viewport-relative, so position alone is not a fact about a page.

    The baseline is captured at the top of the page and the candidate is
    harvested after scrolling to it. Same element, same place in the document,
    900 pixels apart on screen. Before the scroll offset travelled with the
    fingerprint, this scored as though the element had moved half a page.
    """
    top = el(rect={"x": 100, "y": 200, "width": 120, "height": 40},
             scroll_x=0, scroll_y=0, viewport_w=1600, viewport_h=900)
    scrolled = el(rect={"x": 100, "y": -700, "width": 120, "height": 40},
                  scroll_x=0, scroll_y=900, viewport_w=1600, viewport_h=900)
    t = HealTarget(spec=LocatorSpec(key="t", selector="#x"), baseline=top)
    assert geometry_signal(t, scrolled) == 1.0

    # And the same element on a smaller screen is still the same element: the
    # distance is normalised by the viewport that was actually in use, not by
    # a constant that only ever described one developer's window.
    small = el(rect={"x": 100, "y": 200, "width": 120, "height": 40},
               scroll_x=0, scroll_y=0, viewport_w=1280, viewport_h=720)
    t_small = HealTarget(spec=LocatorSpec(key="t", selector="#x"),
                         baseline=small)
    assert geometry_signal(t_small, small) == 1.0

    # A fingerprint recorded before those fields existed reports zero for all
    # of them, and must degrade rather than divide by nothing.
    old = el(rect={"x": 100, "y": 200, "width": 120, "height": 40})
    t_old = HealTarget(spec=LocatorSpec(key="t", selector="#x"), baseline=old)
    assert geometry_signal(t_old, old) == 1.0


def test_paths_are_compared_by_step_not_by_character():
    """Edit distance over an absolute XPath measures mostly punctuation.

    It was also 47% of the entire ranking cost. Steps are the unit that means
    something, and matching from the leaf back up is how a redesign actually
    moves an element: the wrappers near the root get rewritten, the immediate
    surroundings usually do not.
    """
    a = "/html[1]/body[1]/div[2]/form[1]/div[3]/input[1]"
    assert _path_similarity(a, a) == 1.0

    # One wrapper renumbered near the root: most of the path still agrees.
    reshuffled = "/html[1]/body[1]/div[5]/form[1]/div[3]/input[1]"
    assert _path_similarity(a, reshuffled) > 0.6

    # A different element entirely, sharing only the document root.
    elsewhere = "/html[1]/body[1]/header[1]/nav[1]/a[4]"
    assert _path_similarity(a, elsewhere) < 0.3
    assert _path_similarity("", a) == 0.0


def test_the_identity_short_circuit_changes_speed_and_not_answers():
    """difflib's cheap bounds are exact, so skipping on them is safe.

    real_quick_ratio() >= quick_ratio() >= ratio(), so a pair whose upper
    bound is already below the best seen can never win. Getting the *pair
    order* wrong while optimising this silently changed every score, because
    ratio() is not symmetric - hence this test.
    """
    from difflib import SequenceMatcher

    def naive(t, cand):
        wanted = t.target_identifiers()
        if not wanted:
            return None
        have = [collapse_digits(cand.element_id.lower())] if cand.element_id else []
        have += [collapse_digits(c.lower()) for c in cand.classes]
        have = [h for h in have if len(h) > 2]
        if not have:
            return 0.0
        best = 0.0
        for w in wanted:
            for h in have:
                if w == h:
                    return 1.0
                best = max(best, SequenceMatcher(None, w, h).ratio())
        return best ** 3

    t = target(".pd19-product-card__item")
    for cand in (el(classes=["pd21-product-card__item"]),
                 el(classes=["pd21-product-card__title", "js-card"]),
                 el(eid="product-card-item"),
                 el(classes=["totally-unrelated-banner"]),
                 el(classes=[])):
        assert identity_signal(t, cand) == naive(t, cand), cand.classes


def test_a_repair_keeps_the_shape_the_developer_wrote():
    """An author who avoided ids has told us something about the application.

    Usually that its ids are framework-generated and do not survive a deploy.
    Handing that developer an id-based repair overrides knowledge they have
    and we do not, and it is also the code they will have to review.
    """
    assert selector_uses_id("#firstName")
    assert selector_uses_id("input#firstName")
    assert selector_uses_id("div#main .row")
    assert selector_uses_id("xpath=//input[@id='firstName']")

    # A '#' inside an attribute value is a fragment, not an id.
    assert not selector_uses_id('a[href="#top"]')
    assert not selector_uses_id("div.card > a.title")
    assert not selector_uses_id(".pd19-product-card__item")
    assert not selector_uses_id("xpath=//div[@class='card']")
    assert not selector_uses_id("")


# ---------------------------------------------------------------- js_harvest
#
# The in-page JavaScript moved into bis-core when the core was split out, but
# its tests stayed in the toolkit suite and went with it. These two are
# recovered from there: invariant 21 names the first one by name, and a named
# invariant with no test behind it is worse than an unwritten one.


def test_candidates_and_baselines_are_described_by_the_same_code():
    """One describe(), spliced into both entry points.

    They were two implementations of the same fingerprint and they drifted.
    Measured over 738 interactive elements on four live sites, the two
    disagreed about `visible` 40% of the time and about `neighborText` 46%,
    and the baseline copy derived no accessible name from a `<label>` - which
    on a login form is every field.

    The worst of it was where that landed. Both the service store and the
    agent refuse a baseline that is not `visible`, and the baseline copy
    defined `visible` as "has a box", which is what the other half of the same
    guard already checked. The visibility test invariant 12 describes was not
    running on that path at all.

    A candidate and a baseline are compared field against field, so they have
    to come from one function. A browser is needed to prove they agree; this
    proves they cannot disagree.
    """
    from bis_core import js_harvest

    shared = js_harvest._DESCRIBE_JS
    assert "const describe = (el) =>" in shared
    assert shared in js_harvest.HARVEST_JS
    assert shared in js_harvest.DESCRIBE_ONE_JS

    # ... and neither one carries a second copy of anything in it.
    for name in ("HARVEST_JS", "DESCRIBE_ONE_JS"):
        body = getattr(js_harvest, name)
        for fn in ("const describe = (el) =>", "const accName = (el) =>",
                   "const neighborText = (el) =>"):
            assert body.count(fn) == 1, "%s defines %s twice" % (name, fn)


def test_a_fingerprint_keeps_no_value_a_step_could_have_typed():
    """describe() copied the first 60 characters of every element's value
    into every harvest and every baseline - a password box's included. A
    step that broke after the bot had signed in sent what it had typed to
    the service, and baselines.json could keep it, while the models were
    never shown a text box's value at all (invariants 41 and 47).

    A value is kept only where it is the element's own words: a button
    input's caption, a button's or an option's markup. One definition,
    read by the page and by the stores alike."""
    from bis_core import descriptors, js_harvest

    describe = js_harvest._DESCRIBE_JS
    assert "value: ownValue(el)," in describe, "the value is read another way"
    assert describe.count("el.value") == 1, "el.value is read twice"
    kinds = ", ".join("'%s'" % t for t in descriptors.BUTTON_INPUT_TYPES)
    assert "const OWN_VALUE_INPUTS = [%s];" % kinds in describe, kinds

    own = descriptors.own_value
    for tag, kind in (("input", ""), ("input", "text"), ("input", "password"),
                      ("input", "email"), ("input", "hidden"),
                      ("textarea", ""), ("select", ""), ("div", "")):
        assert own(tag, kind, "hunter2") == "", (tag, kind)
    for tag, kind in (("input", "submit"), ("INPUT", "Button"),
                      ("input", "reset"), ("button", ""), ("option", "")):
        assert own(tag, kind, "Sign in") == "Sign in", (tag, kind)


def test_the_harvest_keeps_the_broken_selector_s_family_in_the_cut():
    """The harvest keeps 900 elements, and 35% of 1,358 real pages had more.

    Whatever falls past the cut is never ranked, and the order of the cut
    knew nothing of the step. An element still carrying a fragment of the
    broken selector's name is the family the step names, so it is kept
    ahead of the rest - mined by the same function the presence gate uses,
    spliced into both, so the two can never disagree about what a
    fragment is.
    """
    from bis_core import js_harvest

    shared = js_harvest._FRAGMENTS_JS
    assert "const selectorFragments = (sel) =>" in shared
    for name in ("HARVEST_JS", "PRESENCE_JS"):
        body = getattr(js_harvest, name)
        assert shared in body, "%s does not use the shared fragments" % name
        assert body.count("const selectorFragments") == 1, name
    harvest = js_harvest.HARVEST_JS
    # The step that names the element: a host on the way is not its family.
    assert "selectorFragments(lastStep(String(opts.selector)))" in harvest
    assert "if (carries(el)) pri +=" in harvest, "the cut ignores them"


def test_a_bound_selector_can_be_proved_to_be_the_element_chosen():
    """The service answers with an index; the bot binds a selector the page
    proposed for that element - and checked only that it resolved and was
    visible. A proposal mined from shared classes, or a page that moved
    after the harvest, resolves to some other element, and the step then
    clicks it, confidently and silently.

    So the harvest keeps, in the page, which element each candidate is,
    and SAME_JS answers whether what a selector bound is that element.
    When it cannot tell - the page navigated, a re-render replaced the
    node - it says so rather than guess, and the bind stands as before.
    """
    from bis_core import js_harvest

    harvest = js_harvest.HARVEST_JS
    assert "kept.push(el);" in harvest
    assert "window.__bisHarvest = {id: harvest, els: kept};" in harvest
    assert "harvest: harvest," in harvest
    same = js_harvest.SAME_JS
    assert same.strip().startswith("(opts) =>")
    assert "window.__bisHarvest" in same and "el === want" in same
    assert "isConnected" in same, "a replaced node must read as unknown"
    assert "known: false" in same


def test_a_fingerprint_carries_the_scroll_and_the_viewport():
    """rect is viewport-relative, so it is not a fact about a page on its own.

    Without the scroll offset a baseline taken at the top and a candidate
    harvested at the footer look like the element moved; without the viewport
    the distance between them is normalised by a constant that only ever
    described one developer's window.
    """
    from bis_core import js_harvest

    for field in ("scrollX:", "scrollY:", "viewportW:", "viewportH:"):
        assert field in js_harvest._DESCRIBE_JS


def test_a_backslash_in_the_javascript_survives_into_the_browser():
    """A backslash in a replace() replacement has to be doubled.

    JavaScript drops the backslash in an unrecognised escape, so a
    replacement written with one backslash before $1 is read as a plain $1
    and a replace() meant to escape a character replaces it with itself. The
    CSS.escape fallback in _DESCRIBE_JS was written that way and was an
    identity function for as long as it existed - invisible, because the
    only engines that take that branch are the old ones nobody tests on.

    Checked as text because proving it any other way needs a browser, and
    because text is the exact level the mistake lives at: these are raw
    Python strings, so what is written here is what the engine receives.

    The needle needs a lookbehind rather than a substring test. A single
    backslash before $1 is trivially a substring of a doubled one, so the
    obvious `"backslash + $1" not in body` can never fail - which is a
    neat illustration of why this bug survived in the first place.
    """
    import re

    from bis_core import js_harvest

    slash = chr(92)
    # A backslash that is neither preceded nor followed by another, sitting
    # in front of a replacement group. Built from chr(92) so that no layer
    # of quoting between here and the file can change what is being asked.
    lone = re.compile("(?<!%s)%s[$][1&]" % (slash * 2, slash * 2))
    # The needle has to be able to fail, and a pattern this fiddly is worth
    # proving on a known-bad and a known-good sample before it is trusted
    # to say anything about the real thing.
    assert lone.search("replace(/x/g, '%s$1')" % slash) is not None
    assert lone.search("replace(/x/g, '%s$1')" % (slash * 2)) is None

    for name in ("_DESCRIBE_JS", "_FRAGMENTS_JS", "_RESOLVE_JS", "HARVEST_JS",
                 "PROBE_JS", "DESCRIBE_ONE_JS", "PRESENCE_JS", "SAME_JS"):
        body = getattr(js_harvest, name)
        hit = lone.search(body)
        assert hit is None, (
            "%s has %r, a single backslash before a replacement group. "
            "JavaScript reads that as a plain %r and the escaping does "
            "nothing - double it." % (name, hit.group(0) if hit else "",
                                      (hit.group(0) if hit else "")[1:]))


# ----------------------------------------------- shadow roots and frames
# A selector may step past the page's own document: 'app-shell >>>
# button.save' is the button inside <app-shell>'s open shadow root, and
# 'iframe#main >>> #email' the box inside that frame's document.


def test_a_chain_is_read_by_the_step_that_names_the_element():
    """Everything the ranker and the stores read from a selector - its
    language, its tag, its names, whether it uses an id - is about the
    element it selects. In a chain that is the last step; the hosts and
    frames before it are how to reach the element, as an ancestor step is
    (invariant 7), and their names are not the element's.
    """
    from bis_core.descriptors import chain_steps

    assert chain_steps("app-shell >>> button.save") == ["app-shell",
                                                        "button.save"]
    assert chain_steps("app-shell>>>button.save") == ["app-shell",
                                                      "button.save"]
    # Split only outside quotes and brackets: a value may hold the marker.
    assert chain_steps("a[title='x >>> y'] >>> b") == ["a[title='x >>> y']",
                                                       "b"]
    assert chain_steps("#plain") == ["#plain"]
    assert chain_steps("") == []

    assert selector_tag("app-shell >>> button.save") == "button"
    assert selector_tag("iframe#main >>> xpath=//input[@id='u']") == "input"
    assert selector_syntax("iframe#main >>> xpath=//input[@id='u']") == "xpath"
    assert selector_syntax("xpath=//iframe[@id='m'] >>> #email") == "css"
    assert selector_uses_id("app-shell#shell >>> button.save") is False
    assert selector_uses_id("app-shell >>> #save") is True
    ids, _toks, tag, _text = selector_hints("app-shell#shell >>> button.go")
    assert "shell" not in " ".join(ids) and tag == "button", (ids, tag)
    # Each step keeps its own language, and gets its prefix like any other.
    assert normalize_selector("//iframe[@id='m'] >>> //input") == \
        "xpath=//iframe[@id='m'] >>> xpath=//input"
    assert normalize_selector("app-shell>>>button") == "app-shell >>> button"
    assert normalize_selector("#plain") == "#plain"


def test_one_reading_of_a_chain_serves_every_entry_point():
    """The harvest proposes chains, the probe and the fingerprinter resolve
    them, the bind check compares what they bound and the presence gate
    looks behind the same boundaries. Readings of one notation drift, as
    the two descriptions of an element once did (invariant 16), so there is
    one, spliced into each.
    """
    from bis_core import js_harvest

    shared = js_harvest._RESOLVE_JS
    for fn in ("const splitChain = (sel) =>", "const inside = (el) =>",
               "const resolveAll = (selector) =>",
               "const allScopes = (base) =>"):
        assert fn in shared, fn
    for name in ("HARVEST_JS", "PROBE_JS", "DESCRIBE_ONE_JS", "SAME_JS",
                 "PRESENCE_JS"):
        body = getattr(js_harvest, name)
        assert shared in body, "%s does not use the shared resolver" % name
        assert body.count("const resolveAll") == 1, name
    for name in ("PROBE_JS", "DESCRIBE_ONE_JS"):
        body = getattr(js_harvest, name)
        assert "document.querySelectorAll(selector)" not in body, name
        assert "resolveAll(selector)" in body, name
    # The gate looks for the element's family where the element would be.
    assert "selectorFragments(lastStep(sel))" in js_harvest.PRESENCE_JS
    assert "allScopes(document)" in js_harvest.PRESENCE_JS


def test_the_harvest_reaches_behind_shadow_roots_and_into_frames():
    """Light DOM was all the harvest saw, so a control inside a web
    component or a same-origin frame was never a candidate and could not be
    repaired at all. It now walks every open shadow root and every frame
    whose document the page can read, and an element found there is
    proposed through the hosts and frames on its way. XPath cannot see into
    a shadow root, so an element in one is proposed in CSS only.
    """
    from bis_core import js_harvest

    harvest = js_harvest.HARVEST_JS
    assert "for (const [scope, via] of allScopes(root))" in harvest
    assert "lead.join(' >>> ') + ' >>> ' + s" in harvest
    assert "!(shadow && s.startsWith('xpath='))" in harvest
    # A frame is named by its id before its bare tag, and cssPath - which
    # is the id itself when there is one - is only a last resort when it
    # really is a position. Found live: '#portal' lost to 'iframe'.
    assert "path.indexOf(':nth-of-type(') >= 0 ? path : ''" in harvest
    assert "concat(durable, frame ? [tag] : []" in harvest
    # Everything the page itself holds is described as it always was: the
    # walk only adds what lies behind a boundary, after the page's own.
    describe = js_harvest._DESCRIBE_JS
    assert "if (el.parentElement) return el.parentElement;" in describe
    same = js_harvest.SAME_JS
    assert "resolveAll(String(opts.selector))" in same


# ------------------------------------------- qualified renames, thin names
# Enterprise apps rename by *qualification* far more than by replacement:
# #id becomes #user_id, not #zxcv. Edit distance reads that as a stranger.


def test_a_qualified_rename_is_not_scored_as_a_stranger():
    """#id against #user_id: the authored name survived as a whole token.

    difflib puts this at 0.44, and 0.09 after the cube, because it is
    punishing the length difference. Reported from a Knox login page where
    exactly this rename left the step at 0.62-0.65.
    """
    t = target("#id")
    cand = el(tag="input", eid="user_id", aria="User ID")
    sig = identity_signal(t, cand)
    assert sig is not None, "a two-character id must still produce a signal"
    assert sig >= 0.4, "a qualified rename scored %r - too close to a stranger" % sig


def test_a_two_character_identifier_still_produces_a_signal():
    """#id, #pw, #un are real names. The old floor of three dropped them,
    which silently disabled the highest-weighted signal on the locators
    that need it most."""
    assert target("#id").target_identifiers() == ["id"]
    assert target("#pw").target_identifiers() == ["pw"]


def test_a_shared_component_prefix_is_not_treated_as_a_rename():
    """Regression: partial token overlap scored `cookie-bar__close` at 0.400
    against `cookie-bar__accept-button` because they share {cookie, bar},
    and pushed the correctly-renamed element to last. Containment only."""
    t = target(".cookie-bar__accept-button")
    sibling = el(classes=["cookie-bar__close"], text="close")
    # 0.296 here is plain edit distance and predates the token rule. What
    # must not happen is the sibling collecting the containment boost a
    # genuine qualified rename gets - that is what inverted the ranking.
    assert identity_signal(t, sibling) < 0.4
    renamed = target("#id")
    assert identity_signal(renamed, el(tag="input", eid="user_id")) >= 0.4


# ------------------------------------------- a description that says nothing


def test_a_placeholder_description_is_treated_as_absent():
    """"the field" names no element. Measured over six live sites, carrying
    it into the bag costs the correct candidate 0.06-0.17 of final score,
    because attribute_signal divides by the number of target tokens."""
    assert target("#user_id", "the field").selector_tokens ==         target("#user_id", "").selector_tokens
    assert target("#user_id", "the element").selector_tokens ==         target("#user_id", "").selector_tokens


def test_a_real_description_still_reaches_the_lexical_bag():
    """Invariant 8: dropping every description inverts consent.accept. Only
    the ones carrying no identity at all are dropped."""
    toks = target("#x", "the accept button on the cookie banner").selector_tokens
    assert "accept" in toks and "cookie" in toks


# --------------------------------------- digits that collapse to nothing


def test_a_numeric_id_is_not_a_component_name():
    """Digit collapse turns "12345" into "#", which then matches every other
    numbered element on the page. Measured over 86 live sites: Hacker News'
    90 ids collapse to 5 distinct values, thirty of them to bare "#";
    arXiv's 205 collapse to 10; crates.io's 164 to 5."""
    from bis_core.descriptors import names_a_component
    assert names_a_component("pd#-product-card") is True
    assert names_a_component("c#-title") is True
    assert names_a_component("#") is False
    assert names_a_component("#.#") is False
    assert target("#12345").target_identifiers() == []
    assert target("#comment-12345").target_identifiers() == ["comment-#"]


def test_a_versioned_component_name_still_collapses():
    """The guard must not cost the case digit collapsing exists for."""
    t = target(".pd19-product-card__item")
    cand = el(classes=["pd21-product-card__item"])
    assert identity_signal(t, cand) == 1.0


# ----------------------------------------- a cold heal never auto-applies


def test_an_unproven_locator_never_auto_applies():
    """Measured over 3,760 breakages on 82 live sites: 47% of cold applies
    bound the wrong element, and confidence does not separate right from
    wrong (0.586 correct vs 0.609 incorrect), so no threshold helps."""
    from bis_core.config import HealingConfig
    cfg = HealingConfig()
    for conf in (0.63, 0.80, 0.95, 1.0):
        assert cfg.decide(conf, proven=True) == cfg.APPLY
        assert cfg.decide(conf, proven=False) == cfg.REVIEW, (
            "%.2f auto-applied on a locator never seen working" % conf)
    assert cfg.decide(0.10, proven=False) == cfg.REFUSE
    assert cfg.decide_optional(0.95, proven=False) == cfg.REFUSE


# ------------------------------------------- naming attributes are labels


def test_a_naming_attribute_becomes_the_expected_text():
    """[aria-label="Place order"] says what the element reads as.

    It is not a component name, so scoring it with edit distance against
    class names is meaningless - it belongs to text_signal, which compares
    it against the candidate's accessible name. Measured over 102 live
    sites, routing it there took the cold hit rate for aria-label locators
    from 9% to 66% (CSS) and 22% to 69% (XPath), with every other selector
    kind unchanged.
    """
    for sel in ('[aria-label="Place order"]',
                "//button[@aria-label='Place order']"):
        t = target(sel)
        assert t.reference_text() == "Place order", sel
        cand = el(tag="button", aria="Place order", classes=["cta"])
        assert text_signal(t, cand) == 1.0, sel


def test_the_name_of_a_naming_attribute_is_not_evidence():
    """Every [aria-label=...] locator carries "aria" and "label". A constant
    in the target bag only inflates the denominator of matched/total."""
    toks = target('[aria-label="Place order"]').selector_tokens
    assert "aria" not in toks and "label" not in toks
    assert "place" in toks and "order" in toks


def test_a_descriptive_attribute_name_still_counts():
    """An attribute that is neither identity nor naming keeps its name.

    `type` varies between elements and so is evidence, unlike "aria"/"label"
    which every such locator carries. A data-* attribute is identity, so its
    name has never been tokenised - only its value.
    """
    toks = target('[type="submit"]').selector_tokens
    assert "type" in toks and "submit" in toks
    data = target('[data-omni-type="checkout"]')
    assert data.selector_tokens == ["checkout"]
    assert data.target_identifiers() == ["checkout"]


def test_an_inserted_wrapper_does_not_collapse_the_path_score():
    """A section wrapped in a new container is a normal redesign.

    The leaf-anchored comparison stopped at the first mismatch, so one
    inserted step near the leaf took /ul/li/a against /ul/li/span/a down to
    one step of six. Longest common subsequence keeps the five that are
    genuinely shared. Measured over 102 live sites, this recovers a
    mid-path wrapper from 0.835 back to 0.857 against 0.861 undisturbed,
    and is identical to the old behaviour when nothing is inserted.
    """
    a = "/html[1]/body[1]/div[2]/form[1]/div[3]/input[1]"
    wrapped = "/html[1]/body[1]/div[2]/form[1]/div[3]/span[1]/input[1]"
    assert _path_similarity(a, wrapped) > 0.75


def test_the_document_root_is_not_evidence():
    """Every element sits under /html/body, so crediting those steps lifts
    every candidate equally. Regression: the first subsequence version
    scored a wholly unrelated element at 0.333 because it matched them."""
    a = "/html[1]/body[1]/div[2]/form[1]/div[3]/input[1]"
    elsewhere = "/html[1]/body[1]/header[1]/nav[1]/a[4]"
    assert _path_similarity(a, elsewhere) == 0.0


# ------------------------------------------------- ambiguity is evidence


def test_a_tie_is_not_scored_as_a_decision():
    """A class name is not unique, so twenty elements can score alike.

    The winner used to report full confidence while the runner-up scored
    the same, which reads a coin flip as a decision. Measured over 4,664
    breakages on 102 live sites, discounting by margin took wrong
    auto-applies from 83 to 55 for 36 fewer correct ones.
    """
    from bis_core.config import HealingConfig
    cfg = HealingConfig()
    assert cfg.ambiguity_discount > 0.0

    # Two candidates that are genuinely the same component.
    t = target(".pd19-row__cell")
    twins = [el(classes=["pd21-row__cell"], text="A"),
             el(classes=["pd21-row__cell"], text="B")]
    tied = score_candidates(t, twins, ScoringWeights(), None)
    assert abs(tied[0].score - tied[1].score) < 0.05, (
        "these are meant to tie; the fixture no longer exercises the case")

    # One clear winner: an exact identity match against a stranger.
    clear = score_candidates(target("#checkout-submit"),
                             [el(tag="button", eid="checkout-submit"),
                              el(tag="a", classes=["nav__help"], text="Help")],
                             ScoringWeights(), None)
    assert clear[0].score - clear[1].score > 0.3


# ------------------------------------------------------------ calibration


def _calibration_spec(**over):
    spec = {
        "version": 1,
        "model_sha256": "fcc3a257" + "0" * 56,
        "features": ["bias", "logit_p", "logit_runner", "logit_none",
                     "first", "undescribed", "placeholder", "collect"],
        "weights": [0.5, 0.8, -0.3, -0.2, 0.4, -0.6, -0.9, -0.1],
        "isotonic": {"x": [0.0, 0.5, 0.9, 1.0], "y": [0.0, 0.4, 0.95, 0.99]},
        "bars": {"0.95": {"described": 0.96, "undescribed": 0.98}},
    }
    spec.update(over)
    return spec


def test_a_calibrator_maps_the_odds_to_a_probability_of_being_right():
    """Logistic over the token read and the step's facts, then a monotone
    isotonic map - computed here longhand, so the evaluator is checked
    against the arithmetic rather than against itself."""
    import math
    from bis_core.calibration import Calibrator

    cal = Calibrator(_calibration_spec())

    def logit(x):
        return math.log(x / (1.0 - x))

    z = (0.5 + 0.8 * logit(0.97) - 0.3 * logit(0.02) - 0.2 * logit(0.01)
         + 0.4 * 1.0)
    s = 1.0 / (1.0 + math.exp(-z))
    want = 0.95 + (s - 0.9) / (1.0 - 0.9) * (0.99 - 0.95)
    got = cal.probability(p=0.97, runner=0.02, none=0.01, rank=0,
                          group="described", collect=False)
    assert abs(got - want) < 1e-12, (got, want)

    # Monotone in the AI's own probability, whatever else holds.
    seen = [cal.probability(p=p, runner=0.01, none=0.01, rank=2,
                            group="undescribed", collect=False)
            for p in (0.3, 0.5, 0.7, 0.9, 0.99, 0.999)]
    assert seen == sorted(seen), seen
    assert all(0.0 <= v <= 1.0 for v in seen)
    # A probability of exactly 0 or 1 cannot break a logit.
    assert 0.0 <= cal.probability(p=1.0, runner=0.0, none=0.0, rank=0,
                                  group="described", collect=True) <= 1.0

    assert cal.bar(0.95, "described") == 0.96
    assert cal.bar(0.95, "placeholder") is None, "no evidence, no bar"
    assert cal.bar(0.90, "described") is None


def test_a_calibrator_that_will_not_load_is_no_calibrator():
    """Fail open: a missing, corrupt or unfamiliar file means the raw odds,
    never an exception on a heal."""
    import json
    import os
    import tempfile
    from bis_core.calibration import Calibrator

    folder = tempfile.mkdtemp(prefix="bis-cal-")
    assert Calibrator.load(os.path.join(folder, "missing.json")) is None
    bad = os.path.join(folder, "bad.json")
    open(bad, "w").write("{not json")
    assert Calibrator.load(bad) is None
    for broken in (_calibration_spec(version=99),
                   _calibration_spec(features=["bias", "telepathy"],
                                     weights=[1.0, 2.0]),
                   _calibration_spec(weights=[1.0]),
                   _calibration_spec(isotonic={"x": [0.0, 0.5, 0.4],
                                               "y": [0.0, 0.5, 0.6]}),
                   _calibration_spec(isotonic={"x": [0.0, 1.0],
                                               "y": [0.9, 0.2]})):
        path = os.path.join(folder, "c.json")
        json.dump(broken, open(path, "w"))
        assert Calibrator.load(path) is None, broken
    good = os.path.join(folder, "good.json")
    json.dump(_calibration_spec(), open(good, "w"))
    assert Calibrator.load(good) is not None


def test_the_shipped_calibration_belongs_to_the_shipped_weights():
    """The file beside bis_core is the one the service reads by default.

    It was fitted on 1,200 held-out steps replayed through the token read
    on the stage-3 GGUF, and no map beat the raw odds where a high bar
    sits - so it maps them to themselves and carries the bars. Pinned
    here: the weights it names, that it moves no probability, and that
    every bar it holds is above the floor no bar may go under.
    """
    from bis_core.calibration import Calibrator, DEFAULT_PATH, GROUPS

    cal = Calibrator.load(DEFAULT_PATH)
    assert cal is not None, "the shipped calibration does not load"
    assert cal.model_sha256.startswith("fcc3a257"), cal.model_sha256
    for p in (0.31, 0.62, 0.95, 0.999):
        got = cal.probability(p=p, runner=0.01, none=0.01, rank=3,
                              group="placeholder", collect=True)
        assert abs(got - p) < 1e-6, (p, got)
    for target in (0.90, 0.95):
        for group in GROUPS:
            bar = cal.bar(target, group)
            assert bar is not None and 0.5 <= bar < 1.0, (target, group,
                                                          bar)
    assert all(cal.bar(0.99, g) is None for g in GROUPS), (
        "a 0.99 bar on 788 picks would be claiming what they cannot show")
    assert cal.measured.get("dev_picks", 0) >= 700, cal.measured


def _signup_button(**more):
    raw = {"tag": "input", "id": "", "classes": ["btn", "btn-primary"],
           "name": "", "type": "submit", "role": "", "ariaLabel": "",
           "placeholder": "", "title": "Create your account", "alt": "",
           "href": "", "value": "Sign up", "dataAttrs": {}, "text": "",
           "accName": "", "ancestors": ["body", "main", "form.signup"],
           "rect": {"x": 40.4, "y": 300, "width": 120, "height": 36},
           "visible": True, "enabled": True, "interactive": True,
           "neighborText": "Already have an account? Log in"}
    raw.update(more)
    return ElementDescriptor.coerce(raw)


def test_the_line_both_trained_models_read_does_not_move():
    """compact() is the line the shipped AI was fine-tuned on and the line
    the checker scores (its training pairs were cut from the AI's
    prompts), so a change to it moves every repair off the distribution
    both were trained on. A richer line is compact_v2(), chosen with the
    prompt version."""
    line = _signup_button().compact(3)
    assert line == ('[3] <input class="btn btn-primary" type="submit"> '
                    'at(40,300) 120x36 path=body/main/form.signup/input'), (
        line)


def test_the_richer_line_says_what_tells_two_controls_apart():
    """compact() leaves out the words a person tells two controls apart
    by: what a submit button says (its value), its title, the label a
    field is announced with, the words beside it, where a link goes, and
    whether the control can be used at all."""
    line = _signup_button().compact_v2(3)
    for part in ('value="Sign up"', 'title="Create your account"',
                 "near='Already have an account? Log in'",
                 "at(40,300) 120x36", "path=body/main/form.signup/input"):
        assert part in line, (part, line)
    assert line.startswith('[3] <input class="btn btn-primary" '
                           'type="submit"'), line
    assert " hidden" not in line and " disabled" not in line

    off = _signup_button(visible=False, enabled=False).compact_v2(0)
    assert off.endswith(" hidden disabled"), off
    # A baseline is one element, not one of a list: no index.
    assert _signup_button().compact_v2(None) == line[4:], line

    link = ElementDescriptor.coerce({
        "tag": "a", "href": "/account/settings?tab=billing", "text": "Billing",
        "accName": "Billing", "ancestors": ["nav"], "visible": True,
        "rect": {"x": 0, "y": 0, "width": 60, "height": 20}}).compact_v2(1)
    assert 'href="/account/settings?tab=billing"' in link, link
    assert "label=" not in link, "a name that repeats the text is noise"

    field = ElementDescriptor.coerce({
        "tag": "input", "type": "text", "value": "jane@example.com",
        "accName": "Work email", "ancestors": ["form"], "visible": True,
        "rect": {"x": 0, "y": 0, "width": 200, "height": 30}}).compact_v2(2)
    assert "label='Work email'" in field, field
    assert "jane@example.com" not in field, (
        "what a field holds is what a step typed, not what it is called")

    # A checkbox's label is usually all the text around it, too.
    box = ElementDescriptor.coerce({
        "tag": "input", "type": "checkbox", "accName": "Sports",
        "neighborText": "Sports", "ancestors": ["form"], "visible": True,
        "rect": {"x": 0, "y": 0, "width": 16, "height": 16}}).compact_v2(0)
    assert "label='Sports'" in box and "near=" not in box, box


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
