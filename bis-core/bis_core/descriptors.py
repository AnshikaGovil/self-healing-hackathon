"""Data model: what the bot knows about an element it is trying to reach.

Two objects matter:

LocatorSpec
    The *design-time* contract authored by the RPA developer - the selector to
    try, plus a natural-language statement of business intent.  Intent is what
    lets the agent heal an element it has never successfully seen before.

ElementDescriptor
    A rich, run-time fingerprint of one concrete DOM element harvested from the
    live page.  Used both as a healing *candidate* and, once a step succeeds,
    as the *baseline* stored in memory for future comparisons.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

log = logging.getLogger("bis.descriptors")

# Digits are the single biggest source of false negatives when a front-end team
# ships a redesign: pd19-product-card becomes pd21-product-card,
# nv00-gnb-v3__search becomes nv00-gnb-v4__search.  Collapsing digit runs to a
# placeholder makes those pairs identical under comparison while leaving the
# rest of the token intact.
_DIGITS = re.compile(r"\d+")
_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WS = re.compile(r"\s+")

# Tokens carrying no discriminating power; they only add noise to overlap
# scores. Kept deliberately small - over-filtering hurts more than it helps.
_STOP_TOKENS = {
    "js", "ng", "is", "el", "ui", "wrap", "wrapper", "inner", "outer",
    "container", "div", "span", "active", "show", "hide", "hidden", "clearfix",
}

# English function words, stripped from a prose *description* before it joins
# the lexical token bag - see intent_tokens() for why.
#
# Deliberately closed-class only: articles, pronouns, prepositions,
# conjunctions, auxiliaries, quantifiers. No domain nouns. Words like "button",
# "page", "item", "card", "text" and "search" appear in real class names all
# the time and must survive, or the description stops being able to name the
# thing it is describing.
_PROSE_STOP = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "which",
    "who", "whose", "what", "there", "here",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "within",
    "into", "onto", "over", "under", "above", "below", "beside", "next",
    "near", "inside", "outside", "between", "across", "through", "up", "down",
    "and", "or", "but", "as", "if", "then", "than", "so", "because", "when",
    "while", "where", "after", "before",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must",
    "all", "any", "each", "every", "some", "one", "both", "more", "most",
    "other", "another", "same", "only", "own", "such", "very", "not", "no",
}


def collapse_digits(token: str) -> str:
    """Map pd21 -> pd#, so version-bumped class names stay comparable."""
    return _DIGITS.sub("#", token)


def names_a_component(identifier: str) -> bool:
    """Is there still a component name here once the digits are collapsed?

    Digit collapsing is the highest-yield normalisation there is against a
    versioned design system, and it is ruinous on a page that numbers its
    elements. Measured over 86 live sites: Hacker News' 90 ids collapse to
    5 distinct values, thirty of them to bare "#"; arXiv's 205 collapse to
    10; crates.io's 164 to 5. An id of "12345" carries no name, so after
    collapsing it matches every other numbered element on the page equally
    - and the identity signal, the highest-weighted one, then votes for a
    stranger with full confidence.

    A collapsed identifier has to keep at least two letters to count as
    naming anything. "c#-title" does. "#" and "#.#" do not.
    """
    return sum(1 for ch in identifier if ch.isalpha()) >= 2


def tokenize(value):
    """Split an attribute value into normalised comparison tokens.

    Handles kebab-case, snake_case, BEM (block__elem--mod) and camelCase, then
    lowercases and version-normalises each piece.
    """
    if not value:
        return []
    value = _CAMEL.sub(" ", str(value))
    out = []
    for part in _SPLIT.split(value):
        if not part:
            continue
        part = collapse_digits(part.lower())
        if part in _STOP_TOKENS or len(part) < 2:
            continue
        out.append(part)
    return out


SYNTAX_CSS = "css"
SYNTAX_XPATH = "xpath"
SYNTAX_TEXT = "text"

# A selector can step past the page's own document. Each '>>>' steps into
# what the element before it holds - an open shadow root, or a frame's
# document - and the next step is asked there: 'app-shell >>> button.save'
# is the button in <app-shell>'s shadow root, 'iframe#main >>> #email' the
# box in that frame. js_harvest proposes these and resolves them, and
# bis.py splits them the same way.
CHAIN = ">>>"


def chain_steps(selector) -> List[str]:
    """A selector's steps across shadow roots and frames, outermost first.

    Split on '>>>' only outside quotes and brackets, as the page splits it
    (js_harvest splitChain), since an attribute value may hold one.
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
        elif depth == 0 and body.startswith(CHAIN, i):
            steps.append("".join(buf))
            buf = []
            i += len(CHAIN)
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


def last_step(selector) -> str:
    """The step of a chain that names the element itself.

    Its language, tag and names are the element's; the hosts and frames
    before it are only the way there, as an ancestor step is.
    """
    steps = chain_steps(selector)
    return steps[-1] if steps else ""


def selector_syntax(selector) -> str:
    """Which selector language a locator is written in.

    Recognises an explicit ``xpath=`` prefix and also bare XPath,
    because RPA teams migrating from Selenium or another platform
    habitually paste expressions that start with ``//`` and expect
    them to work. A chain is in the language of its last step.
    """
    if not selector:
        return SYNTAX_CSS
    body = last_step(selector)
    low = body.lower()
    if low.startswith("xpath="):
        return SYNTAX_XPATH
    if low.startswith("text=") or low.startswith("css="):
        return SYNTAX_TEXT if low.startswith("text=") else SYNTAX_CSS
    if body.startswith(("//", "(/", "(//", "./", "/")):
        return SYNTAX_XPATH
    return SYNTAX_CSS


def normalize_selector(selector):
    """Give bare XPath its explicit prefix.

    Everything downstream - the in-page probe, the binder, Playwright itself -
    switches on that prefix, so normalising once here keeps a bare ``//button``
    from being handed to ``querySelectorAll`` and failing as a syntax error.
    Each step of a chain is given its own, and the steps are joined one way.
    """
    if not selector:
        return selector
    steps = chain_steps(selector)
    if len(steps) > 1:
        return (" %s " % CHAIN).join(normalize_selector(s) for s in steps)
    body = str(selector).strip()
    if selector_syntax(body) == SYNTAX_XPATH and not body.lower().startswith("xpath="):
        return "xpath=" + body
    return body


def strip_syntax_prefix(selector) -> str:
    """The selector body, without its engine prefix - for a chain, the
    body of its last step."""
    body = last_step(selector)
    for prefix in ("xpath=", "css=", "text="):
        if body.lower().startswith(prefix):
            return body[len(prefix):]
    return body


CSS_TAG_RE = re.compile(r"^([a-zA-Z][\w\-]*)")
XPATH_TAG_RE = re.compile(r"^(?:[\w\-]+::)?([a-zA-Z][\w\-]*)")

# Node tests that are XPath syntax rather than element names.
_XPATH_NON_TAGS = {"and", "or", "not", "text"}


def split_selector_steps(body: str, syntax: str):
    """Break a selector into its steps, respecting quotes and brackets.

    The **last** step is the element being selected; everything before it
    describes ancestors. That distinction matters: for
    ``//div[@class='card']//a[@class='title']`` the target is the anchor, and
    scoring a candidate against the card's class name would be wrong.
    """
    seps = "/" if syntax == SYNTAX_XPATH else " >+~" + chr(9) + chr(10)
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
        elif depth == 0 and ch in seps:
            if buf:
                steps.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        steps.append("".join(buf))
    return [s for s in (x.strip() for x in steps) if s]


def selector_tag(selector):
    """The tag a selector targets, read from its final step.

    Returns None when the selector names no tag (``.card``, ``//*[@id='x']``).
    """
    if not selector:
        return None
    syntax = selector_syntax(selector)
    if syntax == SYNTAX_TEXT:
        return None
    steps = split_selector_steps(strip_syntax_prefix(selector), syntax)
    if not steps:
        return None
    last = steps[-1]
    if syntax == SYNTAX_XPATH:
        match = XPATH_TAG_RE.match(last.lstrip("./"))
        if match and match.group(1).lower() not in _XPATH_NON_TAGS:
            return match.group(1).lower()
        return None
    match = CSS_TAG_RE.match(last)
    return match.group(1).lower() if match else None


# What kind of thing each tag is, for locators that do not say. Only tags whose
# role is unambiguous appear here: a <div> could be anything, so it stays
# unmapped and the ranker simply scores one signal fewer rather than acting on
# a guess.
_ROLE_BY_TAG = {
    "a": "link",
    "button": "button",
    "input": "textbox",
    "textarea": "textbox",
    "select": "dropdown",
    "img": "image",
    "h1": "text", "h2": "text", "h3": "text",
    "h4": "text", "h5": "text", "h6": "text",
    "p": "text", "span": "text", "td": "text", "li": "text",
}


def intent_tokens(description):
    """Tokens from a prose description, minus English function words.

    The description reaches the ranker twice: whole, as the query for the
    semantic signal and the AI prompt, and tokenised, as extra entries in the
    lexical token bag. Only the second use needs filtering, because
    ``attribute_signal`` scores containment as ``matched / total`` - every
    token in the target that no DOM element could possibly carry inflates the
    denominator and drags *every* candidate down.

    Measured on the shipped Samsung repository, the unfiltered descriptions
    were costing the top-ranked candidate around 0.13 of final score - enough,
    at an auto-apply threshold of 0.62, to turn an applied repair into an
    escalation.

    Deliberately not folded into ``_STOP_TOKENS``: that set is applied by
    ``tokenize`` to attribute values on *both* sides of the comparison, so
    adding "the" or "of" there would quietly change candidate fingerprints and
    selector mining too.
    """
    return [t for t in tokenize(description) if t not in _PROSE_STOP]


# Words that name no element in particular. A description made only of these
# is a placeholder, not evidence - "the field" tells the ranker nothing that
# every candidate does not already satisfy.
_CONTENTLESS = {
    "element", "field", "thing", "item", "control", "widget", "component",
    "object", "area", "box", "page", "screen", "section", "part", "value",
    "data", "content", "one", "stuff", "panel", "text", "block", "wrapper",
    "container", "div", "entry", "record",
}


def informative_intent_tokens(description):
    """Intent tokens, or nothing at all when the description says nothing.

    Measured across six live sites, a placeholder description is worse than
    no description: it costs the correct candidate 0.06-0.17 of final score
    because ``attribute_signal`` divides by the number of target tokens, and
    a word no DOM element carries only inflates that denominator.

        drop_last, junk description "the field"   0.497
        drop_last, no description at all          0.665

    A real description still reaches the bag whole - dropping *every*
    description inverts the consent.accept ranking, which is what invariant 8
    is about. This drops only the ones carrying no identity at all.
    """
    toks = intent_tokens(description)
    if not toks:
        return []
    return [] if all(t in _CONTENTLESS for t in toks) else toks


def selector_uses_id(selector) -> bool:
    """Does this selector identify its element by id?

    Covers the CSS form (``#foo``, ``input#foo``) and the XPath one
    (``//input[@id='foo']``). Used to compare what the *author* wrote against
    what the page proposes. A chain is asked about its last step: an id on
    the host on the way says nothing about how the element is named.
    """
    body = strip_syntax_prefix(selector or "")
    if not body:
        return False
    if selector_syntax(selector) == SYNTAX_XPATH:
        return "@id" in body.lower()
    # A '#' inside an attribute value or a pseudo-class argument is not an id
    # selector - `a[href="#top"]` says nothing about ids - so ignore anything
    # in brackets or quotes before looking.
    bare = re.sub(r"\[[^\]]*\]|'[^']*'|\"[^\"]*\"", " ", body)
    return "#" in bare


def normalize_text(value, limit: int = 160) -> str:
    """Collapse whitespace and trim - used for every text comparison."""
    if not value:
        return ""
    return _WS.sub(" ", str(value)).strip()[:limit]


def page_scope(url) -> str:
    """Normalise a URL into a stable key for anything remembered per page.

    Query strings and fragments are dropped: the same page reached with
    different tracking parameters must not fragment what has been learned
    about it.

    Lives here rather than beside the memory file because the healing
    service keeps baselines for thin clients and has to derive exactly the
    same key the toolkit derives, or a bot would look up its own baseline
    under a name nothing ever wrote.
    """
    try:
        parsed = urlparse(str(url or ""))
        path = (parsed.path or "/").rstrip("/") or "/"
        return "%s%s" % (parsed.netloc, path)
    except Exception:
        return str(url or "") or "unknown"


# How harvest_real.intent_for names the kind of control a step meant. Every
# rich description the AI and the checker learned from was written by it,
# so one written for them from a baseline must say things the same way.
_KIND_BY_TAG = {"input": "box", "textarea": "box", "select": "dropdown",
                "button": "button", "a": "link"}
_KIND_BY_ROLE = {"button": "button", "tab": "tab", "menuitem": "menu item",
                 "link": "link", "checkbox": "checkbox", "switch": "switch"}
# The inputs whose value is the words on them. Anywhere else a value is what
# a step typed, so it is never shown to a model and never kept (own_value);
# the page's describe() reads this same list.
BUTTON_INPUT_TYPES = ("submit", "button", "reset")
_BUTTON_INPUTS = set(BUTTON_INPUT_TYPES)


def own_value(tag: str, input_type: str, value: str) -> str:
    """An element's value if it is the element's own words, else "".

    A button input's caption, a button's or an option's markup. Any other
    value is what a step typed or chose - a password as like as not - and a
    fingerprint travels to the service and is kept as a baseline. The page's
    describe() applies the same rule (js_harvest's ownValue); this one guards
    what a bot still holding the JavaScript from before sends.
    """
    tag = (tag or "").lower()
    kind = (input_type or "").lower()
    if tag in ("button", "option") or (tag == "input"
                                       and kind in BUTTON_INPUT_TYPES):
        return value or ""
    return ""


def describe_from_baseline(desc: "ElementDescriptor") -> str:
    """A step description for an element, phrased as the training set's.

    "the {label} {kind}", built the way harvest_real.intent_for built every
    rich description both models were trained on, from the same facts: the
    aria-label, placeholder, title or <label> first, the element's own
    text after. "" when the element carries no name at all.

    One deliberate difference. The harvest named an input by its value,
    which on a fresh page is a button's caption; in production a text box's
    value is what the bot typed - a name for nothing, and possibly a
    password - so only a button's value is read.
    """
    tag = (desc.tag or "").lower()
    kind_of_input = (desc.input_type or "").lower()
    label = normalize_text(desc.aria_label or desc.placeholder or desc.title
                           or desc.accessible_name, 60)
    text = normalize_text(desc.text, 60)
    if not text and tag == "input" and kind_of_input in _BUTTON_INPUTS:
        text = normalize_text(desc.value, 60)
    name = label or text
    if not name:
        return ""
    if tag == "input" and kind_of_input in ("checkbox", "radio"):
        kind = kind_of_input
    elif tag == "input" and kind_of_input in _BUTTON_INPUTS:
        kind = "button"
    else:
        kind = _KIND_BY_TAG.get(tag) or _KIND_BY_ROLE.get(desc.role or "",
                                                          "control")
    return "the %s %s" % (name[:48], kind)


KNOWN_LOCATOR_KEYS = frozenset({
    "selector", "description", "tag", "role", "expected_text",
    "scope", "fallbacks", "critical", "multiple",
})

# Keys that used to live on a locator and have moved. Worth naming explicitly:
# "unknown key" is unhelpful when the key was documented last release.
_MOVED_LOCATOR_KEYS = {
    "stale": "the repository-level 'stale' list",
}


def _warn_unknown_keys(key: str, raw: Dict[str, Any]) -> None:
    """Say something when a locator carries a key the engine will ignore.

    Warn rather than raise: a typo in a field that was going to be optional
    anyway must not take a running process down. But it must not be silent
    either - a misspelled 'description' costs real healing quality, forever,
    with nothing in the logs to say so.
    """
    for name in raw:
        if name in KNOWN_LOCATOR_KEYS:
            continue
        if name in _MOVED_LOCATOR_KEYS:
            log.warning("locator '%s': '%s' has moved to %s and is ignored "
                        "here", key, name, _MOVED_LOCATOR_KEYS[name])
            continue
        near = difflib.get_close_matches(name, KNOWN_LOCATOR_KEYS, 1, 0.7)
        log.warning("locator '%s': unknown key '%s' is ignored%s", key, name,
                    (" - did you mean '%s'?" % near[0]) if near else "")


@dataclass
class LocatorSpec:
    """A design-time locator plus the business intent behind it.

    ``description`` is not documentation - it is a first-class input to the
    healing engine.  It is what allows a repair on the very first run, before
    any successful baseline has ever been recorded.
    """

    key: str                                   # stable id, e.g. plp.product_name
    selector: str                              # CSS, xpath=..., or text=...
    description: str = ""                      # plain-English business intent
    tag: Optional[str] = None                  # expected tag, if known
    role: Optional[str] = None                 # button | link | textbox | text
    expected_text: Optional[str] = None        # regex a repair must satisfy
    scope: Optional[str] = None                # CSS container to search within
    fallbacks: List[str] = field(default_factory=list)   # tried before any AI
    critical: bool = True                      # False => step may be skipped
    multiple: bool = False
    """True when the step reads a *set* of elements (every product name on the
    page) rather than one. It changes what a good repair looks like: a
    collection locator must stay generic, so a selector pinned to one instance
    is a failed repair even though it resolves."""

    def __post_init__(self) -> None:
        # Accept bare XPath as authored and give it its prefix once, here,
        # rather than making every consumer sniff for it.
        self.selector = normalize_selector(self.selector)
        self.fallbacks = [normalize_selector(f) for f in self.fallbacks]
        # Derive the role rather than asking an author to type what the tag
        # already implies. This is the single seam both sides pass through -
        # LocatorRepository.load on the bot, spec_from_payload on the service -
        # so the two agree without the wire payload growing a field.
        if not self.role:
            self.role = _ROLE_BY_TAG.get(self.tag or selector_tag(self.selector) or "")

    @property
    def syntax(self) -> str:
        return selector_syntax(self.selector)

    def intent_sentence(self) -> str:
        """Render the spec as one sentence for the embedding / LLM stages."""
        bits = [self.description or self.key.replace(".", " ").replace("_", " ")]
        if self.role:
            bits.append("role: " + self.role)
        if self.tag:
            bits.append("tag: " + self.tag)
        if self.expected_text:
            bits.append("text matching: " + self.expected_text)
        return "; ".join(bits)

    @classmethod
    def from_dict(cls, key: str, raw: Dict[str, Any]) -> "LocatorSpec":
        _warn_unknown_keys(key, raw)
        if not raw.get("selector"):
            raise KeyError(
                "locator '%s' has no 'selector' - it is the only required "
                "field" % key)
        return cls(
            key=key,
            selector=raw["selector"],
            description=raw.get("description", ""),
            tag=raw.get("tag"),
            role=raw.get("role"),
            expected_text=raw.get("expected_text"),
            scope=raw.get("scope"),
            fallbacks=list(raw.get("fallbacks", [])),
            critical=bool(raw.get("critical", True)),
            multiple=bool(raw.get("multiple", False)),
        )


# Keys the browser-side JavaScript emits that are not dataclass field names.
# Their presence is what tells the two wire shapes apart - see
# ElementDescriptor.coerce().
_BUTTON_TYPES = BUTTON_INPUT_TYPES

_JS_ONLY_KEYS = ("ariaLabel", "dataAttrs", "accName", "cssPath", "sibIndex",
                 "neighborText", "scrollX", "scrollY", "viewportW",
                 "viewportH")


@dataclass
class ElementDescriptor:
    """Run-time fingerprint of a single DOM element."""

    tag: str = ""
    element_id: str = ""
    classes: List[str] = field(default_factory=list)
    name: str = ""
    input_type: str = ""
    role: str = ""
    aria_label: str = ""
    placeholder: str = ""
    title: str = ""
    alt: str = ""
    href: str = ""
    value: str = ""
    data_attrs: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    accessible_name: str = ""
    xpath: str = ""
    css_path: str = ""
    ancestors: List[str] = field(default_factory=list)
    depth: int = 0
    sibling_index: int = 0
    rect: Dict[str, float] = field(default_factory=dict)
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    viewport_w: float = 0.0
    viewport_h: float = 0.0
    """Where the page was scrolled to, and how big the window was, when this
    element was described. ``rect`` is viewport-relative, so without these two
    a baseline taken at the top of a page and a candidate harvested after
    scrolling look like the element moved. Zero means an older fingerprint
    that predates the fields; geometry_signal falls back rather than guess."""

    visible: bool = True
    enabled: bool = True
    interactive: bool = False
    neighbor_text: str = ""
    selectors: List[str] = field(default_factory=list)
    """Ready-made selector proposals generated in-page, most durable
    first. Because they are derived from a real element, the healer can
    only ever *choose* a selector, never invent one."""

    def attribute_tokens(self) -> List[str]:
        """Every identity-bearing token this element exposes."""
        toks = []
        toks += tokenize(self.element_id)
        for c in self.classes:
            toks += tokenize(c)
        toks += tokenize(self.name)
        toks += tokenize(self.input_type)
        toks += tokenize(self.role)
        for k, v in self.data_attrs.items():
            toks += tokenize(k)
            toks += tokenize(v)
        return toks

    def label_text(self) -> str:
        """Best available human-facing label for this element."""
        for candidate in (
            self.accessible_name, self.aria_label, self.text,
            self.placeholder, self.title, self.alt, self.value,
        ):
            candidate = normalize_text(candidate)
            if candidate:
                return candidate
        return ""

    def short_path(self, depth: int = 4) -> str:
        """Tail of the ancestor chain - enough context without the noise."""
        return "/".join(self.ancestors[-depth:] + [self.tag])

    def semantic_sentence(self) -> str:
        """Natural-language rendering, fed to the embedder and the LLM.

        Deliberately reads like a sentence a person would write, because that
        is the distribution the sentence encoder was trained on.
        """
        kind = self.role or {
            "a": "link", "button": "button", "input": "input field",
            "select": "dropdown", "textarea": "text area", "img": "image",
        }.get(self.tag, self.tag or "element")
        label = self.label_text()
        parts = [kind]
        if label:
            parts.append('labelled "' + label + '"')
        if self.element_id:
            parts.append("with id " + self.element_id)
        if self.classes:
            parts.append("styled " + " ".join(self.classes[:4]))
        if self.neighbor_text:
            parts.append("near " + normalize_text(self.neighbor_text, 60))
        return " ".join(parts)

    def compact(self, index: int) -> str:
        """One-line rendering for the LLM shortlist (token budget matters)."""
        attrs = []
        if self.element_id:
            attrs.append('id="%s"' % self.element_id)
        if self.classes:
            attrs.append('class="%s"' % " ".join(self.classes[:5]))
        if self.name:
            attrs.append('name="%s"' % self.name)
        if self.input_type:
            attrs.append('type="%s"' % self.input_type)
        if self.role:
            attrs.append('role="%s"' % self.role)
        if self.aria_label:
            attrs.append('aria-label="%s"' % self.aria_label)
        if self.placeholder:
            attrs.append('placeholder="%s"' % self.placeholder)
        for k, v in list(self.data_attrs.items())[:2]:
            attrs.append('%s="%s"' % (k, str(v)[:24]))
        text = normalize_text(self.text, 70)
        rect = self.rect or {}
        geom = "at(%d,%d) %dx%d" % (
            int(rect.get("x", 0)), int(rect.get("y", 0)),
            int(rect.get("width", 0)), int(rect.get("height", 0)),
        )
        txt = (" text=" + repr(text)) if text else ""
        return "[%d] <%s %s>%s %s path=%s" % (
            index, self.tag, " ".join(attrs), txt, geom, self.short_path(),
        )

    def compact_v2(self, index: Optional[int]) -> str:
        """The shortlist line of a version 3 prompt; with no index, the
        LAST KNOWN GOOD line, which describes one element, not a list.

        compact() leaves out most of what a person tells two controls
        apart by - what a submit button says, a title or alt text, the
        label a field is announced with, the words beside it, where a link
        goes, and whether the control can be used at all - so this line
        adds each of them when it says something the line does not
        already. compact() itself stays as it is: the shipped AI was
        fine-tuned on it and the checker scores it. What a field holds is
        never shown; it is what a step typed, not what the field is.
        """
        attrs = []
        if self.element_id:
            attrs.append('id="%s"' % self.element_id)
        if self.classes:
            attrs.append('class="%s"' % " ".join(self.classes[:5]))
        if self.name:
            attrs.append('name="%s"' % self.name)
        if self.input_type:
            attrs.append('type="%s"' % self.input_type)
        if self.role:
            attrs.append('role="%s"' % self.role)
        shown = []
        for key, value, room in (("aria-label", self.aria_label, 70),
                                 ("placeholder", self.placeholder, 70),
                                 ("title", self.title, 40),
                                 ("alt", self.alt, 40)):
            value = normalize_text(value, room)
            if value:
                attrs.append('%s="%s"' % (key, value))
                shown.append(value)
        # A submit button's value is the words on it, and nothing else
        # names it: its innerText is empty.
        if self.tag == "input" and self.input_type.lower() in _BUTTON_TYPES:
            value = normalize_text(self.value, 40)
            if value:
                attrs.append('value="%s"' % value)
                shown.append(value)
        for k, v in list(self.data_attrs.items())[:2]:
            attrs.append('%s="%s"' % (k, str(v)[:24]))
        if self.href:
            attrs.append('href="%s"' % self.href[:60])
        text = normalize_text(self.text, 70)
        shown.append(text)
        parts = [" text=" + repr(text)] if text else []
        said = {s.lower() for s in shown}
        label = normalize_text(self.accessible_name, 50)
        if label and label.lower() not in said:
            parts.append(" label=" + repr(label))
            said.add(label.lower())
        near = normalize_text(self.neighbor_text, 50)
        if near and near.lower() not in said:
            parts.append(" near=" + repr(near))
        rect = self.rect or {}
        geom = "at(%d,%d) %dx%d" % (
            int(rect.get("x", 0)), int(rect.get("y", 0)),
            int(rect.get("width", 0)), int(rect.get("height", 0)),
        )
        flags = ("" if self.visible else " hidden") + (
            "" if self.enabled else " disabled")
        line = "<%s %s>%s %s path=%s%s" % (
            self.tag, " ".join(attrs), "".join(parts), geom,
            self.short_path(), flags,
        )
        return line if index is None else "[%d] %s" % (index, line)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_js(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        """Build from the payload returned by js_harvest.HARVEST_JS."""
        return cls(
            tag=(raw.get("tag") or "").lower(),
            element_id=raw.get("id") or "",
            classes=[c for c in (raw.get("classes") or []) if c],
            name=raw.get("name") or "",
            input_type=raw.get("type") or "",
            role=raw.get("role") or "",
            aria_label=raw.get("ariaLabel") or "",
            placeholder=raw.get("placeholder") or "",
            title=raw.get("title") or "",
            alt=raw.get("alt") or "",
            href=raw.get("href") or "",
            value=raw.get("value") or "",
            data_attrs=raw.get("dataAttrs") or {},
            text=normalize_text(raw.get("text")),
            accessible_name=normalize_text(raw.get("accName")),
            xpath=raw.get("xpath") or "",
            css_path=raw.get("cssPath") or "",
            ancestors=raw.get("ancestors") or [],
            depth=int(raw.get("depth") or 0),
            sibling_index=int(raw.get("sibIndex") or 0),
            rect=raw.get("rect") or {},
            scroll_x=float(raw.get("scrollX") or 0),
            scroll_y=float(raw.get("scrollY") or 0),
            viewport_w=float(raw.get("viewportW") or 0),
            viewport_h=float(raw.get("viewportH") or 0),
            visible=bool(raw.get("visible")),
            enabled=bool(raw.get("enabled", True)),
            interactive=bool(raw.get("interactive")),
            neighbor_text=normalize_text(raw.get("neighborText"), 80),
            selectors=list(raw.get("selectors") or []),
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    @classmethod
    def coerce(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        """Build from either wire shape - harvester output or a stored dict.

        The full agent runs from_js() in the bot and sends the *serialised
        descriptor*; a thin client has no descriptor class and sends the
        harvester's own output, which names the same facts differently
        (``id`` for element_id, ``dataAttrs`` for data_attrs, and so on).

        Anything decoding a payload that could have come from either must
        use this. from_dict() silently drops every key it does not
        recognise, so a harvester payload decoded with it arrives with no
        id, no type, no aria-label, no data-* and no accessible name - the
        exact fields the identity and attribute signals are computed from.
        That is not a degraded score, it is a wrong one, and it is invisible:
        the request succeeds and the ranking is quietly built on a third of
        the evidence.
        """
        if not raw:
            return cls()
        return (cls.from_js(raw) if any(k in raw for k in _JS_ONLY_KEYS)
                else cls.from_dict(raw))
