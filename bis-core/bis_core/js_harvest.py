"""Browser-side DOM harvesting.

The healing engine never guesses selectors in the abstract - it looks at what
is *actually on the page right now*.  This module holds the JavaScript that
runs inside the page and returns a structured fingerprint for every plausible
element, together with a ranked list of ready-made selectors for each one.

Generating the selectors in the browser (rather than asking an LLM to invent a
string) is a deliberate design choice: every selector we later apply provably
came from a real element, so the model can only *choose*, never *hallucinate*.
"""

from .descriptors import BUTTON_INPUT_TYPES


# How a selector reaches past the page's own document: a chain of steps
# joined by '>>>'. Each step steps into what the element before it holds -
# an open shadow root, or a frame's document - and the next step is asked
# there. 'app-shell >>> button.save' is the button inside <app-shell>'s
# shadow root; 'iframe#main >>> #email' the box inside that frame. A
# selector with no '>>>' is one step, read exactly as it always was.
#
# Only what a script on the page can reach: a closed shadow root is null to
# everyone but its owner, and a frame from another origin has no document on
# this side of the boundary.
#
# One copy, spliced into every entry point that resolves a selector, for the
# reason _DESCRIBE_JS is one copy: the harvest proposes these chains, the
# probe and the fingerprinter resolve them, the bind check compares what
# they bound and the presence gate looks behind the same boundaries.
# _DESCRIBE_JS uses view(), framesOf() and isShown(), so it is spliced
# after this.
_RESOLVE_JS = r"""
  const CHAIN = '>>>';
  // A selector's steps, split on '>>>' only outside quotes and brackets -
  // an attribute value may hold the characters. descriptors.chain_steps
  // and bis.py's _chain split it the same way.
  const splitChain = (sel) => {
    const s = String(sel || '');
    const out = [];
    let buf = '', quote = '', depth = 0;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (quote) { buf += ch; if (ch === quote) quote = ''; continue; }
      if (ch === '"' || ch === "'") { quote = ch; buf += ch; continue; }
      if (!depth && s.startsWith(CHAIN, i)) {
        out.push(buf); buf = ''; i += CHAIN.length - 1; continue;
      }
      if (ch === '[' || ch === '(') depth++;
      else if (ch === ']' || ch === ')') depth = Math.max(0, depth - 1);
      buf += ch;
    }
    out.push(buf);
    return out.map((p) => p.trim()).filter(Boolean);
  };
  // The step that names the element; the steps before it are the way there.
  const lastStep = (sel) => {
    const steps = splitChain(sel);
    return steps.length ? steps[steps.length - 1] : '';
  };
  // What an element holds for the next step of a chain, or null.
  const inside = (el) => {
    if (!el || el.nodeType !== 1) return null;
    if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
      try { return el.contentDocument || null; } catch (e) { return null; }
    }
    return el.shadowRoot || null;
  };
  const isXPathStep = (s) => /^(xpath=|\/|\(\/|\.\/)/.test(s);
  const queryIn = (scope, step) => {
    if (isXPathStep(step)) {
      // XPath cannot see into a shadow root - nor can WebDriver's search
      // of one - so a step written in it finds nothing there.
      if (scope.nodeType === 11) return [];
      const doc = scope.nodeType === 9 ? scope : scope.ownerDocument;
      const it = doc.evaluate(step.replace(/^xpath=/, ''), scope, null,
                              XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
      const out = [];
      for (let i = 0; i < it.snapshotLength; i++) out.push(it.snapshotItem(i));
      return out;
    }
    return Array.from(scope.querySelectorAll(step.replace(/^css=/, '')));
  };
  // Every element a selector names, in document order within each scope.
  const resolveAll = (selector) => {
    const steps = splitChain(selector);
    let scopes = [document], found = [];
    for (let i = 0; i < steps.length; i++) {
      const seen = new Set();
      found = [];
      for (const scope of scopes) {
        for (const n of queryIn(scope, steps[i])) {
          if (!seen.has(n)) { seen.add(n); found.push(n); }
        }
      }
      if (i < steps.length - 1) scopes = found.map(inside).filter(Boolean);
    }
    return found;
  };
  // [scope, the hosts and frames on the way to it], for `base` and for every
  // open shadow root and readable frame under it - base first.
  const allScopes = (base) => {
    const out = [[base, []]];
    for (let i = 0; i < out.length && out.length < 2000; i++) {
      const [scope, via] = out[i];
      if (via.length >= 8) continue;
      for (const el of scope.querySelectorAll('*')) {
        const held = inside(el);
        const into = held && (held.nodeType === 9
                              ? (held.body || held.documentElement) : held);
        if (into) out.push([into, via.concat([el])]);
      }
    }
    return out;
  };
  // The window an element lives in, and the frames between it and this
  // page, innermost first.
  const view = (el) => (el.ownerDocument && el.ownerDocument.defaultView)
                       || window;
  const framesOf = (el) => {
    const out = [];
    let doc = el.ownerDocument;
    while (doc && doc !== document) {
      let f = null;
      try { f = doc.defaultView && doc.defaultView.frameElement; } catch (e) {}
      if (!f) break;
      out.push(f);
      doc = f.ownerDocument;
    }
    return out;
  };
  // On screen as the visibility test has always read it - a box, not
  // hidden, not transparent - and inside frames that are on screen too.
  const isShown = (el) => {
    for (const n of [el].concat(framesOf(el))) {
      const r = n.getBoundingClientRect();
      const cs = view(n).getComputedStyle(n);
      if (!(r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' &&
            cs.display !== 'none' && parseFloat(cs.opacity || '1') > 0.05)) {
        return false;
      }
    }
    return true;
  };
"""


# One description of an element, spliced into both entry points below.
#
# It used to be implemented twice - here for candidates, and again inside
# DESCRIBE_ONE_JS for baselines - and the two drifted. Measured over 738
# interactive elements on four live sites, they disagreed about `visible` 40%
# of the time and about `neighborText` 46%, and the baseline copy derived no
# accessible name from a <label>, which on a login form is every field.
#
# Worse than the disagreement was where it landed. Both the service store and
# the agent refuse a baseline that is not `visible`, and the baseline copy
# defined that as "has a box" - which is what the other half of the same guard
# already checked. The visibility test invariant 12 describes was not running
# on that path at all.
#
# A candidate and a baseline are compared field against field. They have to be
# produced by the same code, so there is one copy of it.
_DESCRIBE_JS = r"""
  const INTERACTIVE = new Set(['A','BUTTON','INPUT','SELECT','TEXTAREA',
                               'LABEL','SUMMARY','OPTION']);

  const clean = (s) => (s || '').toString().replace(/\s+/g, ' ').trim();

  // --- CSS.escape is not universally available on older engines ------------
  //
  // The backslash in the replacement below is doubled and has to stay that
  // way. JavaScript drops the backslash in an unrecognised escape, so a
  // single one in front of a replacement group is read as a bare group
  // reference - and the fallback then replaced every special character with
  // itself, an identity function standing where the escaping was supposed
  // to be. It went unnoticed because every engine new enough to run the
  // rest of this has CSS.escape, so the branch that was broken is the one
  // that only runs on the engines it was written for. PRESENCE_JS below has
  // always had the doubled form; these two are now the same, and
  // test_a_backslash_in_the_javascript_survives_into_the_browser keeps
  // them that way.
  const esc = (s) => (window.CSS && CSS.escape)
      ? CSS.escape(s) : String(s).replace(/([^\w-])/g, '\\$1');

  // A class is "stable enough" to build a selector from if it is not one of
  // the framework state classes that flip on every interaction.
  const VOLATILE = /^(is-|has-|ng-|v-|active$|open$|show$|hide$|hidden$|selected$|focus|hover|animate|swiper-|slick-)/i;
  const stableClasses = (el) => Array.from(el.classList || [])
      .filter(c => c && !VOLATILE.test(c) && !/^[a-z]{1,2}\d{3,}$/i.test(c));

  const absXPath = (el) => {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 14) {
      let i = 1, sib = node.previousElementSibling;
      while (sib) { if (sib.tagName === node.tagName) i++; sib = sib.previousElementSibling; }
      parts.unshift(node.tagName.toLowerCase() + '[' + i + ']');
      node = node.parentElement;
    }
    return '/' + parts.join('/');
  };

  const cssPath = (el) => {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 7) {
      if (node.id) { parts.unshift('#' + esc(node.id)); break; }
      let i = 1, sib = node.previousElementSibling;
      while (sib) { if (sib.tagName === node.tagName) i++; sib = sib.previousElementSibling; }
      parts.unshift(node.tagName.toLowerCase() + ':nth-of-type(' + i + ')');
      node = node.parentElement;
    }
    return parts.join(' > ');
  };

  // An element's parent in the page as it is drawn: past the top of a
  // shadow root to its host, and past the top of a frame's document to the
  // frame. Everywhere else it is parentElement, so an element of the page
  // itself is described exactly as it always was.
  const up = (el) => {
    if (el.parentElement) return el.parentElement;
    const root = el.getRootNode ? el.getRootNode() : null;
    if (!root || root === document) return null;
    if (root.host) return root.host;
    return framesOf(el)[0] || null;
  };

  // The path from the top of this page, through each host or frame on the
  // way - what the structure signal compares. For an element of the page
  // itself it is absXPath; the steps behind a boundary are relative to it.
  const pagePath = (el) => {
    const own = absXPath(el);
    const root = el.getRootNode ? el.getRootNode() : null;
    if (!root || root === document) return own;
    const holder = root.host || framesOf(el)[0];
    if (!holder) return own;
    return pagePath(holder) + (root.host ? '/#shadow-root' : '/#document') +
           own;
  };

  // The box as this page's viewport sees it. Inside a frame, the frame's
  // content box is added: its box, its border and its padding.
  const pageRect = (el, r) => {
    let x = r.x, y = r.y;
    for (const f of framesOf(el)) {
      const fr = f.getBoundingClientRect();
      const fs = view(f).getComputedStyle(f);
      x += fr.x + f.clientLeft + (parseFloat(fs.paddingLeft) || 0);
      y += fr.y + f.clientTop + (parseFloat(fs.paddingTop) || 0);
    }
    return {x: x, y: y, width: r.width, height: r.height};
  };

  const accName = (el) => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      // Ids are looked up where the element lives: a shadow root and a
      // frame's document each have ids of their own.
      const root = el.getRootNode ? el.getRootNode() : document;
      const byId = (id) => (root && root.getElementById
                            ? root.getElementById(id)
                            : document.getElementById(id));
      const t = lb.split(/\s+/).map(id => {
        const n = byId(id); return n ? n.innerText : '';
      }).join(' ');
      if (clean(t)) return clean(t);
    }
    if (el.tagName === 'INPUT' && el.labels && el.labels.length)
      return clean(el.labels[0].innerText);
    return clean(el.getAttribute('alt') || el.getAttribute('title')
                 || el.getAttribute('placeholder') || '');
  };

  // Text that surrounds the element - a strong disambiguator when class names
  // have been rewritten wholesale but the page copy has not.
  const neighborText = (el) => {
    // Climb until the surroundings actually say something. Stopping at the
    // immediate parent leaves this empty for 77% of the interactive elements
    // on real pages - a form control usually sits alone in a layout div and
    // its label is a level or two up. Three hops takes that to 25%.
    //
    // Climbing further is deliberately not done. The higher you go the more
    // generic the text becomes, until every control in a section shares it,
    // and text shared by fifty candidates discriminates between none of them.
    const own = clean(el.innerText);
    let p = up(el), hops = 0;
    while (p && hops++ < 3) {
      let t = clean(p.innerText);
      if (own && t.startsWith(own)) t = t.slice(own.length);
      t = clean(t);
      if (t.length >= 2) return t.slice(0, 90);
      p = up(p);
    }
    return '';
  };

  // A field's value is what a step typed into it - a password as like as
  // not - and a fingerprint travels to the service and is kept as a
  // baseline. Only a value that is the element's own words is kept: a
  // button input's caption, a button's or an option's markup. The same
  // rule as descriptors.own_value, from the same list.
  const OWN_VALUE_INPUTS = [__OWN_VALUE_INPUTS__];
  const ownValue = (el) => {
    const tag = el.tagName;
    const kind = (el.getAttribute('type') || '').toLowerCase();
    const own = tag === 'BUTTON' || tag === 'OPTION' ||
                (tag === 'INPUT' && OWN_VALUE_INPUTS.indexOf(kind) >= 0);
    return own ? (el.value || '').toString().slice(0, 60) : '';
  };

  const describe = (el) => {
    const r = el.getBoundingClientRect();
    const cs = view(el).getComputedStyle(el);
    const visible = r.width > 0 && r.height > 0 &&
                    cs.visibility !== 'hidden' && cs.display !== 'none' &&
                    parseFloat(cs.opacity || '1') > 0.05 &&
                    framesOf(el).every(isShown);
    const data = {};
    for (const a of el.attributes || []) {
      if (a.name.startsWith('data-') && a.value && a.value.length < 60)
        data[a.name] = a.value;
    }
    const ancestors = [];
    let p = up(el), guard = 0;
    while (p && guard++ < 8) {
      const c = stableClasses(p)[0];
      ancestors.unshift(p.tagName.toLowerCase() + (c ? '.' + c : ''));
      p = up(p);
    }
    let idx = 1, sib = el.previousElementSibling;
    while (sib) { if (sib.tagName === el.tagName) idx++; sib = sib.previousElementSibling; }
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      classes: Array.from(el.classList || []),
      name: el.getAttribute('name') || '',
      type: el.getAttribute('type') || '',
      role: el.getAttribute('role') || '',
      ariaLabel: el.getAttribute('aria-label') || '',
      placeholder: el.getAttribute('placeholder') || '',
      title: el.getAttribute('title') || '',
      alt: el.getAttribute('alt') || '',
      href: (el.getAttribute('href') || '').slice(0, 120),
      value: ownValue(el),
      dataAttrs: data,
      text: clean(el.innerText).slice(0, 160),
      accName: accName(el),
      xpath: pagePath(el),
      cssPath: cssPath(el),
      ancestors: ancestors,
      depth: ancestors.length,
      sibIndex: idx,
      rect: pageRect(el, r),
      // getBoundingClientRect is viewport-relative, so a rect alone cannot be
      // compared across a scroll or across two screen sizes. The scroll offset
      // makes the position absolute; the viewport makes the distance between
      // two positions mean the same thing on a laptop and on a 4K panel.
      scrollX: Math.round(window.scrollX || window.pageXOffset || 0),
      scrollY: Math.round(window.scrollY || window.pageYOffset || 0),
      viewportW: window.innerWidth || 0,
      viewportH: window.innerHeight || 0,
      visible: visible,
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
      // reuse the style object captured above - a second getComputedStyle
      // call here forces a fresh style recalc on every harvested element,
      // which on a page this size costs more than the rest of the harvest
      interactive: INTERACTIVE.has(el.tagName) || !!el.onclick ||
                   ['button','link','textbox','searchbox','tab','menuitem','checkbox']
                     .indexOf(el.getAttribute('role')) >= 0 ||
                   cs.cursor === 'pointer',
      neighborText: neighborText(el)
    };
  };
"""
_DESCRIBE_JS = _DESCRIBE_JS.replace(
    "__OWN_VALUE_INPUTS__", ", ".join("'%s'" % t for t in BUTTON_INPUT_TYPES))


# The durable fragments of a selector's own names, spliced into the presence
# gate and the harvest. One copy, for the reason _DESCRIBE_JS is one copy:
# the gate decides whether a step's family is on the page at all, and the
# harvest keeps that family ahead in its cut, so the two must never disagree
# about what the family is.
_FRAGMENTS_JS = r"""
  const FRAGMENT_MIN = 4;              // shorter than this matches everything
  const GENERIC = new Set(['button','input','link','item','text','icon','list',
                           'title','name','value','label','wrap','wrapper',
                           'container','content','active','hidden','open',
                           'true','false','main','body','page','row','col']);

  const selectorFragments = (sel) => {
    // --- pull identifiers out of either selector language -----------------
    const ids = [];
    const push = (v) => { if (v) ids.push(String(v)); };

    // #id and .class in CSS
    let m;
    const cssIdent = /[#.]([A-Za-z_][\w-]*)/g;
    while ((m = cssIdent.exec(sel)) !== null) push(m[1]);

    // [attr='value'], [attr="value"], [attr*='value']
    const attrVal = /\[[^\]=]+[*^$~|]?=\s*['"]?([^'"\]]+)['"]?\]/g;
    while ((m = attrVal.exec(sel)) !== null) push(m[1]);

    // XPath: @attr='value' and any quoted literal
    const xpAttr = /@[\w-]+\s*=\s*['"]([^'"]+)['"]/g;
    while ((m = xpAttr.exec(sel)) !== null) push(m[1]);

    // --- reduce each identifier to its most durable fragments -------------
    //
    // Two levels, because they answer different questions.
    //
    // The compound fragment ('product-card__title') survives a *version
    // bump*, which is what the presence gate was built for: strip the
    // digits and pd19 and pd21 are the same string.
    //
    // It does not survive a *rename*. Measured: a consent button authored
    // as .cookie-bar__accept-button, re-shipped as .consent-banner__cta,
    // was reported absent - the whole compound was tested as one substring
    // and nothing on the page carried it. The element was right there, and
    // the step stopped being repaired. That is the false 'absent' the gate
    // warns about, and it was the default outcome for every rename.
    //
    // So the component words are tested too. They are weaker evidence on
    // their own, which is the right trade here: a false 'present' costs one
    // wasted heal, a false 'absent' costs the repair forever.
    const frags = [];
    const push_ = (v) => {
      if (!v) return;
      v = v.replace(/^[-_]+|[-_]+$/g, '');
      if (v.length < FRAGMENT_MIN) return;
      if (GENERIC.has(v.toLowerCase())) return;
      if (frags.indexOf(v) < 0) frags.push(v);
    };
    for (const raw of ids) {
      for (const piece of raw.split(/\s+/)) {
        let best = '';
        for (const part of piece.split(/\d+/)) {
          if (part.length > best.length) best = part;
        }
        push_(best);
        // ... and the words inside it, on the separators every naming
        // convention in use here builds compounds out of.
        for (const word of piece.split(/[^A-Za-z]+/)) push_(word);
      }
    }
    return frags;
  };
"""


HARVEST_JS = r"""
(opts) => {
  const MAX = opts.maxCandidates || 900;
  const SKIP = new Set(['SCRIPT','STYLE','LINK','META','HEAD','NOSCRIPT',
                        'TEMPLATE','BR','HR','PATH','G','DEFS','SVG','USE',
                        'SOURCE','TITLE','CIRCLE','RECT','POLYGON','LINE']);
""" + _RESOLVE_JS + _DESCRIBE_JS + _FRAGMENTS_JS + r"""
  const root = opts.scope ? resolveAll(opts.scope)[0] : document.body;
  if (!root) return {error: 'scope-not-found', candidates: []};
  // Ranked selector proposals, most durable first.  The healer validates each
  // against the live DOM and keeps the first that resolves to this element.
  //
  // Each durability tier is offered in both selector languages, with the
  // language the step was *authored* in first. A team standardised on XPath
  // gets XPath repairs back, which is what they will review and maintain -
  // but never at the cost of durability, because tier always wins over
  // language.
  const xpLit = (s) => {
    if (s.indexOf("'") < 0) return "'" + s + "'";
    if (s.indexOf('"') < 0) return '"' + s + '"';
    return null;                       // both quote styles present: skip it
  };

  const proposeSelectors = (el, prefer) => {
    const tiers = [];
    const tag = el.tagName.toLowerCase();
    const cls = stableClasses(el);
    const pair = (css, xp) => tiers.push({css: css, xp: xp});

    if (el.id) {
      const lit = xpLit(el.id);
      pair('#' + esc(el.id), lit ? '//' + tag + '[@id=' + lit + ']' : null);
    }
    for (const a of ['data-testid','data-test','data-qa','data-automation-id','data-omni-type']) {
      const v = el.getAttribute(a);
      if (!v) continue;
      const lit = xpLit(v);
      pair(tag + '[' + a + '="' + v.replace(/"/g, '\\"') + '"]',
           lit ? '//' + tag + '[@' + a + '=' + lit + ']' : null);
    }
    // Component classes before aria-label / name on purpose: an aria-label
    // often carries an *instance* value ("Galaxy Z Flip8"), which would pin a
    // collection locator to a single row; a component class describes the
    // role and stays generic.
    // An XPath class predicate has to AND one contains() per class to be as
    // specific as the CSS form. Offering only the first class would make the
    // XPath side of this tier much looser, and a single-element locator would
    // fail validation and drop to a weaker tier.
    const xpClasses = (t, list) => {
      const parts = [];
      for (const c of list) {
        const lit = xpLit(c);
        if (!lit) return null;
        parts.push('contains(@class,' + lit + ')');
      }
      return parts.length ? '//' + t + '[' + parts.join(' and ') + ']' : null;
    };
    if (cls.length) {
      pair(tag + '.' + cls.slice(0, 3).map(esc).join('.'),
           xpClasses(tag, cls.slice(0, 3)));
      pair(tag + '.' + esc(cls[0]), xpClasses(tag, cls.slice(0, 1)));
      pair('.' + cls.slice(0, 2).map(esc).join('.'),
           xpClasses('*', cls.slice(0, 2)));
    }
    for (const a of ['aria-label', 'name', 'placeholder']) {
      const v = el.getAttribute(a);
      if (!v) continue;
      const lit = xpLit(v);
      pair(tag + '[' + a + '="' + v.replace(/"/g, '\\"') + '"]',
           lit ? '//' + tag + '[@' + a + '=' + lit + ']' : null);
    }
    // Text is a first-class locator in XPath and awkward in CSS, so this tier
    // is XPath-only. Two guards, both learned the hard way:
    //   - keep it short; long copy is edited far more often than a label;
    //   - only bind to text that reads like a *label*, never a *value*. A
    //     price cell healed to //div[normalize-space()='Rs 65,150'] resolves
    //     perfectly today and breaks the next time the price moves, which for
    //     a price monitor is the one thing guaranteed to happen.
    // Rejecting every label with a digit in it was too blunt: it threw away
    // "Add to cart (3)" along with "Rs 65,150", so any control carrying a
    // count or a badge had no text tier at all. The first attempt at a
    // rescue stripped digits from the two *ends* of the label and emitted
    // the remainder as another exact match. Measured, it did neither job:
    //
    //   "Add to cart (3)" -> "Add to cart (3)"  still rejected, digit inside
    //   "Step 2 of 4"     -> "Step of"          would never match anything
    //   "Rs 65,150"       -> "Rs"               ACCEPTED - the exact case
    //                                           the guard above exists to
    //                                           stop, one word wide
    //
    // Two things were wrong. Stripping only at the ends misses an interior
    // count, and an exact match on stripped text cannot bind by
    // construction - the element still has the digits in it. Playwright
    // gets away with the stripped form because its text engine is a
    // substring match; ours was '=' .
    //
    // So: a label with no number in it is used exactly, as before. A label
    // that carries one is only rescued as a contains(), and only when what
    // survives is worth binding to - a contiguous run of the original, two
    // words or more, six letters or more. "Add to cart" clears that; "Rs"
    // and "INR" do not, which is what keeps a price out of a selector.
    //
    // A number in the *middle* of a label still has no text tier, by
    // design rather than by oversight: "Step 2 of 4" leaves "Step of",
    // which is not a run of the original and so could not bind. Claiming
    // that case was fixed is what the previous comment got wrong. These
    // elements fall through to the path tier, as they did before.
    const CURRENCY = /[\u20b9$\u20ac\u00a3\u00a5]/;
    const hasNumber = (t) => /\d/.test(t) || CURRENCY.test(t);
    const stripNumbers = (t) => clean(t
        .replace(/[\u20b9$\u20ac\u00a3\u00a5]/g, ' ')
        .replace(/[\d.,]*\d[\d.,]*/g, ' ')
        .replace(/\(\s*\)|\[\s*\]/g, ' '));
    const label = clean(el.innerText);
    if (label && label.length <= 40 && !hasNumber(label)) {
      const lit = xpLit(label);
      if (lit) pair(null, '//' + tag + '[normalize-space()=' + lit + ']');
    } else if (label && label.length <= 40) {
      const words = stripNumbers(label);
      const letters = (words.match(/[A-Za-z]/g) || []).length;
      // A contiguous run is what makes contains() certain to bind: it is
      // the part of the label the numbers were sitting beside, not a
      // sentence reassembled out of the pieces either side of them.
      if (label.indexOf(words) >= 0 && words.split(' ').length >= 2 &&
          letters >= 6) {
        const lit = xpLit(words);
        if (lit) pair(null, '//' + tag +
                            '[contains(normalize-space(),' + lit + ')]');
      }
    }
    pair(cssPath(el), absXPath(el));

    const asXp = (v) => v ? 'xpath=' + v : null;
    const out = [];
    for (const t of tiers) {
      const ordered = prefer === 'xpath'
          ? [asXp(t.xp), t.css]
          : [t.css, asXp(t.xp)];
      for (const s of ordered) if (s) out.push(s);
    }
    return out.filter((v, i, arr) => v && arr.indexOf(v) === i);
  };

  // ---- collect ------------------------------------------------------------
  // Cheap prefilter first: anything the target could plausibly be.  The
  // expensive similarity maths happens in Python on the survivors only.
  const wantTag = (opts.tagHint || '').toUpperCase();
  const needText = !!opts.needText;
  // The broken selector's own name fragments. An element still carrying
  // one is the family the step names, and on a page with more than MAX
  // candidates it must not be what the cut throws away - 35% of real pages
  // were over it. Tested where the presence gate tests them, as substrings
  // of the same attributes.
  // A chain's hosts and frames are the way to the element, not its family.
  const frags = opts.selector
      ? selectorFragments(lastStep(String(opts.selector))) : [];
  const carries = (el) => {
    if (!frags.length) return false;
    const hay = [el.id, el.getAttribute('class'), el.getAttribute('name'),
                 el.getAttribute('aria-label'), el.getAttribute('data-testid'),
                 el.getAttribute('href')].filter(Boolean).join(' ');
    for (const f of frags) if (hay.indexOf(f) >= 0) return true;
    return false;
  };
  // Everything a step can reach: the page's own elements first, in the
  // order they always came in, then what each open shadow root and each
  // frame the page can read holds - every element with the hosts and
  // frames on its way, which are the steps of a chain to it. Until this,
  // a control inside a web component or a frame was never a candidate.
  const reach = [];
  for (const [scope, via] of allScopes(root)) {
    for (const el of scope.querySelectorAll('*')) reach.push([el, via]);
  }
  const pool = [];
  for (const [el, via] of reach) {
    if (SKIP.has(el.tagName)) continue;
    if (el.childElementCount > 40) continue;          // layout shells
    const r = el.getBoundingClientRect();
    const isInteractive = INTERACTIVE.has(el.tagName) ||
                          !!el.getAttribute('role') || !!el.onclick;
    const hasText = clean(el.innerText).length > 0;
    const onScreen = r.width > 0 && r.height > 0;
    if (!onScreen && !isInteractive) continue;
    if (needText && !hasText && !isInteractive) continue;
    // priority drives which elements survive the MAX cut
    let pri = 0;
    if (onScreen) pri += 4;
    if (isInteractive) pri += 3;
    if (hasText) pri += 1;
    if (wantTag && el.tagName === wantTag) pri += 3;
    if (carries(el)) pri += 3;
    if (el.id) pri += 1;
    pool.push([pri, el, via]);
  }
  pool.sort((a, b) => b[0] - a[0]);

  // A step into a shadow root or a frame names its host or frame by a
  // selector that matches nothing else where that element lives. A custom
  // element's tag comes first - it is the component's name, which outlives
  // its attributes. A frame's id or name comes before its bare tag, which
  // stops being unique the day a chat widget adds a second frame. A
  // positional path is the last resort for either. Asked once per host.
  const stepFor = new Map();
  const stepTo = (holder) => {
    if (stepFor.has(holder)) return stepFor.get(holder);
    const where = holder.getRootNode();
    const tag = holder.tagName.toLowerCase();
    // cssPath stops at an id, and a path that is only an id is not one.
    const path = cssPath(holder);
    const positional = path.indexOf(':nth-of-type(') >= 0 ? path : '';
    const durable = proposeSelectors(holder, 'css').filter(
        (s) => s && !s.startsWith('xpath=') && s !== positional);
    const custom = tag.indexOf('-') > 0;
    const frame = tag === 'iframe' || tag === 'frame';
    const tries = (custom ? [tag] : []).concat(durable, frame ? [tag] : [],
                                               [positional]);
    let got = '';
    for (const s of tries) {
      if (!s) continue;
      try {
        if (where.querySelectorAll(s).length === 1) { got = s; break; }
      } catch (e) { /* a proposal this engine cannot parse */ }
    }
    stepFor.set(holder, got);
    return got;
  };
  // An element behind a boundary is proposed through the hosts and frames
  // on its way, each of its own proposals as the last step. XPath cannot
  // see into a shadow root, so an element in one is proposed in CSS only;
  // a frame's document is a document like any other.
  const reachBy = (el, via, own) => {
    if (!via.length) return own;
    const lead = [];
    for (const holder of via) {
      const s = stepTo(holder);
      if (!s) return [];             // nothing names that host: no way in
      lead.push(s);
    }
    const shadow = el.getRootNode() !== el.ownerDocument;
    return own.filter((s) => !(shadow && s.startsWith('xpath=')))
              .map((s) => lead.join(' >>> ') + ' >>> ' + s);
  };

  const candidates = [];
  const kept = [];
  for (const [, el, via] of pool.slice(0, MAX)) {
    try {
      const d = describe(el);
      d.selectors = reachBy(el, via,
                            proposeSelectors(el, opts.preferSyntax || 'css'));
      candidates.push(d);
      kept.push(el);
    } catch (e) { /* one bad node must never abort the harvest */ }
  }
  // Which element each candidate is, kept in the page. The service answers
  // with an index and the bot binds a selector the page proposed for that
  // element; this is how the bot proves the selector bound *that* element
  // rather than merely something it matches (SAME_JS). Replaced by the
  // next harvest, and gone with the page.
  const harvest = 'h' + Date.now().toString(36) +
                  Math.random().toString(36).slice(2, 8);
  try { window.__bisHarvest = {id: harvest, els: kept}; } catch (e) {}
  return {
    url: location.href,
    title: document.title,
    scanned: reach.length,
    pooled: pool.length,
    harvest: harvest,
    candidates: candidates
  };
}
"""


PROBE_JS = r"""
(selector) => {
  // Resolve a selector the way the healer intends to use it and report back
  // enough for the validation gate to make a decision.
""" + _RESOLVE_JS + r"""
  let nodes = [];
  try {
    nodes = resolveAll(selector);
  } catch (e) {
    return {ok: false, error: 'invalid-selector: ' + e.message, count: 0};
  }
  if (!nodes.length) return {ok: false, error: 'no-match', count: 0};
  const visibleNodes = nodes.filter(isShown);
  const el = visibleNodes[0] || nodes[0];
  const r = el.getBoundingClientRect();
  return {
    ok: true,
    count: nodes.length,
    visibleCount: visibleNodes.length,
    firstVisibleIndex: visibleNodes.length ? nodes.indexOf(visibleNodes[0]) : -1,
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 200),
    enabled: !el.disabled,
    rect: {x: r.x, y: r.y, width: r.width, height: r.height}
  };
}
"""


# The baseline fingerprinter: resolve one element, then hand it to the same
# describe() every candidate goes through.
DESCRIBE_ONE_JS = r"""
(opts) => {
  const selector = opts.selector;
  const wantIndex = opts.index || 0;
""" + _RESOLVE_JS + _DESCRIBE_JS + r"""
  // Compact fingerprint of a single element, captured right after a step
  // succeeds.  This becomes the *baseline* in healing memory: the structural
  // and geometric evidence that turns a future cold heal into a warm one.
  let el = null;
  let nodes = [];
  try {
    nodes = resolveAll(selector);
    // Take the node the step actually used. Re-resolving to "the first
    // visible match" can fingerprint a different element entirely - a
    // selector that matches an off-screen duplicate would store that
    // duplicate as the baseline and poison the structure and geometry
    // signals on every later heal.
    el = nodes[wantIndex] || nodes.find(n => {
      const b = n.getBoundingClientRect();
      return b.width > 0 && b.height > 0;
    }) || nodes[0];
  } catch (e) { return null; }
  if (!el) return null;
  return describe(el);
}
"""


# The bind check: is the element a repair's selector bound the one the
# service chose? Asked of the elements HARVEST_JS kept, by identity.
SAME_JS = r"""
(opts) => {
  // `els` is what the bound selector matched: the one element, or the set
  // for a collection step, which must hold the element chosen.
  //
  // Unknown is an answer too, and the bot binds as it always did on it:
  // no record of this harvest (the page navigated, an older bundle made
  // it), or the chosen element gone from the document - a re-render
  // replaced the node, and identity can no longer be asked about.
""" + _RESOLVE_JS + r"""
  const h = window.__bisHarvest;
  if (!h || !opts || h.id !== opts.harvest) {
    return {known: false, reason: 'no-harvest'};
  }
  const want = h.els[opts.index];
  if (!want) return {known: false, reason: 'no-candidate'};
  if (!want.isConnected) return {known: false, reason: 'replaced'};
  // An element inside a frame cannot be handed to this page's JavaScript
  // from outside the frame, so for a chain the bot sends the selector it
  // bound instead, and the page resolves it as the bot did: the first
  // match on screen - or, for a set, every one.
  let got = opts.els || [];
  if (!opts.els && opts.selector) {
    try {
      got = resolveAll(String(opts.selector)).filter(isShown);
    } catch (e) {
      return {known: false, reason: 'bad-selector'};
    }
    if (!opts.all) got = got.slice(0, 1);
  }
  for (const el of got) {
    if (el === want) return {known: true, same: true};
  }
  const holds = got.some((el) => el && el.contains && el.contains(want));
  return {known: true, same: false, reason: holds ? 'container' : 'other'};
}
"""


PRESENCE_JS = r"""
(opts) => {
  // Is anything of this element's component family on the page at all?
  //
  // This separates the two cases a healer cannot otherwise tell apart: a
  // selector that broke because the site renamed something, and a selector
  // that failed because the element simply is not here this session. They
  // look identical - both match zero nodes - but they want opposite
  // responses. A renamed element should be repaired; an absent optional one
  // should be skipped without spending a harvest and an adjudication on it.
  //
  // The test is deliberately loose. Identifiers are split on digit runs and
  // the longest surviving fragment is matched as a substring, so
  // 'pd19-product-card__item' still recognises 'pd21-product-card__item' as
  // family - the version bump is exactly the redesign this system exists to
  // survive. It is a pre-filter, not a score: nothing here ranks anything.
  //
  // When in doubt it answers true. A false 'present' costs one wasted heal;
  // a false 'absent' silently stops repairing a step forever, which is much
  // worse and much harder to notice.
  const sel = (opts && opts.selector) || '';
  if (!sel) return {present: true, reason: 'no-selector', fragments: []};
""" + _RESOLVE_JS + _FRAGMENTS_JS + r"""
  // The family of the element the step names; a host on the way is not it.
  const frags = selectorFragments(lastStep(sel));

  if (!frags.length) {
    // Nothing distinctive enough to test - do not gate on a guess.
    return {present: true, reason: 'no-usable-fragment', fragments: []};
  }

  // --- does anything on the page carry one of them? -----------------------
  // Behind its shadow roots and inside its frames as well: a control that
  // moved into a web component was renamed, not taken away.
  const scopes = allScopes(document).map((pair) => pair[0]);
  const esc = (s) => s.replace(/["\\]/g, '\\$&');
  let matched = 0;
  const hits = [];
  for (const f of frags) {
    const q = ['class', 'id', 'name', 'aria-label', 'data-testid', 'href']
        .map((a) => '[' + a + '*="' + esc(f) + '"]').join(',');
    let n = 0;
    try {
      for (const scope of scopes) n += scope.querySelectorAll(q).length;
    } catch (e) {
      return {present: true, reason: 'query-failed', fragments: frags};
    }
    if (n > 0) { matched += n; hits.push(f); }
  }

  if (matched > 0) {
    return {present: true, reason: 'family-found', fragments: frags,
            hits: hits, matches: matched};
  }

  // Nothing in the attributes. Before concluding the element was never
  // here, look at what the page *says*. A rename touches class names and
  // leaves the copy alone - that is the whole premise of the neighbour-text
  // signal, and it applies just as well to this question. A consent banner
  // renamed beyond recognition still has the word "cookies" in its own
  // sentence, and a banner that is genuinely not on the page does not.
  //
  // Deliberately last, and deliberately only as a tie-breaker towards
  // 'present': it is looser evidence, and the cost of being wrong in this
  // direction is one wasted heal rather than a step that never heals again.
  try {
    // A shadow root has no innerText of its own; its text content will do
    // for a tie-breaker towards 'present'.
    const said = (s) => (s === document
        ? (document.body && document.body.innerText)
        : (s.innerText !== undefined ? s.innerText : s.textContent)) || '';
    const copy = scopes.map(said).join(' ').toLowerCase();
    for (const f of frags) {
      if (f.length >= FRAGMENT_MIN && copy.indexOf(f.toLowerCase()) >= 0) {
        return {present: true, reason: 'family-in-page-text', fragments: frags,
                hits: [f], matches: 0};
      }
    }
  } catch (e) { /* no body yet - fall through to absent */ }

  return {present: false, reason: 'absent', fragments: frags, hits: hits,
          matches: matched};
}
"""
