# bis-core

The half of the Bot Immune System that both other packages need.

It holds the element and locator descriptors, the seven deterministic
ranking signals, the confidence-band policy, and the adjudication wire
contract. The
RPA toolkit uses it to pre-rank candidates before it posts anything; the
healing service uses it to re-rank them and to speak the same protocol back.

It is a library. It never opens a browser and it never starts a server.

## Why it is separate

The bot and the service run the *same* scoring code. That is not incidental -
two invariants depend on it:

* the confidence bands are defined once, in `HealingConfig.decide()`, so the
  engine that reports a band and the agent that enforces it cannot drift apart
* a step authored in CSS and the same step authored in XPath must score
  identically
* the two client shapes decode to the same descriptor. The full agent runs
  `from_js` in the bot and posts a serialised `ElementDescriptor`; `bis.py`
  has no descriptor class and posts the harvester's own output, which names
  the same facts differently. `ElementDescriptor.coerce()` is the one place
  that reconciles them, and it exists because the strict decoder silently
  dropped every unrecognised key - a thin client's candidates arrived with no
  id, no type, no aria-label and no data attributes, and the request still
  succeeded
* `page_scope()` normalises a URL into a key. The toolkit uses it for its
  memory file and the service uses it for the baselines it keeps on behalf of
  thin clients; two implementations of "the same page" would be two sets of
  keys that never meet

Copying these modules into both deliverables would turn each of those into a
rule someone has to remember. Sharing one package makes them structural.

## Install

```bash
pip install -e .
```

Both deliverables install this, each into its own virtual environment. It is
the only thing they share.

**No third-party dependencies at all**, and a test enforces it, so a bot can
rank candidates on a machine with nothing else installed on it.

The sentence encoder and the GGUF runtime are not here. They live in
`healing-service`, because that is the only host that loads a model, and they
plug into `runtime.py` when that package is imported. A process with nothing
registered simply has no local accelerator half - which is true of every bot
in the fleet, and is why they call the service over HTTP instead.

Used straight from a checkout, both sibling packages find this folder on their
own; no install is required to run the tests or the demos.

## Layout

| module | what it is |
| --- | --- |
| `config.py` | every tunable, and the confidence-band policy |
| `descriptors.py` | `LocatorSpec` / `ElementDescriptor`, selector-syntax helpers, `page_scope` |
| `similarity.py` | the seven signals, selector mining, `HealTarget` |
| `adjudicator.py` | the wire contract plus `LocalAdjudicator` / `RemoteAdjudicator` |
| `js_harvest.py` | the in-page JavaScript, served to thin clients by the service |
| `runtime.py` | the two slots the accelerator half plugs into |

## Tests

```bash
python tests/test_similarity.py     # 63 - ranking signals, CSS + XPath, the JS, chains
```

Offline by design: no browser, no service, no model, no network.
