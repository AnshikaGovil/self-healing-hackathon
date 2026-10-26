"""The fine-tuned checker: is this element the one the step meant?

A LoRA fine-tune of Laya (convaiinnovations/laya, Apache-2.0) trained on
real harvested shortlists - see scripts/harvest_real.py,
scripts/build_checker_from_llm.py and scripts/laya_checker.py. It reads
the step's intent and one candidate's markup and returns a calibrated
probability that the candidate is the element the intent names.

Measured on 4,000 shortlists from 276 sites held out of training (gate
AUC - how well a score separates a right pick from a wrong one):

    step described as          ranker   base Laya   this checker
    a real description          0.772     0.649       0.982
    no description (selector)   0.801     0.600       0.974
    a placeholder ("the field") 0.801     0.547       0.713

A placeholder hides the broken selector from it and names nothing in its
place, so the adjudicator never shows it one: it is shown the selector
instead, as for a step with no description.

`score` judges one element (the "gate" mode); `score_all` scores a whole
shortlist in one pass, which is how it gives the first opinion
(invariant 39).

Everything the model is shown is built HERE - the prompt, the element
text, the network. scripts/laya_checker.py imports these rather than
keeping a copy: a checker served a different prompt than it was trained
on reports numbers that describe some other model.

Importing this module does not import torch. Nothing loads until the
first score, and a deployment that never sets use_checker never pays for
the weights (~1 GB with the base model).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Optional

log = logging.getLogger("bis.service.checker")

DEFAULT_PATH = os.path.join("models", "laya-checker")
BASE_ID = "convaiinnovations/laya"
MAX_LEN, HEAD_MAX_LEN = 512, 192

_WS = re.compile(r"\s+")
_GEOM = re.compile(r"\s+at\(-?\d+,-?\d+\)\s+\d+x\d+")
_INDEX = re.compile(r"^\[\d+\]\s*")


# ------------------------------------------------------------------ prompt
def element_text(compact: str) -> str:
    """A candidate's markup without its shortlist index or pixel geometry.

    The index is where the ranker put it, which would hand the model the
    ranker's opinion for free; geometry is viewport-specific noise.
    """
    return _GEOM.sub("", _INDEX.sub("", compact)).strip()


def _tidy(text, limit):
    return _WS.sub(" ", str(text or "")).strip()[:limit]


def prompt(intent: str, element: str):
    """The (state, question) pair for one check, exactly as trained."""
    intent, element = _tidy(intent, 120), _tidy(element, 320)
    ins = ("Is this element the one described as '%s'? Answer yes only if "
           "this specific element is what the description names. The "
           "element is: %s" % (intent, element))
    state = {"task": "Verify an automation step's repair.",
             "description": intent, "candidate": element}
    return state, {"t": "noul", "ins": ins, "crit": None}


# ------------------------------------------------------------------- model
def make_net(decision_model):
    """Base Laya's noul logit plus a zero-initialised marker-difference term.

        logit = (s_true - s_false) + w . LN(m_true - m_false) + b

    w and b start at zero, so before training this IS the base model.
    Defined inside a function so importing the module stays torch-free.
    """
    import torch
    import torch.nn as nn

    class CheckerNet(nn.Module):
        def __init__(self, dm):
            super().__init__()
            self.dm = dm
            d = dm.encoder.config.hidden_size
            # Normalised first: the marker difference is raw hidden state,
            # and unnormalised it swamped the base logit within 24 updates.
            self.norm = nn.LayerNorm(d)
            self.lin = nn.Linear(d, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

        def forward(self, ids, att, mpos):
            dm = self.dm
            h = dm.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            qtype = torch.full((ids.shape[0],), 2, dtype=torch.long,
                               device=ids.device)
            h = h + dm.type_emb(qtype)[:, None, :]
            pad = ~att.bool()
            for layer in dm.head.layers:
                h = layer(h, src_key_padding_mask=pad)
            m = torch.gather(h, 1, mpos[:, :, None].expand(-1, -1, h.size(-1)))
            s = dm.scorer(m).squeeze(-1).float()
            diff = (m[:, 1] - m[:, 0]).float()
            return (s[:, 1] - s[:, 0]) + self.lin(self.norm(diff)).squeeze(-1)

    return CheckerNet(decision_model)


def add_lora(net, r: int, alpha: int):
    """LoRA on every attention and MLP projection of the encoder."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                     target_modules=["Wqkv", "Wo", "Wi"], bias="none")
    net.dm.encoder = get_peft_model(net.dm.encoder, cfg)
    return net


# ----------------------------------------------------------------- serving
class Checker:
    """Loads the fine-tune lazily and scores elements. Never raises.

    `score` and `score_all` return calibrated probabilities, or None for
    "no opinion" - a missing package, missing weights, a failed load or a
    failed call. None must leave the caller's decision exactly as it
    would have been without a checker.
    """

    def __init__(self, path: str = DEFAULT_PATH, device: str = "") -> None:
        self.path = path or DEFAULT_PATH
        self.device = device or "cpu"
        self._net = None
        self._tok = None
        self._platt = (1.0, 0.0)
        self._lock = threading.Lock()
        self._load_error = ""
        self.typical_latency_s = 0.0
        self._seen = 0
        # A shortlist costs eight rows, not one - ~6s on a laptop CPU
        # against ~0.85s for a single check - so it is budgeted apart.
        self.typical_shortlist_latency_s = 0.0
        self._shortlists = 0

    @property
    def available(self) -> bool:
        return self._net is not None or not self._load_error

    @property
    def loaded(self) -> bool:
        return self._net is not None

    def describe(self) -> dict:
        return {"loaded": self.loaded, "path": self.path,
                "device": self.device, "error": self._load_error,
                "typical_latency_s": round(self.typical_latency_s, 2),
                "typical_shortlist_latency_s":
                    round(self.typical_shortlist_latency_s, 2)}

    def _load(self):
        if self._net is not None or self._load_error:
            return self._net
        with self._lock:
            if self._net is not None or self._load_error:
                return self._net
            try:
                # transformers probes for TensorFlow at import and its
                # abseil runtime can deadlock model construction; and a
                # healing host must never reach out mid-run to resolve a
                # revision, so the base weights come from the local cache.
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                import laya
                from safetensors.torch import load_file
                started = time.time()
                meta = json.load(open(os.path.join(self.path,
                                                   "checker.json")))
                agent = laya.load(meta.get("base", BASE_ID), device="cpu")
                net = add_lora(make_net(agent.model), meta["lora_r"],
                               meta["lora_alpha"])
                missing = net.load_state_dict(
                    load_file(os.path.join(self.path,
                                           "checker.safetensors")),
                    strict=False)
                if missing.unexpected_keys:
                    raise ValueError("checkpoint does not fit the network "
                                     "(%s ...)" % missing.unexpected_keys[0])
                self._tok = agent.tok
                self._platt = tuple(meta["platt"])
                self._net = net.to(self.device).eval()
                log.info("checker ready (%s on %s) in %.1fs", self.path,
                         self.device, time.time() - started)
            except Exception as exc:
                self._load_error = str(exc)[:160]
                log.warning("no checker available (%s) - repairs are "
                            "decided without it", self._load_error)
        return self._net

    def warmup(self) -> bool:
        return self._load() is not None

    def score(self, intent: str, compact: str) -> Optional[float]:
        """P(this element is the one the step meant), calibrated, or None."""
        net = self._load()
        if net is None:
            return None
        element = element_text(compact)
        if not element or not intent:
            return None
        started = time.time()
        try:
            import torch
            from laya.common import build_sequence
            state, q = prompt(intent, element)
            ids, markers = build_sequence(self._tok, state, q, MAX_LEN,
                                          HEAD_MAX_LEN)
            if len(markers) != 2:
                return None
            dev = torch.device(self.device)
            with torch.inference_mode():
                logit = float(net(torch.tensor([ids], device=dev),
                                  torch.ones((1, len(ids)), dtype=torch.long,
                                             device=dev),
                                  torch.tensor([markers], device=dev))[0])
            a, b = self._platt
            value = 1.0 / (1.0 + pow(2.718281828459045, -(a * logit + b)))
        except Exception as exc:
            log.warning("checker failed (%s) - no opinion", str(exc)[:90])
            return None
        took = time.time() - started
        self._seen += 1
        # A running mean, so the adjudicator can budget for it the way it
        # budgets for the primary model.
        self.typical_latency_s += (took - self.typical_latency_s) / self._seen
        return max(0.0, min(1.0, value))

    def score_all(self, intent: str, compacts) -> Optional[list]:
        """P(each candidate is the one the step meant), in order, or None.

        One forward pass over the whole shortlist, padded the way the
        trainer pads, so the first opinion costs one call rather than
        eight. All or nothing: a candidate that cannot be scored would
        leave a hole the bands cannot reason about, so any failure is
        "no opinion" for the shortlist and the caller asks the AI first.
        """
        net = self._load()
        if net is None or not intent or not compacts:
            return None
        started = time.time()
        try:
            import torch
            from laya.common import build_sequence
            seqs = []
            for compact in compacts:
                element = element_text(compact)
                if not element:
                    return None
                state, q = prompt(intent, element)
                ids, markers = build_sequence(self._tok, state, q, MAX_LEN,
                                              HEAD_MAX_LEN)
                if len(markers) != 2:
                    return None
                seqs.append((ids, markers))
            width = max(len(ids) for ids, _ in seqs)
            pad = self._tok.pad_token_id or 0
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for k, (row, _) in enumerate(seqs):
                ids[k, :len(row)] = torch.tensor(row)
                att[k, :len(row)] = 1
            dev = torch.device(self.device)
            with torch.inference_mode():
                logits = net(ids.to(dev), att.to(dev),
                             torch.tensor([m for _, m in seqs],
                                          device=dev)).float().cpu().tolist()
            a, b = self._platt
            out = [max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-(a * x + b)))))
                   for x in logits]
        except Exception as exc:
            log.warning("checker failed on a shortlist (%s) - no opinion",
                        str(exc)[:90])
            return None
        took = time.time() - started
        self._shortlists += 1
        self.typical_shortlist_latency_s += (
            (took - self.typical_shortlist_latency_s) / self._shortlists)
        return out


_INSTANCE: Optional[Checker] = None
_INSTANCE_LOCK = threading.Lock()


def get(path: str = DEFAULT_PATH, device: str = "") -> Checker:
    """One checker per process - it owns about a gigabyte of weights."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = Checker(path, device)
        return _INSTANCE
