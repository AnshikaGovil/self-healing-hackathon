"""What the AI's token odds are worth, as one probability of being right.

The probability the AI puts on the index it picks is a far better number
than the one its reply writes (AUC 0.93 against 0.59), and at >= 0.95 it
holds: picks there were 98.5% right. Below that it runs optimistic - picks
at a mean 0.66 were right 53% of the time, at 0.92 86% - and it says
nothing about the step itself: a placeholder description, a collection, a
pick the ranker put last. So the decision's confidence is this calibrator's
output, fitted on held-out steps replayed through the service's own token
read (scripts/fit_calibration.py):

    logistic over the odds and the step's facts, then an isotonic map

Shipped as JSON beside this file and evaluated here in the standard
library, so bis-core stays pure. It belongs to one set of AI weights,
recorded as `model_sha256`, and has to be fitted again when they change -
as the checker's bands do.

`bars` say, per group of steps and per target precision, the confidence a
repair must clear for the 95% lower confidence bound on precision to reach
that target on the fitting data. A group with too little evidence has no
bar, and its repairs go to review.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional

log = logging.getLogger("bis.calibration")

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration.json")
VERSION = 1

GROUPS = ("described", "undescribed", "placeholder")


def _logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


# What each feature is, from the facts of one decision. A fitted file names
# the ones it uses; one naming anything else is refused.
_FEATURES = {
    "bias": lambda f: 1.0,
    "logit_p": lambda f: _logit(f["p"]),
    "logit_runner": lambda f: _logit(f["runner"]),
    "logit_none": lambda f: _logit(f["none"]),
    "first": lambda f: 1.0 if f["rank"] == 0 else 0.0,
    "undescribed": lambda f: 1.0 if f["group"] == "undescribed" else 0.0,
    "placeholder": lambda f: 1.0 if f["group"] == "placeholder" else 0.0,
    "collect": lambda f: 1.0 if f["collect"] else 0.0,
}


class Calibrator:
    """A fitted calibration, evaluated without anything outside stdlib."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        if spec.get("version") != VERSION:
            raise ValueError("calibration version %r, this reads %d"
                             % (spec.get("version"), VERSION))
        self.features = list(spec["features"])
        self.weights = [float(w) for w in spec["weights"]]
        unknown = [f for f in self.features if f not in _FEATURES]
        if unknown or len(self.features) != len(self.weights):
            raise ValueError("unknown features %s or a weight count that "
                             "does not match" % unknown)
        knots = spec["isotonic"]
        self.xs = [float(x) for x in knots["x"]]
        self.ys = [float(y) for y in knots["y"]]
        if (len(self.xs) < 2 or len(self.xs) != len(self.ys)
                or any(b < a for a, b in zip(self.xs, self.xs[1:]))
                or any(b < a for a, b in zip(self.ys, self.ys[1:]))):
            raise ValueError("isotonic knots must be two or more, sorted, "
                             "and non-decreasing")
        self.bars = {str(t): dict(g) for t, g in
                     (spec.get("bars") or {}).items()}
        self.model_sha256 = str(spec.get("model_sha256") or "")
        # The odds were read after one prompt (build_prompt's version);
        # every file before the field was fitted on 2.
        self.prompt_version = int(spec.get("prompt_version") or 2)
        self.measured = spec.get("measured") or {}

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> Optional["Calibrator"]:
        """The calibration at ``path``, or None - never raises."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return cls(json.load(handle))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("ignoring the calibration in %s (%s) - the AI's raw "
                        "odds are used", path, str(exc)[:120])
            return None

    def _isotonic(self, s: float) -> float:
        xs, ys = self.xs, self.ys
        if s <= xs[0]:
            return ys[0]
        if s >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if s <= xs[i]:
                span = xs[i] - xs[i - 1]
                if span <= 0.0:
                    return ys[i]
                return ys[i - 1] + (s - xs[i - 1]) / span * (ys[i] - ys[i - 1])
        return ys[-1]

    def probability(self, p: float, runner: float, none: float, rank: int,
                    group: str, collect: bool) -> float:
        """P(the pick is right), from the token read and the step's facts.

        ``p`` is the AI's probability for its pick, ``runner`` the most it
        put on anything else, ``none`` what it put on "none"; ``rank`` is
        the pick's place in the ranker's shortlist; ``group`` is how the
        step was described to the models (GROUPS).
        """
        facts = {"p": p, "runner": runner, "none": none, "rank": rank,
                 "group": group, "collect": bool(collect)}
        z = sum(w * _FEATURES[name](facts)
                for name, w in zip(self.features, self.weights))
        z = max(-40.0, min(40.0, z))
        return max(0.0, min(1.0, self._isotonic(1.0 / (1.0 + math.exp(-z)))))

    def bar(self, target: float, group: str) -> Optional[float]:
        """The confidence a repair in ``group`` must clear for ``target``.

        None when the fitting data could not show that precision for the
        group at any bar - and then nothing in it applies on its own.
        """
        bars = self.bars.get(str(target))
        if bars is None:
            for key, value in self.bars.items():
                try:
                    if abs(float(key) - float(target)) < 1e-9:
                        bars = value
                        break
                except ValueError:
                    continue
        if not bars or bars.get(group) is None:
            return None
        return float(bars[group])
