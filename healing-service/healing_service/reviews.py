"""Reviewers' verdicts on repairs the service held for review.

Why keep them
-------------
A repair below the bar used to be a failed step and nothing more. With a
review hook (``bis.configure(review_hook=...)``) a person decides it, and
what they decide is the one label this system can get from production. The
training corpus is mutated selectors on pages that never changed; a person
saying "yes, that is the field" or "no, that is the search box" on a real
break is what a calibration fitted to production needs, and what the
confidence bands can be checked against.

So every verdict is kept, approved or rejected, with what the models were
shown - the shortlist in their order and which entry was offered - as one
JSON line. Append-only, under the same run-memory rule as the other stores:
learned run state, not part of the deliverable.

No page, no DOM, no typed values: the shortlist lines are the same compact
fingerprints the AI is shown on every adjudication.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict

log = logging.getLogger("bis.service.reviews")

DEFAULT_PATH = os.path.join("run-memory", "reviews.jsonl")


class ReviewStore:
    """Review verdicts, one JSON line each, appended and synced."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._count = self._existing()

    def _existing(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            log.warning("could not read %s (%s) - counting from zero",
                        self.path, exc)
            return 0

    def record(self, row: Dict[str, Any]) -> bool:
        """Append one verdict. Returns whether it was written; never raises.

        Synced before it counts. This is a person's time, which a crash
        must not cost - and this machine has blue-screened mid-run before.
        """
        line = json.dumps(row, sort_keys=True)
        with self._lock:
            try:
                folder = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(folder, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                log.warning("could not keep a review verdict in %s (%s)",
                            self.path, exc)
                return False
            self._count += 1
        return True

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"labels": self._count}
