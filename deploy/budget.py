"""What the judge's GPU has cost this month, and when to stop serving.

Modal bills a container per second for its GPU, its CPU and its memory, from
the moment it boots. The ledger only sees the seconds it runs, so it charges
the GPU's rate plus a margin for the rest: it reaches the cap before the
account's credit does, never after. Pure standard library, so the container
and the offline tests import the same file.
"""

from __future__ import annotations

import time
from typing import Optional

# Dollars a second - Modal's published rates, 2026-10: a GPU's card alone,
# and "CPU" for the front door's whole container, one core ($0.0473 an
# hour) and 2 GiB ($0.008 a GiB).
RATES = {"T4": 0.000164, "L4": 0.000222, "CPU": 0.0000176}
# A card with no rate here is charged as an H100, the dearest Modal sells,
# so a GPU changed in one file and not the other can only stop early.
UNKNOWN_RATE = 0.0011
# CPU and memory on top of the card: Modal billed a T4 container about
# $0.69 an hour all in, against $0.59 for the GPU.
OVERHEAD = 1.3
# Modal bills from a container's boot, and the ledger starts only when the
# container's own code does - 15-25 s later, measured. Charged once a
# container, at its start.
BOOT_S = 30.0


def charge(spent: float, seconds: float, gpu: str) -> float:
    """``spent`` plus ``seconds`` of a container holding this GPU."""
    rate = RATES.get(gpu, UNKNOWN_RATE)
    return spent + max(0.0, seconds) * rate * OVERHEAD


def over(spent: float, cap: float) -> bool:
    """Has the spend reached the cap? A cap of 0 or less refuses all."""
    return spent >= cap


def month_key(now: Optional[float] = None) -> str:
    """The ledger's key for a calendar month (UTC): Modal's credit is
    monthly, so a new month starts a new ledger."""
    return time.strftime("spent-%Y-%m",
                         time.gmtime(time.time() if now is None else now))
