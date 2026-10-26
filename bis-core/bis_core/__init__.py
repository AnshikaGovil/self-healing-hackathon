"""Shared core of the Bot Immune System.

Everything both sides of the network seam need: the locator and element
descriptors, the six deterministic ranking signals, the confidence-band
policy, and the adjudication wire contract.

Nothing here touches a browser, and nothing here starts a server. The RPA
toolkit imports it to pre-rank candidates; the healing service imports it to
re-rank them and to speak the same wire protocol back.

``embeddings`` and ``inference`` are the optional local runtime - install
``bis-core[local]`` for them. Without those extras the adjudicator degrades
to the deterministic signals rather than failing, which is the behaviour the
whole system is built around.
"""

from .config import HealingConfig, ScoringWeights
from .descriptors import ElementDescriptor, LocatorSpec
from .similarity import HealTarget, score_candidates
from .adjudicator import (PROTOCOL_VERSION, AdjudicationDecision,
                          AdjudicationRequest, LocalAdjudicator,
                          RemoteAdjudicator, build_adjudicator)

__all__ = [
    "HealingConfig", "ScoringWeights",
    "ElementDescriptor", "LocatorSpec",
    "HealTarget", "score_candidates",
    "PROTOCOL_VERSION", "AdjudicationDecision", "AdjudicationRequest",
    "LocalAdjudicator", "RemoteAdjudicator", "build_adjudicator",
]
