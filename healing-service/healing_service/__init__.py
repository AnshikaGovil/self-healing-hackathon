"""The healing service - the GPU-bound half of the agent, behind an API.

Run it on one accelerated host and point any number of bots at it:

    python -m healing_service --host 0.0.0.0 --port 8077
"""

import os as _os
import sys as _sys

# Used straight from a checkout, bis-core sits beside this package rather
# than in site-packages. Appended, never inserted: a properly installed
# bis_core must always win over the folder next door.
_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sibling = _os.path.join(_os.path.dirname(_pkg_root), "bis-core")
if _os.path.isdir(_sibling) and _sibling not in _sys.path:
    _sys.path.append(_sibling)


from bis_core import runtime as _runtime

from .embeddings import default_encoder as _default_encoder
from .inference import AdjudicatorModel as _AdjudicatorModel
from .checker import get as _checker

# Importing this package is what gives a process its accelerator half. The
# core declares the slots; filling them here keeps the dependency running
# one way, so bis-core never has to know this package exists.
_runtime.register_encoder(_default_encoder)
_runtime.register_model(_AdjudicatorModel)
# Registered, not loaded. The checker is a second set of weights and the
# config decides whether to ask for it; a deployment that leaves
# use_checker off never imports torch for it.
_runtime.register_checker(_checker)

from .app import create_app

__all__ = ["create_app"]
