"""Compatibility shim for fingerprints module.

Use 'from infrastructure.fingerprints import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.fingerprints is deprecated, use infrastructure.fingerprints",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.fingerprints import *
