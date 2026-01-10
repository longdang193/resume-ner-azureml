"""Compatibility shim for tracking module.

Use 'from infrastructure.tracking import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.tracking is deprecated, use infrastructure.tracking",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.tracking import *

