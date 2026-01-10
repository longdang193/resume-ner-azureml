"""Compatibility shim for config module.

Use 'from infrastructure.config import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.config is deprecated, use infrastructure.config",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.config import *

