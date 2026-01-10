"""Compatibility shim for storage module.

Use 'from infrastructure.storage import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.storage is deprecated, use infrastructure.storage",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.storage import *

