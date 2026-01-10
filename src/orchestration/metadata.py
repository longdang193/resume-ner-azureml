"""Compatibility shim for metadata module.

Use 'from infrastructure.metadata import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.metadata is deprecated, use infrastructure.metadata",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.metadata import *

