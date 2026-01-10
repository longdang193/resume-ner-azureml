"""Compatibility shim for platform_adapters module.

Use 'from infrastructure.platform.adapters import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.platform_adapters is deprecated, use infrastructure.platform.adapters",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.platform.adapters import *

