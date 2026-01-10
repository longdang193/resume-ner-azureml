"""Compatibility shim for constants module.

Use 'from common.constants import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.constants is deprecated, use common.constants",
    DeprecationWarning,
    stacklevel=2
)

from common.constants import *

