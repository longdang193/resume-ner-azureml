"""Compatibility shim for shared module.

Use 'from common.shared import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.shared is deprecated, use common.shared",
    DeprecationWarning,
    stacklevel=2
)

from common.shared import *

