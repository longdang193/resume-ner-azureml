"""Compatibility shim for training.data module.

Use 'from data.loaders import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "training.data is deprecated, use data.loaders",
    DeprecationWarning,
    stacklevel=2
)

from data.loaders import *
