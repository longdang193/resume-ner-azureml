"""Compatibility shim for azureml module.

Use 'from infrastructure.platform.azureml import ...' instead.
This will be removed in 2 releases.
"""

import warnings

warnings.warn(
    "orchestration.azureml is deprecated, use infrastructure.platform.azureml",
    DeprecationWarning,
    stacklevel=2
)

from infrastructure.platform.azureml import *

