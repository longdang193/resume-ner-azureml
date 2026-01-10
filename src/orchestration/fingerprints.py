"""Legacy facade for fingerprints module.

This module provides backward compatibility by re-exporting from fingerprints.
All imports from this module are deprecated.
"""

import warnings
from fingerprints import (
    compute_spec_fp,
    compute_exec_fp,
    compute_conv_fp,
    compute_bench_fp,
    compute_hardware_fp,
)

__all__ = [
    "compute_spec_fp",
    "compute_exec_fp",
    "compute_conv_fp",
    "compute_bench_fp",
    "compute_hardware_fp",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'fingerprints' from 'orchestration' is deprecated. "
    "Please import from 'fingerprints' instead.",
    DeprecationWarning,
    stacklevel=2
)
