"""Data loaders module.

This module provides data loading utilities for different use cases.
"""

# Import from dataset_loader (moved from training/data.py)
from .dataset_loader import (
    load_dataset,
    build_label_list,
    ResumeNERDataset,
)

# Import from benchmark_loader (moved from benchmarking/data_loader.py)
from .benchmark_loader import (
    load_test_texts,
)

__all__ = [
    "load_dataset",
    "build_label_list",
    "ResumeNERDataset",
    "load_test_texts",
]

