"""
@meta
name: benchmarking_data_loader
type: utility
domain: benchmarking
responsibility:
  - Load test texts from JSON files
  - Handle multiple data formats (list of strings or list of dicts)
inputs:
  - JSON file path
outputs:
  - List of test text strings
tags:
  - utility
  - benchmarking
  - data-loading
ci:
  runnable: true
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Test data loading utilities for benchmarking."""

import json
from pathlib import Path
from typing import List


def load_test_texts(file_path: Path) -> List[str]:
    """
    Load test texts from a JSON file.

    Args:
        file_path: Path to JSON file containing test data.

    Returns:
        List of test text strings.

    Raises:
        ValueError: If test data format is invalid or no texts found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # List of dicts with 'text' field
            return [item.get("text", "") for item in data if item.get("text")]
        else:
            # List of strings
            return [str(item) for item in data if item]
    else:
        raise ValueError("Test data must be a list of texts or list of dicts with 'text' field")

