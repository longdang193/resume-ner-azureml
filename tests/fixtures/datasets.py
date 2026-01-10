"""Shared dataset fixtures for tests."""

import json
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a tiny dataset for testing.
    
    Creates a minimal dataset structure with train.json, validation.json, and test.json
    in the format expected by the dataset loader.
    
    Args:
        tmp_path: Pytest temporary directory fixture
        
    Returns:
        Path to dataset directory (dataset_tiny/seed0/)
    """
    dataset_dir = tmp_path / "dataset_tiny" / "seed0"
    dataset_dir.mkdir(parents=True)
    
    # Create minimal train.json
    train_data = [
        {
            "text": f"Sample text {i} with entity",
            "annotations": [[0, 10, "SKILL"]] if i % 2 == 0 else []
        }
        for i in range(20)
    ]
    (dataset_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    
    # Create optional validation.json
    val_data = [
        {
            "text": f"Validation text {i}",
            "annotations": []
        }
        for i in range(5)
    ]
    (dataset_dir / "validation.json").write_text(json.dumps(val_data, indent=2))
    
    # Create optional test.json for benchmarking
    test_data = [
        {
            "text": f"Test text {i}",
            "annotations": []
        }
        for i in range(5)
    ]
    (dataset_dir / "test.json").write_text(json.dumps(test_data, indent=2))
    
    return dataset_dir


def create_dataset_structure(
    base_path: Path,
    seed: Optional[int] = None,
    train_size: int = 20,
    val_size: int = 5,
    test_size: int = 5,
) -> Path:
    """Create a dataset structure with configurable sizes.
    
    Args:
        base_path: Base directory for dataset
        seed: Optional seed for subdirectory (e.g., seed0, seed1)
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        
    Returns:
        Path to dataset directory
    """
    if seed is not None:
        dataset_dir = base_path / "dataset_tiny" / f"seed{seed}"
    else:
        dataset_dir = base_path / "dataset"
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train.json
    train_data = [
        {
            "text": f"Sample text {i} with entity",
            "annotations": [[0, 10, "SKILL"]] if i % 2 == 0 else []
        }
        for i in range(train_size)
    ]
    (dataset_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    
    # Create validation.json
    if val_size > 0:
        val_data = [
            {
                "text": f"Validation text {i}",
                "annotations": []
            }
            for i in range(val_size)
        ]
        (dataset_dir / "validation.json").write_text(json.dumps(val_data, indent=2))
    
    # Create test.json
    if test_size > 0:
        test_data = [
            {
                "text": f"Test text {i}",
                "annotations": []
            }
            for i in range(test_size)
        ]
        (dataset_dir / "test.json").write_text(json.dumps(test_data, indent=2))
    
    return dataset_dir









