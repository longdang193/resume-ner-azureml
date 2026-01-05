"""Dataset combination utilities for continued training."""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional


def combine_datasets(
    old_dataset_path: Optional[Path],
    new_dataset_path: Path,
    strategy: str = "combined",
    validation_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Combine old and new datasets according to specified strategy.
    
    Args:
        old_dataset_path: Path to old dataset directory (optional).
        new_dataset_path: Path to new dataset directory.
        strategy: Combination strategy - "new_only", "combined", or "append".
        validation_ratio: Ratio for validation split if creating new validation set.
        random_seed: Random seed for shuffling.
    
    Returns:
        Dictionary with "train" and optionally "validation" keys containing data lists.
    
    Raises:
        ValueError: If strategy is invalid or required dataset is missing.
        FileNotFoundError: If dataset paths don't exist.
    """
    if strategy not in ["new_only", "combined", "append"]:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of: 'new_only', 'combined', 'append'"
        )
    
    # Load new dataset
    if not new_dataset_path.exists():
        raise FileNotFoundError(f"New dataset path not found: {new_dataset_path}")
    
    new_train_file = new_dataset_path / "train.json"
    if not new_train_file.exists():
        raise FileNotFoundError(f"New training file not found: {new_train_file}")
    
    with open(new_train_file, "r", encoding="utf-8") as f:
        new_train = json.load(f)
    
    new_val_file = new_dataset_path / "validation.json"
    new_val = []
    if new_val_file.exists():
        with open(new_val_file, "r", encoding="utf-8") as f:
            new_val = json.load(f)
    
    # Handle strategy
    if strategy == "new_only":
        # Use only new dataset
        return {
            "train": new_train,
            "validation": new_val if new_val else [],
        }
    
    # For "combined" or "append", we need old dataset
    if old_dataset_path is None or not old_dataset_path.exists():
        raise ValueError(
            f"Old dataset path required for strategy '{strategy}' but not provided or doesn't exist"
        )
    
    old_train_file = old_dataset_path / "train.json"
    if not old_train_file.exists():
        raise FileNotFoundError(f"Old training file not found: {old_train_file}")
    
    with open(old_train_file, "r", encoding="utf-8") as f:
        old_train = json.load(f)
    
    old_val_file = old_dataset_path / "validation.json"
    old_val = []
    if old_val_file.exists():
        with open(old_val_file, "r", encoding="utf-8") as f:
            old_val = json.load(f)
    
    # Combine training data
    if strategy == "append":
        # Append new to old without shuffling
        combined_train = old_train + new_train
    else:  # strategy == "combined"
        # Merge and shuffle
        combined_train = old_train + new_train
        random.seed(random_seed)
        random.shuffle(combined_train)
    
    # Combine validation data (if both exist)
    combined_val = []
    if old_val and new_val:
        combined_val = old_val + new_val
        if strategy == "combined":
            random.seed(random_seed + 1)  # Different seed for validation shuffle
            random.shuffle(combined_val)
    elif old_val:
        combined_val = old_val
    elif new_val:
        combined_val = new_val
    
    # If no validation data and validation_ratio > 0, create split from train
    if not combined_val and validation_ratio > 0:
        from sklearn.model_selection import train_test_split
        
        split_size = max(1, int(len(combined_train) * validation_ratio))
        if len(combined_train) > split_size:
            combined_train, combined_val = train_test_split(
                combined_train,
                test_size=split_size,
                random_state=random_seed,
                shuffle=True,
            )
        else:
            # Dataset too small to split
            combined_val = []
    
    return {
        "train": combined_train,
        "validation": combined_val,
    }

