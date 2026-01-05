"""K-fold cross-validation validation service for HPO pipeline tests.

This module is responsible solely for validating k-fold CV splits.
It contains no orchestration, presentation, or configuration logic.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from training.cv_utils import create_kfold_splits
from training.data import load_dataset

from testing.fixtures.config.test_config_loader import DEFAULT_RANDOM_SEED


def validate_kfold_splits(
    dataset_path: Path,
    k: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """
    Test k-fold CV creation and validate splits.

    Args:
        dataset_path: Path to dataset directory
        k: Number of folds
        random_seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting

    Returns:
        Dictionary with validation results
    """
    results = {
        "dataset_path": str(dataset_path),
        "k": k,
        "random_seed": random_seed,
        "shuffle": shuffle,
        "success": False,
        "n_samples": 0,
        "splits_created": 0,
        "splits_valid": False,
        "all_folds_non_empty": False,
        "errors": [],
    }

    try:
        dataset_dict = load_dataset(str(dataset_path))
        train_data = dataset_dict["train"]
        results["n_samples"] = len(train_data)

        if k > results["n_samples"]:
            try:
                create_kfold_splits(
                    train_data, k=k, random_seed=random_seed, shuffle=shuffle
                )
                results["errors"].append(
                    f"Expected error for k={k} > n_samples={results['n_samples']}, "
                    "but splits were created"
                )
                results["success"] = False
            except ValueError as e:
                results["success"] = True
                results["errors"] = []
                results["expected_error"] = str(e)
            return results

        splits = create_kfold_splits(
            train_data, k=k, random_seed=random_seed, shuffle=shuffle
        )
        results["splits_created"] = len(splits)

        all_valid = True
        all_non_empty = True

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            if not all(0 <= idx < results["n_samples"] for idx in train_indices + val_indices):
                results["errors"].append(f"Fold {fold_idx}: Invalid indices")
                all_valid = False

            if set(train_indices) & set(val_indices):
                results["errors"].append(
                    f"Fold {fold_idx}: Overlapping train/val indices"
                )
                all_valid = False

            if not train_indices:
                results["errors"].append(
                    f"Fold {fold_idx}: Empty training set")
                all_non_empty = False
            if not val_indices:
                results["errors"].append(
                    f"Fold {fold_idx}: Empty validation set")
                all_non_empty = False

            all_indices = set(train_indices) | set(val_indices)
            if len(all_indices) != results["n_samples"]:
                results["errors"].append(
                    f"Fold {fold_idx}: Not all samples used "
                    f"(expected {results['n_samples']}, got {len(all_indices)})"
                )
                all_valid = False

        results["splits_valid"] = all_valid
        results["all_folds_non_empty"] = all_non_empty
        results["success"] = all_valid and all_non_empty

        fold_sizes = []
        for train_indices, val_indices in splits:
            fold_sizes.append({
                "train_size": len(train_indices),
                "val_size": len(val_indices),
            })
        results["fold_sizes"] = fold_sizes

    except Exception as e:
        results["errors"].append(f"K-fold validation failed: {str(e)}")
        results["success"] = False

    return results

