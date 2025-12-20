"""Edge case detection service for HPO pipeline tests.

This module is responsible solely for detecting edge cases in test configuration.
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

from tests.fixtures.config.test_config_loader import (
    DEFAULT_RANDOM_SEED,
    MINIMAL_K_FOLDS,
    VERY_SMALL_VALIDATION_THRESHOLD,
)
from tests.integration.services.kfold_validator import validate_kfold_splits


def detect_edge_cases(
    dataset_path: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect edge cases: minimal k, small validation sets, batch size issues.

    Args:
        dataset_path: Path to dataset directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary

    Returns:
        Dictionary with edge case detection results
    """
    results = {
        "dataset_path": str(dataset_path),
        "edge_cases": {},
        "overall_success": False,
    }

    try:
        dataset_dict = load_dataset(str(dataset_path))
        train_data = dataset_dict["train"]
        n_samples = len(train_data)

        minimal_k_results = validate_kfold_splits(
            dataset_path=dataset_path,
            k=MINIMAL_K_FOLDS,
            random_seed=DEFAULT_RANDOM_SEED,
            shuffle=True,
        )
        results["edge_cases"]["minimal_k"] = minimal_k_results

        batch_sizes = hpo_config["search_space"]["batch_size"]["values"]
        k_configured = hpo_config["k_fold"]["n_splits"]

        splits = create_kfold_splits(
            train_data,
            k=k_configured,
            random_seed=hpo_config["k_fold"]["random_seed"],
            shuffle=hpo_config["k_fold"]["shuffle"],
        )

        val_sizes = [len(val_indices) for _, val_indices in splits]
        min_val_size = min(val_sizes)
        max_val_size = max(val_sizes)

        batch_size_issues = []
        for batch_size in batch_sizes:
            if batch_size >= min_val_size:
                batch_size_issues.append(
                    f"batch_size={batch_size} >= min_val_size={min_val_size} "
                    "(may cause issues)"
                )

        results["edge_cases"]["batch_size_validation"] = {
            "batch_sizes": batch_sizes,
            "min_val_size": min_val_size,
            "max_val_size": max_val_size,
            "val_sizes": val_sizes,
            "potential_issues": batch_size_issues,
        }

        very_small_val = [
            size for size in val_sizes if size <= VERY_SMALL_VALIDATION_THRESHOLD
        ]
        results["edge_cases"]["very_small_validation_sets"] = {
            "count": len(very_small_val),
            "sizes": very_small_val,
            "warning": "Validation sets with 1-2 samples may produce unstable metrics",
        }

        results["overall_success"] = (
            minimal_k_results["success"] and len(batch_size_issues) == 0
        )

    except Exception as e:
        results["error"] = str(e)
        results["overall_success"] = False

    return results

