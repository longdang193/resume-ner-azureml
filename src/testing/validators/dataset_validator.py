"""Dataset validation for HPO pipeline tests.

This module is responsible solely for validating dataset existence and structure.
It contains no orchestration, presentation, or business logic beyond validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from data.loaders import load_dataset


def check_dataset_exists(dataset_path: Path) -> bool:
    """
    Check if dataset directory exists.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        True if dataset exists, False otherwise
    """
    return dataset_path.exists()


def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
    """
    Validate dataset structure and return validation results.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dictionary with validation results:
        - exists: bool - Whether dataset directory exists
        - has_train: bool - Whether train.json exists
        - has_validation: bool - Whether validation.json exists
        - n_train_samples: Optional[int] - Number of training samples
        - n_val_samples: Optional[int] - Number of validation samples
        - errors: List[str] - List of validation errors
        - valid: bool - Whether dataset structure is valid
    """
    results: Dict[str, Any] = {
        "exists": False,
        "has_train": False,
        "has_validation": False,
        "n_train_samples": None,
        "n_val_samples": None,
        "errors": [],
        "valid": False,
    }

    if not dataset_path.exists():
        results["errors"].append(
            f"Dataset directory not found: {dataset_path}")
        return results

    results["exists"] = True

    train_file = dataset_path / "train.json"
    validation_file = dataset_path / "validation.json"

    if not train_file.exists():
        results["errors"].append(f"train.json not found in {dataset_path}")
    else:
        results["has_train"] = True
        try:
            dataset_dict = load_dataset(str(dataset_path))
            if "train" in dataset_dict:
                results["n_train_samples"] = len(dataset_dict["train"])
            else:
                results["errors"].append(
                    "train.json exists but 'train' key not found")
        except Exception as e:
            results["errors"].append(f"Error loading train data: {str(e)}")

    if not validation_file.exists():
        results["errors"].append(
            f"validation.json not found in {dataset_path}")
    else:
        results["has_validation"] = True
        try:
            dataset_dict = load_dataset(str(dataset_path))
            if "validation" in dataset_dict:
                results["n_val_samples"] = len(dataset_dict["validation"])
            else:
                results["errors"].append(
                    "validation.json exists but 'validation' key not found"
                )
        except Exception as e:
            results["errors"].append(
                f"Error loading validation data: {str(e)}")

    results["valid"] = (
        results["exists"]
        and results["has_train"]
        and len(results["errors"]) == 0
    )

    return results

