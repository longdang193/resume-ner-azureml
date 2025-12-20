"""Test orchestration for HPO pipeline tests.

This module is responsible solely for coordinating test execution flow.
It contains no environment setup, validation, presentation, or business logic.
"""

from tests.integration.validators.dataset_validator import check_dataset_exists
from tests.integration.setup.environment_setup import setup_test_environment
from tests.integration.services.kfold_validator import validate_kfold_splits
from tests.integration.services.hpo_executor import run_hpo_sweep_for_dataset
from tests.integration.services.edge_case_detector import detect_edge_cases
from tests.integration.comparators.result_comparator import compare_results
from tests.integration.aggregators.result_aggregator import (
    build_test_details,
    collect_test_results,
)
from tests.fixtures.config.test_config_loader import (
    DEFAULT_BACKBONE,
    BACKBONES_LIST,
    DEFAULT_RANDOM_SEED,
    get_test_config,
)
from training.data import load_dataset
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))


def test_deterministic_hpo(
    dataset_path: Path,
    config_dir: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    backbone: str = DEFAULT_BACKBONE,
) -> Optional[Dict[str, Any]]:
    """
    Test HPO with deterministic dataset.

    Args:
        dataset_path: Path to deterministic dataset directory
        config_dir: Path to configuration directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary
        output_dir: Test output directory
        backbone: Model backbone name

    Returns:
        Dictionary with HPO results, or None if dataset doesn't exist
    """
    if not check_dataset_exists(dataset_path):
        return None

    deterministic_output = output_dir / "deterministic"
    mlflow_experiment = "test-hpo-deterministic"

    results = run_hpo_sweep_for_dataset(
        dataset_path=dataset_path,
        config_dir=config_dir,
        hpo_config=hpo_config,
        train_config=train_config,
        output_dir=deterministic_output,
        mlflow_experiment_name=mlflow_experiment,
        backbone=backbone,
    )

    return results


def test_random_seed_variants(
    dataset_base_path: Path,
    seeds: List[int],
    config_dir: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    backbone: str = DEFAULT_BACKBONE,
) -> Dict[int, Dict[str, Any]]:
    """
    Test HPO with multiple random seed variants.

    Args:
        dataset_base_path: Base path to dataset directory (e.g., dataset_tiny)
        seeds: List of seed numbers to test
        config_dir: Path to configuration directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary
        output_dir: Test output directory
        backbone: Model backbone name

    Returns:
        Dictionary mapping seed numbers to their HPO results
    """
    results = {}

    if not seeds:
        return results

    for seed in seeds:
        random_dataset_path = dataset_base_path / f"seed{seed}"

        if not check_dataset_exists(random_dataset_path):
            continue

        random_output = output_dir / f"random_seed{seed}"
        mlflow_experiment = f"test-hpo-random-seed{seed}"

        seed_results = run_hpo_sweep_for_dataset(
            dataset_path=random_dataset_path,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=random_output,
            mlflow_experiment_name=mlflow_experiment,
            backbone=backbone,
        )

        results[seed] = seed_results

    return results


def test_deterministic_hpo_multiple_backbones(
    dataset_path: Path,
    config_dir: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    backbones: List[str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Test HPO with deterministic dataset for multiple backbones.

    Args:
        dataset_path: Path to deterministic dataset directory
        config_dir: Path to configuration directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary
        output_dir: Test output directory
        backbones: List of model backbone names to test

    Returns:
        Dictionary mapping backbone names to their HPO results
    """
    results = {}
    for backbone in backbones:
        backbone_output = output_dir / f"deterministic_{backbone}"
        mlflow_experiment = f"test-hpo-deterministic-{backbone}"

        backbone_results = run_hpo_sweep_for_dataset(
            dataset_path=dataset_path,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=backbone_output,
            mlflow_experiment_name=mlflow_experiment,
            backbone=backbone,
        )
        results[backbone] = backbone_results

    return results


def test_random_seed_variants_multiple_backbones(
    dataset_base_path: Path,
    seeds: List[int],
    config_dir: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    backbones: List[str],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Test HPO with multiple random seed variants for multiple backbones.

    Args:
        dataset_base_path: Base path to dataset directory (e.g., dataset_tiny)
        seeds: List of seed numbers to test
        config_dir: Path to configuration directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary
        output_dir: Test output directory
        backbones: List of model backbone names to test

    Returns:
        Dictionary mapping backbone names to dictionaries of seed results
    """
    all_results = {}
    for backbone in backbones:
        backbone_results = test_random_seed_variants(
            dataset_base_path=dataset_base_path,
            seeds=seeds,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir / f"random_{backbone}",
            backbone=backbone,
        )
        all_results[backbone] = backbone_results

    return all_results


def test_kfold_validation(
    dataset_path: Path,
    hpo_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Test k-fold CV validation.

    Args:
        dataset_path: Path to dataset directory
        hpo_config: HPO configuration dictionary

    Returns:
        Dictionary with k-fold validation results, or None if dataset doesn't exist
    """
    if not check_dataset_exists(dataset_path):
        return None

    k_configured = hpo_config["k_fold"]["n_splits"]
    results = validate_kfold_splits(
        dataset_path=dataset_path,
        k=k_configured,
        random_seed=hpo_config["k_fold"]["random_seed"],
        shuffle=hpo_config["k_fold"]["shuffle"],
    )

    return results


def test_edge_case_k_too_large(
    dataset_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Test edge case: k > n_samples.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dictionary with validation results, or None if dataset doesn't exist
    """
    if not check_dataset_exists(dataset_path):
        return None

    dataset_dict = load_dataset(str(dataset_path))
    n_samples = len(dataset_dict["train"])
    k_too_large = n_samples + 1

    results = validate_kfold_splits(
        dataset_path=dataset_path,
        k=k_too_large,
        random_seed=DEFAULT_RANDOM_SEED,
        shuffle=True,
    )

    # Store metadata for presentation layer
    results["_metadata"] = {
        "k": k_too_large,
        "n_samples": n_samples,
    }

    return results


def test_edge_cases_suite(
    dataset_path: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Test all edge cases.

    Args:
        dataset_path: Path to dataset directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary

    Returns:
        Dictionary with edge case test results, or None if dataset doesn't exist
    """
    if not check_dataset_exists(dataset_path):
        return None

    results = detect_edge_cases(
        dataset_path=dataset_path,
        hpo_config=hpo_config,
        train_config=train_config,
    )

    return results


def run_all_tests(
    root_dir: Path,
    random_seeds: List[int] = [0],
    hpo_config_path: Optional[Path] = None,
    train_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_deterministic: bool = False,
    skip_random: bool = False,
    skip_kfold: bool = False,
    skip_edge_cases: bool = False,
    backbone: str = DEFAULT_BACKBONE,
) -> Dict[str, Any]:
    """
    Main orchestrator that runs all tests and returns aggregated results.

    Args:
        root_dir: Project root directory
        random_seeds: List of seed numbers to test
        hpo_config_path: Path to HPO config file (default: config/hpo/smoke.yaml)
        train_config_path: Path to train config file (default: config/train.yaml)
        output_dir: Test output directory (default: outputs/hpo_tests)
        skip_deterministic: Skip deterministic dataset test
        skip_random: Skip random seed variant tests
        skip_kfold: Skip k-fold validation tests
        skip_edge_cases: Skip edge case tests
        backbone: Model backbone name

    Returns:
        Dictionary with all test results and summary
    """
    # Setup test environment
    env = setup_test_environment(
        root_dir=root_dir,
        hpo_config_path=hpo_config_path,
        train_config_path=train_config_path,
        output_dir=output_dir,
    )

    config_dir = env["config_dir"]
    hpo_config = env["hpo_config"]
    train_config = env["train_config"]
    output_dir = env["output_dir"]
    deterministic_dataset = env["deterministic_dataset"]

    # Use default random seeds from config if using default value
    test_config = get_test_config(root_dir)
    datasets_section = test_config.get("datasets", {})
    config_seeds = datasets_section.get("random_seeds", [0])
    if random_seeds == [0] and config_seeds != [0]:
        random_seeds = config_seeds

    deterministic_results = None
    random_seed_results = None
    kfold_results = None
    k_too_large_results = None
    edge_case_results = None

    if not skip_deterministic:
        deterministic_results = test_deterministic_hpo(
            dataset_path=deterministic_dataset,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            backbone=backbone,
        )

    if not skip_random:
        random_seed_results = test_random_seed_variants(
            dataset_base_path=deterministic_dataset,
            seeds=random_seeds,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            backbone=backbone,
        )

    if not skip_kfold:
        kfold_results = test_kfold_validation(
            dataset_path=deterministic_dataset,
            hpo_config=hpo_config,
        )

    if not skip_edge_cases:
        k_too_large_results = test_edge_case_k_too_large(
            dataset_path=deterministic_dataset,
        )

        edge_case_results = test_edge_cases_suite(
            dataset_path=deterministic_dataset,
            hpo_config=hpo_config,
            train_config=train_config,
        )

    test_summary = collect_test_results(
        deterministic_results=deterministic_results,
        random_seed_results=random_seed_results,
        kfold_results=kfold_results,
        k_too_large_results=k_too_large_results,
        edge_case_results=edge_case_results,
    )

    test_details = build_test_details(
        deterministic_results=deterministic_results,
        random_seed_results=random_seed_results,
        kfold_results=kfold_results,
    )

    # Compute comparison data (for presentation layer)
    comparison = compare_results(
        deterministic_results=deterministic_results,
        random_seed_results=random_seed_results,
    )

    return {
        "test_summary": test_summary,
        "test_details": test_details,
        "comparison": comparison,
        "deterministic_results": deterministic_results,
        "random_seed_results": random_seed_results,
        "kfold_results": kfold_results,
        "k_too_large_results": k_too_large_results,
        "edge_case_results": edge_case_results,
    }
