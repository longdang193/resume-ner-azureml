"""Result aggregation for HPO pipeline tests.

This module is responsible solely for aggregating and transforming test results.
It contains no business logic, orchestration, or presentation functionality.
"""

from typing import Any, Dict, Optional

from testing.fixtures.config.test_config_loader import METRIC_DECIMAL_PLACES


def collect_test_results(
    deterministic_results: Optional[Dict[str, Any]] = None,
    random_seed_results: Optional[Dict[int, Dict[str, Any]]] = None,
    kfold_results: Optional[Dict[str, Any]] = None,
    k_too_large_results: Optional[Dict[str, Any]] = None,
    edge_case_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[bool]]:
    """
    Collect all test results into a summary dictionary.

    Args:
        deterministic_results: Results from deterministic HPO test
        random_seed_results: Results from random seed variant tests
        kfold_results: Results from k-fold validation test
        k_too_large_results: Results from k > n_samples edge case test
        edge_case_results: Results from edge cases suite test

    Returns:
        Dictionary mapping test names to their pass/fail status
    """
    random_hpo_success = None
    if random_seed_results:
        random_hpo_success = all(
            res.get("success", False) for res in random_seed_results.values()
        )

    return {
        "deterministic_hpo": (
            deterministic_results.get("success", False)
            if deterministic_results else None
        ),
        "random_hpo": random_hpo_success,
        "kfold_validation": (
            kfold_results.get("success", False) if kfold_results else None
        ),
        "edge_case_k_too_large": (
            k_too_large_results.get("success", False)
            if k_too_large_results else None
        ),
        "edge_cases_overall": (
            edge_case_results.get("overall_success", False)
            if edge_case_results else None
        ),
    }


def build_test_details(
    deterministic_results: Optional[Dict[str, Any]] = None,
    random_seed_results: Optional[Dict[int, Dict[str, Any]]] = None,
    kfold_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build details dictionary for each test result.

    Args:
        deterministic_results: Results from deterministic HPO test
        random_seed_results: Results from random seed variant tests
        kfold_results: Results from k-fold validation test

    Returns:
        Dictionary with detailed information for each test
    """
    details = {}

    if deterministic_results:
        details["deterministic"] = {
            "Trials completed": deterministic_results.get("trials_completed", 0),
        }
        if deterministic_results.get("best_value") is not None:
            details["deterministic"]["Best value"] = (
                f"{deterministic_results['best_value']:.{METRIC_DECIMAL_PLACES}f}"
            )

    if random_seed_results:
        total_trials = sum(
            res.get("trials_completed", 0) for res in random_seed_results.values()
        )
        details["random"] = {
            "Seeds tested": f"{list(random_seed_results.keys())}",
            "Total trials completed": total_trials,
        }
        successful_seeds = [
            seed
            for seed, res in random_seed_results.items()
            if res.get("success")
        ]
        if successful_seeds:
            best_values = {
                seed: res["best_value"]
                for seed, res in random_seed_results.items()
                if res.get("best_value") is not None
            }
            if best_values:
                details["random"]["Best values"] = ", ".join([
                    f"seed{seed}: {val:.{METRIC_DECIMAL_PLACES}f}"
                    for seed, val in best_values.items()
                ])

    if kfold_results:
        details["kfold"] = {
            "Splits valid": kfold_results.get("splits_valid", False),
            "All folds non-empty": kfold_results.get("all_folds_non_empty", False),
        }

    return details

