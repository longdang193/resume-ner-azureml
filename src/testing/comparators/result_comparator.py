"""Result comparison for HPO pipeline tests.

This module is responsible solely for comparing test results and computing
comparison metrics. It contains no orchestration, presentation, or business logic.
"""

from typing import Any, Dict, List, Optional

from testing.fixtures.config.test_config_loader import METRIC_DECIMAL_PLACES


def compute_comparison_data(
    deterministic_results: Dict[str, Any],
    random_seed_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute comparison data between deterministic and random seed variant results.

    Args:
        deterministic_results: Results from deterministic dataset
        random_seed_results: Dictionary mapping seed numbers to their results

    Returns:
        Dictionary with comparison data:
        - deterministic_best_value: Optional[float]
        - random_seed_results: Dict[int, Dict] - Successful random seed results
        - differences: Dict[int, float] - Differences from deterministic
        - random_values: List[float] - All random seed best values
        - min_random: Optional[float]
        - max_random: Optional[float]
        - variability: Optional[float]
        - has_comparison: bool
    """
    comparison: Dict[str, Any] = {
        "deterministic_best_value": None,
        "random_seed_results": {},
        "differences": {},
        "random_values": [],
        "min_random": None,
        "max_random": None,
        "variability": None,
        "has_comparison": False,
    }

    if not deterministic_results or not deterministic_results.get("success"):
        return comparison

    deterministic_best = deterministic_results.get("best_value")
    if deterministic_best is None:
        return comparison

    comparison["deterministic_best_value"] = deterministic_best

    # Filter successful random seed results
    successful_random = {
        seed: res
        for seed, res in random_seed_results.items()
        if res.get("success") and res.get("best_value") is not None
    }

    if not successful_random:
        return comparison

    comparison["random_seed_results"] = successful_random

    # Compute differences
    for seed, res in successful_random.items():
        diff = abs(deterministic_best - res["best_value"])
        comparison["differences"][seed] = diff

    # Compute variability if multiple random seeds
    if len(successful_random) > 1:
        random_values = [res["best_value"] for res in successful_random.values()]
        comparison["random_values"] = random_values
        comparison["min_random"] = min(random_values)
        comparison["max_random"] = max(random_values)
        comparison["variability"] = comparison["max_random"] - comparison["min_random"]

    comparison["has_comparison"] = True
    return comparison


def compare_results(
    deterministic_results: Optional[Dict[str, Any]],
    random_seed_results: Optional[Dict[int, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Compare deterministic and random seed variant results.

    Args:
        deterministic_results: Results from deterministic dataset
        random_seed_results: Dictionary mapping seed numbers to their results

    Returns:
        Dictionary with comparison data, or None if comparison cannot be performed
    """
    if not deterministic_results or not random_seed_results:
        return None

    comparison = compute_comparison_data(
        deterministic_results=deterministic_results,
        random_seed_results=random_seed_results,
    )

    if not comparison["has_comparison"]:
        return None

    return comparison

