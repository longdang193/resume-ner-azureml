"""Presentation and formatting utilities for HPO pipeline tests.

This module is responsible solely for formatting and printing test results.
It contains no business logic or configuration loading.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from testing.fixtures.config.test_config_loader import (
    METRIC_DECIMAL_PLACES,
    MINIMAL_K_FOLDS,
    SEPARATOR_WIDTH,
    VERY_SMALL_VALIDATION_THRESHOLD,
)


def print_hpo_results(results: Dict[str, Any], title: str) -> None:
    """
    Print HPO results in a formatted way.
    
    Args:
        results: Dictionary with HPO results
        title: Title for the results section
    """
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}")
    print(f"{title}:")
    print(separator)
    print(f"Success: {results['success']}")
    print(f"Trials completed: {results['trials_completed']}")
    print(f"Trials failed: {results['trials_failed']}")
    
    if results['best_trial'] is not None:
        print(f"Best trial: {results['best_trial']}")
        print(f"Best value: {results['best_value']:.{METRIC_DECIMAL_PLACES}f}")
        print(f"Best params: {results['best_params']}")
    
    if results['errors']:
        print(f"Errors: {results['errors']}")
    
    print(separator)


def print_comparison(comparison: Dict[str, Any]) -> None:
    """
    Print comparison between deterministic and random seed variants.
    
    Args:
        comparison: Comparison data dictionary from result_comparator
    """
    if not comparison or not comparison.get("has_comparison"):
        return
    
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}")
    print("Comparison: Seed0 vs Random Seed Variants")
    print(separator)
    
    seed0_best = comparison["deterministic_best_value"]  # Note: key name kept for backward compatibility
    print(f"Seed0 best value: {seed0_best:.{METRIC_DECIMAL_PLACES}f}")
    
    for seed, res in comparison["random_seed_results"].items():
        print(f"Random seed {seed} best value: {res['best_value']:.{METRIC_DECIMAL_PLACES}f}")
        diff = comparison["differences"][seed]
        print(f"  Difference from seed0: {diff:.{METRIC_DECIMAL_PLACES}f}")
    
    if comparison.get("variability") is not None:
        print(f"\nRandom seed variants range: {comparison['min_random']:.{METRIC_DECIMAL_PLACES}f} to {comparison['max_random']:.{METRIC_DECIMAL_PLACES}f}")
        print(f"Variability: {comparison['variability']:.{METRIC_DECIMAL_PLACES}f}")
    
    print(separator)


def print_kfold_results(results: Dict[str, Any], k: int) -> None:
    """
    Print k-fold CV validation results.
    
    Args:
        results: Dictionary with k-fold validation results
        k: Number of folds tested
    """
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}")
    print(f"K-Fold CV Validation (k={k}):")
    print(separator)
    print(f"Dataset: {results['dataset_path']}")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Splits created: {results['splits_created']}")
    print(f"Splits valid: {results['splits_valid']}")
    print(f"All folds non-empty: {results['all_folds_non_empty']}")
    print(f"Success: {results['success']}")
    
    if results.get('fold_sizes'):
        print("\nFold sizes:")
        for i, sizes in enumerate(results['fold_sizes']):
            print(f"  Fold {i}: train={sizes['train_size']}, val={sizes['val_size']}")
    
    if results['errors']:
        print(f"\nErrors: {results['errors']}")
    print(separator)


def print_edge_case_k_too_large_results(
    results: Dict[str, Any],
) -> None:
    """
    Print results for edge case where k > n_samples.
    
    Args:
        results: Validation results dictionary (should contain _metadata with k and n_samples)
    """
    metadata = results.get("_metadata", {})
    k = metadata.get("k", "unknown")
    n_samples = metadata.get("n_samples", "unknown")
    
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}")
    print(f"Edge Case: k > n_samples (k={k}, n_samples={n_samples}):")
    print(separator)
    print(f"Success (expected error caught): {results['success']}")
    if results.get('expected_error'):
        print(f"Expected error: {results['expected_error']}")
    if results['errors']:
        print(f"Unexpected errors: {results['errors']}")
    print(separator)


def print_edge_case_results(results: Dict[str, Any]) -> None:
    """
    Print edge case test results in a formatted way.
    
    Args:
        results: Dictionary with edge case test results
    """
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}")
    print("Edge Case Test Results:")
    print(separator)
    
    if "minimal_k" in results["edge_cases"]:
        minimal_k = results["edge_cases"]["minimal_k"]
        print(f"\n1. Minimal k (k={MINIMAL_K_FOLDS}):")
        print(f"   Success: {minimal_k['success']}")
        print(f"   Splits valid: {minimal_k['splits_valid']}")
        if minimal_k.get('fold_sizes'):
            for i, sizes in enumerate(minimal_k['fold_sizes']):
                print(f"   Fold {i}: train={sizes['train_size']}, val={sizes['val_size']}")
    
    if "batch_size_validation" in results["edge_cases"]:
        batch_info = results["edge_cases"]["batch_size_validation"]
        print(f"\n2. Batch Size vs Validation Set Size:")
        print(f"   Batch sizes: {batch_info['batch_sizes']}")
        print(f"   Validation set sizes: {batch_info['val_sizes']}")
        print(f"   Min val size: {batch_info['min_val_size']}, Max val size: {batch_info['max_val_size']}")
        if batch_info['potential_issues']:
            print(f"   ⚠️  Potential issues:")
            for issue in batch_info['potential_issues']:
                print(f"      - {issue}")
        else:
            print(f"   ✓ No batch size issues detected")
    
    if "very_small_validation_sets" in results["edge_cases"]:
        small_val = results["edge_cases"]["very_small_validation_sets"]
        print(f"\n3. Very Small Validation Sets (≤{VERY_SMALL_VALIDATION_THRESHOLD} samples):")
        print(f"   Count: {small_val['count']}")
        if small_val['sizes']:
            print(f"   Sizes: {small_val['sizes']}")
            print(f"   ⚠️  {small_val['warning']}")
        else:
            print(f"   ✓ No very small validation sets")
    
    print(f"\nOverall success: {results['overall_success']}")
    if "error" in results:
        print(f"Error: {results['error']}")
    print(separator)


def format_test_result(
    test_name: str,
    passed: bool,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a single test result line.
    
    Args:
        test_name: Name of the test
        passed: Whether the test passed
        details: Optional dictionary with additional details
        
    Returns:
        Formatted test result string
    """
    status = "✓ PASS" if passed else "✗ FAIL"
    result = f"{test_name}: {status}"
    if details:
        for key, value in details.items():
            result += f"\n   - {key}: {value}"
    return result


def print_test_summary(
    test_summary: Dict[str, Optional[bool]],
    test_details: Dict[str, Dict[str, Any]]
) -> None:
    """
    Print the complete test summary.
    
    Args:
        test_summary: Dictionary mapping test names to their pass/fail status
        test_details: Dictionary with detailed information for each test
    """
    separator = "=" * SEPARATOR_WIDTH
    print(separator)
    print("HPO Pipeline Testing Summary")
    print(separator)
    
    print("\nTest Results:")
    print("-" * SEPARATOR_WIDTH)
    
    if test_summary.get("deterministic_hpo") is not None:
        details = test_details.get("deterministic", {})
        print(format_test_result(
            "1. HPO with seed0 dataset",
            test_summary["deterministic_hpo"],
            details
        ))
    
    if test_summary.get("random_hpo") is not None:
        details = test_details.get("random", {})
        print(format_test_result(
            "2. HPO with random seed variant(s)",
            test_summary["random_hpo"],
            details
        ))
    
    if test_summary.get("kfold_validation") is not None:
        details = test_details.get("kfold", {})
        print(format_test_result(
            "3. K-fold CV validation",
            test_summary["kfold_validation"],
            details
        ))
    
    if test_summary.get("edge_case_k_too_large") is not None:
        print(format_test_result(
            "4. Edge case (k > n_samples)",
            test_summary["edge_case_k_too_large"]
        ))
    
    if test_summary.get("edge_cases_overall") is not None:
        print(format_test_result(
            "5. Edge cases overall",
            test_summary["edge_cases_overall"]
        ))
    
    print("-" * SEPARATOR_WIDTH)
    
    all_tests = [v for v in test_summary.values() if v is not None]
    if all_tests:
        overall_success = all(v for v in all_tests)
        overall_status = "✓ ALL TESTS PASSED" if overall_success else "✗ SOME TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        
        if not overall_success:
            print("\n⚠️  Some tests failed. Review the detailed output above.")
            print("   Common issues:")
            print("   - Dataset files not found (run tests/00_make_tiny_dataset.ipynb)")
            print("   - K-fold CV issues with small datasets")
            print("   - Batch size >= validation set size")
    else:
        print("\n⚠️  No tests were run. Check dataset paths and prerequisites.")
    
    print(separator)

