"""Backward compatibility module for HPO pipeline testing.

This module re-exports functions and constants from the refactored modules
to maintain backward compatibility. New code should import directly from
the specific modules:
- tests.fixtures.config.test_config_loader (config loading)
- tests.fixtures.presenters.result_formatters (formatting/printing)
"""

# Re-export config loader functions and constants
from tests.fixtures.config.test_config_loader import (
    DEFAULT_BACKBONE,
    DEFAULT_RANDOM_SEED,
    METRIC_DECIMAL_PLACES,
    MINIMAL_K_FOLDS,
    SEPARATOR_WIDTH,
    VERY_SMALL_VALIDATION_THRESHOLD,
    get_test_config,
    load_hpo_test_config,
)

# Re-export presenter functions
from tests.fixtures.presenters.result_formatters import (
    format_test_result,
    print_comparison,
    print_edge_case_k_too_large_results,
    print_edge_case_results,
    print_hpo_results,
    print_kfold_results,
    print_test_summary,
)

# Re-export validator functions (moved from presenters)
from tests.integration.validators.dataset_validator import check_dataset_exists

__all__ = [
    # Config constants
    "DEFAULT_BACKBONE",
    "DEFAULT_RANDOM_SEED",
    "METRIC_DECIMAL_PLACES",
    "MINIMAL_K_FOLDS",
    "SEPARATOR_WIDTH",
    "VERY_SMALL_VALIDATION_THRESHOLD",
    # Config functions
    "get_test_config",
    "load_hpo_test_config",
    # Validator functions (moved from presenters)
    "check_dataset_exists",
    # Presenter functions
    "format_test_result",
    "print_comparison",
    "print_edge_case_k_too_large_results",
    "print_edge_case_results",
    "print_hpo_results",
    "print_kfold_results",
    "print_test_summary",
]
