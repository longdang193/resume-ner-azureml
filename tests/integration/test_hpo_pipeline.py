"""Pytest test wrappers for HPO pipeline integration tests.

This module contains only pytest test wrappers that call the test orchestrator
and services. All business logic, orchestration, and presentation are in
separate modules following Single Responsibility Principle.
"""

import pytest

from tests.integration.orchestrators.test_orchestrator import (
    test_deterministic_hpo,
    test_edge_case_k_too_large,
    test_edge_cases_suite,
    test_kfold_validation,
    test_random_seed_variants,
)


@pytest.mark.integration
class TestHPOPipeline:
    """pytest test class for HPO pipeline integration tests."""

    @pytest.mark.slow
    def test_deterministic_hpo_pytest(
        self,
        root_dir,
        config_dir,
        hpo_config,
        train_config,
        deterministic_dataset,
        test_output_dir,
    ):
        """pytest wrapper for deterministic HPO test."""
        results = test_deterministic_hpo(
            dataset_path=deterministic_dataset,
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=test_output_dir,
        )
        assert results is not None, "Deterministic dataset not found"
        assert results["success"], f"HPO test failed: {results.get('errors', [])}"

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_seed_variants_pytest(
        self,
        seed,
        root_dir,
        config_dir,
        hpo_config,
        train_config,
        deterministic_dataset,
        test_output_dir,
    ):
        """pytest wrapper for random seed variant test with parametrization."""
        results = test_random_seed_variants(
            dataset_base_path=deterministic_dataset,
            seeds=[seed],
            config_dir=config_dir,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=test_output_dir,
        )
        assert seed in results, f"Seed {seed} dataset not found"
        assert results[seed]["success"], (
            f"HPO test failed for seed {seed}: {results[seed].get('errors', [])}"
        )

    def test_kfold_validation_pytest(
        self,
        deterministic_dataset,
        hpo_config,
    ):
        """pytest wrapper for k-fold validation test."""
        results = test_kfold_validation(
            dataset_path=deterministic_dataset,
            hpo_config=hpo_config,
        )
        assert results is not None, "Deterministic dataset not found"
        assert results["success"], f"K-fold validation failed: {results.get('errors', [])}"

    def test_edge_case_k_too_large_pytest(
        self,
        deterministic_dataset,
    ):
        """pytest wrapper for k > n_samples edge case test."""
        results = test_edge_case_k_too_large(
            dataset_path=deterministic_dataset,
        )
        assert results is not None, "Deterministic dataset not found"
        assert results["success"], "Edge case k > n_samples should succeed (expected error caught)"

    def test_edge_cases_suite_pytest(
        self,
        deterministic_dataset,
        hpo_config,
        train_config,
    ):
        """pytest wrapper for edge cases suite test."""
        results = test_edge_cases_suite(
            dataset_path=deterministic_dataset,
            hpo_config=hpo_config,
            train_config=train_config,
        )
        assert results is not None, "Deterministic dataset not found"
        assert results[
            "overall_success"], f"Edge cases test failed: {results.get('error', 'Unknown error')}"
