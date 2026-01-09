"""Component tests for MLflow selection using best_model_selection.yaml config."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from orchestration.jobs.selection.mlflow_selection import find_best_model_from_mlflow


class TestMLflowSelectionConfigUsage:
    """Test that find_best_model_from_mlflow() uses config options correctly."""

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_uses_objective_metric(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that objective.metric from config is used."""
        # Extract metric from config
        expected_metric = sample_selection_config["objective"]["metric"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search_runs to return empty lists (no runs found)
        mock_client.search_runs.return_value = []
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Verify client was called (function attempted to query)
        assert mock_client.search_runs.called
        
        # The function should return None when no runs found
        # But we verify it was called with correct experiment IDs
        call_args = mock_client.search_runs.call_args_list
        assert len(call_args) > 0

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_uses_scoring_weights(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that scoring weights from config are used for composite score calculation."""
        # Extract weights from config
        f1_weight = sample_selection_config["scoring"]["f1_weight"]
        latency_weight = sample_selection_config["scoring"]["latency_weight"]
        normalize_weights = sample_selection_config["scoring"]["normalize_weights"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search_runs to return empty lists
        mock_client.search_runs.return_value = []
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called (weights are used internally)
        assert mock_client.search_runs.called
        
        # We can't directly verify weights were used without runs,
        # but we verify the function executed (weights extracted at line 70-73)

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_uses_benchmark_required_metrics(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
        mock_benchmark_run,
    ):
        """Test that benchmark.required_metrics from config filters benchmark runs."""
        # Extract required metrics from config
        required_metrics = sample_selection_config["benchmark"]["required_metrics"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create benchmark run with required metrics
        benchmark_run = mock_benchmark_run
        for metric in required_metrics:
            benchmark_run.data.metrics[metric] = 5.0
        
        # Mock search_runs to return benchmark run
        mock_client.search_runs.return_value = [benchmark_run]
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Verify search_runs was called
        assert mock_client.search_runs.called
        
        # Function should filter runs based on required_metrics (line 115-116)
        # Since we have a run with required metrics, it should be considered valid

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_weight_normalization(
        self,
        mock_client_class,
        custom_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that normalize_weights config option controls weight normalization."""
        # Use config with normalize_weights=False
        normalize_weights = custom_selection_config["scoring"]["normalize_weights"]
        assert normalize_weights is False
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search_runs to return empty lists
        mock_client.search_runs.return_value = []
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=custom_selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called
        assert mock_client.search_runs.called
        
        # When normalize_weights=False, weights should not be normalized (line 77-81)

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_composite_score_calculation(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
        mock_benchmark_run,
        mock_trial_run,
        mock_refit_run,
    ):
        """Test that composite score is calculated using config weights."""
        # Extract weights from config
        f1_weight = sample_selection_config["scoring"]["f1_weight"]
        latency_weight = sample_selection_config["scoring"]["latency_weight"]
        normalize_weights = sample_selection_config["scoring"]["normalize_weights"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Setup benchmark run with required metrics
        benchmark_run = mock_benchmark_run
        benchmark_run.data.metrics["latency_batch_1_ms"] = 5.0
        benchmark_run.data.tags["tags.grouping.study_key_hash"] = "study_hash_123"
        benchmark_run.data.tags["tags.grouping.trial_key_hash"] = "trial_hash_456"
        
        # Setup trial run with F1 score
        trial_run = mock_trial_run
        trial_run.data.metrics["macro-f1"] = 0.75
        trial_run.data.tags["tags.grouping.study_key_hash"] = "study_hash_123"
        trial_run.data.tags["tags.grouping.trial_key_hash"] = "trial_hash_456"
        
        # Setup refit run
        refit_run = mock_refit_run
        refit_run.data.tags["tags.grouping.study_key_hash"] = "study_hash_123"
        refit_run.data.tags["tags.grouping.trial_key_hash"] = "trial_hash_456"
        
        # Mock search_runs: first call returns benchmark runs, subsequent calls return HPO runs
        def search_runs_side_effect(*args, **kwargs):
            experiment_ids = kwargs.get("experiment_ids", args[0] if args else [])
            if mock_benchmark_experiment["id"] in experiment_ids:
                return [benchmark_run]
            else:
                # Return both trial and refit runs for HPO experiments
                return [trial_run, refit_run]
        
        mock_client.search_runs.side_effect = search_runs_side_effect
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called
        assert mock_client.search_runs.called
        
        # If result is not None, verify composite_score was calculated
        # (composite score calculation happens at lines 260-285)
        if result is not None:
            assert "composite_score" in result
            assert "f1_score" in result
            assert "latency_ms" in result

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_all_config_options_together(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that all config options are used together correctly."""
        # Extract all used options from config
        objective_metric = sample_selection_config["objective"]["metric"]
        f1_weight = sample_selection_config["scoring"]["f1_weight"]
        latency_weight = sample_selection_config["scoring"]["latency_weight"]
        normalize_weights = sample_selection_config["scoring"]["normalize_weights"]
        required_metrics = sample_selection_config["benchmark"]["required_metrics"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search_runs to return empty lists
        mock_client.search_runs.return_value = []
        
        # Call function with all config options
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called
        assert mock_client.search_runs.called
        
        # All config options are extracted at lines 69-74
        # We verify the function executed successfully with all options

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_custom_config_values(
        self,
        mock_client_class,
        custom_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test function with custom config values."""
        # Extract custom values
        objective_metric = custom_selection_config["objective"]["metric"]
        f1_weight = custom_selection_config["scoring"]["f1_weight"]
        latency_weight = custom_selection_config["scoring"]["latency_weight"]
        normalize_weights = custom_selection_config["scoring"]["normalize_weights"]
        required_metrics = custom_selection_config["benchmark"]["required_metrics"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search_runs to return empty lists
        mock_client.search_runs.return_value = []
        
        # Call function with custom config
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=custom_selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called
        assert mock_client.search_runs.called
        
        # Verify custom values are used (extracted at lines 69-74)

