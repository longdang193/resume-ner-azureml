"""Edge case and validation tests for best_model_selection.yaml configuration."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestration.jobs.selection.cache import compute_selection_cache_key
from orchestration.jobs.selection.mlflow_selection import find_best_model_from_mlflow


class TestSelectionConfigEdgeCases:
    """Test edge cases and validation for selection configuration."""

    def test_missing_run_section_uses_defaults(self):
        """Test that missing run section uses defaults."""
        selection_config = {
            "objective": {
                "metric": "macro-f1"
            }
        }
        
        # Extract with default fallback
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        
        assert mode == "reuse_if_exists"

    def test_missing_objective_section_handled(self):
        """Test that missing objective section is handled."""
        selection_config = {
            "scoring": {
                "f1_weight": 0.7
            }
        }
        
        # Extract with defaults
        metric = selection_config.get("objective", {}).get("metric", "macro-f1")
        goal = selection_config.get("objective", {}).get("goal", "maximize")
        
        assert metric == "macro-f1"
        assert goal == "maximize"

    def test_missing_scoring_section_uses_defaults(self):
        """Test that missing scoring section uses defaults."""
        selection_config = {
            "objective": {
                "metric": "macro-f1"
            }
        }
        
        # Extract with defaults
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        normalize_weights = selection_config.get("scoring", {}).get("normalize_weights", True)
        
        assert f1_weight == 0.7
        assert latency_weight == 0.3
        assert normalize_weights is True

    def test_negative_weights_handled(self):
        """Test that negative weights are possible (validation should catch)."""
        selection_config = {
            "scoring": {
                "f1_weight": -0.5,
                "latency_weight": 0.5
            }
        }
        
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        
        # Config loader doesn't validate, so negative value is returned
        assert f1_weight == -0.5
        assert latency_weight == 0.5

    def test_zero_weights_handled(self):
        """Test that zero weights are possible."""
        selection_config = {
            "scoring": {
                "f1_weight": 0.0,
                "latency_weight": 1.0
            }
        }
        
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        
        assert f1_weight == 0.0
        assert latency_weight == 1.0

    def test_weight_normalization_zero_sum(self):
        """Test weight normalization when sum is zero."""
        selection_config = {
            "scoring": {
                "f1_weight": 0.0,
                "latency_weight": 0.0,
                "normalize_weights": True
            }
        }
        
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        normalize_weights = selection_config.get("scoring", {}).get("normalize_weights", True)
        
        # When normalize_weights=True and sum=0, normalization should handle gracefully
        # (Implementation at line 77-81 in mlflow_selection.py checks total_weight > 0)
        if normalize_weights:
            total_weight = f1_weight + latency_weight
            if total_weight > 0:
                f1_weight = f1_weight / total_weight
                latency_weight = latency_weight / total_weight
        
        # When sum is zero, weights remain unchanged (division by zero avoided)
        assert f1_weight == 0.0
        assert latency_weight == 0.0

    def test_empty_required_metrics_list(self):
        """Test that empty required_metrics list is handled."""
        selection_config = {
            "benchmark": {
                "required_metrics": []
            }
        }
        
        required_metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        assert required_metrics == []
        assert isinstance(required_metrics, list)

    def test_missing_benchmark_section_uses_defaults(self):
        """Test that missing benchmark section uses defaults."""
        selection_config = {
            "objective": {
                "metric": "macro-f1"
            }
        }
        
        # Extract with defaults
        required_metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        assert required_metrics == []

    def test_cache_key_with_missing_sections(self):
        """Test that cache key computation handles missing config sections."""
        minimal_config = {
            "objective": {
                "metric": "macro-f1",
                "goal": "maximize"
            },
            "scoring": {
                "f1_weight": 0.7,
                "latency_weight": 0.3
            },
            "benchmark": {
                "required_metrics": ["latency_batch_1_ms"]
            }
        }
        
        tags_config = {
            "grouping": {
                "study_key_hash": "tags.grouping.study_key_hash"
            }
        }
        
        # Should compute cache key without error
        cache_key = compute_selection_cache_key(
            experiment_name="test_experiment",
            selection_config=minimal_config,
            tags_config=tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 16

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_missing_required_metrics(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
        mock_benchmark_run,
    ):
        """Test that missing required metrics filters out benchmark runs."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Benchmark run without required metrics
        benchmark_run = mock_benchmark_run
        # Remove required metric
        del benchmark_run.data.metrics["latency_batch_1_ms"]
        
        # Mock search_runs to return benchmark run without required metrics
        mock_client.search_runs.return_value = [benchmark_run]
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=sample_selection_config,
            use_python_filtering=True,
        )
        
        # Should return None because no valid benchmark runs (filtered out at line 115-116)
        assert result is None

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_find_best_model_empty_required_metrics(
        self,
        mock_client_class,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that empty required_metrics list allows all runs."""
        # Modify config to have empty required_metrics
        config = sample_selection_config.copy()
        config["benchmark"]["required_metrics"] = []
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search_runs.return_value = []
        
        # Call function
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=config,
            use_python_filtering=True,
        )
        
        # Function should execute (empty list means no filtering)
        assert mock_client.search_runs.called

    def test_invalid_run_mode_value(self):
        """Test that invalid run.mode value is possible (not validated)."""
        selection_config = {
            "run": {
                "mode": "invalid_mode"
            }
        }
        
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        
        # Config loader doesn't validate mode values
        assert mode == "invalid_mode"



