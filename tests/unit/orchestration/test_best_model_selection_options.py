"""Unit tests for best_model_selection.yaml config option extraction and defaults."""

import pytest


class TestBestModelSelectionConfigOptions:
    """Test extraction of individual config options from best_model_selection.yaml."""

    def test_run_mode_extraction(self):
        """Test run.mode extraction from config."""
        selection_config = {
            "run": {
                "mode": "force_new"
            }
        }
        
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        
        assert mode == "force_new"
        assert isinstance(mode, str)

    def test_run_mode_default(self):
        """Test run.mode default value when missing."""
        selection_config = {
            "run": {}
        }
        
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        
        assert mode == "reuse_if_exists"

    def test_run_mode_reuse_if_exists(self):
        """Test run.mode with reuse_if_exists value."""
        selection_config = {
            "run": {
                "mode": "reuse_if_exists"
            }
        }
        
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        
        assert mode == "reuse_if_exists"

    def test_objective_metric_extraction(self):
        """Test objective.metric extraction from config."""
        selection_config = {
            "objective": {
                "metric": "macro-f1"
            }
        }
        
        metric = selection_config.get("objective", {}).get("metric", "macro-f1")
        
        assert metric == "macro-f1"
        assert isinstance(metric, str)

    def test_objective_metric_default(self):
        """Test objective.metric default value when missing."""
        selection_config = {
            "objective": {}
        }
        
        metric = selection_config.get("objective", {}).get("metric", "macro-f1")
        
        assert metric == "macro-f1"

    def test_objective_goal_extraction(self):
        """Test objective.goal extraction from config."""
        selection_config = {
            "objective": {
                "goal": "maximize"
            }
        }
        
        goal = selection_config.get("objective", {}).get("goal", "maximize")
        
        assert goal == "maximize"
        assert isinstance(goal, str)

    def test_objective_goal_minimize(self):
        """Test objective.goal with minimize value."""
        selection_config = {
            "objective": {
                "goal": "minimize"
            }
        }
        
        goal = selection_config.get("objective", {}).get("goal", "maximize")
        
        assert goal == "minimize"

    def test_objective_goal_default(self):
        """Test objective.goal default value when missing."""
        selection_config = {
            "objective": {}
        }
        
        goal = selection_config.get("objective", {}).get("goal", "maximize")
        
        assert goal == "maximize"

    def test_scoring_f1_weight_extraction(self):
        """Test scoring.f1_weight extraction from config."""
        selection_config = {
            "scoring": {
                "f1_weight": 0.7
            }
        }
        
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        
        assert f1_weight == 0.7
        assert isinstance(f1_weight, (int, float))

    def test_scoring_f1_weight_default(self):
        """Test scoring.f1_weight default value when missing."""
        selection_config = {
            "scoring": {}
        }
        
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        
        assert f1_weight == 0.7

    def test_scoring_latency_weight_extraction(self):
        """Test scoring.latency_weight extraction from config."""
        selection_config = {
            "scoring": {
                "latency_weight": 0.3
            }
        }
        
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        
        assert latency_weight == 0.3
        assert isinstance(latency_weight, (int, float))

    def test_scoring_latency_weight_default(self):
        """Test scoring.latency_weight default value when missing."""
        selection_config = {
            "scoring": {}
        }
        
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        
        assert latency_weight == 0.3

    def test_scoring_normalize_weights_extraction_true(self):
        """Test scoring.normalize_weights extraction when true."""
        selection_config = {
            "scoring": {
                "normalize_weights": True
            }
        }
        
        normalize = selection_config.get("scoring", {}).get("normalize_weights", True)
        
        assert normalize is True
        assert isinstance(normalize, bool)

    def test_scoring_normalize_weights_extraction_false(self):
        """Test scoring.normalize_weights extraction when false."""
        selection_config = {
            "scoring": {
                "normalize_weights": False
            }
        }
        
        normalize = selection_config.get("scoring", {}).get("normalize_weights", True)
        
        assert normalize is False

    def test_scoring_normalize_weights_default(self):
        """Test scoring.normalize_weights default value when missing."""
        selection_config = {
            "scoring": {}
        }
        
        normalize = selection_config.get("scoring", {}).get("normalize_weights", True)
        
        assert normalize is True

    def test_benchmark_required_metrics_extraction(self):
        """Test benchmark.required_metrics extraction from config."""
        selection_config = {
            "benchmark": {
                "required_metrics": ["latency_batch_1_ms"]
            }
        }
        
        metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        assert metrics == ["latency_batch_1_ms"]
        assert isinstance(metrics, list)
        assert all(isinstance(m, str) for m in metrics)

    def test_benchmark_required_metrics_multiple(self):
        """Test benchmark.required_metrics with multiple metrics."""
        selection_config = {
            "benchmark": {
                "required_metrics": ["latency_batch_1_ms", "throughput_samples_per_sec"]
            }
        }
        
        metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        assert len(metrics) == 2
        assert "latency_batch_1_ms" in metrics
        assert "throughput_samples_per_sec" in metrics

    def test_benchmark_required_metrics_default(self):
        """Test benchmark.required_metrics default value when missing."""
        selection_config = {
            "benchmark": {}
        }
        
        metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        assert metrics == []

    def test_all_options_together(self):
        """Test extracting all options from a complete config."""
        selection_config = {
            "run": {
                "mode": "force_new"
            },
            "objective": {
                "metric": "macro-f1",
                "goal": "maximize"
            },
            "scoring": {
                "f1_weight": 0.7,
                "latency_weight": 0.3,
                "normalize_weights": True
            },
            "benchmark": {
                "required_metrics": ["latency_batch_1_ms"]
            }
        }
        
        # Extract all options
        mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
        metric = selection_config.get("objective", {}).get("metric", "macro-f1")
        goal = selection_config.get("objective", {}).get("goal", "maximize")
        f1_weight = selection_config.get("scoring", {}).get("f1_weight", 0.7)
        latency_weight = selection_config.get("scoring", {}).get("latency_weight", 0.3)
        normalize = selection_config.get("scoring", {}).get("normalize_weights", True)
        required_metrics = selection_config.get("benchmark", {}).get("required_metrics", [])
        
        # Verify all values
        assert mode == "force_new"
        assert metric == "macro-f1"
        assert goal == "maximize"
        assert f1_weight == 0.7
        assert latency_weight == 0.3
        assert normalize is True
        assert required_metrics == ["latency_batch_1_ms"]

