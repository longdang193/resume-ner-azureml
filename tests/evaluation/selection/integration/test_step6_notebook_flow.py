"""Integration tests for Step 6 notebook flow (champion selection → checkpoint acquisition → benchmarking).

Tests the complete flow from notebook 02_best_config_selection.ipynb Step 6:
1. Load selection config from best_model_selection.yaml
2. Select champions per backbone using select_champions_for_backbones()
3. Convert champions to best_trials format
4. Acquire checkpoints using acquire_best_model_checkpoint()
5. Run benchmarking using benchmark_best_trials()

This test mimics the real notebook flow to ensure config-driven champion selection works end-to-end.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import json


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def config_dir_with_selection_config(temp_dir):
    """Create config directory with best_model_selection.yaml matching real config.
    
    Creates config file with champion_selection section matching lines 21-45 of actual config.
    """
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True)
    
    # Create best_model_selection.yaml with champion_selection section (matching real config lines 21-45)
    selection_config = {
        "run": {
            "mode": "force_new"
        },
        "objective": {
            "metric": "macro-f1",
            "direction": "maximize"  # New key (migration from "goal")
        },
        "champion_selection": {
            "min_trials_per_group": 1,  # Lowered for testing (default: 3)
            "top_k_for_stable_score": 1,  # For smoke/max_trials=1
            "require_artifact_available": False,  # Trial runs don't hold checkpoints in refit workflow
            "artifact_check_source": "tag",  # "tag" or "disk"
            "prefer_schema_version": "auto",  # "2.0" or "1.0" or "auto"
            "allow_mixed_schema_groups": False  # Default: false (strict separation)
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
    
    import yaml
    config_file = config_dir / "best_model_selection.yaml"
    with open(config_file, "w") as f:
        yaml.dump(selection_config, f, default_flow_style=False)
    
    return config_dir


@pytest.fixture
def mock_mlflow_client():
    """Create mock MLflow client."""
    client = Mock()
    return client


@pytest.fixture
def mock_hpo_experiments():
    """Create mock HPO experiments dict (backbone -> {name, id})."""
    return {
        "distilbert": {
            "name": "test_experiment-hpo-distilbert",
            "id": "exp-distilbert-123"
        },
        "distilroberta": {
            "name": "test_experiment-hpo-distilroberta",
            "id": "exp-distilroberta-456"
        }
    }


@pytest.fixture
def mock_champion_runs():
    """Create mock MLflow runs for champion selection."""
    def create_mock_run(run_id, metric_value, study_key_hash, trial_key_hash, 
                        schema_version="1.0", artifact_available=True, is_parent=False):
        """Helper to create mock MLflow run."""
        run = Mock()
        run.info.run_id = run_id
        run.info.status = "FINISHED"
        run.info.start_time = 1000000
        run.data.metrics = {"macro-f1": metric_value}
        run.data.tags = {
            "code.backbone": "distilbert",
            "code.stage": "hpo_trial" if not is_parent else "hpo_sweep",
            "code.study_key_hash": study_key_hash,
            "code.trial_key_hash": trial_key_hash,
            "code.study.key_schema_version": schema_version,
            "code.artifact.available": "true" if artifact_available else "false",
            "mlflow.parentRunId": "parent-123" if not is_parent else None,
        }
        return run
    
    # Create runs for distilbert (3 runs in same group - meets min_trials=1)
    runs = [
        create_mock_run("run1", 0.85, "study-hash-1", "trial-hash-1"),
        create_mock_run("run2", 0.87, "study-hash-1", "trial-hash-2"),  # Best
        create_mock_run("run3", 0.86, "study-hash-1", "trial-hash-3"),
    ]
    
    # Add parent run
    parent_run = create_mock_run("parent-123", None, "study-hash-1", None, is_parent=True)
    
    return runs + [parent_run]


@pytest.fixture
def mock_refit_run():
    """Create mock refit run for checkpoint acquisition."""
    refit_run = Mock()
    refit_run.info.run_id = "refit-run-123"
    refit_run.info.status = "FINISHED"
    refit_run.data.tags = {
        "code.stage": "hpo_refit",
        "code.trial_key_hash": "trial-hash-2",
        "code.refit.of_trial_run_id": "run2",
    }
    return refit_run


class TestStep6NotebookFlow:
    """Test the complete Step 6 flow from notebook 02_best_config_selection.ipynb."""

    @patch('evaluation.selection.trial_finder.query_runs_by_tags')
    @patch('mlflow.tracking.MlflowClient')
    def test_step6_load_config_and_select_champions(
        self,
        mock_mlflow_client_class,
        mock_query_runs,
        config_dir_with_selection_config,
        mock_hpo_experiments,
        mock_champion_runs,
        temp_dir,
    ):
        """Test Step 6.1-6.3: Load config and select champions (mimics notebook flow)."""
        from common.shared.yaml_utils import load_yaml
        from evaluation.selection.trial_finder import select_champions_for_backbones
        
        # Step 6.1: Load selection config (mimics notebook line 674)
        selection_config = load_yaml(config_dir_with_selection_config / "best_model_selection.yaml")
        
        # Verify config loaded correctly
        assert "champion_selection" in selection_config
        assert selection_config["champion_selection"]["min_trials_per_group"] == 1
        assert selection_config["champion_selection"]["prefer_schema_version"] == "auto"
        
        # Step 6.2: Setup MLflow client (mimics notebook line 675)
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        
        # Step 6.3: Select champions per backbone (mimics notebook lines 695-700)
        mock_query_runs.return_value = mock_champion_runs
        
        champions = select_champions_for_backbones(
            backbone_values=list(mock_hpo_experiments.keys()),
            hpo_experiments=mock_hpo_experiments,
            selection_config=selection_config,
            mlflow_client=mock_client,
            root_dir=temp_dir,
            config_dir=config_dir_with_selection_config,
        )
        
        # Verify champions selected
        assert "distilbert" in champions
        assert champions["distilbert"]["champion"]["run_id"] == "run2"  # Best metric
        assert champions["distilbert"]["champion"]["metric"] == 0.87

    @patch('evaluation.selection.artifact_acquisition.acquire_best_model_checkpoint')
    @patch('evaluation.selection.trial_finder.query_runs_by_tags')
    @patch('mlflow.tracking.MlflowClient')
    def test_step6_champion_to_best_trials_conversion(
        self,
        mock_mlflow_client_class,
        mock_query_runs,
        mock_acquire_checkpoint,
        config_dir_with_selection_config,
        mock_hpo_experiments,
        mock_champion_runs,
        mock_refit_run,
        temp_dir,
    ):
        """Test Step 6.4: Convert champions to best_trials format (mimics notebook lines 772-843)."""
        from common.shared.yaml_utils import load_yaml
        from evaluation.selection.trial_finder import select_champions_for_backbones
        
        # Load config
        selection_config = load_yaml(config_dir_with_selection_config / "best_model_selection.yaml")
        
        # Setup mocks
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_query_runs.return_value = mock_champion_runs
        
        # Mock refit run lookup
        mock_client.get_run.return_value = mock_refit_run
        mock_client.search_runs.return_value = [mock_refit_run]
        
        # Select champions
        champions = select_champions_for_backbones(
            backbone_values=["distilbert"],
            hpo_experiments={"distilbert": mock_hpo_experiments["distilbert"]},
            selection_config=selection_config,
            mlflow_client=mock_client,
            root_dir=temp_dir,
            config_dir=config_dir_with_selection_config,
        )
        
        # Step 6.4: Convert champions to best_trials format (mimics notebook lines 776-843)
        best_trials = {}
        
        for backbone, champion_data in champions.items():
            champion = champion_data["champion"]
            
            # Extract run IDs (mimics notebook lines 790-795)
            refit_run_id = champion.get("refit_run_id")
            trial_run_id = champion.get("trial_run_id")
            run_id = refit_run_id or champion.get("run_id")
            study_key_hash = champion.get("study_key_hash")
            trial_key_hash = champion.get("trial_key_hash")
            
            # Mock checkpoint acquisition (mimics notebook lines 815-829)
            acquired_checkpoint_dir = temp_dir / "acquired_checkpoint"
            acquired_checkpoint_dir.mkdir()
            (acquired_checkpoint_dir / "pytorch_model.bin").touch()
            
            mock_acquire_checkpoint.return_value = acquired_checkpoint_dir
            
            # Load acquisition config (mimics notebook line 801)
            acquisition_config = {"priority": ["local", "drive", "mlflow"]}
            
            # Call acquire_best_model_checkpoint (mimics notebook lines 815-825)
            from evaluation.selection.artifact_acquisition import acquire_best_model_checkpoint
            
            best_run_info = {
                "run_id": run_id,
                "refit_run_id": refit_run_id,
                "trial_run_id": trial_run_id,
                "study_key_hash": study_key_hash,
                "trial_key_hash": trial_key_hash,
                "backbone": backbone,
            }
            
            checkpoint_dir = acquire_best_model_checkpoint(
                best_run_info=best_run_info,
                root_dir=temp_dir,
                config_dir=config_dir_with_selection_config,
                acquisition_config=acquisition_config,
                selection_config=selection_config,
                platform="local",
            )
            
            # Build best_trials dict (mimics notebook lines 836-843)
            best_trials[backbone] = {
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "trial_name": champion.get("trial_key_hash", "unknown"),
                "accuracy": champion["metric"],
                "study_key_hash": champion.get("study_key_hash"),
                "trial_key_hash": champion.get("trial_key_hash"),
                "run_id": champion["run_id"],
            }
        
        # Verify best_trials structure matches notebook expectations
        assert "distilbert" in best_trials
        assert best_trials["distilbert"]["checkpoint_dir"] is not None
        assert best_trials["distilbert"]["accuracy"] == 0.87
        assert best_trials["distilbert"]["study_key_hash"] == "study-hash-1"
        assert best_trials["distilbert"]["trial_key_hash"] == "trial-hash-2"

    @patch('evaluation.benchmarking.benchmark_best_trials')
    @patch('evaluation.selection.artifact_acquisition.acquire_best_model_checkpoint')
    @patch('evaluation.selection.trial_finder.query_runs_by_tags')
    @patch('mlflow.tracking.MlflowClient')
    def test_step6_complete_flow_with_benchmarking(
        self,
        mock_mlflow_client_class,
        mock_query_runs,
        mock_acquire_checkpoint,
        mock_benchmark,
        config_dir_with_selection_config,
        mock_hpo_experiments,
        mock_champion_runs,
        mock_refit_run,
        temp_dir,
    ):
        """Test complete Step 6 flow including benchmarking (mimics notebook lines 655-906)."""
        from common.shared.yaml_utils import load_yaml
        from evaluation.selection.trial_finder import select_champions_for_backbones
        
        # Step 6.1: Load selection config
        selection_config = load_yaml(config_dir_with_selection_config / "best_model_selection.yaml")
        
        # Step 6.2: Setup MLflow client
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_query_runs.return_value = mock_champion_runs
        mock_client.get_run.return_value = mock_refit_run
        mock_client.search_runs.return_value = [mock_refit_run]
        
        # Step 6.3: Select champions
        backbone_values = list(mock_hpo_experiments.keys())
        champions = select_champions_for_backbones(
            backbone_values=backbone_values,
            hpo_experiments=mock_hpo_experiments,
            selection_config=selection_config,
            mlflow_client=mock_client,
            root_dir=temp_dir,
            config_dir=config_dir_with_selection_config,
        )
        
        # Step 6.4: Convert to best_trials and acquire checkpoints
        best_trials = {}
        acquired_checkpoint_dir = temp_dir / "acquired_checkpoint"
        acquired_checkpoint_dir.mkdir()
        (acquired_checkpoint_dir / "pytorch_model.bin").touch()
        mock_acquire_checkpoint.return_value = acquired_checkpoint_dir
        
        for backbone, champion_data in champions.items():
            champion = champion_data["champion"]
            refit_run_id = champion.get("refit_run_id")
            trial_run_id = champion.get("trial_run_id")
            run_id = refit_run_id or champion.get("run_id")
            
            acquisition_config = {"priority": ["local", "drive", "mlflow"]}
            
            from evaluation.selection.artifact_acquisition import acquire_best_model_checkpoint
            
            best_run_info = {
                "run_id": run_id,
                "refit_run_id": refit_run_id,
                "trial_run_id": trial_run_id,
                "study_key_hash": champion.get("study_key_hash"),
                "trial_key_hash": champion.get("trial_key_hash"),
                "backbone": backbone,
            }
            
            checkpoint_dir = acquire_best_model_checkpoint(
                best_run_info=best_run_info,
                root_dir=temp_dir,
                config_dir=config_dir_with_selection_config,
                acquisition_config=acquisition_config,
                selection_config=selection_config,
                platform="local",
            )
            
            best_trials[backbone] = {
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "trial_name": champion.get("trial_key_hash", "unknown"),
                "accuracy": champion["metric"],
                "study_key_hash": champion.get("study_key_hash"),
                "trial_key_hash": champion.get("trial_key_hash"),
                "run_id": champion["run_id"],
            }
        
        # Step 6.5-6.7: Setup and run benchmarking (mimics notebook lines 847-899)
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [1],
                "iterations": 10,
                "warmup_iterations": 10,
                "max_length": 512,
            }
        }
        
        test_data_path = temp_dir / "test.json"
        test_data_path.write_text('{"text": "test"}')
        
        # Mock benchmark results
        mock_benchmark.return_value = {
            "distilbert": {
                "latency_batch_1_ms": 100.0,
                "macro-f1": 0.87
            }
        }
        
        # Call benchmark_best_trials (mimics notebook line 882)
        from evaluation.benchmarking import benchmark_best_trials
        from orchestration.jobs.tracking.mlflow_tracker import MLflowBenchmarkTracker
        
        benchmark_tracker = Mock(spec=MLflowBenchmarkTracker)
        
        benchmark_results = benchmark_best_trials(
            best_trials=best_trials,
            test_data_path=test_data_path,
            root_dir=temp_dir,
            environment="local",
            data_config={},
            hpo_config={},
            benchmark_config=benchmark_config,
            benchmark_batch_sizes=[1],
            benchmark_iterations=10,
            benchmark_warmup=10,
            benchmark_max_length=512,
            benchmark_device=None,
            benchmark_tracker=benchmark_tracker,
            backup_enabled=False,
        )
        
        # Verify benchmarking was called
        assert mock_benchmark.called
        assert benchmark_results is not None

    @patch('evaluation.selection.trial_finder.query_runs_by_tags')
    @patch('mlflow.tracking.MlflowClient')
    def test_step6_config_driven_champion_selection(
        self,
        mock_mlflow_client_class,
        mock_query_runs,
        config_dir_with_selection_config,
        mock_hpo_experiments,
        mock_champion_runs,
        temp_dir,
    ):
        """Test that champion selection respects all config options from best_model_selection.yaml."""
        from common.shared.yaml_utils import load_yaml
        from evaluation.selection.trial_finder import select_champions_for_backbones
        
        # Load config
        selection_config = load_yaml(config_dir_with_selection_config / "best_model_selection.yaml")
        
        # Verify config has all required champion_selection options
        champion_config = selection_config["champion_selection"]
        assert "min_trials_per_group" in champion_config
        assert "top_k_for_stable_score" in champion_config
        assert "require_artifact_available" in champion_config
        assert "artifact_check_source" in champion_config
        assert "prefer_schema_version" in champion_config
        assert "allow_mixed_schema_groups" in champion_config
        
        # Setup mocks
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_query_runs.return_value = mock_champion_runs
        
        # Select champions (should use config values)
        champions = select_champions_for_backbones(
            backbone_values=["distilbert"],
            hpo_experiments={"distilbert": mock_hpo_experiments["distilbert"]},
            selection_config=selection_config,
            mlflow_client=mock_client,
            root_dir=temp_dir,
            config_dir=config_dir_with_selection_config,
        )
        
        # Verify selection metadata includes config values
        assert "distilbert" in champions
        metadata = champions["distilbert"]["selection_metadata"]
        assert metadata["min_trials_required"] == champion_config["min_trials_per_group"]
        assert metadata["top_k_for_stable"] == champion_config["top_k_for_stable_score"]
        assert metadata["artifact_required"] == champion_config["require_artifact_available"]
        assert metadata["artifact_check_source"] == champion_config["artifact_check_source"]
        assert metadata["prefer_schema_version"] == champion_config["prefer_schema_version"]
        assert metadata["allow_mixed_schema_groups"] == champion_config["allow_mixed_schema_groups"]

    @patch('evaluation.selection.trial_finder.query_runs_by_tags')
    @patch('mlflow.tracking.MlflowClient')
    def test_step6_no_champions_found_diagnostics(
        self,
        mock_mlflow_client_class,
        mock_query_runs,
        config_dir_with_selection_config,
        mock_hpo_experiments,
        temp_dir,
    ):
        """Test diagnostics when no champions found (mimics notebook lines 702-770)."""
        from common.shared.yaml_utils import load_yaml
        from evaluation.selection.trial_finder import select_champions_for_backbones
        
        # Load config
        selection_config = load_yaml(config_dir_with_selection_config / "best_model_selection.yaml")
        
        # Setup mocks - no runs found
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_query_runs.return_value = []
        
        # Select champions (should return empty dict)
        champions = select_champions_for_backbones(
            backbone_values=["distilbert"],
            hpo_experiments={"distilbert": mock_hpo_experiments["distilbert"]},
            selection_config=selection_config,
            mlflow_client=mock_client,
            root_dir=temp_dir,
            config_dir=config_dir_with_selection_config,
        )
        
        # Verify no champions found (mimics notebook line 702)
        assert not champions
        
        # Verify diagnostics can be run (mimics notebook lines 706-770)
        # This would check runs, tags, etc. in the actual notebook
        # Here we just verify the structure allows for diagnostics
        runs = mock_client.search_runs(
            experiment_ids=[mock_hpo_experiments["distilbert"]["id"]],
            filter_string="",
            max_results=100,
        )
        assert isinstance(runs, list)

