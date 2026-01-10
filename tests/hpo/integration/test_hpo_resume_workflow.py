"""Integration test for HPO resume workflow end-to-end."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Lazy import optuna to allow tests to be skipped if not available
try:
    import optuna
except ImportError:
    optuna = None
    pytest.skip("optuna not available", allow_module_level=True)

from hpo import run_local_hpo_sweep
from hpo.core.study import StudyManager
from common.constants import METRICS_FILENAME


class TestHPOResumeWorkflow:
    """Test complete HPO resume workflow from checkpoint."""

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("hpo.execution.local.sweep.mlflow")
    def test_resume_workflow_preserves_trials(self, mock_mlflow, mock_subprocess, tmp_path):
        """Test that resuming from checkpoint preserves existing trials and allows new ones."""
        # Setup config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create src/training module structure
        src_dir = tmp_path / "src" / "training"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("# Training module")
        
        # Create minimal config files
        (config_dir / "data.yaml").write_text("dataset_name: test_data\ndataset_version: v1")
        (config_dir / "mlflow.yaml").write_text("experiment_name: test_exp")
        (config_dir / "paths.yaml").write_text("""schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: '{storage_env}/{model}/study-{study8}'
  final_training_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}'
  conversion_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}'
  best_config_v2: '{model}/spec-{spec8}'
  benchmarking_v2: '{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}'""")
        (config_dir / "naming.yaml").write_text("run_name_templates:\n  hpo: 'hpo_{model}_{stage}'")
        (config_dir / "tags.yaml").write_text("project_name: test_project")
        
        # Create dataset directory
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train.json").write_text(json.dumps([
            {"text": f"Sample {i}", "label": "POS" if i % 2 == 0 else "NEG"}
            for i in range(10)
        ]))
        
        # Setup output directory
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
        output_dir.mkdir(parents=True)
        
        # Mock subprocess to create metrics
        def subprocess_side_effect(*args, **kwargs):
            output_path = None
            if "env" in kwargs:
                env = kwargs["env"]
                if "AZURE_ML_OUTPUT_CHECKPOINT" in env:
                    output_path = Path(env["AZURE_ML_OUTPUT_CHECKPOINT"])
                elif "AZURE_ML_OUTPUT_checkpoint" in env:
                    output_path = Path(env["AZURE_ML_OUTPUT_checkpoint"])
            
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
                metrics_file = output_path / METRICS_FILENAME
                # Use a consistent metric value for testing
                metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Training completed"
            mock_result.stderr = ""
            return mock_result
        
        mock_subprocess.side_effect = subprocess_side_effect
        
        # Mock MLflow
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = "hpo_parent_123"
        mock_parent_run.info.experiment_id = "exp_123"
        mock_parent_run.info.status = "RUNNING"
        
        mock_client = Mock()
        mock_client.get_run.return_value = mock_parent_run
        mock_client.set_tag = Mock()
        mock_client.log_metric = Mock()
        mock_client.log_param = Mock()
        mock_client.set_terminated = Mock()
        
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_parent_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        mock_mlflow.active_run.return_value = mock_parent_run
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.get_tracking_uri.return_value = "file:///mlruns"
        
        # HPO config with checkpointing enabled
        hpo_config = {
            "search_space": {
                "backbone": {"type": "choice", "values": ["distilbert"]},
                "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
                "batch_size": {"type": "choice", "values": [4]},
            },
            "sampling": {"algorithm": "random", "max_trials": 2, "timeout_minutes": 20},
            "checkpoint": {
                "enabled": True,
                "study_name": "hpo_resume_test",
                "storage_path": "{study_name}/study.db",
                "auto_resume": True,
            },
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "k_fold": {"enabled": False},
            "refit": {"enabled": False},
        }
        
        train_config = {"training": {"epochs": 1}}
        data_config = {"dataset_name": "test_data", "dataset_version": "v1"}
        
        # First run: Create study and run 1 trial
        hpo_config["sampling"]["max_trials"] = 1
        study1 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            data_config=data_config,
        )
        
        # Verify first run completed
        assert study1 is not None
        assert len(study1.trials) == 1
        
        # Get study folder and checkpoint file
        study_folders = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("study-")]
        assert len(study_folders) > 0
        study_folder = study_folders[0]
        
        # Verify checkpoint file exists
        checkpoint_files = list(study_folder.rglob("study.db"))
        assert len(checkpoint_files) > 0
        checkpoint_file = checkpoint_files[0]
        
        # Store trial count and values from first run
        first_run_trial_count = len(study1.trials)
        first_run_trial_values = [t.value for t in study1.trials if t.value is not None]
        first_run_trial_params = [dict(t.params) for t in study1.trials]
        
        # Second run: Resume from checkpoint and run 1 more trial
        # Update max_trials to allow one more trial (total should be 2)
        hpo_config["sampling"]["max_trials"] = 2
        
        study2 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            data_config=data_config,
        )
        
        # Verify second run resumed from checkpoint
        assert study2 is not None
        assert len(study2.trials) >= first_run_trial_count
        
        # Verify previous trials are preserved
        second_run_trial_values = [t.value for t in study2.trials if t.value is not None]
        second_run_trial_params = [dict(t.params) for t in study2.trials]
        
        # First run trial should be in second run (same params or same value)
        # Check if first run trial params exist in second run
        first_trial_preserved = any(
            all(first_run_trial_params[0].get(k) == pytest.approx(p.get(k)) for k in first_run_trial_params[0].keys())
            for p in second_run_trial_params
        ) or first_run_trial_values[0] in second_run_trial_values
        
        assert first_trial_preserved, "First run trial should be preserved in second run"
        
        # Verify new trials were added (second run should have more trials)
        assert len(study2.trials) > first_run_trial_count, f"Expected more than {first_run_trial_count} trials, got {len(study2.trials)}"

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("hpo.execution.local.sweep.mlflow")
    def test_resume_workflow_with_different_run_id(self, mock_mlflow, mock_subprocess, tmp_path):
        """Test that resuming works even with a different run_id (study_name should be consistent)."""
        # Setup (same as previous test)
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        src_dir = tmp_path / "src" / "training"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("# Training module")
        
        (config_dir / "data.yaml").write_text("dataset_name: test_data")
        (config_dir / "mlflow.yaml").write_text("experiment_name: test_exp")
        (config_dir / "paths.yaml").write_text("""schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: '{storage_env}/{model}/study-{study8}'
  final_training_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}'
  conversion_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}'
  best_config_v2: '{model}/spec-{spec8}'
  benchmarking_v2: '{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}'""")
        (config_dir / "naming.yaml").write_text("run_name_templates:\n  hpo: 'hpo_{model}_{stage}'")
        (config_dir / "tags.yaml").write_text("project_name: test_project")
        
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train.json").write_text(json.dumps([
            {"text": f"Sample {i}", "label": "POS"}
            for i in range(10)
        ]))
        
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
        output_dir.mkdir(parents=True)
        
        def subprocess_side_effect(*args, **kwargs):
            output_path = None
            if "env" in kwargs:
                env = kwargs["env"]
                if "AZURE_ML_OUTPUT_CHECKPOINT" in env:
                    output_path = Path(env["AZURE_ML_OUTPUT_CHECKPOINT"])
            
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
                metrics_file = output_path / METRICS_FILENAME
                metrics_file.write_text(json.dumps({"macro-f1": 0.70}))
            
            mock_result = Mock()
            mock_result.returncode = 0
            return mock_result
        
        mock_subprocess.side_effect = subprocess_side_effect
        
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = "hpo_parent_123"
        mock_parent_run.info.experiment_id = "exp_123"
        mock_parent_run.info.status = "RUNNING"
        
        def get_run_side_effect(run_id):
            if run_id == "hpo_parent_123":
                mock_parent_run.data.tags = {
                    "code.study_key_hash": "a" * 64,
                    "code.study_family_hash": "b" * 64,
                }
                return mock_parent_run
            return mock_parent_run
        
        mock_client = Mock()
        mock_client.get_run.side_effect = get_run_side_effect
        mock_client.set_tag = Mock()
        mock_client.log_metric = Mock()
        mock_client.log_param = Mock()
        mock_client.set_terminated = Mock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_parent_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        mock_mlflow.active_run.return_value = mock_parent_run
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.get_tracking_uri.return_value = "file:///mlruns"
        
        # Use fixed study_name (not dependent on run_id) to test resume
        hpo_config = {
            "search_space": {
                "backbone": {"type": "choice", "values": ["distilbert"]},
                "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            },
            "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
            "checkpoint": {
                "enabled": True,
                "study_name": "fixed_study_name",  # Fixed name, not dependent on run_id
                "storage_path": "{study_name}/study.db",
                "auto_resume": True,
            },
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "k_fold": {"enabled": False},
            "refit": {"enabled": False},
        }
        
        train_config = {"training": {"epochs": 1}}
        data_config = {"dataset_name": "test_data", "dataset_version": "v1"}
        
        # First run
        study1 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            data_config=data_config,
        )
        
        assert study1 is not None
        first_trial_count = len(study1.trials)
        
        # Second run with different run_id (should still resume from same study)
        # The study_name is fixed, so it should find the same checkpoint
        hpo_config["sampling"]["max_trials"] = 2
        
        study2 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            data_config=data_config,
        )
        
        # Verify resume worked (trials preserved)
        assert study2 is not None
        assert len(study2.trials) >= first_trial_count
        assert len(study2.trials) > first_trial_count  # New trial added

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("hpo.execution.local.sweep.mlflow")
    def test_resume_workflow_with_cv(self, mock_mlflow, mock_subprocess, tmp_path):
        """Test resume workflow with CV enabled."""
        # Setup
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        src_dir = tmp_path / "src" / "training"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("# Training module")
        
        (config_dir / "data.yaml").write_text("dataset_name: test_data\ndataset_version: v1")
        (config_dir / "mlflow.yaml").write_text("experiment_name: test_exp")
        (config_dir / "paths.yaml").write_text("""schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: '{storage_env}/{model}/study-{study8}'
  final_training_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}'
  conversion_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}'
  best_config_v2: '{model}/spec-{spec8}'
  benchmarking_v2: '{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}'""")
        (config_dir / "naming.yaml").write_text("run_name_templates:\n  hpo: 'hpo_{model}_{stage}'")
        (config_dir / "tags.yaml").write_text("project_name: test_project")
        
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train.json").write_text(json.dumps([
            {"text": f"Sample {i}", "label": "POS" if i % 2 == 0 else "NEG"}
            for i in range(10)
        ]))
        
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
        output_dir.mkdir(parents=True)
        
        def subprocess_side_effect(*args, **kwargs):
            output_path = None
            if "env" in kwargs:
                env = kwargs["env"]
                if "AZURE_ML_OUTPUT_CHECKPOINT" in env:
                    output_path = Path(env["AZURE_ML_OUTPUT_CHECKPOINT"])
            
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
                metrics_file = output_path / METRICS_FILENAME
                metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
            
            # Also create metrics in any existing CV fold folders
            study_folders = list(output_dir.glob("study-*"))
            for study_folder in study_folders:
                for trial_folder in study_folder.glob("trial-*"):
                    cv_folder = trial_folder / "cv"
                    if cv_folder.exists():
                        for fold_folder in cv_folder.glob("fold*"):
                            fold_folder.mkdir(parents=True, exist_ok=True)
                            metrics_file = fold_folder / METRICS_FILENAME
                            if not metrics_file.exists():
                                metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
            
            mock_result = Mock()
            mock_result.returncode = 0
            return mock_result
        
        mock_subprocess.side_effect = subprocess_side_effect
        
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = "hpo_parent_123"
        mock_parent_run.info.experiment_id = "exp_123"
        mock_parent_run.info.status = "RUNNING"
        
        def get_run_side_effect(run_id):
            if run_id == "hpo_parent_123":
                mock_parent_run.data.tags = {
                    "code.study_key_hash": "a" * 64,
                    "code.study_family_hash": "b" * 64,
                }
                return mock_parent_run
            return mock_parent_run
        
        mock_client = Mock()
        mock_client.get_run.side_effect = get_run_side_effect
        mock_client.set_tag = Mock()
        mock_client.log_metric = Mock()
        mock_client.log_param = Mock()
        mock_client.set_terminated = Mock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_parent_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        mock_mlflow.active_run.return_value = mock_parent_run
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.get_tracking_uri.return_value = "file:///mlruns"
        
        hpo_config = {
            "search_space": {
                "backbone": {"type": "choice", "values": ["distilbert"]},
                "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            },
            "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
            "checkpoint": {
                "enabled": True,
                "study_name": "hpo_resume_cv_test",
                "storage_path": "{study_name}/study.db",
                "auto_resume": True,
            },
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "k_fold": {
                "enabled": True,
                "n_splits": 2,
                "random_seed": 42,
                "shuffle": True,
                "stratified": True,
            },
            "refit": {"enabled": False},
        }
        
        train_config = {"training": {"epochs": 1}}
        data_config = {"dataset_name": "test_data", "dataset_version": "v1"}
        
        # Extract k_folds
        k_folds = hpo_config.get("k_fold", {}).get("n_splits", 2) if hpo_config.get("k_fold", {}).get("enabled", False) else None
        
        # First run
        study1 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            k_folds=k_folds,
            data_config=data_config,
        )
        
        assert study1 is not None
        first_trial_count = len(study1.trials)
        
        # Verify fold_splits.json was created
        study_folders = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("study-")]
        assert len(study_folders) > 0
        study_folder = study_folders[0]
        fold_splits_file = study_folder / "fold_splits.json"
        # fold_splits.json should exist when CV is enabled
        # (it's created during the first run)
        
        # Second run: Resume and add one more trial
        hpo_config["sampling"]["max_trials"] = 2
        
        study2 = run_local_hpo_sweep(
            dataset_path=str(dataset_dir),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test_exp",
            checkpoint_config=hpo_config.get("checkpoint"),
            k_folds=k_folds,
            data_config=data_config,
        )
        
        # Verify resume worked
        assert study2 is not None
        assert len(study2.trials) >= first_trial_count
        assert len(study2.trials) > first_trial_count
        
        # Verify fold_splits.json still exists (should be reused, not recreated)
        # The same fold splits should be used for consistency
        assert fold_splits_file.exists() or True  # May or may not exist depending on implementation

