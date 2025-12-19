"""Tests for local HPO sweep utilities."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, mock_open
from orchestration.jobs.local_sweeps import (
    translate_search_space_to_optuna,
    create_optuna_pruner,
    run_training_trial,
)


class TestTranslateSearchSpaceToOptuna:
    """Tests for translate_search_space_to_optuna function."""

    def test_choice_distribution(self):
        """Test translating choice distribution."""
        hpo_config = {
            "search_space": {
                "backbone": {
                    "type": "choice",
                    "values": ["distilbert", "deberta"],
                },
            },
        }
        
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.return_value = "distilbert"
        
        params = translate_search_space_to_optuna(hpo_config, mock_trial)
        
        assert params["backbone"] == "distilbert"
        mock_trial.suggest_categorical.assert_called_once_with(
            "backbone", ["distilbert", "deberta"]
        )

    def test_uniform_distribution(self):
        """Test translating uniform distribution."""
        hpo_config = {
            "search_space": {
                "dropout": {
                    "type": "uniform",
                    "min": 0.1,
                    "max": 0.3,
                },
            },
        }
        
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.2
        
        params = translate_search_space_to_optuna(hpo_config, mock_trial)
        
        assert params["dropout"] == 0.2
        mock_trial.suggest_float.assert_called_once_with(
            "dropout", 0.1, 0.3
        )

    def test_loguniform_distribution(self):
        """Test translating loguniform distribution."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
            },
        }
        
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 2e-5
        
        params = translate_search_space_to_optuna(hpo_config, mock_trial)
        
        assert params["learning_rate"] == 2e-5
        mock_trial.suggest_float.assert_called_once_with(
            "learning_rate", 1e-5, 5e-5, log=True
        )

    def test_multiple_parameters(self):
        """Test translating multiple parameters."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
                "batch_size": {
                    "type": "choice",
                    "values": [4, 8, 16],
                },
                "dropout": {
                    "type": "uniform",
                    "min": 0.1,
                    "max": 0.3,
                },
            },
        }
        
        mock_trial = MagicMock()
        mock_trial.suggest_float.side_effect = [2e-5, 0.2]
        mock_trial.suggest_categorical.return_value = 8
        
        params = translate_search_space_to_optuna(hpo_config, mock_trial)
        
        assert params["learning_rate"] == 2e-5
        assert params["batch_size"] == 8
        assert params["dropout"] == 0.2


class TestCreateOptunaPruner:
    """Tests for create_optuna_pruner function."""

    @patch("orchestration.jobs.local_sweeps._import_optuna")
    def test_create_median_pruner(self, mock_import_optuna):
        """Test creating median pruner."""
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import RandomSampler
        from optuna.trial import Trial
        
        mock_import_optuna.return_value = (optuna, MedianPruner, RandomSampler, Trial)
        
        hpo_config = {
            "early_termination": {
                "policy": "median",
            },
        }
        
        pruner = create_optuna_pruner(hpo_config)
        
        assert isinstance(pruner, MedianPruner)

    @patch("orchestration.jobs.local_sweeps._import_optuna")
    def test_no_pruner(self, mock_import_optuna):
        """Test that None is returned when no pruner is configured."""
        hpo_config = {}
        
        pruner = create_optuna_pruner(hpo_config)
        
        assert pruner is None

    @patch("orchestration.jobs.local_sweeps._import_optuna")
    def test_no_early_termination(self, mock_import_optuna):
        """Test that None is returned when early_termination is not configured."""
        hpo_config = {
            "search_space": {},
        }
        
        pruner = create_optuna_pruner(hpo_config)
        
        assert pruner is None


class TestRunTrainingTrial:
    """Tests for run_training_trial function."""

    @patch("orchestration.jobs.local_sweeps.subprocess.run")
    @patch("orchestration.jobs.local_sweeps.mlflow")
    def test_run_training_trial_success(self, mock_mlflow, mock_subprocess, temp_dir):
        """Test successful execution of training trial."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        trial_params = {
            "learning_rate": 3e-5,
            "dropout": 0.2,
            "trial_number": 1
        }
        config_dir = temp_dir / "config"
        config_dir.mkdir(parents=True)
        output_dir = temp_dir / "outputs"
        output_dir.mkdir(parents=True)
        
        train_config = {"training": {"epochs": 1}}
        
        # Create metrics.json file in the trial output directory (created by run_training_trial)
        # The function creates: output_dir / f"trial_{trial_number}"
        trial_output_dir = output_dir / "trial_1"
        trial_output_dir.mkdir(parents=True)
        metrics_file = trial_output_dir / "metrics.json"
        import json
        with open(metrics_file, "w") as f:
            json.dump({"macro-f1": 0.95}, f)
        
        result = run_training_trial(
            trial_params=trial_params,
            dataset_path="/data/dataset",
            config_dir=config_dir,
            backbone="bert-base-uncased",
            output_dir=output_dir,
            train_config=train_config,
            mlflow_experiment_name="test-experiment",
            objective_metric="macro-f1"
        )
        
        assert result == 0.95
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert "--backbone" in call_args[0][0]
        assert "bert-base-uncased" in call_args[0][0]

    @patch("orchestration.jobs.local_sweeps.subprocess.run")
    @patch("orchestration.jobs.local_sweeps.mlflow")
    def test_run_training_trial_failure(self, mock_mlflow, mock_subprocess, temp_dir):
        """Test handling of failed training trial."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Training failed"
        mock_subprocess.return_value = mock_result
        
        trial_params = {"trial_number": 1}
        config_dir = temp_dir / "config"
        config_dir.mkdir(parents=True)
        output_dir = temp_dir / "outputs"
        
        train_config = {"training": {"epochs": 1}}
        
        with pytest.raises(RuntimeError, match="Training trial failed"):
            run_training_trial(
                trial_params=trial_params,
                dataset_path="/data/dataset",
                config_dir=config_dir,
                backbone="bert-base-uncased",
                output_dir=output_dir,
                train_config=train_config,
                mlflow_experiment_name="test-experiment",
                objective_metric="macro-f1"
            )

    @patch("orchestration.jobs.local_sweeps.subprocess.run")
    @patch("orchestration.jobs.local_sweeps.mlflow")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_run_training_trial_fallback_to_mlflow(self, mock_file, mock_mlflow, mock_subprocess, temp_dir):
        """Test fallback to MLflow when metrics.json is not found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        mock_runs = MagicMock()
        mock_runs.empty = False
        mock_runs.iloc = [MagicMock()]
        mock_runs.iloc[0] = {"run_id": "run-123"}
        mock_mlflow.search_runs.return_value = mock_runs
        
        mock_run = MagicMock()
        mock_run.data.metrics = {"macro-f1": 0.92}
        mock_mlflow.get_run.return_value = mock_run
        
        trial_params = {"trial_number": 1}
        config_dir = temp_dir / "config"
        config_dir.mkdir(parents=True)
        output_dir = temp_dir / "outputs"
        
        train_config = {"training": {"epochs": 1}}
        
        result = run_training_trial(
            trial_params=trial_params,
            dataset_path="/data/dataset",
            config_dir=config_dir,
            backbone="bert-base-uncased",
            output_dir=output_dir,
            train_config=train_config,
            mlflow_experiment_name="test-experiment",
            objective_metric="macro-f1"
        )
        
        assert result == 0.92
        mock_mlflow.get_experiment_by_name.assert_called_once_with("test-experiment")

    @patch("orchestration.jobs.local_sweeps.subprocess.run")
    @patch("orchestration.jobs.local_sweeps.mlflow")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_run_training_trial_no_metrics(self, mock_file, mock_mlflow, mock_subprocess, temp_dir):
        """Test that 0.0 is returned when no metrics are found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        mock_mlflow.get_experiment_by_name.return_value = None
        
        trial_params = {"trial_number": 1}
        config_dir = temp_dir / "config"
        config_dir.mkdir(parents=True)
        output_dir = temp_dir / "outputs"
        
        train_config = {"training": {"epochs": 1}}
        
        result = run_training_trial(
            trial_params=trial_params,
            dataset_path="/data/dataset",
            config_dir=config_dir,
            backbone="bert-base-uncased",
            output_dir=output_dir,
            train_config=train_config,
            mlflow_experiment_name="test-experiment",
            objective_metric="macro-f1"
        )
        
        assert result == 0.0

    @patch("orchestration.jobs.local_sweeps.subprocess.run")
    @patch("orchestration.jobs.local_sweeps.mlflow")
    @patch("builtins.open", new_callable=mock_open, read_data='{"precision": 0.92}')
    def test_run_training_trial_missing_objective_metric(self, mock_file, mock_mlflow, mock_subprocess, temp_dir):
        """Test handling when objective metric is missing from metrics.json."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        trial_params = {"trial_number": 1}
        config_dir = temp_dir / "config"
        config_dir.mkdir(parents=True)
        output_dir = temp_dir / "outputs"
        
        train_config = {"training": {"epochs": 1}}
        
        result = run_training_trial(
            trial_params=trial_params,
            dataset_path="/data/dataset",
            config_dir=config_dir,
            backbone="bert-base-uncased",
            output_dir=output_dir,
            train_config=train_config,
            mlflow_experiment_name="test-experiment",
            objective_metric="macro-f1"  # Not in metrics.json
        )
        
        # Should fallback to MLflow or return 0.0
        assert result == 0.0

