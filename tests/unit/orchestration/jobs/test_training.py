"""Tests for final training job creation and configuration."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from orchestration.jobs.training import (
    build_final_training_config,
    validate_final_training_job,
    create_final_training_job,
    _build_data_input_from_asset,
    DEFAULT_RANDOM_SEED,
)


class TestBuildFinalTrainingConfig:
    """Tests for build_final_training_config function."""

    def test_build_config_with_hpo_overrides(self):
        """Test building config with HPO hyperparameter overrides."""
        best_config = {
            "backbone": "bert-base-uncased",
            "hyperparameters": {
                "learning_rate": 3e-5,
                "dropout": 0.2,
                "weight_decay": 0.05,
            }
        }
        train_config = {
            "training": {
                "learning_rate": 2e-5,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "batch_size": 16,
                "epochs": 5,
            }
        }
        
        result = build_final_training_config(best_config, train_config)
        
        assert result["backbone"] == "bert-base-uncased"
        assert result["learning_rate"] == 3e-5  # From HPO
        assert result["dropout"] == 0.2  # From HPO
        assert result["weight_decay"] == 0.05  # From HPO
        assert result["batch_size"] == 16  # From train.yaml
        assert result["epochs"] == 5  # From train.yaml
        assert result["random_seed"] == DEFAULT_RANDOM_SEED
        assert result["early_stopping_enabled"] is False
        assert result["use_combined_data"] is True
        assert result["use_all_data"] is True

    def test_build_config_with_defaults(self):
        """Test building config when HPO doesn't override all params."""
        best_config = {
            "backbone": "bert-base-uncased",
            "hyperparameters": {
                "learning_rate": 3e-5,
            }
        }
        train_config = {
            "training": {
                "dropout": 0.1,
                "weight_decay": 0.01,
                "batch_size": 32,
                "epochs": 10,
            }
        }
        
        result = build_final_training_config(best_config, train_config)
        
        assert result["learning_rate"] == 3e-5  # From HPO
        assert result["dropout"] == 0.1  # From train.yaml (HPO didn't override)
        assert result["weight_decay"] == 0.01  # From train.yaml
        assert result["batch_size"] == 32  # From train.yaml
        assert result["epochs"] == 10  # From train.yaml

    def test_build_config_with_custom_seed(self):
        """Test building config with custom random seed."""
        best_config = {"backbone": "bert-base-uncased", "hyperparameters": {}}
        train_config = {"training": {}}
        
        result = build_final_training_config(best_config, train_config, random_seed=123)
        
        assert result["random_seed"] == 123

    def test_build_config_with_empty_hyperparameters(self):
        """Test building config when hyperparameters dict is empty."""
        best_config = {
            "backbone": "bert-base-uncased",
            "hyperparameters": {}
        }
        train_config = {
            "training": {
                "learning_rate": 2e-5,
                "dropout": 0.1,
            }
        }
        
        result = build_final_training_config(best_config, train_config)
        
        assert result["learning_rate"] == 2e-5  # From train.yaml
        assert result["dropout"] == 0.1  # From train.yaml

    def test_build_config_with_missing_training_section(self):
        """Test building config when train_config has no training section."""
        best_config = {
            "backbone": "bert-base-uncased",
            "hyperparameters": {"learning_rate": 3e-5}
        }
        train_config = {}
        
        result = build_final_training_config(best_config, train_config)
        
        assert result["learning_rate"] == 3e-5  # From HPO
        assert result["batch_size"] == 16  # Default
        assert result["epochs"] == 5  # Default


class TestValidateFinalTrainingJob:
    """Tests for validate_final_training_job function."""

    def test_validate_success(self):
        """Test successful validation of completed job."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        
        validate_final_training_job(mock_job)
        
        # Should not raise any exception

    def test_validate_failed_job(self):
        """Test that ValueError is raised for failed job."""
        mock_job = MagicMock()
        mock_job.status = "Failed"
        
        with pytest.raises(ValueError, match="Final training job failed with status: Failed"):
            validate_final_training_job(mock_job)

    def test_validate_canceled_job(self):
        """Test that ValueError is raised for canceled job."""
        mock_job = MagicMock()
        mock_job.status = "Canceled"
        
        with pytest.raises(ValueError, match="Final training job failed with status: Canceled"):
            validate_final_training_job(mock_job)


class TestBuildDataInputFromAsset:
    """Tests for _build_data_input_from_asset function."""

    def test_build_input_uses_asset_reference(self):
        """Test that input uses azureml:name:version format."""
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        result = _build_data_input_from_asset(mock_data_asset)
        
        assert result.path == "azureml:resume-ner-data:2"
        assert result.path.startswith("azureml:")
        assert result.type == "uri_folder"
        assert result.mode == "mount"

    def test_build_input_not_datastore_path(self):
        """Test that input is NOT a manual datastore path."""
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        result = _build_data_input_from_asset(mock_data_asset)
        
        assert not result.path.startswith("azureml://datastores/")


class TestCreateFinalTrainingJob:
    """Tests for create_final_training_job function."""

    def test_create_job_success(self, temp_dir):
        """Test successful creation of final training job."""
        script_path = temp_dir / "train.py"
        script_path.touch()
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        mock_environment = MagicMock()
        final_config = {
            "backbone": "bert-base-uncased",
            "learning_rate": 3e-5,
            "dropout": 0.2,
            "weight_decay": 0.05,
            "batch_size": 16,
            "epochs": 5,
            "random_seed": 42,
            "early_stopping_enabled": False,
            "use_combined_data": True,
        }
        
        with patch("orchestration.jobs.training.command") as mock_command:
            mock_job = MagicMock()
            mock_command.return_value = mock_job
            
            result = create_final_training_job(
                script_path=script_path,
                data_asset=mock_data_asset,
                environment=mock_environment,
                compute_cluster="gpu-cluster",
                final_config=final_config,
                aml_experiment_name="test-experiment",
                tags={"key": "value"}
            )
            
            assert result == mock_job
            mock_command.assert_called_once()
            call_kwargs = mock_command.call_args[1]
            assert call_kwargs["compute"] == "gpu-cluster"
            assert call_kwargs["experiment_name"] == "test-experiment"
            assert call_kwargs["tags"] == {"key": "value"}
            assert call_kwargs["display_name"] == "final-training"

    def test_create_job_script_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised when script doesn't exist."""
        script_path = temp_dir / "nonexistent.py"
        
        with pytest.raises(FileNotFoundError, match="Training script not found"):
            create_final_training_job(
                script_path=script_path,
                data_asset=MagicMock(),
                environment=MagicMock(),
                compute_cluster="gpu-cluster",
                final_config={},
                aml_experiment_name="test-experiment",
                tags={}
            )

    def test_create_job_command_args(self, temp_dir):
        """Test that command arguments are correctly formatted."""
        script_path = temp_dir / "train.py"
        script_path.touch()
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        final_config = {
            "backbone": "bert-base-uncased",
            "learning_rate": 3e-5,
            "dropout": 0.2,
            "weight_decay": 0.05,
            "batch_size": 16,
            "epochs": 5,
            "random_seed": 42,
            "early_stopping_enabled": False,
            "use_combined_data": True,
        }
        
        with patch("orchestration.jobs.training.command") as mock_command:
            create_final_training_job(
                script_path=script_path,
                data_asset=mock_data_asset,
                environment=MagicMock(),
                compute_cluster="gpu-cluster",
                final_config=final_config,
                aml_experiment_name="test-experiment",
                tags={}
            )
            
            call_args = mock_command.call_args
            command_str = call_args[1]["command"]
            assert "--backbone bert-base-uncased" in command_str
            assert "--learning-rate 3e-05" in command_str
            assert "--dropout 0.2" in command_str
            assert "--weight-decay 0.05" in command_str
            assert "--batch-size 16" in command_str
            assert "--epochs 5" in command_str
            assert "--random-seed 42" in command_str
            assert "--early-stopping-enabled false" in command_str
            assert "--use-combined-data true" in command_str

    def test_create_job_uses_asset_reference(self, temp_dir):
        """Test that job uses asset reference format for data input."""
        script_path = temp_dir / "train.py"
        script_path.touch()
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        final_config = {
            "backbone": "bert-base-uncased",
            "learning_rate": 3e-5,
            "dropout": 0.2,
            "weight_decay": 0.05,
            "batch_size": 16,
            "epochs": 5,
            "random_seed": 42,
            "early_stopping_enabled": False,
            "use_combined_data": True,
        }
        
        with patch("orchestration.jobs.training.command") as mock_command:
            create_final_training_job(
                script_path=script_path,
                data_asset=mock_data_asset,
                environment=MagicMock(),
                compute_cluster="gpu-cluster",
                final_config=final_config,
                aml_experiment_name="test-experiment",
                tags={}
            )
            
            call_kwargs = mock_command.call_args[1]
            inputs = call_kwargs["inputs"]
            assert "data" in inputs
            assert inputs["data"].path == "azureml:resume-ner-data:2"
            assert inputs["data"].path.startswith("azureml:")
            assert not inputs["data"].path.startswith("azureml://datastores/")

    def test_create_job_outputs_checkpoint(self, temp_dir):
        """Test that job has checkpoint output."""
        script_path = temp_dir / "train.py"
        script_path.touch()
        
        final_config = {
            "backbone": "bert-base-uncased",
            "learning_rate": 3e-5,
            "dropout": 0.2,
            "weight_decay": 0.05,
            "batch_size": 16,
            "epochs": 5,
            "random_seed": 42,
            "early_stopping_enabled": False,
            "use_combined_data": True,
        }
        
        with patch("orchestration.jobs.training.command") as mock_command:
            create_final_training_job(
                script_path=script_path,
                data_asset=MagicMock(),
                environment=MagicMock(),
                compute_cluster="gpu-cluster",
                final_config=final_config,
                aml_experiment_name="test-experiment",
                tags={}
            )
            
            call_kwargs = mock_command.call_args[1]
            outputs = call_kwargs["outputs"]
            assert "checkpoint" in outputs
            assert outputs["checkpoint"].type == "uri_folder"

