"""Integration tests for artifact acquisition end-to-end workflow."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from selection.artifact_acquisition import acquire_best_model_checkpoint


class TestArtifactAcquisitionWorkflow:
    """Test complete artifact acquisition workflow."""

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_complete_workflow_with_default_config(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test complete acquisition workflow with default config."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
            # Call function with default config
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=sample_acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Should successfully acquire checkpoint
            assert result is not None
            assert mock_find_checkpoint.called
            assert mock_validate.called

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_workflow_with_custom_priority_order(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test acquisition workflow with custom priority order."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Custom priority: mlflow first, then local
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["mlflow", "local"]
        
        # Mock MLflow to fail, so it falls back to local
        with patch("mlflow.tracking.MlflowClient") as mock_client_class, \
             patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.list_artifacts.side_effect = Exception("MLflow failed")
            mock_build_dir.return_value = mock_checkpoint_path
            
            # Call function
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Should fall back to local strategy
            assert result is not None
            assert mock_find_checkpoint.called

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_workflow_with_validation_disabled(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test acquisition workflow with validation disabled."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = False  # Validation would fail
        
        # Disable validation
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["local"]["validate"] = False
        
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
            # Call function
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Should succeed even with validation disabled
            assert result is not None
            # Note: validate may still be called internally, but result is not checked when validate=False

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_workflow_with_all_sources_enabled(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test acquisition workflow with all sources enabled."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Ensure all sources are enabled
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["local", "drive", "mlflow"]
        acquisition_config["drive"]["enabled"] = True
        acquisition_config["mlflow"]["enabled"] = True
        
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
            # Call function
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Should succeed using local (first in priority)
            assert result is not None
            assert mock_find_checkpoint.called

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_workflow_with_all_sources_disabled(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
    ):
        """Test acquisition workflow with all sources disabled."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks - local fails
        mock_find_checkpoint.return_value = None
        mock_validate.return_value = True
        
        # Disable all sources
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["local", "drive", "mlflow"]
        acquisition_config["drive"]["enabled"] = False
        acquisition_config["mlflow"]["enabled"] = False
        
        # Should raise ValueError when all strategies fail
        with pytest.raises(ValueError, match="Could not acquire checkpoint"):
            acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("mlflow.tracking.MlflowClient")
    def test_workflow_mlflow_fallback_to_manual(
        self,
        mock_client_class,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that workflow falls back to manually placed checkpoint when all strategies fail."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks - MLflow fails
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_artifacts.side_effect = Exception("MLflow failed")
        mock_validate.return_value = True
        
        # Only mlflow in priority, but it fails
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["mlflow"]
        acquisition_config["mlflow"]["enabled"] = True
        
        # Create manually placed checkpoint
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir:
            manual_checkpoint_dir = root_dir / "best_model_selection" / "local" / "distilbert" / "run_test_run"
            manual_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_subdir = manual_checkpoint_dir / "checkpoint"
            checkpoint_subdir.mkdir()
            (checkpoint_subdir / "config.json").write_text('{"model_type": "bert"}')
            (checkpoint_subdir / "pytorch_model.bin").write_bytes(b"fake_model_data")
            
            # Mock _build_checkpoint_dir to return the checkpoint directory
            # The function will look for a "checkpoint" subdirectory
            mock_build_dir.return_value = manual_checkpoint_dir
            
            # Call function
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Should find manually placed checkpoint
            assert result is not None

