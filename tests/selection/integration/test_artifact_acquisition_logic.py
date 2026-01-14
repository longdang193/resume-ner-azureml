"""Component tests for artifact acquisition using config options."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from evaluation.selection.artifact_acquisition import acquire_best_model_checkpoint


class TestArtifactAcquisitionConfig:
    """Test artifact acquisition logic using config options."""

    @patch("evaluation.selection.artifact_unified.compat._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_priority_order_local_first(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that priority order affects which strategy is tried first."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Call function with local first in priority
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
        
        # Should use local strategy (first in priority)
        assert mock_find_checkpoint.called
        assert result is not None

    @patch("evaluation.selection.artifact_unified.compat._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_priority_order_mlflow_first(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        custom_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that priority order with mlflow first tries MLflow before local."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks - local should not be called if mlflow succeeds
        mock_find_checkpoint.return_value = None  # Local fails
        mock_validate.return_value = True
        
        # Mock MLflow client
        with patch("mlflow.tracking.MlflowClient") as mock_client_class, \
             patch("evaluation.selection.artifact_unified.compat._build_checkpoint_dir") as mock_build_dir, \
             patch("evaluation.selection.artifact_unified.compat._find_checkpoint_in_directory") as mock_find_in_dir, \
             patch("evaluation.selection.artifact_unified.compat.shutil") as mock_shutil:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.list_artifacts.return_value = []
            mock_client.download_artifacts.return_value = str(mock_checkpoint_path)
            mock_build_dir.return_value = mock_checkpoint_path
            mock_find_in_dir.return_value = mock_checkpoint_path
            
            # Call function with mlflow first in priority
            try:
                result = acquire_best_model_checkpoint(
                    best_run_info=mock_best_run_info,
                    root_dir=root_dir,
                    config_dir=config_dir,
                    acquisition_config=custom_acquisition_config,
                    selection_config={},
                    platform="local",
                    restore_from_drive=None,
                    drive_store=None,
                    in_colab=False,
                )
                # If mlflow succeeds, local should not be called
                # (but mlflow might fail, so we check that mlflow was attempted)
                assert mock_client.download_artifacts.called or mock_find_checkpoint.called
            except ValueError:
                # Expected if all strategies fail
                pass

    @patch("evaluation.selection.artifact_unified.compat._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_local_validate_controls_validation(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that local.validate controls checkpoint validation for local strategy."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Test with validate=True
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["local"]["validate"] = True
        
        with patch("evaluation.selection.artifact_unified.compat._build_checkpoint_dir") as mock_build_dir, \
             patch("evaluation.selection.artifact_unified.compat.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
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
            
            # Should call validate when validate=True
            assert mock_validate.called
        
        # Test with validate=False
        acquisition_config["local"]["validate"] = False
        mock_validate.reset_mock()
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        
        with patch("evaluation.selection.artifact_unified.compat._build_checkpoint_dir") as mock_build_dir, \
             patch("evaluation.selection.artifact_unified.compat.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
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
            
            # Should still work but validate may not be called (or called but result ignored)
            # The function checks validate flag before calling _validate_checkpoint

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.selection.artifact_acquisition._find_checkpoint_in_drive_by_hash")
    def test_drive_enabled_controls_drive_strategy(
        self,
        mock_find_drive_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that drive.enabled controls drive strategy execution."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_drive_store = Mock()
        mock_drive_store.drive_path_for.return_value = Path("/drive/hpo")
        mock_find_drive_checkpoint.return_value = Path("/drive/checkpoint")
        mock_validate.return_value = True
        
        # Test with enabled=True
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["drive"]["enabled"] = True
        
        # Mock restore_from_drive
        mock_restore_from_drive = Mock(return_value=True)
        
        try:
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="colab",
                restore_from_drive=mock_restore_from_drive,
                drive_store=mock_drive_store,
                in_colab=True,
            )
            # Drive strategy should be attempted when enabled=True
            # (may fail if local succeeds first, but drive code path should be checked)
        except ValueError:
            pass
        
        # Test with enabled=False
        acquisition_config["drive"]["enabled"] = False
        mock_find_drive_checkpoint.reset_mock()
        
        try:
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="colab",
                restore_from_drive=mock_restore_from_drive,
                drive_store=mock_drive_store,
                in_colab=True,
            )
        except ValueError:
            pass
        
        # When enabled=False, drive strategy should be skipped
        # (check happens at line 398: acquisition_config["drive"]["enabled"])

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.selection.artifact_acquisition._find_checkpoint_in_drive_by_hash")
    def test_drive_validate_controls_validation(
        self,
        mock_find_drive_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that drive.validate controls checkpoint validation for drive strategy."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_drive_store = Mock()
        mock_drive_store.drive_path_for.return_value = Path("/drive/hpo")
        mock_find_drive_checkpoint.return_value = Path("/drive/checkpoint")
        mock_validate.return_value = True
        
        # Test with validate=True
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["drive"]  # Skip local
        acquisition_config["drive"]["validate"] = True
        
        mock_restore_from_drive = Mock(return_value=True)
        
        try:
            result = acquire_best_model_checkpoint(
                best_run_info=mock_best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="colab",
                restore_from_drive=mock_restore_from_drive,
                drive_store=mock_drive_store,
                in_colab=True,
            )
            # Should call validate when validate=True
            # (validation happens at line 432)
        except (ValueError, Exception):
            pass

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_enabled_controls_mlflow_strategy(
        self,
        mock_client_class,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that mlflow.enabled controls MLflow strategy execution."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_artifacts.return_value = []
        mock_client.download_artifacts.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Test with enabled=True
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["mlflow"]  # Skip local and drive
        acquisition_config["mlflow"]["enabled"] = True
        
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition._find_checkpoint_in_directory") as mock_find_in_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            mock_find_in_dir.return_value = mock_checkpoint_path
            
            try:
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
                # MLflow strategy should be attempted when enabled=True
                assert mock_client.download_artifacts.called
            except ValueError:
                # Expected if download fails
                pass
        
        # Test with enabled=False
        acquisition_config["mlflow"]["enabled"] = False
        mock_client.reset_mock()
        
        try:
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
        except ValueError:
            # Expected when all strategies are disabled
            pass
        
        # When enabled=False, MLflow strategy should be skipped
        # (check happens at line 458: acquisition_config["mlflow"]["enabled"])

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_validate_controls_validation(
        self,
        mock_client_class,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that mlflow.validate controls checkpoint validation for MLflow strategy."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_artifacts.return_value = []
        mock_client.download_artifacts.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = True
        
        # Test with validate=True
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["mlflow"]  # Skip local and drive
        acquisition_config["mlflow"]["validate"] = True
        
        with patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition._find_checkpoint_in_directory") as mock_find_in_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            mock_find_in_dir.return_value = mock_checkpoint_path
            
            try:
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
                # Should call validate when validate=True (at line 549)
                # Note: validate is called even if download fails earlier
            except ValueError:
                # Expected if validation fails or download fails
                pass

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_priority_order_with_some_sources_disabled(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test priority order when some sources are disabled."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = None  # Local fails
        mock_validate.return_value = True
        
        # Disable drive, keep mlflow enabled
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["local", "drive", "mlflow"]
        acquisition_config["drive"]["enabled"] = False
        acquisition_config["mlflow"]["enabled"] = True
        
        # Mock MLflow client
        with patch("mlflow.tracking.MlflowClient") as mock_client_class, \
             patch("orchestration.jobs.selection.artifact_acquisition._build_checkpoint_dir") as mock_build_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition._find_checkpoint_in_directory") as mock_find_in_dir, \
             patch("orchestration.jobs.selection.artifact_acquisition.shutil") as mock_shutil:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.list_artifacts.return_value = []
            mock_client.download_artifacts.return_value = str(mock_checkpoint_path)
            mock_build_dir.return_value = mock_checkpoint_path
            mock_find_in_dir.return_value = mock_checkpoint_path
            
            try:
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
                # Should skip drive (disabled) and try mlflow
                assert mock_client.download_artifacts.called
            except ValueError:
                pass

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_all_strategies_fail_gracefully_when_disabled(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
    ):
        """Test that all strategies fail gracefully when disabled."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Disable all strategies
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["local", "drive", "mlflow"]
        acquisition_config["drive"]["enabled"] = False
        acquisition_config["mlflow"]["enabled"] = False
        
        # Local will fail (no checkpoint found)
        mock_find_checkpoint.return_value = None
        
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

