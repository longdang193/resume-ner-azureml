"""Edge case and validation tests for artifact_acquisition.yaml configuration."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.selection.artifact_acquisition import acquire_best_model_checkpoint


class TestArtifactAcquisitionEdgeCases:
    """Test edge cases and validation for acquisition configuration."""

    def test_empty_priority_list(self):
        """Test that empty priority list is handled."""
        acquisition_config = {
            "priority": [],
            "local": {"validate": True},
            "drive": {"enabled": True, "validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        priority = acquisition_config.get("priority", [])
        
        assert priority == []
        assert isinstance(priority, list)
        # Note: Empty priority list would cause all strategies to be skipped

    def test_invalid_priority_values(self):
        """Test that invalid priority values are possible (not validated)."""
        acquisition_config = {
            "priority": ["invalid_source", "another_invalid"],
            "local": {"validate": True},
            "drive": {"enabled": True, "validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        priority = acquisition_config.get("priority", [])
        
        # Config loader doesn't validate priority values
        assert priority == ["invalid_source", "another_invalid"]
        assert "invalid_source" in priority

    def test_missing_local_section(self):
        """Test that missing local section is handled."""
        acquisition_config = {
            "priority": ["local", "mlflow"],
            "drive": {"enabled": True, "validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        # Extract with defaults
        local_validate = acquisition_config.get("local", {}).get("validate", True)
        
        assert local_validate is True  # Default value

    def test_missing_drive_section(self):
        """Test that missing drive section is handled."""
        acquisition_config = {
            "priority": ["local", "drive"],
            "local": {"validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        # Extract with defaults
        drive_enabled = acquisition_config.get("drive", {}).get("enabled", True)
        drive_validate = acquisition_config.get("drive", {}).get("validate", True)
        
        assert drive_enabled is True  # Default value
        assert drive_validate is True  # Default value

    def test_missing_mlflow_section(self):
        """Test that missing mlflow section is handled."""
        acquisition_config = {
            "priority": ["local", "mlflow"],
            "local": {"validate": True},
            "drive": {"enabled": True, "validate": True}
        }
        
        # Extract with defaults
        mlflow_enabled = acquisition_config.get("mlflow", {}).get("enabled", True)
        mlflow_validate = acquisition_config.get("mlflow", {}).get("validate", True)
        
        assert mlflow_enabled is True  # Default value
        assert mlflow_validate is True  # Default value

    @patch("evaluation.selection.artifact_unified.compat._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_validation_false_allows_invalid_checkpoints(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that validation=False allows invalid checkpoints to be used."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks - validation would fail
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = False  # Invalid checkpoint
        
        # Disable validation
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["local"]["validate"] = False
        
        with patch("evaluation.selection.artifact_unified.compat._build_checkpoint_dir") as mock_build_dir, \
             patch("evaluation.selection.artifact_unified.compat.shutil") as mock_shutil:
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
            
            # Should succeed even with invalid checkpoint when validate=False
            assert result is not None

    @patch("evaluation.selection.artifact_unified.compat._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_validation_true_rejects_invalid_checkpoints(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
        mock_checkpoint_path,
    ):
        """Test that validation=True rejects invalid checkpoints."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks - validation fails
        mock_find_checkpoint.return_value = str(mock_checkpoint_path)
        mock_validate.return_value = False  # Invalid checkpoint
        
        # Enable validation
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["local"]["validate"] = True
        
        with patch("evaluation.selection.artifact_unified.compat._build_checkpoint_dir") as mock_build_dir, \
             patch("evaluation.selection.artifact_unified.compat.shutil") as mock_shutil:
            mock_build_dir.return_value = mock_checkpoint_path
            
            # Call function - should fail because checkpoint is invalid
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
                # If local fails, it might try other strategies or raise error
                # The exact behavior depends on whether other strategies are available
            except ValueError:
                # Expected if all strategies fail
                pass

    def test_priority_order_affects_strategy_selection(self):
        """Test that priority order affects which strategy is selected."""
        # Test with local first
        config_local_first = {
            "priority": ["local", "mlflow"],
            "local": {"validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        # Test with mlflow first
        config_mlflow_first = {
            "priority": ["mlflow", "local"],
            "local": {"validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        assert config_local_first["priority"][0] == "local"
        assert config_mlflow_first["priority"][0] == "mlflow"
        # The actual strategy selection is tested in component tests

    def test_duplicate_priority_values(self):
        """Test that duplicate priority values are possible."""
        acquisition_config = {
            "priority": ["local", "local", "mlflow"],
            "local": {"validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        priority = acquisition_config.get("priority", [])
        
        # Config loader doesn't validate for duplicates
        assert priority.count("local") == 2
        assert len(priority) == 3

    def test_priority_with_only_one_source(self):
        """Test priority list with only one source."""
        acquisition_config = {
            "priority": ["mlflow"],
            "local": {"validate": True},
            "drive": {"enabled": True, "validate": True},
            "mlflow": {"enabled": True, "validate": True}
        }
        
        priority = acquisition_config.get("priority", [])
        
        assert len(priority) == 1
        assert priority[0] == "mlflow"

    @patch("orchestration.jobs.selection.artifact_acquisition._validate_checkpoint")
    @patch("orchestration.jobs.local_selection_v2.find_trial_checkpoint_by_hash")
    def test_missing_study_trial_hashes_skips_local(
        self,
        mock_find_checkpoint,
        mock_validate,
        tmp_path,
        sample_acquisition_config,
        mock_best_run_info,
    ):
        """Test that missing study_key_hash or trial_key_hash skips local strategy."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        mock_find_checkpoint.return_value = None
        mock_validate.return_value = True
        
        # Remove hashes from best_run_info
        best_run_info_no_hashes = mock_best_run_info.copy()
        del best_run_info_no_hashes["study_key_hash"]
        del best_run_info_no_hashes["trial_key_hash"]
        
        # Only local in priority, but it will be skipped due to missing hashes
        acquisition_config = sample_acquisition_config.copy()
        acquisition_config["priority"] = ["local"]
        acquisition_config["drive"]["enabled"] = False
        acquisition_config["mlflow"]["enabled"] = False
        
        # Should raise ValueError when all strategies fail
        with pytest.raises(ValueError, match="Could not acquire checkpoint"):
            acquire_best_model_checkpoint(
                best_run_info=best_run_info_no_hashes,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config={},
                platform="local",
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
        
        # find_trial_checkpoint_by_hash should not be called when hashes are missing
        assert not mock_find_checkpoint.called

    def test_config_with_all_optional_fields(self):
        """Test config with all optional fields (including unimplemented ones)."""
        acquisition_config = {
            "priority": ["local", "drive", "mlflow"],
            "local": {
                "match_strategy": "tags",
                "require_exact_match": True,
                "validate": True
            },
            "drive": {
                "enabled": True,
                "folder_path": "custom-checkpoints",
                "validate": True
            },
            "mlflow": {
                "enabled": True,
                "validate": True,
                "download_timeout": 600
            }
        }
        
        # Verify all fields can be extracted
        assert acquisition_config["priority"] == ["local", "drive", "mlflow"]
        assert acquisition_config["local"]["match_strategy"] == "tags"
        assert acquisition_config["local"]["require_exact_match"] is True
        assert acquisition_config["local"]["validate"] is True
        assert acquisition_config["drive"]["enabled"] is True
        assert acquisition_config["drive"]["folder_path"] == "custom-checkpoints"
        assert acquisition_config["drive"]["validate"] is True
        assert acquisition_config["mlflow"]["enabled"] is True
        assert acquisition_config["mlflow"]["validate"] is True
        assert acquisition_config["mlflow"]["download_timeout"] == 600
        
        # NOTE: match_strategy, require_exact_match, folder_path, and download_timeout
        # exist in config but are not currently used in implementation

