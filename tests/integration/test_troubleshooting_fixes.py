"""Integration tests to verify fixes for common troubleshooting issues."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import os


class TestConfigDirectoryResolution:
    """Tests for config directory path resolution to prevent FileNotFoundError."""

    def test_config_dir_relative_to_code_root(self, temp_dir):
        """Test that config directory is resolved relative to code root."""
        # Simulate project structure
        project_root = temp_dir / "project"
        project_root.mkdir()
        (project_root / "src").mkdir()
        (project_root / "config").mkdir()
        (project_root / "config" / "train.yaml").write_text("training:\n  epochs: 1\n")
        
        # When code=".." (project root), config should be at "config"
        config_dir = project_root / "config"
        assert config_dir.exists()
        assert (config_dir / "train.yaml").exists()

    def test_config_dir_not_found_raises_error(self, temp_dir):
        """Test that missing config directory raises FileNotFoundError."""
        from training.config import load_config_file
        
        non_existent_config_dir = temp_dir / "nonexistent_config"
        
        with pytest.raises(FileNotFoundError):
            load_config_file(non_existent_config_dir, "train.yaml")


class TestDataAssetReferenceFormat:
    """Integration tests for data asset reference format."""

    def test_training_job_uses_asset_reference(self):
        """Test that training job creation uses azureml:name:version format."""
        from orchestration.jobs.training import _build_data_input_from_asset
        from azure.ai.ml import Input
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        data_input = _build_data_input_from_asset(mock_data_asset)
        
        # Verify it's an Input object with correct path format
        assert isinstance(data_input, Input)
        assert data_input.path.startswith("azureml:")
        assert ":" in data_input.path
        assert not data_input.path.startswith("azureml://datastores/")

    def test_sweep_job_uses_asset_reference(self):
        """Test that sweep job creation uses azureml:name:version format."""
        from orchestration.jobs.sweeps import _build_data_input_from_asset
        from azure.ai.ml import Input
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        data_input = _build_data_input_from_asset(mock_data_asset)
        
        assert isinstance(data_input, Input)
        assert data_input.path.startswith("azureml:")
        assert not data_input.path.startswith("azureml://datastores/")


class TestMLflowContextManagement:
    """Integration tests for MLflow context management."""

    @patch("platform_adapters.mlflow_context.mlflow", create=True)
    def test_azure_ml_does_not_create_nested_run(self, mock_mlflow):
        """Test that Azure ML context does not create nested runs."""
        from platform_adapters.adapters import get_platform_adapter
        
        # Simulate Azure ML environment
        with patch.dict(os.environ, {"AZURE_ML_OUTPUT_DIR": "/mnt/outputs"}):
            adapter = get_platform_adapter()
            mlflow_context = adapter.get_mlflow_context_manager()
            
            with mlflow_context.get_context():
                # Should not call start_run
                pass
            
            # Verify start_run was NOT called
            mock_mlflow.start_run.assert_not_called()

    def test_local_creates_mlflow_run(self):
        """Test that local execution creates MLflow run."""
        from platform_adapters.adapters import get_platform_adapter
        
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Ensure no Azure ML environment variables
        env_vars_to_remove = [k for k in os.environ.keys() if k.startswith("AZURE_ML_")]
        with patch.dict(os.environ, {k: None for k in env_vars_to_remove}, clear=False):
            with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
                adapter = get_platform_adapter()
                mlflow_context = adapter.get_mlflow_context_manager()
                
                with mlflow_context.get_context():
                    pass
                
                # Verify start_run WAS called
                mock_mlflow.start_run.assert_called_once()


class TestCheckpointPathResolution:
    """Integration tests for checkpoint path resolution."""

    def test_checkpoint_resolver_searches_nested_directories(self, temp_dir):
        """Test that checkpoint resolver searches nested checkpoint/ directories."""
        from platform_adapters.checkpoint_resolver import LocalCheckpointResolver
        
        # Create nested structure: root/checkpoint/config.json
        checkpoint_root = temp_dir / "checkpoint_root"
        checkpoint_dir = checkpoint_root / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.safetensors").write_text("dummy")
        
        resolver = LocalCheckpointResolver()
        resolved = resolver.resolve_checkpoint_dir(str(checkpoint_root))
        
        assert resolved.exists()
        assert (resolved / "config.json").exists()

    def test_checkpoint_resolver_finds_hf_model_files(self, temp_dir):
        """Test that checkpoint resolver finds Hugging Face model files."""
        from platform_adapters.checkpoint_resolver import LocalCheckpointResolver
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.safetensors").write_text("dummy")
        
        resolver = LocalCheckpointResolver()
        resolved = resolver.resolve_checkpoint_dir(str(checkpoint_dir))
        
        assert resolved == checkpoint_dir
        assert (resolved / "config.json").exists()
        assert (resolved / "model.safetensors").exists()

