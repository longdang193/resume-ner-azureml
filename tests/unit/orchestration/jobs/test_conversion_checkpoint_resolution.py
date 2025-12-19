"""Tests for checkpoint resolution in conversion jobs to prevent FileNotFoundError."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from orchestration.jobs.conversion import (
    get_checkpoint_output_from_training_job,
    _get_job_output_reference,
    create_conversion_job,
    validate_conversion_job,
)


class TestGetCheckpointOutputFromTrainingJob:
    """Tests for checkpoint output extraction from training jobs."""

    def test_uses_asset_reference_format(self):
        """Test that checkpoint reference uses azureml:name:version format."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.name = "training_job_123"
        mock_job.outputs = {}
        
        # When ml_client is provided and asset exists
        mock_ml_client = MagicMock()
        mock_data_asset = MagicMock()
        mock_data_asset.name = "azureml_training_job_123_output_data_checkpoint"
        mock_data_asset.version = "1"
        mock_ml_client.data.get.return_value = mock_data_asset
        
        result = get_checkpoint_output_from_training_job(mock_job, mock_ml_client)
        
        assert result.startswith("azureml:")
        assert result == "azureml:azureml_training_job_123_output_data_checkpoint:1"

    def test_fallback_to_direct_construction(self):
        """Test fallback when ml_client cannot fetch asset."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.name = "training_job_123"
        mock_job.outputs = {}
        
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = Exception("Asset not found")
        
        result = get_checkpoint_output_from_training_job(mock_job, mock_ml_client)
        
        # Should still return azureml: format
        assert result.startswith("azureml:")
        assert "azureml_training_job_123_output_data_checkpoint" in result

    def test_uses_job_output_uri_if_available(self):
        """Test that job output URI is used if available."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.name = "training_job_123"
        
        mock_output = MagicMock()
        mock_output.uri = "azureml:checkpoint_asset:1"
        mock_job.outputs = {"checkpoint": mock_output}
        
        result = get_checkpoint_output_from_training_job(mock_job, ml_client=None)
        
        assert result == "azureml:checkpoint_asset:1"

    def test_raises_error_if_job_not_completed(self):
        """Test that error is raised if training job did not complete."""
        mock_job = MagicMock()
        mock_job.status = "Failed"
        mock_job.name = "training_job_123"
        
        with pytest.raises(ValueError, match="did not complete successfully"):
            get_checkpoint_output_from_training_job(mock_job, ml_client=None)

    def test_asset_reference_not_datastore_path(self):
        """Test that returned reference is NOT a manual datastore path."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.name = "training_job_123"
        mock_job.outputs = {}
        
        result = get_checkpoint_output_from_training_job(mock_job, ml_client=None)
        
        # Should NOT be a datastore path
        assert not result.startswith("azureml://datastores/")
        # Should be asset reference
        assert result.startswith("azureml:")


class TestGetJobOutputReference:
    """Tests for _get_job_output_reference function."""

    def test_returns_asset_uri_format(self):
        """Test that job output reference uses azureml: format."""
        mock_job = MagicMock()
        mock_output = MagicMock()
        mock_output.uri = "azureml:output_asset:1"
        mock_job.outputs = {"checkpoint": mock_output}
        
        result = _get_job_output_reference(mock_job, "checkpoint", ml_client=None)
        
        assert result.startswith("azureml:")
        assert result == "azureml:output_asset:1"

    def test_fallback_to_path_if_uri_not_available(self):
        """Test fallback to path if URI is not available."""
        mock_job = MagicMock()
        mock_output = MagicMock()
        mock_output.uri = None
        mock_output.path = "/some/path"
        mock_job.outputs = {"checkpoint": mock_output}
        
        result = _get_job_output_reference(mock_job, "checkpoint", ml_client=None)
        
        # Should return path as fallback
        assert result == "/some/path"

    def test_fallback_to_asset_reference_when_no_outputs(self):
        """Test fallback to asset reference when job has no outputs."""
        mock_job = MagicMock()
        mock_job.name = "test_job"
        mock_job.outputs = {}
        
        result = _get_job_output_reference(mock_job, "checkpoint", ml_client=None)
        
        assert result.startswith("azureml:")
        assert "azureml_test_job_output_data_checkpoint" in result

    def test_uses_ml_client_when_provided(self):
        """Test that ml_client is used to fetch asset when provided."""
        mock_job = MagicMock()
        mock_job.name = "test_job"
        mock_job.outputs = {}
        
        mock_ml_client = MagicMock()
        mock_data_asset = MagicMock()
        mock_data_asset.name = "azureml_test_job_output_data_checkpoint"
        mock_data_asset.version = "1"
        mock_ml_client.data.get.return_value = mock_data_asset
        
        result = _get_job_output_reference(mock_job, "checkpoint", ml_client=mock_ml_client)
        
        assert result == "azureml:azureml_test_job_output_data_checkpoint:1"
        mock_ml_client.data.get.assert_called_once_with(name="azureml_test_job_output_data_checkpoint", version="1")


class TestCreateConversionJob:
    """Tests for create_conversion_job function."""

    def test_create_conversion_job_success(self, temp_dir):
        """Test successful creation of conversion job."""
        script_path = temp_dir / "convert_to_onnx.py"
        script_path.touch()
        
        mock_environment = MagicMock()
        checkpoint_uri = "azureml:checkpoint_asset:1"
        
        with patch("orchestration.jobs.conversion.command") as mock_command:
            mock_job = MagicMock()
            mock_command.return_value = mock_job
            
            result = create_conversion_job(
                script_path=script_path,
                checkpoint_uri=checkpoint_uri,
                environment=mock_environment,
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment",
                tags={"key": "value"}
            )
            
            assert result == mock_job
            mock_command.assert_called_once()
            call_kwargs = mock_command.call_args[1]
            assert call_kwargs["compute"] == "cpu-cluster"
            assert call_kwargs["experiment_name"] == "test-experiment"
            assert call_kwargs["tags"] == {"key": "value"}

    def test_create_conversion_job_script_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised when script doesn't exist."""
        script_path = temp_dir / "nonexistent.py"
        
        with pytest.raises(FileNotFoundError, match="Conversion script not found"):
            create_conversion_job(
                script_path=script_path,
                checkpoint_uri="azureml:checkpoint:1",
                environment=MagicMock(),
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment"
            )

    def test_create_conversion_job_invalid_checkpoint_uri(self, temp_dir):
        """Test that ValueError is raised for invalid checkpoint URI format."""
        script_path = temp_dir / "convert_to_onnx.py"
        script_path.touch()
        
        with pytest.raises(ValueError, match="Unexpected checkpoint URI format"):
            create_conversion_job(
                script_path=script_path,
                checkpoint_uri="/local/path",
                environment=MagicMock(),
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment"
            )

    def test_create_conversion_job_with_datastore_uri(self, temp_dir):
        """Test creating job with datastore URI format."""
        script_path = temp_dir / "convert_to_onnx.py"
        script_path.touch()
        
        mock_environment = MagicMock()
        checkpoint_uri = "azureml://datastores/workspaceblobstore/paths/checkpoint"
        
        with patch("orchestration.jobs.conversion.command") as mock_command:
            mock_job = MagicMock()
            mock_command.return_value = mock_job
            
            result = create_conversion_job(
                script_path=script_path,
                checkpoint_uri=checkpoint_uri,
                environment=mock_environment,
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment"
            )
            
            assert result == mock_job
            # Verify Input was created with correct path
            call_args = mock_command.call_args
            inputs = call_args[1]["inputs"]
            assert "checkpoint" in inputs
            assert inputs["checkpoint"].path == checkpoint_uri

    def test_create_conversion_job_command_args(self, temp_dir):
        """Test that command arguments are correctly formatted."""
        script_path = temp_dir / "convert_to_onnx.py"
        script_path.touch()
        
        mock_environment = MagicMock()
        checkpoint_uri = "azureml:checkpoint:1"
        
        with patch("orchestration.jobs.conversion.command") as mock_command:
            create_conversion_job(
                script_path=script_path,
                checkpoint_uri=checkpoint_uri,
                environment=mock_environment,
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment"
            )
            
            call_args = mock_command.call_args
            command_str = call_args[1]["command"]
            assert "--checkpoint-path" in command_str
            assert "--config-dir config" in command_str
            assert "--backbone bert-base-uncased" in command_str
            assert "--quantize-int8" in command_str
            assert "--run-smoke-test" in command_str

    def test_create_conversion_job_empty_tags(self, temp_dir):
        """Test creating job with empty tags dict."""
        script_path = temp_dir / "convert_to_onnx.py"
        script_path.touch()
        
        mock_environment = MagicMock()
        checkpoint_uri = "azureml:checkpoint:1"
        
        with patch("orchestration.jobs.conversion.command") as mock_command:
            create_conversion_job(
                script_path=script_path,
                checkpoint_uri=checkpoint_uri,
                environment=mock_environment,
                compute_cluster="cpu-cluster",
                backbone="bert-base-uncased",
                experiment_name="test-experiment",
                tags={}
            )
            
            call_kwargs = mock_command.call_args[1]
            assert call_kwargs["tags"] == {}


class TestValidateConversionJob:
    """Tests for validate_conversion_job function."""

    def test_validate_success(self):
        """Test successful validation of completed conversion job."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_output = MagicMock()
        mock_output.uri = "azureml:onnx_model:1"
        mock_job.outputs = {"onnx_model": mock_output}
        
        validate_conversion_job(mock_job, ml_client=None)
        
        # Should not raise any exception

    def test_validate_failed_job(self):
        """Test that ValueError is raised for failed job."""
        mock_job = MagicMock()
        mock_job.status = "Failed"
        
        with pytest.raises(ValueError, match="Conversion job failed with status"):
            validate_conversion_job(mock_job, ml_client=None)

    def test_validate_no_outputs(self):
        """Test that ValueError is raised when job has no outputs."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.outputs = {}
        
        with pytest.raises(ValueError, match="Conversion job produced no outputs"):
            validate_conversion_job(mock_job, ml_client=None)

    def test_validate_missing_onnx_output(self):
        """Test that ValueError is raised when onnx_model output is missing."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.outputs = {"other_output": MagicMock()}
        
        with pytest.raises(ValueError, match="Conversion job missing required output: onnx_model"):
            validate_conversion_job(mock_job, ml_client=None)

    def test_validate_invalid_output_reference(self):
        """Test that ValueError is raised for invalid output reference."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_output = MagicMock()
        mock_output.uri = None
        mock_output.path = "/invalid/path"
        mock_job.outputs = {"onnx_model": mock_output}
        
        with pytest.raises(ValueError, match="Invalid ONNX model output reference"):
            validate_conversion_job(mock_job, ml_client=None)

    def test_validate_uses_ml_client(self):
        """Test that ml_client is used when provided."""
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.name = "conversion_job"
        mock_output = MagicMock()
        mock_output.uri = None
        mock_output.path = None
        mock_job.outputs = {"onnx_model": mock_output}
        
        mock_ml_client = MagicMock()
        mock_data_asset = MagicMock()
        mock_data_asset.name = "azureml_conversion_job_output_data_onnx_model"
        mock_data_asset.version = "1"
        mock_ml_client.data.get.return_value = mock_data_asset
        
        validate_conversion_job(mock_job, ml_client=mock_ml_client)
        
        # Should not raise any exception
        mock_ml_client.data.get.assert_called_once()

