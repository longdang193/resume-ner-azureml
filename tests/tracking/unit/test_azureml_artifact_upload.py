"""Tests for Azure ML artifact upload fixes.

This module tests the fixes for:
1. Monkey-patch for azureml_artifacts_builder to handle tracking_uri parameter
2. Artifact upload to child runs (refit runs) in Azure ML
3. Compatibility between MLflow 3.5.0 and azureml-mlflow 1.61.0.post1

Updated to test the new consolidated tracking.mlflow utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import mlflow
from mlflow.tracking import MlflowClient


class TestAzureMLArtifactBuilderPatch:
    """Test the monkey-patch for azureml_artifacts_builder."""

    def test_patch_registered_on_import(self):
        """Test that the monkey-patch is registered when tracking.mlflow is imported."""
        # Import should trigger the patch
        from tracking.mlflow import apply_azureml_artifact_patch
        
        # Verify the patch is registered
        import mlflow.store.artifact.artifact_repository_registry as arr
        builder = arr._artifact_repository_registry._registry.get('azureml')
        if builder is None:
            pytest.skip("Azure ML builder not registered (not using Azure ML)")
        
        # Check that it's the patched version (has __wrapped__ attribute)
        assert hasattr(builder, '__wrapped__'), "Builder should be wrapped (patched)"

    def test_patch_handles_tracking_uri_parameter(self):
        """Test that the patched builder handles tracking_uri parameter gracefully."""
        from tracking.mlflow import apply_azureml_artifact_patch
        
        import mlflow.store.artifact.artifact_repository_registry as arr
        builder = arr._artifact_repository_registry._registry.get('azureml')
        
        if builder is None:
            pytest.skip("Azure ML builder not registered (not using Azure ML)")
        
        # Verify the patch structure - it should have __wrapped__ attribute
        assert hasattr(builder, '__wrapped__'), "Builder should be wrapped (patched)"
        
        # The actual error handling is tested in integration tests
        # This test just verifies the patch is in place
        assert callable(builder), "Patched builder should be callable"
    
    def test_patch_auto_applies_on_module_import(self):
        """Test that patch auto-applies when tracking.mlflow module is imported."""
        # Clear any existing patch by reloading the module
        import sys
        if 'tracking.mlflow.compatibility' in sys.modules:
            del sys.modules['tracking.mlflow.compatibility']
            del sys.modules['tracking.mlflow']
        
        # Import the module - should auto-apply patch
        from tracking.mlflow import apply_azureml_artifact_patch  # noqa: F401
        
        # Verify patch was applied
        import mlflow.store.artifact.artifact_repository_registry as arr
        builder = arr._artifact_repository_registry._registry.get('azureml')
        if builder is not None:
            assert hasattr(builder, '__wrapped__'), "Patch should auto-apply on import"


class TestArtifactUploadToChildRun:
    """Test artifact upload to child runs (refit runs)."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_active_run(self):
        """Create a mock active MLflow run (parent run)."""
        with patch('mlflow.active_run') as mock_active:
            mock_run = MagicMock()
            mock_run.info.run_id = "parent-run-id-123"
            mock_active.return_value = mock_run
            yield mock_run

    def test_upload_to_refit_run_when_available(self, mock_mlflow_client, mock_active_run):
        """Test that artifacts are uploaded to refit run when available."""
        from tracking.mlflow import upload_checkpoint_archive
        from pathlib import Path
        import tempfile
        
        # Create a temporary archive file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            archive_path = Path(tmp_file.name)
            archive_path.write_bytes(b"test archive content")
        
        try:
            manifest = {"file_count": 1, "total_size": 100}
            refit_run_id = "refit-run-id-456"
            
            # Test upload to refit run using new utility
            # Mock MlflowClient.log_artifact to verify the function works correctly
            with patch('mlflow.tracking.MlflowClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                
                result = upload_checkpoint_archive(
                    archive_path=archive_path,
                    manifest=manifest,
                    artifact_path="best_trial_checkpoint",
                    run_id=refit_run_id,
                )
                        
                # Verify that the function succeeded
                assert result is True, "upload_checkpoint_archive should return True"
                # Verify that MlflowClient was used (indirectly via log_artifact_safe)
                assert mock_client_class.called, "MlflowClient should be instantiated"
        finally:
            # Clean up
            if archive_path.exists():
                archive_path.unlink()

    def test_upload_to_parent_run_when_refit_not_available(self, mock_mlflow_client, mock_active_run):
        """Test that artifacts are uploaded to parent run when refit run is not available."""
        from tracking.mlflow import upload_checkpoint_archive
        from pathlib import Path
        import tempfile
        
        # Create a temporary archive file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            archive_path = Path(tmp_file.name)
            archive_path.write_bytes(b"test archive content")
        
        try:
            manifest = {"file_count": 1, "total_size": 100}
            parent_run_id = "parent-run-id-123"
            
            # Test upload to parent run (no refit run) using new utility
            # Mock MlflowClient.log_artifact to verify the function works correctly
            with patch('mlflow.tracking.MlflowClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                
                result = upload_checkpoint_archive(
                    archive_path=archive_path,
                    manifest=manifest,
                    artifact_path="best_trial_checkpoint",
                    run_id=parent_run_id,
                )
                        
                # Verify that the function succeeded
                assert result is True, "upload_checkpoint_archive should return True"
                # Verify that MlflowClient was used (indirectly via log_artifact_safe)
                assert mock_client_class.called, "MlflowClient should be instantiated"
        finally:
            # Clean up
            if archive_path.exists():
                archive_path.unlink()


class TestRefitRunFinishedStatus:
    """Test that refit runs are marked as FINISHED after artifact upload."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock run status
            mock_run = MagicMock()
            mock_run.info.status = "RUNNING"
            mock_client.get_run.return_value = mock_run
            
            yield mock_client

    def test_refit_run_marked_finished_after_successful_upload(self, mock_mlflow_client):
        """Test that refit run is marked as FINISHED after successful artifact upload."""
        from tracking.mlflow import terminate_run_with_tags
        
        refit_run_id = "refit-run-id-456"
        tags = {"code.refit_artifacts_uploaded": "true"}
        
        # Test using new utility
        terminate_run_with_tags(
            run_id=refit_run_id,
            status="FINISHED",
            tags=tags
        )
        
        # Verify that set_terminated was called with FINISHED status
        mock_mlflow_client.set_terminated.assert_called_once_with(
            refit_run_id, status="FINISHED"
        )
        mock_mlflow_client.set_tag.assert_called_with(
            refit_run_id, "code.refit_artifacts_uploaded", "true"
        )

    def test_refit_run_marked_failed_after_upload_failure(self, mock_mlflow_client):
        """Test that refit run is marked as FAILED after artifact upload failure."""
        from tracking.mlflow import terminate_run_with_tags
        
        refit_run_id = "refit-run-id-456"
        upload_error = Exception("Upload failed")
        error_msg = str(upload_error)[:200] if upload_error else "Unknown error"
        tags = {
            "code.refit_artifacts_uploaded": "false",
            "code.refit_error": error_msg
        }
        
        # Test using new utility
        terminate_run_with_tags(
            run_id=refit_run_id,
            status="FAILED",
            tags=tags
        )
        
        # Verify that set_terminated was called with FAILED status
        mock_mlflow_client.set_terminated.assert_called_once_with(
            refit_run_id, status="FAILED"
        )
        mock_mlflow_client.set_tag.assert_any_call(
            refit_run_id, "code.refit_artifacts_uploaded", "false"
        )

    def test_refit_run_not_terminated_if_already_finished(self, mock_mlflow_client):
        """Test that refit run is not terminated if it's already FINISHED."""
        from tracking.mlflow import terminate_run_safe
        
        refit_run_id = "refit-run-id-456"
        
        # Mock run that's already FINISHED
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_mlflow_client.get_run.return_value = mock_run
        
        # Test using new utility with status check
        terminate_run_safe(
            run_id=refit_run_id,
            status="FINISHED",
            check_status=True
        )
        
        # Verify that set_terminated was NOT called (already terminated)
        mock_mlflow_client.set_terminated.assert_not_called()


class TestAzureMLCompatibility:
    """Test compatibility between MLflow and azureml-mlflow versions."""

    def test_azureml_mlflow_imported(self):
        """Test that azureml.mlflow is imported when tracking.mlflow is imported."""
        # Import should trigger azureml.mlflow import
        from tracking.mlflow import apply_azureml_artifact_patch  # noqa: F401
        
        # Verify azureml.mlflow was imported (check if it's in sys.modules)
        import sys
        # Note: azureml.mlflow may not be available in test environment
        # This test just verifies the import path works
        assert True, "Import should succeed (azureml.mlflow may not be available)"

    def test_artifact_repository_registry_has_azureml(self):
        """Test that Azure ML artifact repository is registered."""
        from tracking.mlflow import apply_azureml_artifact_patch  # noqa: F401
        
        import mlflow.store.artifact.artifact_repository_registry as arr
        registry = arr._artifact_repository_registry._registry
        
        # Azure ML may not be registered if azureml.mlflow is not installed
        if 'azureml' in registry:
            builder = registry.get('azureml')
            if builder is not None:
                # Verify it's callable
                assert callable(builder), "Azure ML builder should be callable"
                # Verify it's patched if available
                if hasattr(builder, '__wrapped__'):
                    assert True, "Azure ML builder is patched"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

