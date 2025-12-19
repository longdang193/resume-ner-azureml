"""Tests for job runtime operations."""

import pytest
from unittest.mock import MagicMock, patch
from orchestration.jobs.runtime import submit_and_wait_for_job


class TestSubmitAndWaitForJob:
    """Tests for submit_and_wait_for_job function."""

    def test_submit_and_wait_success(self):
        """Test successful job submission and completion."""
        mock_ml_client = MagicMock()
        mock_job = MagicMock()
        mock_submitted = MagicMock()
        mock_submitted.name = "test_job"
        mock_completed = MagicMock()
        mock_completed.status = "Completed"
        mock_completed.name = "test_job"
        
        mock_ml_client.jobs.create_or_update.return_value = mock_submitted
        mock_ml_client.jobs.get.return_value = mock_completed
        
        result = submit_and_wait_for_job(mock_ml_client, mock_job)
        
        assert result == mock_completed
        mock_ml_client.jobs.create_or_update.assert_called_once_with(mock_job)
        mock_ml_client.jobs.stream.assert_called_once_with("test_job")
        mock_ml_client.jobs.get.assert_called_once_with("test_job")

    def test_submit_and_wait_failed_job(self):
        """Test that RuntimeError is raised when job fails."""
        mock_ml_client = MagicMock()
        mock_job = MagicMock()
        mock_submitted = MagicMock()
        mock_submitted.name = "test_job"
        mock_completed = MagicMock()
        mock_completed.status = "Failed"
        mock_completed.name = "test_job"
        
        mock_ml_client.jobs.create_or_update.return_value = mock_submitted
        mock_ml_client.jobs.get.return_value = mock_completed
        
        with pytest.raises(RuntimeError, match="Job test_job failed with status: Failed"):
            submit_and_wait_for_job(mock_ml_client, mock_job)

    def test_submit_and_wait_canceled_job(self):
        """Test that RuntimeError is raised when job is canceled."""
        mock_ml_client = MagicMock()
        mock_job = MagicMock()
        mock_submitted = MagicMock()
        mock_submitted.name = "test_job"
        mock_completed = MagicMock()
        mock_completed.status = "Canceled"
        mock_completed.name = "test_job"
        
        mock_ml_client.jobs.create_or_update.return_value = mock_submitted
        mock_ml_client.jobs.get.return_value = mock_completed
        
        with pytest.raises(RuntimeError, match="Job test_job failed with status: Canceled"):
            submit_and_wait_for_job(mock_ml_client, mock_job)

    def test_submit_and_wait_streams_logs(self):
        """Test that job logs are streamed during execution."""
        mock_ml_client = MagicMock()
        mock_job = MagicMock()
        mock_submitted = MagicMock()
        mock_submitted.name = "test_job"
        mock_completed = MagicMock()
        mock_completed.status = "Completed"
        mock_completed.name = "test_job"
        
        mock_ml_client.jobs.create_or_update.return_value = mock_submitted
        mock_ml_client.jobs.get.return_value = mock_completed
        
        submit_and_wait_for_job(mock_ml_client, mock_job)
        
        # Verify that stream was called
        mock_ml_client.jobs.stream.assert_called_once_with("test_job")

    def test_submit_and_wait_returns_completed_job(self):
        """Test that the completed job object is returned."""
        mock_ml_client = MagicMock()
        mock_job = MagicMock()
        mock_submitted = MagicMock()
        mock_submitted.name = "test_job"
        mock_completed = MagicMock()
        mock_completed.status = "Completed"
        mock_completed.name = "test_job"
        mock_completed.id = "job-123"
        
        mock_ml_client.jobs.create_or_update.return_value = mock_submitted
        mock_ml_client.jobs.get.return_value = mock_completed
        
        result = submit_and_wait_for_job(mock_ml_client, mock_job)
        
        assert result.id == "job-123"
        assert result.status == "Completed"

