"""Tests for data asset reference format to prevent ScriptExecution.StreamAccess.NotFound errors."""

import pytest
from orchestration.data_assets import build_data_asset_reference
from orchestration.jobs.training import _build_data_input_from_asset
from orchestration.jobs.sweeps import _build_data_input_from_asset as sweeps_build_data_input
from unittest.mock import MagicMock


class TestDataAssetReferenceFormat:
    """Tests that data asset references use correct azureml:name:version format."""

    def test_build_data_asset_reference_format(self):
        """Test that build_data_asset_reference returns correct asset URI format."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        
        assert "asset_uri" in result
        assert result["asset_uri"] == "azureml:resume-ner-data:2"
        assert result["asset_uri"].startswith("azureml:")

    def test_build_data_input_uses_asset_reference(self):
        """Test that _build_data_input_from_asset uses azureml:name:version format."""
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        data_input = _build_data_input_from_asset(mock_data_asset)
        
        assert data_input.path == "azureml:resume-ner-data:2"
        assert data_input.path.startswith("azureml:")
        assert ":" in data_input.path  # Should have name:version format
        assert not data_input.path.startswith("azureml://datastores/")  # Should NOT use manual datastore path

    def test_sweeps_build_data_input_uses_asset_reference(self):
        """Test that sweeps _build_data_input_from_asset uses correct format."""
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        data_input = sweeps_build_data_input(mock_data_asset)
        
        assert data_input.path == "azureml:resume-ner-data:2"
        assert data_input.path.startswith("azureml:")
        assert not data_input.path.startswith("azureml://datastores/")  # Should NOT use manual path

    def test_asset_reference_not_manual_datastore_path(self):
        """Test that asset references are NOT manually constructed datastore paths."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "resume-ner-data"
        mock_data_asset.version = "2"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        asset_uri = result["asset_uri"]
        
        # Should NOT be a manual datastore path
        assert not asset_uri.startswith("azureml://datastores/")
        # Should be asset reference format
        assert asset_uri.startswith("azureml:")
        assert asset_uri.count(":") == 2  # azureml:name:version

