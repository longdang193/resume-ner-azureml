"""Tests for Azure ML sweep job creation."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from orchestration.jobs.sweeps import (
    create_search_space,
    _build_data_input_from_asset,
)


class TestCreateSearchSpace:
    """Tests for create_search_space function."""

    def test_create_search_space_choice(self):
        """Test creating search space with choice distribution."""
        hpo_config = {
            "search_space": {
                "backbone": {
                    "type": "choice",
                    "values": ["bert-base-uncased", "distilbert-base-uncased"]
                }
            }
        }
        
        with patch("orchestration.jobs.sweeps.Choice") as mock_choice:
            result = create_search_space(hpo_config)
            
            assert "backbone" in result
            mock_choice.assert_called_once_with(
                values=["bert-base-uncased", "distilbert-base-uncased"]
            )

    def test_create_search_space_uniform(self):
        """Test creating search space with uniform distribution."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "uniform",
                    "min": 1e-5,
                    "max": 5e-5
                }
            }
        }
        
        with patch("orchestration.jobs.sweeps.Uniform") as mock_uniform:
            result = create_search_space(hpo_config)
            
            assert "learning_rate" in result
            mock_uniform.assert_called_once_with(
                min_value=1e-5,
                max_value=5e-5
            )

    def test_create_search_space_loguniform(self):
        """Test creating search space with loguniform distribution."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-6,
                    "max": 1e-3
                }
            }
        }
        
        with patch("orchestration.jobs.sweeps.LogUniform") as mock_loguniform:
            result = create_search_space(hpo_config)
            
            assert "learning_rate" in result
            mock_loguniform.assert_called_once_with(
                min_value=1e-6,
                max_value=1e-3
            )

    def test_create_search_space_multiple_params(self):
        """Test creating search space with multiple parameters."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "uniform",
                    "min": 1e-5,
                    "max": 5e-5
                },
                "dropout": {
                    "type": "uniform",
                    "min": 0.1,
                    "max": 0.3
                },
                "backbone": {
                    "type": "choice",
                    "values": ["bert", "distilbert"]
                }
            }
        }
        
        result = create_search_space(hpo_config)
        
        assert len(result) == 3
        assert "learning_rate" in result
        assert "dropout" in result
        assert "backbone" in result


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

