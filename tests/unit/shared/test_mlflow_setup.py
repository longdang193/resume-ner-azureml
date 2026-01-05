"""Unit tests for MLflow setup utility."""

from shared.mlflow_setup import (
    setup_mlflow_cross_platform,
    setup_mlflow_from_config,
    create_ml_client_from_config,
    _get_azure_ml_tracking_uri,
    _get_local_tracking_uri,
)
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add project root to path
import sys
ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    import shutil
    shutil.rmtree(temp_path)


class TestGetLocalTrackingUri:
    """Tests for _get_local_tracking_uri function."""

    @patch("shared.mlflow_setup.detect_platform")
    def test_local_platform(self, mock_detect):
        """Test local platform returns SQLite URI in ./mlruns."""
        mock_detect.return_value = "local"

        uri = _get_local_tracking_uri()

        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri
        # Should be in current directory's mlruns folder
        assert Path(uri.replace("sqlite:///", "")).parent.name == "mlruns"

    @patch("shared.mlflow_setup.detect_platform")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    def test_colab_with_drive(self, mock_is_dir, mock_exists, mock_detect):
        """Test Colab platform with Drive mounted uses Drive path."""
        mock_detect.return_value = "colab"
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        uri = _get_local_tracking_uri()

        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri
        assert "/content/drive/MyDrive" in uri or "resume-ner-mlflow" in uri

    @patch("shared.mlflow_setup.detect_platform")
    @patch("pathlib.Path.exists")
    def test_colab_without_drive(self, mock_exists, mock_detect):
        """Test Colab platform without Drive uses /content."""
        mock_detect.return_value = "colab"
        mock_exists.return_value = False

        uri = _get_local_tracking_uri()

        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri
        assert "/content" in uri

    @patch("shared.mlflow_setup.detect_platform")
    def test_kaggle_platform(self, mock_detect):
        """Test Kaggle platform uses /kaggle/working."""
        mock_detect.return_value = "kaggle"

        uri = _get_local_tracking_uri()

        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri
        assert "/kaggle/working" in uri


class TestGetAzureMlTrackingUri:
    """Tests for _get_azure_ml_tracking_uri function."""

    def test_success(self):
        """Test successful Azure ML workspace URI retrieval."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspace_name = "test-ws"
        mock_ml_client.workspaces.get.return_value = mock_workspace

        with patch("shared.mlflow_setup.azureml") as mock_azureml:
            uri = _get_azure_ml_tracking_uri(mock_ml_client)

            assert uri == "azureml://workspace/experiments"
            mock_ml_client.workspaces.get.assert_called_once_with(
                name="test-ws")

    def test_import_error(self):
        """Test ImportError when azureml.mlflow is not available."""
        mock_ml_client = MagicMock()

        with patch("builtins.__import__", side_effect=ImportError("No module named 'azureml'")):
            with pytest.raises(ImportError) as exc_info:
                _get_azure_ml_tracking_uri(mock_ml_client)

            assert "azureml.mlflow" in str(exc_info.value)

    def test_workspace_access_error(self):
        """Test RuntimeError when workspace access fails."""
        mock_ml_client = MagicMock()
        mock_ml_client.workspace_name = "test-ws"
        mock_ml_client.workspaces.get.side_effect = Exception("Access denied")

        with patch("shared.mlflow_setup.azureml"):
            with pytest.raises(RuntimeError) as exc_info:
                _get_azure_ml_tracking_uri(mock_ml_client)

            assert "Failed to get Azure ML workspace tracking URI" in str(
                exc_info.value)


class TestSetupMlflowCrossPlatform:
    """Tests for setup_mlflow_cross_platform function."""

    @patch("shared.mlflow_setup.mlflow")
    @patch("shared.mlflow_setup._get_local_tracking_uri")
    def test_local_fallback_no_ml_client(self, mock_get_local, mock_mlflow):
        """Test local fallback when no ML client provided."""
        mock_get_local.return_value = "sqlite:///./mlruns/mlflow.db"

        uri = setup_mlflow_cross_platform(
            experiment_name="test-experiment",
            ml_client=None
        )

        assert uri == "sqlite:///./mlruns/mlflow.db"
        mock_get_local.assert_called_once()
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "sqlite:///./mlruns/mlflow.db")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("shared.mlflow_setup.mlflow")
    @patch("shared.mlflow_setup._get_azure_ml_tracking_uri")
    def test_azure_ml_success(self, mock_get_azure, mock_mlflow):
        """Test successful Azure ML setup."""
        mock_ml_client = MagicMock()
        mock_get_azure.return_value = "azureml://workspace/experiments"

        uri = setup_mlflow_cross_platform(
            experiment_name="test-experiment",
            ml_client=mock_ml_client
        )

        assert uri == "azureml://workspace/experiments"
        mock_get_azure.assert_called_once_with(mock_ml_client)
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "azureml://workspace/experiments")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("shared.mlflow_setup.mlflow")
    @patch("shared.mlflow_setup._get_azure_ml_tracking_uri")
    @patch("shared.mlflow_setup._get_local_tracking_uri")
    def test_azure_ml_failure_with_fallback(self, mock_get_local, mock_get_azure, mock_mlflow):
        """Test Azure ML failure with fallback enabled."""
        mock_ml_client = MagicMock()
        mock_get_azure.side_effect = Exception("Azure ML unavailable")
        mock_get_local.return_value = "sqlite:///./mlruns/mlflow.db"

        uri = setup_mlflow_cross_platform(
            experiment_name="test-experiment",
            ml_client=mock_ml_client,
            fallback_to_local=True
        )

        assert uri == "sqlite:///./mlruns/mlflow.db"
        mock_get_azure.assert_called_once()
        mock_get_local.assert_called_once()
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "sqlite:///./mlruns/mlflow.db")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("shared.mlflow_setup._get_azure_ml_tracking_uri")
    def test_azure_ml_failure_no_fallback(self, mock_get_azure):
        """Test Azure ML failure with fallback disabled raises error."""
        mock_ml_client = MagicMock()
        mock_get_azure.side_effect = Exception("Azure ML unavailable")

        with pytest.raises(RuntimeError) as exc_info:
            setup_mlflow_cross_platform(
                experiment_name="test-experiment",
                ml_client=mock_ml_client,
                fallback_to_local=False
            )

        assert "Azure ML tracking failed and fallback disabled" in str(
            exc_info.value)

    @patch("shared.mlflow_setup.mlflow", side_effect=ImportError("No module named 'mlflow'"))
    def test_mlflow_not_installed(self, mock_mlflow):
        """Test ImportError when mlflow is not installed."""
        with pytest.raises(ImportError) as exc_info:
            setup_mlflow_cross_platform(experiment_name="test-experiment")

        assert "mlflow is required" in str(exc_info.value)


class TestPlatformSpecificBehavior:
    """Integration-style tests for platform-specific behavior."""

    @patch("shared.mlflow_setup.mlflow")
    @patch.dict(os.environ, {"COLAB_GPU": "1"})
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    def test_colab_drive_mounted(self, mock_is_dir, mock_exists, mock_mlflow):
        """Test Colab with Drive mounted uses Drive path."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        uri = setup_mlflow_cross_platform(experiment_name="test")

        # Should use Drive path
        assert "drive" in uri.lower() or "mydrive" in uri.lower()

    @patch("shared.mlflow_setup.mlflow")
    @patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"})
    def test_kaggle_platform(self, mock_mlflow):
        """Test Kaggle platform uses /kaggle/working."""
        uri = setup_mlflow_cross_platform(experiment_name="test")

        assert "/kaggle/working" in uri

    @patch("shared.mlflow_setup.mlflow")
    @patch.dict(os.environ, {}, clear=True)
    def test_local_platform(self, mock_mlflow):
        """Test local platform uses ./mlruns."""
        uri = setup_mlflow_cross_platform(experiment_name="test")

        assert uri.startswith("sqlite:///")
        # Should create mlruns directory
        assert "mlruns" in uri


class TestCreateMlClientFromConfig:
    """Tests for create_ml_client_from_config function."""

    @patch("shared.mlflow_setup.load_yaml")
    @patch("shared.mlflow_setup.MLClient")
    @patch("shared.mlflow_setup.DefaultAzureCredential")
    @patch.dict(os.environ, {
        "AZURE_SUBSCRIPTION_ID": "test-sub-id",
        "AZURE_RESOURCE_GROUP": "test-rg"
    })
    def test_success_with_env_vars(self, mock_cred, mock_mlclient, mock_load_yaml):
        """Test successful MLClient creation with environment variables."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": True,
                "workspace_name": "test-ws"
            }
        }
        mock_cred_instance = MagicMock()
        mock_cred.return_value = mock_cred_instance
        mock_client_instance = MagicMock()
        mock_mlclient.return_value = mock_client_instance

        config_dir = Path("config")
        client = create_ml_client_from_config(config_dir)

        assert client is not None
        mock_mlclient.assert_called_once_with(
            credential=mock_cred_instance,
            subscription_id="test-sub-id",
            resource_group_name="test-rg",
            workspace_name="test-ws"
        )

    @patch("shared.mlflow_setup.load_yaml")
    def test_azure_ml_disabled(self, mock_load_yaml):
        """Test returns None when Azure ML is disabled."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": False
            }
        }

        config_dir = Path("config")
        client = create_ml_client_from_config(config_dir)

        assert client is None

    @patch("shared.mlflow_setup.load_yaml")
    def test_config_missing_azure_ml_section(self, mock_load_yaml):
        """Test returns None when Azure ML section is missing."""
        mock_load_yaml.return_value = {}

        config_dir = Path("config")
        client = create_ml_client_from_config(config_dir)

        assert client is None

    @patch("shared.mlflow_setup.load_yaml")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials(self, mock_load_yaml):
        """Test returns None when credentials are missing."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": True,
                "workspace_name": "test-ws"
            }
        }

        config_dir = Path("config")
        client = create_ml_client_from_config(config_dir)

        assert client is None

    @patch("shared.mlflow_setup.load_yaml")
    def test_import_error(self, mock_load_yaml):
        """Test raises ImportError when Azure ML SDK not available."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": True
            }
        }

        with patch("builtins.__import__", side_effect=ImportError("No module named 'azure'")):
            with pytest.raises(ImportError):
                create_ml_client_from_config(Path("config"))


class TestSetupMlflowFromConfig:
    """Tests for setup_mlflow_from_config function."""

    @patch("shared.mlflow_setup.setup_mlflow_cross_platform")
    @patch("shared.mlflow_setup.load_yaml")
    def test_config_file_exists_azure_enabled(self, mock_load_yaml, mock_setup):
        """Test setup with config file and Azure ML enabled."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": True,
                "workspace_name": "test-ws"
            }
        }
        mock_setup.return_value = "azureml://workspace/experiments"

        with patch("shared.mlflow_setup.create_ml_client_from_config") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            uri = setup_mlflow_from_config(
                experiment_name="test-experiment",
                config_dir=Path("config")
            )

            assert uri == "azureml://workspace/experiments"
            mock_setup.assert_called_once_with(
                experiment_name="test-experiment",
                ml_client=mock_client,
                fallback_to_local=True
            )

    @patch("shared.mlflow_setup.setup_mlflow_cross_platform")
    @patch("shared.mlflow_setup.load_yaml")
    def test_config_file_exists_azure_disabled(self, mock_load_yaml, mock_setup):
        """Test setup with config file and Azure ML disabled."""
        mock_load_yaml.return_value = {
            "azure_ml": {
                "enabled": False
            }
        }
        mock_setup.return_value = "sqlite:///./mlruns/mlflow.db"

        uri = setup_mlflow_from_config(
            experiment_name="test-experiment",
            config_dir=Path("config")
        )

        assert uri == "sqlite:///./mlruns/mlflow.db"
        mock_setup.assert_called_once_with(
            experiment_name="test-experiment",
            ml_client=None,
            fallback_to_local=True
        )

    @patch("shared.mlflow_setup.setup_mlflow_cross_platform")
    @patch("pathlib.Path.exists")
    def test_config_file_missing(self, mock_exists, mock_setup):
        """Test setup when config file doesn't exist."""
        mock_exists.return_value = False
        mock_setup.return_value = "sqlite:///./mlruns/mlflow.db"

        uri = setup_mlflow_from_config(
            experiment_name="test-experiment",
            config_dir=Path("config")
        )

        assert uri == "sqlite:///./mlruns/mlflow.db"
        mock_setup.assert_called_once_with(
            experiment_name="test-experiment",
            ml_client=None,
            fallback_to_local=True
        )
