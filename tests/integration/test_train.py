"""Integration tests for main training script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest


class TestParseArguments:
    """Tests for parse_arguments function."""

    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "distilbert"])
    def test_parse_required_arguments(self):
        """Test parsing of required arguments."""
        from train import parse_arguments
        
        args = parse_arguments()
        
        assert args.data_asset == "/data"
        assert args.config_dir == "/config"
        assert args.backbone == "distilbert"
        assert args.learning_rate is None
        assert args.batch_size is None

    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "deberta", "--learning-rate", "0.001", "--batch-size", "16",
                        "--dropout", "0.1", "--weight-decay", "0.01", "--epochs", "5",
                        "--random-seed", "42", "--early-stopping-enabled", "true",
                        "--use-combined-data", "false", "--fold-idx", "0", "--k-folds", "5"])
    def test_parse_all_arguments(self):
        """Test parsing of all optional arguments."""
        from train import parse_arguments
        
        args = parse_arguments()
        
        assert args.data_asset == "/data"
        assert args.config_dir == "/config"
        assert args.backbone == "deberta"
        assert args.learning_rate == 0.001
        assert args.batch_size == 16
        assert args.dropout == 0.1
        assert args.weight_decay == 0.01
        assert args.epochs == 5
        assert args.random_seed == 42
        assert args.early_stopping_enabled == "true"
        assert args.use_combined_data == "false"
        assert args.fold_idx == 0
        assert args.k_folds == 5

    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "distilbert", "--use-all-data", "true"])
    def test_parse_use_all_data(self):
        """Test parsing use-all-data argument."""
        from train import parse_arguments
        
        args = parse_arguments()
        
        assert args.use_all_data == "true"


class TestLogTrainingParameters:
    """Tests for log_training_parameters function."""

    def test_log_training_parameters_all_params(self):
        """Test logging all training parameters."""
        from train import log_training_parameters
        
        config = {
            "training": {
                "learning_rate": 0.001,
                "batch_size": 16,
                "weight_decay": 0.01,
                "epochs": 5,
            },
            "model": {
                "dropout": 0.1,
                "backbone": "distilbert",
            },
        }
        
        mock_logging_adapter = MagicMock()
        
        log_training_parameters(config, mock_logging_adapter)
        
        mock_logging_adapter.log_params.assert_called_once()
        logged_params = mock_logging_adapter.log_params.call_args[0][0]
        assert logged_params["learning_rate"] == 0.001
        assert logged_params["batch_size"] == 16
        assert logged_params["dropout"] == 0.1
        assert logged_params["weight_decay"] == 0.01
        assert logged_params["epochs"] == 5
        assert logged_params["backbone"] == "distilbert"

    def test_log_training_parameters_partial_params(self):
        """Test logging with partial parameters."""
        from train import log_training_parameters
        
        config = {
            "training": {
                "learning_rate": 0.001,
            },
            "model": {
                "backbone": "distilbert",
            },
        }
        
        mock_logging_adapter = MagicMock()
        
        log_training_parameters(config, mock_logging_adapter)
        
        mock_logging_adapter.log_params.assert_called_once()
        logged_params = mock_logging_adapter.log_params.call_args[0][0]
        assert logged_params["learning_rate"] == 0.001
        assert logged_params["backbone"] == "distilbert"
        assert "batch_size" not in logged_params or logged_params.get("batch_size") is None

    def test_log_training_parameters_filters_none(self):
        """Test that None values are filtered out."""
        from train import log_training_parameters
        
        config = {
            "training": {
                "learning_rate": 0.001,
                "batch_size": None,
            },
            "model": {
                "backbone": "distilbert",
                "dropout": None,
            },
        }
        
        mock_logging_adapter = MagicMock()
        
        log_training_parameters(config, mock_logging_adapter)
        
        mock_logging_adapter.log_params.assert_called_once()
        logged_params = mock_logging_adapter.log_params.call_args[0][0]
        assert logged_params["learning_rate"] == 0.001
        assert logged_params["backbone"] == "distilbert"
        assert None not in logged_params.values()


class TestMain:
    """Tests for main function."""

    @patch("train.log_metrics")
    @patch("train.train_model")
    @patch("train.load_dataset")
    @patch("train.set_seed")
    @patch("train.build_training_config")
    @patch("train.get_platform_adapter")
    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "distilbert"])
    def test_main_success(self, mock_get_adapter, mock_build_config, mock_set_seed,
                          mock_load_dataset, mock_train_model, mock_log_metrics):
        """Test successful main execution."""
        from train import main
        
        mock_config = {
            "training": {"random_seed": 42},
            "model": {"backbone": "distilbert"},
        }
        mock_build_config.return_value = mock_config
        
        mock_dataset = [{"text": "test", "annotations": []}]
        mock_load_dataset.return_value = mock_dataset
        
        mock_adapter = MagicMock()
        mock_output_resolver = MagicMock()
        mock_output_resolver.resolve_output_path.return_value = Path("/output")
        mock_output_resolver.ensure_output_directory.return_value = Path("/output")
        mock_logging_adapter = MagicMock()
        mock_mlflow_context = MagicMock()
        mock_context = MagicMock()
        mock_mlflow_context.get_context.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow_context.get_context.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_adapter.get_output_path_resolver.return_value = mock_output_resolver
        mock_adapter.get_logging_adapter.return_value = mock_logging_adapter
        mock_adapter.get_mlflow_context_manager.return_value = mock_mlflow_context
        mock_get_adapter.return_value = mock_adapter
        
        mock_metrics = {"f1": 0.85, "loss": 0.5}
        mock_train_model.return_value = mock_metrics
        
        with patch("train.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance
            
            main()
        
        mock_build_config.assert_called_once()
        mock_set_seed.assert_called_once_with(42)
        mock_load_dataset.assert_called_once_with("/data")
        mock_train_model.assert_called_once()
        mock_log_metrics.assert_called_once()

    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/nonexistent",
                        "--backbone", "distilbert"])
    def test_main_config_dir_not_found(self):
        """Test main execution when config directory doesn't exist."""
        from train import main
        
        with patch("train.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path_class.return_value = mock_path_instance
            
            with pytest.raises(FileNotFoundError, match="Config directory not found"):
                main()

    @patch("train.log_metrics")
    @patch("train.train_model")
    @patch("train.load_dataset")
    @patch("train.set_seed")
    @patch("train.build_training_config")
    @patch("train.get_platform_adapter")
    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "distilbert", "--learning-rate", "0.001"])
    def test_main_with_overrides(self, mock_get_adapter, mock_build_config, mock_set_seed,
                                 mock_load_dataset, mock_train_model, mock_log_metrics):
        """Test main execution with argument overrides."""
        from train import main
        
        mock_config = {
            "training": {"random_seed": 42, "learning_rate": 0.001},
            "model": {"backbone": "distilbert"},
        }
        mock_build_config.return_value = mock_config
        
        mock_dataset = [{"text": "test", "annotations": []}]
        mock_load_dataset.return_value = mock_dataset
        
        mock_adapter = MagicMock()
        mock_output_resolver = MagicMock()
        mock_output_resolver.resolve_output_path.return_value = Path("/output")
        mock_output_resolver.ensure_output_directory.return_value = Path("/output")
        mock_logging_adapter = MagicMock()
        mock_mlflow_context = MagicMock()
        mock_mlflow_context.get_context.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow_context.get_context.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_adapter.get_output_path_resolver.return_value = mock_output_resolver
        mock_adapter.get_logging_adapter.return_value = mock_logging_adapter
        mock_adapter.get_mlflow_context_manager.return_value = mock_mlflow_context
        mock_get_adapter.return_value = mock_adapter
        
        mock_metrics = {"f1": 0.85}
        mock_train_model.return_value = mock_metrics
        
        with patch("train.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance
            
            main()
        
        call_args = mock_build_config.call_args[0]
        assert call_args[0].learning_rate == 0.001

    @patch("train.log_metrics")
    @patch("train.train_model")
    @patch("train.load_dataset")
    @patch("train.set_seed")
    @patch("train.build_training_config")
    @patch("train.get_platform_adapter")
    @patch("sys.argv", ["train.py", "--data-asset", "/data", "--config-dir", "/config",
                        "--backbone", "distilbert"])
    def test_main_uses_platform_adapter(self, mock_get_adapter, mock_build_config, mock_set_seed,
                                       mock_load_dataset, mock_train_model, mock_log_metrics):
        """Test that main uses platform adapter correctly."""
        from train import main
        
        mock_config = {
            "training": {"random_seed": 42},
            "model": {"backbone": "distilbert"},
        }
        mock_build_config.return_value = mock_config
        
        mock_dataset = [{"text": "test", "annotations": []}]
        mock_load_dataset.return_value = mock_dataset
        
        mock_adapter = MagicMock()
        mock_output_resolver = MagicMock()
        mock_output_resolver.resolve_output_path.return_value = Path("/output")
        mock_output_resolver.ensure_output_directory.return_value = Path("/output")
        mock_logging_adapter = MagicMock()
        mock_mlflow_context = MagicMock()
        mock_mlflow_context.get_context.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow_context.get_context.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_adapter.get_output_path_resolver.return_value = mock_output_resolver
        mock_adapter.get_logging_adapter.return_value = mock_logging_adapter
        mock_adapter.get_mlflow_context_manager.return_value = mock_mlflow_context
        mock_get_adapter.return_value = mock_adapter
        
        mock_metrics = {"f1": 0.85}
        mock_train_model.return_value = mock_metrics
        
        with patch("train.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance
            
            main()
        
        mock_get_adapter.assert_called_once()
        mock_output_resolver.resolve_output_path.assert_called_once()
        mock_output_resolver.ensure_output_directory.assert_called_once()
        mock_logging_adapter.log_params.assert_called_once()
        mock_mlflow_context.get_context.assert_called_once()

