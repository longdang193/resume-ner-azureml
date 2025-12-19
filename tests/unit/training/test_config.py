"""Tests for training configuration utilities."""

import argparse
import pytest
from pathlib import Path
from training.config import build_training_config, _apply_argument_overrides


class TestBuildTrainingConfig:
    """Tests for build_training_config function."""

    def test_basic_config_loading(self, mock_configs):
        """Test loading basic training configuration."""
        config_dir = mock_configs["root"]
        
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert config["model"]["backbone"] == "distilbert-base-uncased"

    def test_missing_config_file(self, temp_dir):
        """Test that missing config file raises FileNotFoundError."""
        args = argparse.Namespace(backbone="distilbert")
        
        with pytest.raises(FileNotFoundError):
            build_training_config(args, temp_dir)

    def test_config_structure(self, mock_configs):
        """Test that config has expected structure."""
        config_dir = mock_configs["root"]
        
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert "training" in config
        assert "epochs" in config["training"]
        assert "batch_size" in config["training"]
        assert "learning_rate" in config["training"]


class TestApplyArgumentOverrides:
    """Tests for _apply_argument_overrides function."""

    def test_learning_rate_override(self, mock_configs):
        """Test learning rate override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=1e-4,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["learning_rate"] == 1e-4

    def test_batch_size_override(self, mock_configs):
        """Test batch size override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=8,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["batch_size"] == 8

    def test_dropout_override(self, mock_configs):
        """Test dropout override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=0.2,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["model"]["dropout"] == 0.2

    def test_weight_decay_override(self, mock_configs):
        """Test weight decay override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=0.05,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["weight_decay"] == 0.05

    def test_epochs_override(self, mock_configs):
        """Test epochs override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=5,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["epochs"] == 5

    def test_random_seed_override(self, mock_configs):
        """Test random seed override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=123,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["random_seed"] == 123

    def test_early_stopping_enabled_true(self, mock_configs):
        """Test early stopping enabled with 'true' string."""
        config_dir = mock_configs["root"]
        # First ensure early_stopping exists in config
        import yaml
        train_config_path = mock_configs["train"]
        with open(train_config_path, "r") as f:
            train_config = yaml.safe_load(f)
        train_config["training"]["early_stopping"] = {"enabled": False, "patience": 3}
        with open(train_config_path, "w") as f:
            yaml.dump(train_config, f)
        
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled="true",
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["early_stopping"]["enabled"] is True

    def test_early_stopping_enabled_false(self, mock_configs):
        """Test early stopping disabled with 'false' string."""
        config_dir = mock_configs["root"]
        import yaml
        train_config_path = mock_configs["train"]
        with open(train_config_path, "r") as f:
            train_config = yaml.safe_load(f)
        train_config["training"]["early_stopping"] = {"enabled": True, "patience": 3}
        with open(train_config_path, "w") as f:
            yaml.dump(train_config, f)
        
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled="false",
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["early_stopping"]["enabled"] is False

    def test_use_combined_data_override(self, mock_configs):
        """Test use_combined_data override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data="true",
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["data"]["use_combined_data"] is True

    def test_fold_idx_override(self, mock_configs):
        """Test fold_idx override."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=None,
            batch_size=None,
            dropout=None,
            weight_decay=None,
            epochs=None,
            random_seed=None,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=2,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["fold_idx"] == 2

    def test_multiple_overrides(self, mock_configs):
        """Test multiple argument overrides at once."""
        config_dir = mock_configs["root"]
        args = argparse.Namespace(
            backbone="distilbert",
            learning_rate=1e-4,
            batch_size=16,
            dropout=0.3,
            weight_decay=0.02,
            epochs=10,
            random_seed=42,
            early_stopping_enabled=None,
            use_combined_data=None,
            fold_idx=None,
            fold_splits_file=None,
            k_folds=None,
            use_all_data=None,
        )
        
        config = build_training_config(args, config_dir)
        
        assert config["training"]["learning_rate"] == 1e-4
        assert config["training"]["batch_size"] == 16
        assert config["model"]["dropout"] == 0.3
        assert config["training"]["weight_decay"] == 0.02
        assert config["training"]["epochs"] == 10
        assert config["training"]["random_seed"] == 42

