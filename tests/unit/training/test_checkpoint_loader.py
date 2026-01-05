"""Unit tests for checkpoint loader module."""

import os
import json
import tempfile
from pathlib import Path
import pytest

from training.checkpoint_loader import resolve_checkpoint_path, validate_checkpoint


class TestValidateCheckpoint:
    """Test checkpoint validation."""
    
    def test_validate_checkpoint_valid(self, tmp_path):
        """Test validation of a valid checkpoint."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        
        # Create required files
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "pytorch_model.bin").write_text("fake model")
        
        assert validate_checkpoint(checkpoint_dir) is True
    
    def test_validate_checkpoint_with_safetensors(self, tmp_path):
        """Test validation with safetensors file."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "model.safetensors").write_text("fake model")
        
        assert validate_checkpoint(checkpoint_dir) is True
    
    def test_validate_checkpoint_missing_config(self, tmp_path):
        """Test validation fails when config.json is missing."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        
        (checkpoint_dir / "pytorch_model.bin").write_text("fake model")
        
        assert validate_checkpoint(checkpoint_dir) is False
    
    def test_validate_checkpoint_missing_model(self, tmp_path):
        """Test validation fails when model file is missing."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        
        assert validate_checkpoint(checkpoint_dir) is False
    
    def test_validate_checkpoint_nonexistent(self):
        """Test validation fails for non-existent path."""
        assert validate_checkpoint(Path("/nonexistent/path")) is False


class TestResolveCheckpointPath:
    """Test checkpoint path resolution."""
    
    def test_resolve_from_env_var(self, tmp_path, monkeypatch):
        """Test resolution from CHECKPOINT_PATH environment variable."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "model.bin").write_text("fake model")
        
        monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint_dir))
        
        config = {"training": {}}
        result = resolve_checkpoint_path(config)
        
        assert result == checkpoint_dir.resolve()
    
    def test_resolve_from_config(self, tmp_path):
        """Test resolution from config file."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "model.bin").write_text("fake model")
        
        config = {
            "training": {
                "checkpoint": {
                    "source_path": str(checkpoint_dir)
                }
            },
            "_config_dir": tmp_path,
        }
        
        result = resolve_checkpoint_path(config)
        
        assert result == checkpoint_dir.resolve()
    
    def test_resolve_from_cache(self, tmp_path):
        """Test resolution from previous training cache."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "model.bin").write_text("fake model")
        
        output_dir = checkpoint_dir.parent
        
        # Create cache file
        cache_file = tmp_path / "cache.json"
        with open(cache_file, "w") as f:
            json.dump({"output_dir": str(output_dir)}, f)
        
        config = {"training": {}}
        result = resolve_checkpoint_path(
            config,
            previous_cache_path=cache_file
        )
        
        assert result == checkpoint_dir.resolve()
    
    def test_resolve_with_pattern(self, tmp_path):
        """Test resolution with pattern replacement."""
        backbone = "distilbert"
        run_id = "20251227_220407"
        
        checkpoint_dir = tmp_path / f"{backbone}_{run_id}" / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text('{"test": "config"}')
        (checkpoint_dir / "model.bin").write_text("fake model")
        
        pattern_path = tmp_path / "{backbone}_{run_id}" / "checkpoint"
        config = {
            "training": {
                "checkpoint": {
                    "source_path": str(pattern_path)
                }
            },
            "_config_dir": tmp_path,
        }
        
        result = resolve_checkpoint_path(
            config,
            backbone=backbone,
            run_id=run_id
        )
        
        assert result == checkpoint_dir.resolve()
    
    def test_resolve_none_when_invalid(self, tmp_path):
        """Test returns None when checkpoint is invalid."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        # Missing required files
        
        config = {
            "training": {
                "checkpoint": {
                    "source_path": str(invalid_dir)
                }
            },
            "_config_dir": tmp_path,
        }
        
        result = resolve_checkpoint_path(config)
        
        assert result is None
    
    def test_resolve_none_when_not_configured(self):
        """Test returns None when no checkpoint is configured."""
        config = {"training": {}}
        result = resolve_checkpoint_path(config)
        
        assert result is None
    
    def test_resolve_priority_env_over_config(self, tmp_path, monkeypatch):
        """Test environment variable takes priority over config."""
        env_checkpoint = tmp_path / "env_checkpoint"
        env_checkpoint.mkdir()
        (env_checkpoint / "config.json").write_text('{"test": "config"}')
        (env_checkpoint / "model.bin").write_text("fake model")
        
        config_checkpoint = tmp_path / "config_checkpoint"
        config_checkpoint.mkdir()
        (config_checkpoint / "config.json").write_text('{"test": "config"}')
        (config_checkpoint / "model.bin").write_text("fake model")
        
        monkeypatch.setenv("CHECKPOINT_PATH", str(env_checkpoint))
        
        config = {
            "training": {
                "checkpoint": {
                    "source_path": str(config_checkpoint)
                }
            },
            "_config_dir": tmp_path,
        }
        
        result = resolve_checkpoint_path(config)
        
        assert result == env_checkpoint.resolve()

