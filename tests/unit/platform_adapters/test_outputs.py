"""Tests for output path resolvers."""

import os
import pytest
from pathlib import Path
from platform_adapters.outputs import (
    AzureMLOutputPathResolver,
    LocalOutputPathResolver,
)


class TestAzureMLOutputPathResolver:
    """Tests for AzureMLOutputPathResolver."""

    def test_resolve_named_output(self, monkeypatch):
        """Test resolving named output from environment variable."""
        monkeypatch.setenv("AZURE_ML_OUTPUT_checkpoint", "/mnt/outputs/checkpoint")
        
        resolver = AzureMLOutputPathResolver()
        path = resolver.resolve_output_path("checkpoint")
        
        assert path == Path("/mnt/outputs/checkpoint")

    def test_resolve_fallback_to_generic(self, monkeypatch):
        """Test fallback to generic output directory."""
        monkeypatch.delenv("AZURE_ML_OUTPUT_checkpoint", raising=False)
        monkeypatch.setenv("AZURE_ML_OUTPUT_DIR", "/mnt/outputs")
        
        resolver = AzureMLOutputPathResolver()
        path = resolver.resolve_output_path("checkpoint")
        
        assert path == Path("/mnt/outputs")

    def test_resolve_fallback_to_default(self, monkeypatch):
        """Test fallback to default path."""
        monkeypatch.delenv("AZURE_ML_OUTPUT_checkpoint", raising=False)
        monkeypatch.delenv("AZURE_ML_OUTPUT_DIR", raising=False)
        
        resolver = AzureMLOutputPathResolver()
        path = resolver.resolve_output_path("checkpoint", default=Path("./outputs"))
        
        assert path == Path("./outputs")

    def test_resolve_fallback_to_current_dir(self, monkeypatch):
        """Test fallback to current directory when no defaults."""
        monkeypatch.delenv("AZURE_ML_OUTPUT_checkpoint", raising=False)
        monkeypatch.delenv("AZURE_ML_OUTPUT_DIR", raising=False)
        
        resolver = AzureMLOutputPathResolver()
        path = resolver.resolve_output_path("checkpoint")
        
        assert path == Path("./outputs")

    def test_ensure_output_directory(self, temp_dir):
        """Test ensuring output directory exists with placeholder."""
        resolver = AzureMLOutputPathResolver()
        output_path = temp_dir / "checkpoint"
        
        result = resolver.ensure_output_directory(output_path)
        
        assert result.exists()
        assert result.is_dir()
        placeholder = result / "output_placeholder.txt"
        assert placeholder.exists()

    def test_ensure_output_directory_existing(self, temp_dir):
        """Test ensuring existing output directory."""
        output_path = temp_dir / "checkpoint"
        output_path.mkdir()
        
        resolver = AzureMLOutputPathResolver()
        result = resolver.ensure_output_directory(output_path)
        
        assert result == output_path
        assert result.exists()


class TestLocalOutputPathResolver:
    """Tests for LocalOutputPathResolver."""

    def test_resolve_output_path_with_default(self):
        """Test resolving output path with default directory."""
        resolver = LocalOutputPathResolver(default_output_dir=Path("./outputs"))
        path = resolver.resolve_output_path("checkpoint")
        
        assert path == Path("./outputs/checkpoint")

    def test_resolve_output_path_with_provided_default(self):
        """Test resolving output path with provided default."""
        resolver = LocalOutputPathResolver(default_output_dir=Path("./outputs"))
        path = resolver.resolve_output_path("checkpoint", default=Path("./custom"))
        
        assert path == Path("./custom/checkpoint")

    def test_resolve_output_path_no_default(self):
        """Test resolving output path without default."""
        resolver = LocalOutputPathResolver(default_output_dir=Path("./outputs"))
        path = resolver.resolve_output_path("checkpoint", default=None)
        
        assert path == Path("./outputs/checkpoint")

    def test_ensure_output_directory(self, temp_dir):
        """Test ensuring output directory exists."""
        resolver = LocalOutputPathResolver(default_output_dir=temp_dir)
        output_path = temp_dir / "checkpoint"
        
        result = resolver.ensure_output_directory(output_path)
        
        assert result.exists()
        assert result.is_dir()

    def test_ensure_output_directory_existing(self, temp_dir):
        """Test ensuring existing output directory."""
        output_path = temp_dir / "checkpoint"
        output_path.mkdir()
        
        resolver = LocalOutputPathResolver(default_output_dir=temp_dir)
        result = resolver.ensure_output_directory(output_path)
        
        assert result == output_path
        assert result.exists()

    def test_ensure_output_directory_nested(self, temp_dir):
        """Test ensuring nested output directory."""
        resolver = LocalOutputPathResolver(default_output_dir=temp_dir)
        output_path = temp_dir / "nested" / "deep" / "checkpoint"
        
        result = resolver.ensure_output_directory(output_path)
        
        assert result.exists()
        assert result.is_dir()

