"""Tests for platform adapter selection."""

import os
import pytest
from pathlib import Path
from platform_adapters.adapters import (
    get_platform_adapter,
    AzureMLAdapter,
    LocalAdapter,
)


class TestGetPlatformAdapter:
    """Tests for get_platform_adapter function."""

    def test_detect_azureml_adapter(self, monkeypatch):
        """Test that Azure ML adapter is detected from environment."""
        monkeypatch.setenv("AZURE_ML_OUTPUT_DIR", "/mnt/outputs")
        
        adapter = get_platform_adapter()
        
        assert isinstance(adapter, AzureMLAdapter)

    def test_detect_local_adapter(self, monkeypatch):
        """Test that local adapter is used when Azure ML env vars are absent."""
        # Remove any Azure ML environment variables
        for key in list(os.environ.keys()):
            if key.startswith("AZURE_ML_"):
                monkeypatch.delenv(key, raising=False)
        
        adapter = get_platform_adapter()
        
        assert isinstance(adapter, LocalAdapter)

    def test_local_adapter_with_default_dir(self, monkeypatch):
        """Test local adapter with custom default output directory."""
        for key in list(os.environ.keys()):
            if key.startswith("AZURE_ML_"):
                monkeypatch.delenv(key, raising=False)
        
        default_dir = Path("./custom_outputs")
        adapter = get_platform_adapter(default_output_dir=default_dir)
        
        assert isinstance(adapter, LocalAdapter)
        output_resolver = adapter.get_output_path_resolver()
        path = output_resolver.resolve_output_path("checkpoint")
        assert "custom_outputs" in str(path)


class TestAzureMLAdapter:
    """Tests for AzureMLAdapter."""

    def test_is_platform_job(self, monkeypatch):
        """Test that Azure ML adapter detects platform job."""
        monkeypatch.setenv("AZURE_ML_OUTPUT_DIR", "/mnt/outputs")
        
        adapter = AzureMLAdapter()
        
        assert adapter.is_platform_job() is True

    def test_is_not_platform_job(self, monkeypatch):
        """Test that adapter returns False when no Azure ML env vars."""
        for key in list(os.environ.keys()):
            if key.startswith("AZURE_ML_"):
                monkeypatch.delenv(key, raising=False)
        
        adapter = AzureMLAdapter()
        
        assert adapter.is_platform_job() is False

    def test_get_output_path_resolver(self):
        """Test getting output path resolver."""
        adapter = AzureMLAdapter()
        resolver = adapter.get_output_path_resolver()
        
        from platform_adapters.outputs import AzureMLOutputPathResolver
        assert isinstance(resolver, AzureMLOutputPathResolver)

    def test_get_logging_adapter(self):
        """Test getting logging adapter."""
        adapter = AzureMLAdapter()
        logging_adapter = adapter.get_logging_adapter()
        
        from platform_adapters.logging_adapter import AzureMLLoggingAdapter
        assert isinstance(logging_adapter, AzureMLLoggingAdapter)


class TestLocalAdapter:
    """Tests for LocalAdapter."""

    def test_is_platform_job(self):
        """Test that local adapter always returns False."""
        adapter = LocalAdapter()
        
        assert adapter.is_platform_job() is False

    def test_get_output_path_resolver(self):
        """Test getting output path resolver."""
        adapter = LocalAdapter(default_output_dir=Path("./outputs"))
        resolver = adapter.get_output_path_resolver()
        
        from platform_adapters.outputs import LocalOutputPathResolver
        assert isinstance(resolver, LocalOutputPathResolver)

    def test_get_logging_adapter(self):
        """Test getting logging adapter."""
        adapter = LocalAdapter()
        logging_adapter = adapter.get_logging_adapter()
        
        from platform_adapters.logging_adapter import LocalLoggingAdapter
        assert isinstance(logging_adapter, LocalLoggingAdapter)

    def test_default_output_dir(self):
        """Test default output directory."""
        adapter = LocalAdapter(default_output_dir=Path("./custom"))
        resolver = adapter.get_output_path_resolver()
        path = resolver.resolve_output_path("checkpoint")
        
        assert "custom" in str(path)

