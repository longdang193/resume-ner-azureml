"""Unit tests for conversion.yaml config loading."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from shared.yaml_utils import load_yaml
from config.conversion import load_conversion_config


class TestConversionConfigLoading:
    """Test loading conversion.yaml configuration."""

    def test_load_yaml_loads_conversion_config(self, tmp_path):
        """Test that load_yaml can load conversion.yaml."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 18
  quantization: "none"
  run_smoke_test: true
output:
  filename_pattern: "model_{quantization}.onnx"
""")
        
        config = load_yaml(conversion_yaml)
        
        assert isinstance(config, dict)
        assert "target" in config
        assert "onnx" in config
        assert "output" in config

    def test_conversion_config_structure(self, tmp_path):
        """Test that conversion.yaml has correct structure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 18
  quantization: "none"
  run_smoke_test: true
output:
  filename_pattern: "model_{quantization}.onnx"
""")
        
        config = load_yaml(conversion_yaml)
        
        # Verify target section
        assert isinstance(config["target"], dict)
        assert "format" in config["target"]
        
        # Verify onnx section
        assert isinstance(config["onnx"], dict)
        assert "opset_version" in config["onnx"]
        assert "quantization" in config["onnx"]
        assert "run_smoke_test" in config["onnx"]
        
        # Verify output section
        assert isinstance(config["output"], dict)
        assert "filename_pattern" in config["output"]

    def test_conversion_config_default_values(self, tmp_path):
        """Test that config has expected default values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 18
  quantization: "none"
  run_smoke_test: true
output:
  filename_pattern: "model_{quantization}.onnx"
""")
        
        config = load_yaml(conversion_yaml)
        
        # Verify default values match actual config file
        assert config["target"]["format"] == "onnx"
        assert config["onnx"]["opset_version"] == 18
        assert config["onnx"]["quantization"] == "none"
        assert config["onnx"]["run_smoke_test"] is True
        assert config["output"]["filename_pattern"] == "model_{quantization}.onnx"

    def test_conversion_config_custom_values(self, tmp_path):
        """Test loading config with custom values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 19
  quantization: "int8"
  run_smoke_test: false
output:
  filename_pattern: "custom_{quantization}_model.onnx"
""")
        
        config = load_yaml(conversion_yaml)
        
        # Verify custom values
        assert config["target"]["format"] == "onnx"
        assert config["onnx"]["opset_version"] == 19
        assert config["onnx"]["quantization"] == "int8"
        assert config["onnx"]["run_smoke_test"] is False
        assert config["output"]["filename_pattern"] == "custom_{quantization}_model.onnx"

    def test_conversion_config_missing_sections(self, tmp_path):
        """Test that missing sections are handled gracefully."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 18
""")
        
        config = load_yaml(conversion_yaml)
        
        # Should load successfully even with missing sections
        assert "target" in config
        assert "onnx" in config
        # output section is missing, but that's OK for this test
        # (actual usage would need it, but config loader doesn't validate)

    def test_conversion_config_types(self, tmp_path):
        """Test that config values have correct types."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        conversion_yaml = config_dir / "conversion.yaml"
        conversion_yaml.write_text("""
target:
  format: "onnx"
onnx:
  opset_version: 18
  quantization: "none"
  run_smoke_test: true
output:
  filename_pattern: "model_{quantization}.onnx"
""")
        
        config = load_yaml(conversion_yaml)
        
        # Verify types
        assert isinstance(config["target"]["format"], str)
        assert isinstance(config["onnx"]["opset_version"], int)
        assert isinstance(config["onnx"]["quantization"], str)
        assert isinstance(config["onnx"]["run_smoke_test"], bool)
        assert isinstance(config["output"]["filename_pattern"], str)

    def test_load_actual_conversion_yaml(self):
        """Test loading the actual conversion.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        conversion_yaml = config_dir / "conversion.yaml"
        
        if not conversion_yaml.exists():
            pytest.skip("conversion.yaml not found in config directory")
        
        config = load_yaml(conversion_yaml)
        
        # Verify it loads successfully
        assert isinstance(config, dict)
        assert "target" in config
        assert "onnx" in config
        assert "output" in config
        
        # Verify structure matches expected
        assert isinstance(config["target"], dict)
        assert isinstance(config["onnx"], dict)
        assert isinstance(config["output"], dict)

