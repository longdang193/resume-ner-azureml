"""Unit tests for extracting options from conversion.yaml."""

import pytest


class TestConversionOptions:
    """Test extraction and defaults for options in conversion.yaml."""

    def test_target_format_extraction(self):
        """Test target.format extraction from config."""
        conversion_config = {
            "target": {
                "format": "onnx"
            }
        }
        
        format_value = conversion_config.get("target", {}).get("format", "onnx")
        
        assert format_value == "onnx"
        assert isinstance(format_value, str)

    def test_target_format_default(self):
        """Test target.format default value when missing."""
        conversion_config = {
            "target": {}
        }
        
        format_value = conversion_config.get("target", {}).get("format", "onnx")
        
        assert format_value == "onnx"

    def test_onnx_opset_version_extraction(self):
        """Test onnx.opset_version extraction from config."""
        conversion_config = {
            "onnx": {
                "opset_version": 18
            }
        }
        
        opset_version = conversion_config.get("onnx", {}).get("opset_version", 18)
        
        assert opset_version == 18
        assert isinstance(opset_version, int)

    def test_onnx_opset_version_custom(self):
        """Test onnx.opset_version with custom value."""
        conversion_config = {
            "onnx": {
                "opset_version": 19
            }
        }
        
        opset_version = conversion_config.get("onnx", {}).get("opset_version", 18)
        
        assert opset_version == 19

    def test_onnx_opset_version_default(self):
        """Test onnx.opset_version default value when missing."""
        conversion_config = {
            "onnx": {}
        }
        
        opset_version = conversion_config.get("onnx", {}).get("opset_version", 18)
        
        assert opset_version == 18

    def test_onnx_quantization_extraction(self):
        """Test onnx.quantization extraction from config."""
        conversion_config = {
            "onnx": {
                "quantization": "none"
            }
        }
        
        quantization = conversion_config.get("onnx", {}).get("quantization", "none")
        
        assert quantization == "none"
        assert isinstance(quantization, str)

    def test_onnx_quantization_int8(self):
        """Test onnx.quantization with int8 value."""
        conversion_config = {
            "onnx": {
                "quantization": "int8"
            }
        }
        
        quantization = conversion_config.get("onnx", {}).get("quantization", "none")
        
        assert quantization == "int8"

    def test_onnx_quantization_dynamic(self):
        """Test onnx.quantization with dynamic value."""
        conversion_config = {
            "onnx": {
                "quantization": "dynamic"
            }
        }
        
        quantization = conversion_config.get("onnx", {}).get("quantization", "none")
        
        assert quantization == "dynamic"

    def test_onnx_quantization_default(self):
        """Test onnx.quantization default value when missing."""
        conversion_config = {
            "onnx": {}
        }
        
        quantization = conversion_config.get("onnx", {}).get("quantization", "none")
        
        assert quantization == "none"

    def test_onnx_run_smoke_test_extraction(self):
        """Test onnx.run_smoke_test extraction from config."""
        conversion_config = {
            "onnx": {
                "run_smoke_test": True
            }
        }
        
        run_smoke_test = conversion_config.get("onnx", {}).get("run_smoke_test", True)
        
        assert run_smoke_test is True
        assert isinstance(run_smoke_test, bool)

    def test_onnx_run_smoke_test_false(self):
        """Test onnx.run_smoke_test with false value."""
        conversion_config = {
            "onnx": {
                "run_smoke_test": False
            }
        }
        
        run_smoke_test = conversion_config.get("onnx", {}).get("run_smoke_test", True)
        
        assert run_smoke_test is False

    def test_onnx_run_smoke_test_default(self):
        """Test onnx.run_smoke_test default value when missing."""
        conversion_config = {
            "onnx": {}
        }
        
        run_smoke_test = conversion_config.get("onnx", {}).get("run_smoke_test", True)
        
        assert run_smoke_test is True

    def test_output_filename_pattern_extraction(self):
        """Test output.filename_pattern extraction from config."""
        conversion_config = {
            "output": {
                "filename_pattern": "model_{quantization}.onnx"
            }
        }
        
        filename_pattern = conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx")
        
        assert filename_pattern == "model_{quantization}.onnx"
        assert isinstance(filename_pattern, str)

    def test_output_filename_pattern_custom(self):
        """Test output.filename_pattern with custom value."""
        conversion_config = {
            "output": {
                "filename_pattern": "custom_{quantization}_model.onnx"
            }
        }
        
        filename_pattern = conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx")
        
        assert filename_pattern == "custom_{quantization}_model.onnx"

    def test_output_filename_pattern_default(self):
        """Test output.filename_pattern default value when missing."""
        conversion_config = {
            "output": {}
        }
        
        filename_pattern = conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx")
        
        assert filename_pattern == "model_{quantization}.onnx"

    def test_all_options_together(self):
        """Test extracting all options from a complete config."""
        conversion_config = {
            "target": {
                "format": "onnx"
            },
            "onnx": {
                "opset_version": 18,
                "quantization": "none",
                "run_smoke_test": True
            },
            "output": {
                "filename_pattern": "model_{quantization}.onnx"
            }
        }
        
        # Extract all options
        format_value = conversion_config.get("target", {}).get("format", "onnx")
        opset_version = conversion_config.get("onnx", {}).get("opset_version", 18)
        quantization = conversion_config.get("onnx", {}).get("quantization", "none")
        run_smoke_test = conversion_config.get("onnx", {}).get("run_smoke_test", True)
        filename_pattern = conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx")
        
        # Verify all values
        assert format_value == "onnx"
        assert opset_version == 18
        assert quantization == "none"
        assert run_smoke_test is True
        assert filename_pattern == "model_{quantization}.onnx"

