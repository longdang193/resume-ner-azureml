"""Unit tests for benchmark.yaml config option extraction and defaults."""

import pytest
from pathlib import Path


class TestBenchmarkConfigOptions:
    """Test extraction of individual config options from benchmark.yaml."""

    def test_batch_sizes_extraction(self):
        """Test batch_sizes extraction from config."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [1, 8, 16]
            }
        }
        
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        
        assert batch_sizes == [1, 8, 16]
        assert isinstance(batch_sizes, list)
        assert all(isinstance(bs, int) for bs in batch_sizes)

    def test_batch_sizes_default(self):
        """Test batch_sizes default value when missing."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        
        assert batch_sizes == [1, 8, 16]

    def test_batch_sizes_custom_values(self):
        """Test batch_sizes with custom values."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [2, 4, 32, 64]
            }
        }
        
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        
        assert batch_sizes == [2, 4, 32, 64]

    def test_iterations_extraction(self):
        """Test iterations extraction from config."""
        benchmark_config = {
            "benchmarking": {
                "iterations": 100
            }
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        assert iterations == 100
        assert isinstance(iterations, int)

    def test_iterations_default(self):
        """Test iterations default value when missing."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        assert iterations == 100

    def test_iterations_custom_value(self):
        """Test iterations with custom value."""
        benchmark_config = {
            "benchmarking": {
                "iterations": 200
            }
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        assert iterations == 200

    def test_warmup_iterations_extraction(self):
        """Test warmup_iterations extraction from config."""
        benchmark_config = {
            "benchmarking": {
                "warmup_iterations": 10
            }
        }
        
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        
        assert warmup == 10
        assert isinstance(warmup, int)

    def test_warmup_iterations_default(self):
        """Test warmup_iterations default value when missing."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        
        assert warmup == 10

    def test_warmup_iterations_custom_value(self):
        """Test warmup_iterations with custom value."""
        benchmark_config = {
            "benchmarking": {
                "warmup_iterations": 20
            }
        }
        
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        
        assert warmup == 20

    def test_max_length_extraction(self):
        """Test max_length extraction from config."""
        benchmark_config = {
            "benchmarking": {
                "max_length": 512
            }
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert max_length == 512
        assert isinstance(max_length, int)

    def test_max_length_default(self):
        """Test max_length default value when missing."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert max_length == 512

    def test_max_length_custom_value(self):
        """Test max_length with custom value."""
        benchmark_config = {
            "benchmarking": {
                "max_length": 256
            }
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert max_length == 256

    def test_device_extraction_null(self):
        """Test device extraction when null (auto-detect)."""
        benchmark_config = {
            "benchmarking": {
                "device": None
            }
        }
        
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        assert device is None

    def test_device_extraction_cuda(self):
        """Test device extraction with cuda."""
        benchmark_config = {
            "benchmarking": {
                "device": "cuda"
            }
        }
        
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        assert device == "cuda"

    def test_device_extraction_cpu(self):
        """Test device extraction with cpu."""
        benchmark_config = {
            "benchmarking": {
                "device": "cpu"
            }
        }
        
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        assert device == "cpu"

    def test_device_default(self):
        """Test device default value when missing (should be None for auto-detect)."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        assert device is None

    def test_test_data_extraction_null(self):
        """Test test_data extraction when null."""
        benchmark_config = {
            "benchmarking": {
                "test_data": None
            }
        }
        
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        
        assert test_data is None

    def test_test_data_extraction_relative_path(self):
        """Test test_data extraction with relative path."""
        benchmark_config = {
            "benchmarking": {
                "test_data": "dataset/test.json"
            }
        }
        
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        
        assert test_data == "dataset/test.json"

    def test_test_data_extraction_absolute_path(self):
        """Test test_data extraction with absolute path."""
        benchmark_config = {
            "benchmarking": {
                "test_data": "/absolute/path/test.json"
            }
        }
        
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        
        assert test_data == "/absolute/path/test.json"

    def test_test_data_default(self):
        """Test test_data default value when missing (should be None)."""
        benchmark_config = {
            "benchmarking": {}
        }
        
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        
        assert test_data is None

    def test_output_filename_extraction(self):
        """Test output.filename extraction from config."""
        benchmark_config = {
            "output": {
                "filename": "benchmark.json"
            }
        }
        
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        
        assert filename == "benchmark.json"
        assert isinstance(filename, str)

    def test_output_filename_default(self):
        """Test output.filename default value when missing."""
        benchmark_config = {
            "output": {}
        }
        
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        
        assert filename == "benchmark.json"

    def test_output_filename_custom_value(self):
        """Test output.filename with custom value."""
        benchmark_config = {
            "output": {
                "filename": "custom_benchmark.json"
            }
        }
        
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        
        assert filename == "custom_benchmark.json"


    def test_all_options_together(self):
        """Test extracting all options from a complete config."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [1, 8, 16],
                "iterations": 100,
                "warmup_iterations": 10,
                "max_length": 512,
                "device": None,
                "test_data": None
            },
            "output": {
                "filename": "benchmark.json",
                "save_summary": True
            }
        }
        
        # Extract all options
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        save_summary = benchmark_config.get("output", {}).get("save_summary", True)
        
        # Verify all values
        assert batch_sizes == [1, 8, 16]
        assert iterations == 100
        assert warmup == 10
        assert max_length == 512
        assert device is None
        assert test_data is None
        assert filename == "benchmark.json"

