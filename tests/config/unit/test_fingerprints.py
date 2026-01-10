"""Unit tests for fingerprint computation."""

import pytest
from infrastructure.fingerprints import (
    compute_spec_fp,
    compute_exec_fp,
    compute_conv_fp,
    compute_bench_fp,
    compute_hardware_fp,
)


def test_compute_spec_fp_deterministic():
    """Test that spec_fp is deterministic."""
    model_config = {"backbone": "distilbert", "hidden_size": 768}
    data_config = {"dataset": "resume_ner", "max_length": 512}
    train_config = {"learning_rate": 2e-5, "batch_size": 16}
    seed = 42
    
    fp1 = compute_spec_fp(model_config, data_config, train_config, seed)
    fp2 = compute_spec_fp(model_config, data_config, train_config, seed)
    
    assert fp1 == fp2
    assert len(fp1) == 16  # CONFIG_HASH_LENGTH


def test_compute_spec_fp_without_seed():
    """Test spec_fp computation with seed=0."""
    model_config = {"backbone": "distilbert"}
    data_config = {"dataset": "resume_ner"}
    train_config = {"learning_rate": 2e-5}
    
    fp = compute_spec_fp(model_config, data_config, train_config, seed=0)
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_spec_fp_different_seeds():
    """Test that different seeds produce different spec_fp."""
    model_config = {"backbone": "distilbert"}
    data_config = {"dataset": "resume_ner"}
    train_config = {"learning_rate": 2e-5}
    
    fp1 = compute_spec_fp(model_config, data_config, train_config, seed=42)
    fp2 = compute_spec_fp(model_config, data_config, train_config, seed=43)
    
    assert fp1 != fp2


def test_compute_exec_fp_basic():
    """Test exec_fp computation with provided values."""
    git_sha = "abc12345"
    env_config = {"platform": "azureml", "compute": "cpu-cluster"}
    
    fp = compute_exec_fp(git_sha=git_sha, env_config=env_config)
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_exec_fp_auto_detect():
    """Test exec_fp computation with None values."""
    fp = compute_exec_fp(git_sha=None, env_config={})
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_exec_fp_different_precision():
    """Test that different precision settings produce different exec_fp."""
    git_sha = "abc12345"
    env_config = {"platform": "azureml"}
    
    fp1 = compute_exec_fp(git_sha, env_config, include_precision=True)
    fp2 = compute_exec_fp(git_sha, env_config, include_precision=False)
    
    assert fp1 != fp2


def test_compute_conv_fp():
    """Test conv_fp computation."""
    parent_spec_fp = "abc123def4567890"
    parent_exec_fp = "xyz789abc1234567"
    conversion_config = {"quantization": "int8", "opset": 14}
    
    fp = compute_conv_fp(
        parent_spec_fp=parent_spec_fp,
        parent_exec_fp=parent_exec_fp,
        conversion_config=conversion_config
    )
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_conv_fp_different_parents():
    """Test that different parent fingerprints produce different conv_fp."""
    conversion_config = {"quantization": "int8", "opset": 14}
    
    fp1 = compute_conv_fp("spec_abc_exec_xyz", "exec_abc", conversion_config)
    fp2 = compute_conv_fp("spec_def_exec_uvw", "exec_def", conversion_config)
    
    assert fp1 != fp2


def test_compute_bench_fp():
    """Test bench_fp computation."""
    model_config = {"backbone": "distilbert"}
    benchmark_config = {"batch_sizes": [1, 4, 8], "iterations": 100}
    
    fp = compute_bench_fp(
        model_config=model_config,
        benchmark_config=benchmark_config
    )
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_hardware_fp():
    """Test hardware_fp computation."""
    hardware_info = {
        "gpu_model": "Tesla T4",
        "cpu_model": "Intel Xeon",
        "ram_gb": 16
    }
    
    fp = compute_hardware_fp(hardware_info=hardware_info)
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_hardware_fp_empty():
    """Test hardware_fp with empty dict."""
    fp = compute_hardware_fp(hardware_info={})
    
    assert fp is not None
    assert len(fp) == 16


