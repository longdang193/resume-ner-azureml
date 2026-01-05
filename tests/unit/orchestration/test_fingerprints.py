"""Unit tests for fingerprint computation."""

import pytest
from orchestration.fingerprints import (
    compute_spec_fp,
    compute_exec_fp,
    compute_conv_fp,
    compute_bench_fp,
    compute_hardware_fp,
)


def test_compute_spec_fp_deterministic():
    """Test that spec_fp is deterministic."""
    model_hash = "abc123def4567890"
    data_hash = "xyz789abc1234567"
    train_hash = "def456xyz7890123"
    seed = 42
    
    fp1 = compute_spec_fp(model_hash, data_hash, train_hash, seed)
    fp2 = compute_spec_fp(model_hash, data_hash, train_hash, seed)
    
    assert fp1 == fp2
    assert len(fp1) == 16  # CONFIG_HASH_LENGTH


def test_compute_spec_fp_without_seed():
    """Test spec_fp computation without seed."""
    model_hash = "abc123def4567890"
    data_hash = "xyz789abc1234567"
    train_hash = "def456xyz7890123"
    
    fp = compute_spec_fp(model_hash, data_hash, train_hash)
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_spec_fp_different_seeds():
    """Test that different seeds produce different spec_fp."""
    model_hash = "abc123def4567890"
    data_hash = "xyz789abc1234567"
    train_hash = "def456xyz7890123"
    
    fp1 = compute_spec_fp(model_hash, data_hash, train_hash, seed=42)
    fp2 = compute_spec_fp(model_hash, data_hash, train_hash, seed=43)
    
    assert fp1 != fp2


def test_compute_exec_fp_basic():
    """Test exec_fp computation with provided values."""
    git_sha = "abc12345"
    env_hash = "xyz78901"
    torch_ver = "2.0"
    transformers_ver = "4.30"
    precision = "fp16"
    
    fp = compute_exec_fp(
        git_sha=git_sha,
        env_config_hash=env_hash,
        torch_version=torch_ver,
        transformers_version=transformers_ver,
        precision=precision
    )
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_exec_fp_auto_detect():
    """Test exec_fp computation with auto-detection."""
    fp = compute_exec_fp()
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_exec_fp_different_precision():
    """Test that different precision produces different exec_fp."""
    git_sha = "abc12345"
    env_hash = "xyz78901"
    torch_ver = "2.0"
    transformers_ver = "4.30"
    
    fp1 = compute_exec_fp(git_sha, env_hash, torch_ver, transformers_ver, "fp32")
    fp2 = compute_exec_fp(git_sha, env_hash, torch_ver, transformers_ver, "fp16")
    
    assert fp1 != fp2


def test_compute_conv_fp():
    """Test conv_fp computation."""
    parent_id = "spec_abc_exec_xyz/v1"
    conv_config_hash = "conv1234567890"
    optimum_ver = "1.10"
    ort_ver = "1.15"
    
    fp = compute_conv_fp(
        parent_training_id=parent_id,
        conversion_config_hash=conv_config_hash,
        optimum_version=optimum_ver,
        onnxruntime_version=ort_ver
    )
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_conv_fp_different_parents():
    """Test that different parent IDs produce different conv_fp."""
    conv_config_hash = "conv1234567890"
    optimum_ver = "1.10"
    ort_ver = "1.15"
    
    fp1 = compute_conv_fp("spec_abc_exec_xyz/v1", conv_config_hash, optimum_ver, ort_ver)
    fp2 = compute_conv_fp("spec_def_exec_uvw/v1", conv_config_hash, optimum_ver, ort_ver)
    
    assert fp1 != fp2


def test_compute_bench_fp():
    """Test bench_fp computation."""
    spec_fp = "abc123def4567890"
    bench_config_hash = "bench1234567890"
    hardware_fp = "gpu_t4_cpu_xeon"
    runtime_fp = "ort_1.15"
    
    fp = compute_bench_fp(
        spec_fp=spec_fp,
        benchmark_config_hash=bench_config_hash,
        hardware_fp=hardware_fp,
        runtime_fp=runtime_fp
    )
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_hardware_fp():
    """Test hardware_fp computation."""
    gpu_model = "Tesla T4"
    cpu_model = "Intel Xeon"
    ram_gb = 16
    
    fp = compute_hardware_fp(gpu_model=gpu_model, cpu_model=cpu_model, ram_gb=ram_gb)
    
    assert fp is not None
    assert len(fp) == 16


def test_compute_hardware_fp_empty():
    """Test hardware_fp with no information returns 'unknown'."""
    fp = compute_hardware_fp()
    
    assert fp == "unknown"


