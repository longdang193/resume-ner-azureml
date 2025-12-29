"""Unit tests for centralized naming system."""

import pytest
from pathlib import Path
from orchestration.naming_centralized import (
    NamingContext,
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)


def test_naming_context_validation():
    """Test that NamingContext validates inputs."""
    # Valid context
    context = NamingContext(
        process_type="hpo",
        model="distilbert",
        environment="local",
        trial_id="trial_1_20251229_100000"
    )
    assert context.process_type == "hpo"
    
    # Invalid process_type
    with pytest.raises(ValueError, match="Invalid process_type"):
        NamingContext(
            process_type="invalid",
            model="distilbert",
            environment="local"
        )
    
    # Invalid environment
    with pytest.raises(ValueError, match="Invalid environment"):
        NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="invalid"
        )
    
    # Invalid variant
    with pytest.raises(ValueError, match="Variant must be >= 1"):
        NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="abc123",
            exec_fp="xyz789",
            variant=0
        )


def test_naming_context_final_training_requires_fingerprints():
    """Test that final_training requires spec_fp and exec_fp."""
    with pytest.raises(ValueError, match="final_training requires spec_fp and exec_fp"):
        NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local"
        )


def test_naming_context_conversion_requires_parent_and_conv_fp():
    """Test that conversion requires parent_training_id and conv_fp."""
    with pytest.raises(ValueError, match="conversion requires parent_training_id and conv_fp"):
        NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local"
        )


def test_create_naming_context_auto_detect():
    """Test that create_naming_context auto-detects environment."""
    context = create_naming_context(
        process_type="hpo",
        model="distilbert",
        trial_id="trial_1_20251229_100000"
    )
    
    assert context.environment in ["local", "colab", "kaggle", "azure"]


def test_build_output_path_hpo():
    """Test path building for HPO."""
    context = NamingContext(
        process_type="hpo",
        model="distilbert",
        environment="local",
        trial_id="trial_1_20251229_100000"
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert str(path) == "/root/outputs/hpo/local/distilbert/trial_1_20251229_100000"


def test_build_output_path_benchmarking():
    """Test path building for benchmarking."""
    context = NamingContext(
        process_type="benchmarking",
        model="distilbert",
        environment="colab",
        trial_id="trial_1_20251229_100000"
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert str(path) == "/root/outputs/benchmarking/colab/distilbert/trial_1_20251229_100000"


def test_build_output_path_final_training():
    """Test path building for final training."""
    context = NamingContext(
        process_type="final_training",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890",
        exec_fp="xyz789abc1234567",
        variant=1
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/final_training/local/distilbert/spec_abc123def4567890_exec_xyz789abc1234567/v1"
    assert str(path) == expected


def test_build_output_path_final_training_variant():
    """Test path building for final training with variant."""
    context = NamingContext(
        process_type="final_training",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890",
        exec_fp="xyz789abc1234567",
        variant=2
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert "v2" in str(path)


def test_build_output_path_conversion():
    """Test path building for conversion."""
    context = NamingContext(
        process_type="conversion",
        model="distilbert",
        environment="local",
        parent_training_id="spec_abc_exec_xyz/v1",
        conv_fp="conv1234567890123"
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/conversion/local/distilbert/spec_abc_exec_xyz/v1/conv_conv1234567890123"
    assert str(path) == expected


def test_build_output_path_best_configurations():
    """Test path building for best configurations."""
    context = NamingContext(
        process_type="best_configurations",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890"
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/cache/best_configurations/distilbert/spec_abc123def4567890"
    assert str(path) == expected


def test_build_parent_training_id():
    """Test building parent training ID."""
    spec_fp = "abc123def4567890"
    exec_fp = "xyz789abc1234567"
    variant = 1
    
    parent_id = build_parent_training_id(spec_fp, exec_fp, variant)
    
    assert parent_id == "spec_abc123def4567890_exec_xyz789abc1234567/v1"

