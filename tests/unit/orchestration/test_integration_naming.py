"""Integration tests for centralized naming system end-to-end workflow."""

import pytest
from pathlib import Path
from orchestration.fingerprints import compute_spec_fp, compute_exec_fp, compute_conv_fp
from orchestration.naming_centralized import (
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)
from orchestration.metadata_manager import save_metadata_with_fingerprints
from orchestration.index_manager import update_index, find_by_spec_fp


def test_end_to_end_final_training(tmp_path):
    """Test complete workflow: fingerprints → context → path → metadata → index."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Step 1: Compute fingerprints
    model_hash = "abc123def4567890"
    data_hash = "xyz789abc1234567"
    train_hash = "def456xyz7890123"
    seed = 42
    
    spec_fp = compute_spec_fp(model_hash, data_hash, train_hash, seed)
    exec_fp = compute_exec_fp(git_sha="test123", torch_version="2.0", transformers_version="4.30")
    
    assert spec_fp is not None
    assert exec_fp is not None
    
    # Step 2: Create naming context
    context = create_naming_context(
        process_type="final_training",
        model="distilbert",
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=1,
        environment="local"
    )
    
    # Step 3: Build output path
    output_path = build_output_path(root_dir, context)
    output_path.mkdir(parents=True, exist_ok=True)
    
    assert output_path.exists()
    assert "final_training" in str(output_path)
    assert "local" in str(output_path)
    assert "distilbert" in str(output_path)
    assert spec_fp in str(output_path)
    assert exec_fp in str(output_path)
    assert "v1" in str(output_path)
    
    # Step 4: Save metadata
    metadata_path = output_path / "metadata.json"
    save_metadata_with_fingerprints(
        metadata_path=metadata_path,
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=1,
        environment="local",
        model="distilbert",
        status={"training": {"completed": True}}
    )
    
    assert metadata_path.exists()
    
    # Step 5: Update index
    metadata = {
        "_path": str(output_path),
        "created_at": "2025-01-01T12:00:00Z",
        "status": {"training": {"completed": True}}
    }
    index_path = update_index(root_dir, "final_training", context, metadata)
    
    assert index_path.exists()
    
    # Step 6: Query index
    results = find_by_spec_fp(root_dir, spec_fp, "final_training")
    assert len(results) > 0
    assert results[0]["spec_fp"] == spec_fp
    assert results[0]["environment"] == "local"


def test_end_to_end_conversion(tmp_path):
    """Test conversion workflow with parent training reference."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Step 1: Create parent training context
    spec_fp = "abc123def4567890"
    exec_fp = "xyz789abc1234567"
    parent_id = build_parent_training_id(spec_fp, exec_fp, variant=1)
    
    # Step 2: Compute conversion fingerprint
    conv_fp = compute_conv_fp(
        parent_training_id=parent_id,
        conversion_config_hash="conv1234567890",
        optimum_version="1.10",
        onnxruntime_version="1.15"
    )
    
    # Step 3: Create conversion context
    context = create_naming_context(
        process_type="conversion",
        model="distilbert",
        parent_training_id=parent_id,
        conv_fp=conv_fp,
        environment="local"
    )
    
    # Step 4: Build conversion path
    output_path = build_output_path(root_dir, context)
    output_path.mkdir(parents=True, exist_ok=True)
    
    assert output_path.exists()
    assert "conversion" in str(output_path)
    assert parent_id in str(output_path)
    assert conv_fp in str(output_path)


def test_cross_platform_same_spec_fp(tmp_path):
    """Test that same spec_fp produces same identity across environments."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Same fingerprints
    spec_fp = "abc123def4567890"
    exec_fp = "xyz789abc1234567"
    
    # Different environments
    local_context = create_naming_context(
        process_type="final_training",
        model="distilbert",
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=1,
        environment="local"
    )
    
    colab_context = create_naming_context(
        process_type="final_training",
        model="distilbert",
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=1,
        environment="colab"
    )
    
    # Same spec_fp and exec_fp
    assert local_context.spec_fp == colab_context.spec_fp
    assert local_context.exec_fp == colab_context.exec_fp
    
    # Different paths (different environments)
    local_path = build_output_path(root_dir, local_context)
    colab_path = build_output_path(root_dir, colab_context)
    
    assert local_path != colab_path
    assert "local" in str(local_path)
    assert "colab" in str(colab_path)


