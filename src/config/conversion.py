"""Load and resolve conversion configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from shared.yaml_utils import load_yaml
from shared.json_cache import load_json
from config.loader import ExperimentConfig
from fingerprints import compute_conv_fp


def load_conversion_config(
    root_dir: Path,
    config_dir: Path,
    parent_training_output_dir: Path,
    parent_spec_fp: str,
    parent_exec_fp: str,
    experiment_config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Load and resolve conversion configuration from YAML.
    
    This function:
    1. Loads config/conversion.yaml
    2. Extracts parent training information (backbone, checkpoint path) from metadata or path
    3. Computes conv_fp (conversion fingerprint) based on conversion settings
    4. Extracts parent_training_id from path structure
    5. Returns resolved config dict
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        parent_training_output_dir: Output directory of parent training run.
        parent_spec_fp: Specification fingerprint of parent training.
        parent_exec_fp: Execution fingerprint of parent training.
        experiment_config: Experiment configuration.
    
    Returns:
        Resolved conversion config dict with:
        - conv_fp: Conversion fingerprint
        - parent_training_id: Parent training identifier (e.g., "spec_{spec_fp}_exec_{exec_fp}/v1")
        - backbone: Canonical backbone name from metadata
        - checkpoint_path: Path to checkpoint directory
        - onnx: ONNX conversion settings (quantization, opset_version, run_smoke_test)
        - output: Output settings (filename_pattern)
    """
    # Load conversion.yaml
    conversion_yaml_path = config_dir / "conversion.yaml"
    conversion_config = load_yaml(conversion_yaml_path)
    
    # Extract parent training information
    checkpoint_path = parent_training_output_dir / "checkpoint"
    backbone = _extract_backbone_from_metadata(parent_training_output_dir)
    
    # Extract parent_training_id from path
    parent_training_id = _extract_parent_training_id(parent_training_output_dir)
    
    # Compute conv_fp based on conversion settings
    conv_fp = compute_conv_fp(
        parent_spec_fp=parent_spec_fp,
        parent_exec_fp=parent_exec_fp,
        conversion_config=conversion_config,
    )
    
    # Build resolved config
    resolved_config = {
        "conv_fp": conv_fp,
        "parent_training_id": parent_training_id,
        "backbone": backbone,
        "checkpoint_path": str(checkpoint_path),
        "format": conversion_config.get("target", {}).get("format", "onnx"),  # Top-level for easy access
        "onnx": {
            "opset_version": conversion_config.get("onnx", {}).get("opset_version", 18),
            "quantization": conversion_config.get("onnx", {}).get("quantization", "none"),
            "run_smoke_test": conversion_config.get("onnx", {}).get("run_smoke_test", True),
        },
        "output": {
            "filename_pattern": conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx"),
        },
    }
    
    return resolved_config


def _extract_backbone_from_metadata(parent_training_output_dir: Path) -> str:
    """
    Extract canonical backbone name from metadata.json.
    
    Falls back to path extraction if metadata is missing.
    
    Args:
        parent_training_output_dir: Parent training output directory.
    
    Returns:
        Canonical backbone name (e.g., "distilbert-base-uncased").
    """
    metadata_file = parent_training_output_dir / "metadata.json"
    if metadata_file.exists():
        try:
            metadata = load_json(metadata_file, default={})
            backbone = metadata.get("backbone")
            if backbone:
                return backbone
        except Exception:
            pass  # Fall through to path extraction
    
    # Fallback: extract from path
    # Path structure: .../final_training/{env}/{model}/...
    parts = parent_training_output_dir.parts
    try:
        ft_idx = parts.index("final_training")
        if len(parts) > ft_idx + 1:
            model_part = parts[ft_idx + 1]
            # Return as-is (don't strip, keep canonical form)
            return model_part
    except (ValueError, IndexError):
        pass
    
    # Last resort: use directory name
    return parent_training_output_dir.name


def _extract_parent_training_id(output_dir: Path) -> str:
    """
    Extract parent training ID from output directory path.
    
    Path structure: .../final_training/{env}/{model}/spec_{spec_fp}_exec_{exec_fp}/v{variant}
    
    Returns: "spec_{spec_fp}_exec_{exec_fp}/v{variant}"
    
    Args:
        output_dir: Parent training output directory.
    
    Returns:
        Parent training ID string.
    """
    parts = output_dir.parts
    try:
        ft_idx = parts.index("final_training")
        if len(parts) > ft_idx + 3:
            spec_exec_part = parts[ft_idx + 3]  # spec_{spec_fp}_exec_{exec_fp}
            if len(parts) > ft_idx + 4:
                variant_part = parts[ft_idx + 4]  # v1, v2, etc.
                return f"{spec_exec_part}/{variant_part}"
            return spec_exec_part
    except (ValueError, IndexError):
        pass
    
    # Fallback: use directory name
    return output_dir.name

