"""ONNX model export and quantization."""

from pathlib import Path
from typing import Dict, Iterable

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from shared.logging_utils import get_script_logger

_log = get_script_logger("convert_to_onnx")


def _dynamic_axes_for(inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[int, str]]:
    """Build dynamic axes dictionary for ONNX export."""
    axes: Dict[str, Dict[int, str]] = {}
    for name, tensor in inputs.items():
        if tensor.dim() >= 2:
            axes[name] = {0: "batch", 1: "seq"}
        elif tensor.dim() == 1:
            axes[name] = {0: "batch"}
    # Output logits are [batch, seq, num_labels]
    axes["logits"] = {0: "batch", 1: "seq"}
    return axes


def export_to_onnx(
    checkpoint_dir: Path,
    output_dir: Path,
    quantize_int8: bool,
    opset_version: int = 18,
) -> Path:
    """
    Export a token-classification model to ONNX (and optionally quantize).
    
    Args:
        checkpoint_dir: Path to checkpoint directory.
        output_dir: Output directory for ONNX model.
        quantize_int8: Whether to apply int8 quantization.
        opset_version: ONNX opset version (default: 18).
    
    Returns:
        Path to exported ONNX model.
    """
    _log.info(f"Starting ONNX export. quantize_int8={quantize_int8}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _log.info(f"Output directory created at '{output_dir}'")
    
    # Load model + tokenizer from the saved checkpoint directory
    _log.info(f"Loading tokenizer and model from checkpoint directory '{checkpoint_dir}'")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.eval()
    _log.info("Model and tokenizer successfully loaded; building example inputs for tracing")
    
    # Build a tiny example batch for tracing
    example = tokenizer(
        "John Doe works at Microsoft in Seattle.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    # Some models don't use token_type_ids; export only what the model accepts.
    allowed_keys: Iterable[str] = ("input_ids", "attention_mask", "token_type_ids")
    inputs: Dict[str, torch.Tensor] = {
        k: v for k, v in example.items() if k in allowed_keys
    }
    # Ensure stable ordering aligned with HF forward signature
    input_names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in inputs]
    output_names = ["logits"]
    dynamic_axes = _dynamic_axes_for(inputs)
    
    fp32_path = output_dir / "model.onnx"
    _log.info(f"Exporting FP32 ONNX model to '{fp32_path}' (opset={opset_version}, dynamo=False)")
    try:
        torch.onnx.export(
            model,
            args=tuple(inputs[name] for name in input_names),
            f=str(fp32_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,  # Use parameter instead of hardcoded
            do_constant_folding=True,
            dynamo=False,  # use classic exporter; avoids dynamic_axes+dynamo issues
        )
    except Exception as e:
        _log.info(
            f"FP32 ONNX export failed with {type(e).__name__}: {e}. "
            "Raising to fail the job so logs capture the stack trace."
        )
        raise
    _log.info("FP32 ONNX export completed")
    
    if not quantize_int8:
        _log.info("Int8 quantization not requested; returning FP32 model")
        return fp32_path
    
    int8_path = output_dir / "model_int8.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        
        _log.info(f"Starting dynamic int8 quantization. Output path: '{int8_path}'")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        _log.info("Int8 quantization completed successfully")
        return int8_path
    except Exception as e:
        # Don't fail the whole conversion if quantization tooling isn't available.
        # The caller requested int8, so we keep fp32 but emit a clear warning.
        import warnings
        
        warnings.warn(
            f"Int8 quantization requested but failed ({type(e).__name__}: {e}). "
            f"Falling back to fp32 ONNX at {fp32_path}."
        )
        _log.info(
            f"Int8 quantization failed with {type(e).__name__}: {e}. "
            f"Falling back to FP32 model at '{fp32_path}'"
        )
        return fp32_path

