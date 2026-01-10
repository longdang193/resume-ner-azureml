"""Smoke testing for exported ONNX models.

This module provides validation functionality for exported ONNX models by
running a minimal inference test to ensure the model works correctly.
"""

from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer

from shared.logging_utils import get_script_logger
from shared.tokenization_utils import prepare_onnx_inputs

_log = get_script_logger("conversion.testing")


def run_smoke_test(onnx_path: Path, checkpoint_dir: Path) -> None:
    """
    Run a minimal ONNX Runtime inference to validate the exported graph.
    
    Args:
        onnx_path: Path to ONNX model file.
        checkpoint_dir: Path to checkpoint directory containing tokenizer.
    """
    _log.info(f"Running ONNX smoke test for '{onnx_path}'")
    try:
        import onnxruntime as ort
    except Exception as e:
        import warnings
        
        warnings.warn(
            f"Smoke test skipped because onnxruntime is not available: {e}"
        )
        _log.info(
            "Smoke test skipped because onnxruntime is not available. "
            f"Reason: {type(e).__name__}: {e}"
        )
        return
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    
    # Get input names from ONNX model
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    
    # Prepare inputs using shared utilities
    feeds = prepare_onnx_inputs(
        tokenizer,
        "Jane Doe is a software engineer.",
        16,  # max_length
        input_names,
    )
    
    # Only feed expected inputs (some models omit token_type_ids)
    feeds = {k: v for k, v in feeds.items() if k in input_names}
    
    outputs = sess.run(None, feeds)
    if not outputs or outputs[0] is None:
        raise RuntimeError("ONNX smoke test failed: no logits returned")
    _log.info("ONNX smoke test completed successfully")

