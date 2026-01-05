"""Integration tests for ONNX inference speed."""

import pytest
import time
import numpy as np
from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer


@pytest.mark.integration
def test_onnx_inference_speed(onnx_model_path, checkpoint_dir, sample_text):
    """
    Test ONNX inference speed and verify it completes in reasonable time.
    
    This test isolates ONNX inference performance from other components.
    """
    onnx_path = Path(onnx_model_path).resolve()
    checkpoint = Path(checkpoint_dir).resolve()
    
    if not onnx_path.exists():
        pytest.skip(f"ONNX model not found: {onnx_path}")
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint directory not found: {checkpoint}")
    
    print(f"Loading ONNX model from: {onnx_path}")
    start = time.time()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    load_time = time.time() - start
    print(f"[OK] ONNX model loaded in {load_time:.3f}s")
    
    # Check inputs
    input_names = [i.name for i in session.get_inputs()]
    print(f"Model requires inputs: {input_names}")
    
    print(f"\nLoading tokenizer from: {checkpoint}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        use_fast=True,
    )
    load_time = time.time() - start
    print(f"[OK] Tokenizer loaded in {load_time:.3f}s")
    
    # Tokenize
    print(f"\nTokenizing text (length: {len(sample_text)})...")
    start = time.time()
    tokenizer_output = tokenizer(
        sample_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    token_time = time.time() - start
    print(f"[OK] Tokenization completed in {token_time:.3f}s")
    print(f"  Keys: {list(tokenizer_output.keys())}")
    
    assert token_time < 1.0, f"Tokenization took {token_time:.2f} seconds, expected < 1.0"
    
    # Prepare feeds
    print(f"\nPreparing ONNX feeds...")
    feeds = {}
    for k, v in tokenizer_output.items():
        if k in input_names:
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                feeds[k] = v.astype(np.int64)
            else:
                feeds[k] = v
    
    # Ensure attention_mask if required
    if "attention_mask" in input_names and "attention_mask" not in feeds:
        if "input_ids" in tokenizer_output:
            input_ids = tokenizer_output["input_ids"]
            attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
            feeds["attention_mask"] = attention_mask
            print(f"  Created attention_mask from input_ids")
    
    print(f"  Feeds: {list(feeds.keys())}")
    print(f"  Feed shapes: {[(k, v.shape) for k, v in feeds.items()]}")
    
    # Run inference
    print(f"\nRunning ONNX inference...")
    start = time.time()
    outputs = session.run(None, feeds)
    inference_time = time.time() - start
    logits = outputs[0]
    print(f"[OK] Inference completed in {inference_time:.3f}s")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    assert inference_time < 1.0, f"Inference took {inference_time:.2f} seconds, expected < 1.0"
    assert logits.shape[0] == 1, "Batch size should be 1"
    assert logits.shape[1] > 0, "Sequence length should be > 0"
    assert logits.shape[2] > 0, "Number of labels should be > 0"

