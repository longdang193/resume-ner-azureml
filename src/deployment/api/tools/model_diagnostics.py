"""
@meta
name: model_diagnostics
type: script
domain: api
responsibility:
  - Check model predictions and label mappings
  - Validate ONNX model behavior
  - Diagnose model issues
inputs:
  - ONNX model file
  - Checkpoint directory
  - Input text
outputs:
  - Diagnostic information
tags:
  - utility
  - diagnostics
  - onnx
ci:
  runnable: true
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Model diagnostic tools for checking predictions and label mappings."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoTokenizer, AutoConfig
import onnxruntime as ort


def check_predictions(
    onnx_path: Path,
    checkpoint_dir: Path,
    text: str,
    max_length: int = 128,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Check model predictions and label mappings.
    
    Args:
        onnx_path: Path to ONNX model file.
        checkpoint_dir: Path to checkpoint directory containing tokenizer and config.
        text: Text to analyze.
        max_length: Maximum sequence length for tokenization.
        verbose: If True, print diagnostic information.
    
    Returns:
        Dictionary with diagnostic information, or None if error occurred.
    """
    if verbose:
        print("=" * 60)
        print("Model Prediction Diagnostic")
        print("=" * 60)
    
    # Load config to see label mappings
    if verbose:
        print(f"\n1. Loading config from: {checkpoint_dir}")
    config = AutoConfig.from_pretrained(checkpoint_dir)
    
    if hasattr(config, "id2label") and config.id2label:
        id2label = {int(k): v for k, v in config.id2label.items()}
        if verbose:
            print(f"   [OK] Found {len(id2label)} labels in config")
            print(f"   Label mappings:")
            for label_id, label_name in sorted(id2label.items()):
                print(f"     {label_id}: {label_name}")
    else:
        if verbose:
            print("   [ERROR] No id2label found in config!")
        return None
    
    # Load tokenizer
    if verbose:
        print(f"\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if verbose:
        print(f"   [OK] Tokenizer loaded")
    
    # Tokenize
    if verbose:
        print(f"\n3. Tokenizing text: {text[:50]}...")
    tokenizer_output = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    if verbose:
        print(f"   [OK] Tokenized: {tokenizer_output['input_ids'].shape}")
    
    # Load ONNX model
    if verbose:
        print(f"\n4. Loading ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    if verbose:
        print(f"   [OK] Model loaded")
        print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"   Outputs: {[o.name for o in session.get_outputs()]}")
    
    # Prepare feeds
    feeds = {}
    input_names = [i.name for i in session.get_inputs()]
    for k, v in tokenizer_output.items():
        if k in input_names:
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                feeds[k] = v.astype(np.int64)
            else:
                feeds[k] = v
    
    # Ensure attention_mask
    if "attention_mask" not in feeds and "attention_mask" in input_names:
        if "input_ids" in feeds:
            input_ids = feeds["input_ids"]
            attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
            feeds["attention_mask"] = attention_mask
    
    # Run inference
    if verbose:
        print(f"\n5. Running inference...")
    outputs = session.run(None, feeds)
    logits = outputs[0]  # Shape: (batch_size, seq_len, num_labels)
    if verbose:
        print(f"   [OK] Inference complete")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Get predictions
    if verbose:
        print(f"\n6. Analyzing predictions...")
    predictions = np.argmax(logits, axis=-1)[0]  # Shape: (seq_len,)
    if verbose:
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Unique predictions: {np.unique(predictions)}")
    
    # Convert to labels
    labels = [id2label.get(int(p), "O") for p in predictions]
    unique_labels = set(labels)
    if verbose:
        print(f"   Unique labels: {sorted(unique_labels)}")
    
    # Count labels
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    if verbose:
        print(f"\n   Label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"     {label}: {count}")
    
    # Show sample predictions
    if verbose:
        print(f"\n7. Sample predictions (first 20 non-padding tokens):")
    attention_mask = tokenizer_output["attention_mask"][0]
    tokens = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"][0])
    
    sample_predictions = []
    non_padding_count = 0
    for i, (token, label, mask) in enumerate(zip(tokens, labels, attention_mask)):
        if mask == 1 and non_padding_count < 20:
            if verbose:
                print(f"   {i:3d}. {token:15s} -> {label}")
            sample_predictions.append({"index": i, "token": token, "label": label})
            non_padding_count += 1
    
    # Check if any entities predicted
    non_o_count = sum(1 for l in labels if l != "O")
    if verbose:
        print(f"\n8. Summary:")
        print(f"   Total tokens: {len(tokens)}")
        print(f"   Non-padding tokens: {int(attention_mask.sum())}")
        print(f"   Non-O labels: {non_o_count}")
        if non_o_count == 0:
            print(f"   [WARNING] Model is predicting all 'O' labels!")
            print(f"   This suggests:")
            print(f"     - Model may not be trained properly")
            print(f"     - Model checkpoint may be incorrect")
            print(f"     - Input text may not match training data format")
        else:
            print(f"   [OK] Model is predicting {non_o_count} entity tokens")
    
    # Return diagnostic information
    return {
        "id2label": id2label,
        "predictions": predictions.tolist(),
        "labels": labels,
        "label_counts": label_counts,
        "unique_labels": sorted(unique_labels),
        "non_o_count": non_o_count,
        "sample_predictions": sample_predictions,
        "logits_shape": logits.shape,
        "logits_range": [float(logits.min()), float(logits.max())],
    }


def main():
    """CLI entry point for model diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check model predictions")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--text",
        type=str,
        default="John Doe is a software engineer at Google. Email: john.doe@example.com Phone: +1-555-123-4567 Location: Seattle, WA",
        help="Text to check"
    )
    
    args = parser.parse_args()
    
    check_predictions(Path(args.onnx), Path(args.checkpoint), args.text)


if __name__ == "__main__":
    main()

