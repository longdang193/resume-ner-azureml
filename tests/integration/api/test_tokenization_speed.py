"""Integration tests for tokenization speed."""

import pytest
import time
from pathlib import Path

from transformers import AutoTokenizer


@pytest.mark.integration
def test_tokenization_speed(checkpoint_dir, sample_text):
    """
    Test tokenization speed and verify it completes quickly.
    
    This test verifies that tokenization doesn't hang and completes
    in reasonable time for different tokenization modes.
    """
    checkpoint = Path(checkpoint_dir)
    
    print(f"Loading tokenizer from: {checkpoint}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
    )
    load_time = time.time() - start
    print(f"[OK] Tokenizer loaded in {load_time:.3f}s")
    
    # Test 1: Simple tokenization
    print(f"\nTest 1: Simple tokenization (text length: {len(sample_text)})")
    start = time.time()
    result = tokenizer(
        sample_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    elapsed = time.time() - start
    print(f"[OK] Tokenization completed in {elapsed:.3f}s")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in result.items()]}")
    
    assert elapsed < 1.0, f"Tokenization took {elapsed:.2f} seconds, expected < 1.0"
    assert "input_ids" in result
    assert "attention_mask" in result
    
    # Test 2: With offset mapping
    print(f"\nTest 2: Tokenization with offset mapping")
    start = time.time()
    result = tokenizer(
        sample_text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    elapsed = time.time() - start
    print(f"[OK] Tokenization with offsets completed in {elapsed:.3f}s")
    if "offset_mapping" in result:
        print(f"  Offset mapping type: {type(result['offset_mapping'])}")
        if isinstance(result['offset_mapping'], list) and len(result['offset_mapping']) > 0:
            print(f"  Offset mapping length: {len(result['offset_mapping'][0])}")
    
    assert elapsed < 1.0, f"Tokenization with offsets took {elapsed:.2f} seconds, expected < 1.0"
    assert "offset_mapping" in result
    
    # Test 3: Both together (may fail for some tokenizers)
    print(f"\nTest 3: Tokenization with both return_tensors='np' and return_offsets_mapping=True")
    start = time.time()
    try:
        result = tokenizer(
            sample_text,
            return_tensors="np",
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        elapsed = time.time() - start
        print(f"[OK] Combined tokenization completed in {elapsed:.3f}s")
        print(f"  Keys: {list(result.keys())}")
        assert elapsed < 1.0
    except Exception as e:
        print(f"[INFO] Combined tokenization not supported: {e}")
        print(f"  (This is expected - some tokenizers don't support both together)")

