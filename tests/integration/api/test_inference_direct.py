"""Integration tests for direct inference engine testing."""

import pytest
import time
from pathlib import Path

from src.api.inference import ONNXInferenceEngine


@pytest.mark.integration
def test_direct_inference(onnx_model_path, checkpoint_dir, sample_text):
    """
    Test inference engine directly without FastAPI.
    
    This test verifies that the inference engine works correctly
    when called directly, including step-by-step timing analysis.
    """
    onnx_path = Path(onnx_model_path).resolve()
    checkpoint = Path(checkpoint_dir).resolve()
    
    if not onnx_path.exists():
        pytest.skip(f"ONNX model not found: {onnx_path}")
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint directory not found: {checkpoint}")
    
    print(f"Loading model from: {onnx_path}")
    print(f"Loading tokenizer from: {checkpoint}")
    
    start = time.time()
    engine = ONNXInferenceEngine(onnx_path, checkpoint)
    load_time = time.time() - start
    print(f"[OK] Model loaded in {load_time:.3f}s\n")
    
    print(f"Testing prediction for text: {sample_text[:50]}...")
    print(f"Text length: {len(sample_text)}\n")
    
    # Test full prediction
    start = time.time()
    entities = engine.predict(sample_text, return_confidence=True)
    elapsed = time.time() - start
    print(f"[OK] Full prediction completed in {elapsed:.3f}s")
    print(f"Found {len(entities)} entities:")
    for i, entity in enumerate(entities[:5], 1):
        print(f"  {i}. {entity['text']} -> {entity['label']} [{entity['start']}:{entity['end']}]")
    
    assert isinstance(entities, list)
    assert elapsed < 5.0, f"Prediction took {elapsed:.2f} seconds, expected < 5.0"
    
    # Test step by step
    print(f"\n" + "=" * 60)
    print("Step-by-step timing:")
    print("=" * 60)
    
    start = time.time()
    logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens(sample_text)
    token_time = time.time() - start
    print(f"predict_tokens: {token_time:.3f}s")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Offset mapping: {offset_mapping is not None}")
    
    assert logits is not None
    assert len(tokens) > 0
    
    start = time.time()
    entities = engine.decode_entities(
        sample_text,
        logits,
        tokens,
        tokenizer_output,
        offset_mapping,
        return_confidence=True,
    )
    decode_time = time.time() - start
    print(f"decode_entities: {decode_time:.3f}s")
    print(f"  Entities found: {len(entities)}")
    
    total_time = token_time + decode_time
    print(f"\nTotal time: {total_time:.3f}s")
    
    assert len(entities) > 0, "Should extract at least some entities"

