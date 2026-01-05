"""Shared tokenization utilities for ONNX inference and testing."""

from typing import Dict, Optional, Set
import numpy as np
from transformers import PreTrainedTokenizer


def prepare_onnx_inputs(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
    input_names: Optional[Set[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare tokenized inputs for ONNX inference.
    
    Args:
        tokenizer: Tokenizer instance.
        text: Input text to tokenize.
        max_length: Maximum sequence length.
        input_names: Optional set of expected input names from ONNX model.
                     If provided, only includes inputs that the model expects.
    
    Returns:
        Dictionary of tokenized inputs ready for ONNX Runtime (int64 arrays).
    """
    # Tokenize text
    tokenizer_output = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    # Prepare feeds for ONNX
    feeds: Dict[str, np.ndarray] = {}
    
    # If input_names provided, filter to only what model expects
    if input_names:
        for k, v in tokenizer_output.items():
            if k in input_names:
                # Convert integer inputs to int64 for ONNX Runtime
                if k in ("input_ids", "attention_mask", "token_type_ids"):
                    feeds[k] = v.astype(np.int64)
                else:
                    feeds[k] = v
    else:
        # Include all tokenizer outputs, converting integers to int64
        for k, v in tokenizer_output.items():
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                feeds[k] = v.astype(np.int64)
            else:
                feeds[k] = v
    
    # Ensure attention_mask if required but not provided
    if input_names and "attention_mask" in input_names and "attention_mask" not in feeds:
        if "input_ids" in tokenizer_output:
            input_ids = tokenizer_output["input_ids"]
            # Create attention mask: 1 for non-padding tokens, 0 for padding
            attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
            feeds["attention_mask"] = attention_mask
    
    return feeds


def get_offset_mapping(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
) -> Optional[np.ndarray]:
    """
    Get token offset mapping from tokenizer.
    
    Args:
        tokenizer: Tokenizer instance.
        text: Input text.
        max_length: Maximum sequence length.
    
    Returns:
        Offset mapping array of shape (seq_len, 2) or None if unavailable.
        Each row contains [start, end] character offsets in the original text.
    """
    try:
        tokenizer_output = tokenizer(
            text,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        if "offset_mapping" in tokenizer_output:
            offset_mapping_list = tokenizer_output["offset_mapping"]
            if offset_mapping_list and len(offset_mapping_list) > 0:
                return np.array(offset_mapping_list[0], dtype=np.int32)
    except Exception:
        # Offset mapping is optional - return None if unavailable
        pass
    
    return None


def prepare_onnx_inputs_with_offsets(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
    input_names: Optional[Set[str]] = None,
) -> tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Prepare ONNX inputs and offset mapping in one call.
    
    This is more efficient than calling prepare_onnx_inputs and get_offset_mapping
    separately, as it can reuse tokenization results.
    
    Args:
        tokenizer: Tokenizer instance.
        text: Input text to tokenize.
        max_length: Maximum sequence length.
        input_names: Optional set of expected input names from ONNX model.
    
    Returns:
        Tuple of (feeds_dict, offset_mapping).
        - feeds_dict: Dictionary of tokenized inputs for ONNX.
        - offset_mapping: Offset mapping array or None.
    """
    # First tokenize for ONNX (without offsets for speed)
    feeds = prepare_onnx_inputs(tokenizer, text, max_length, input_names)
    
    # Then get offset mapping separately (can be slower, but done once)
    offset_mapping = get_offset_mapping(tokenizer, text, max_length)
    
    return feeds, offset_mapping

