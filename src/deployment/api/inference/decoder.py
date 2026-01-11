"""Entity decoding from token predictions."""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class EntityDecoder:
    """Decodes token predictions into entity spans."""
    
    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize entity decoder.
        
        Args:
            id2label: Mapping from label IDs to label strings.
        """
        self.id2label = id2label
    
    def decode_entities(
        self,
        text: str,
        logits: np.ndarray,
        tokens: List[str],
        tokenizer_output: Dict[str, np.ndarray],
        offset_mapping: Optional[np.ndarray] = None,
        return_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Decode token predictions to entity spans.
        
        Args:
            text: Original input text.
            logits: Model logits (seq_len, num_labels).
            tokens: Token list from tokenizer.
            tokenizer_output: Tokenizer output dictionary.
            offset_mapping: Token offset mapping from tokenizer (seq_len, 2).
            return_confidence: Whether to compute confidence scores.
        
        Returns:
            List of entity dictionaries with text, label, start, end, confidence.
        """
        # Get predictions (argmax over labels)
        predictions = np.argmax(logits, axis=-1)  # Shape: (seq_len,)
        
        # Get attention mask
        attention_mask = tokenizer_output.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask[0]
        else:
            attention_mask = np.ones(len(predictions), dtype=np.int64)
        
        # Convert predictions to labels (aligned with all tokens)
        labels = []
        for i, pred_id in enumerate(predictions):
            label = self.id2label.get(int(pred_id), "O")
            labels.append(label)
        
        # Debug: Log prediction statistics (only in debug mode)
        if logger.isEnabledFor(logging.DEBUG):
            non_o_labels = [l for l in labels if l != "O"]
            logger.debug(
                f"Predictions: {len(labels)} total, {len(non_o_labels)} non-O labels: {non_o_labels[:10]}")
        
        # Extract entities from BIO tags
        entities = self._extract_entities_from_bio(
            text,
            tokens,
            labels,
            logits if return_confidence else None,
            attention_mask,
            offset_mapping,
            return_confidence=return_confidence,
        )
        
        return entities
    
    def _extract_entities_from_bio(
        self,
        text: str,
        tokens: List[str],
        labels: List[str],
        logits: Optional[np.ndarray],
        attention_mask: np.ndarray,
        offset_mapping: Optional[np.ndarray] = None,
        return_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from BIO-tagged sequence.
        
        Args:
            text: Original text.
            tokens: Token list.
            labels: Label list (BIO format).
            logits: Optional logits for confidence calculation.
            attention_mask: Attention mask.
            offset_mapping: Token offset mapping from tokenizer.
        
        Returns:
            List of entity dictionaries.
        """
        entities = []
        current_entity = None
        
        # Get token offsets in original text
        token_offsets = self._process_offset_mapping(
            offset_mapping, text, tokens, attention_mask)
        
        # Ensure token_offsets has same length as tokens
        while len(token_offsets) < len(tokens):
            token_offsets.append(None)
        
        for i, (token, label, offset) in enumerate(zip(tokens, labels, token_offsets)):
            # Skip padding tokens (attention_mask == 0)
            if i >= len(attention_mask) or attention_mask[i] == 0:
                continue
            # Skip special tokens (offset is None or (0, 0))
            if offset is None:
                continue
            
            token_start, token_end = offset
            
            # Safety check: ensure valid offset
            if token_start < 0 or token_end < 0 or token_start >= token_end or token_end > len(text):
                continue
            
            # Handle labels - check if using BIO format or flat labels
            if label == "O":
                # "O" tag: end current entity if any
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-") or label.startswith("I-"):
                # BIO format
                if label.startswith("B-"):
                    # Start new entity
                    if current_entity is not None:
                        # Save previous entity
                        entities.append(current_entity)
                    entity_type = label[2:]  # Remove "B-" prefix
                    current_entity = {
                        "start": token_start,
                        "end": token_end,
                        "label": entity_type,
                        "token_indices": [i],
                    }
                else:  # I- label
                    # Continue entity
                    entity_type = label[2:]  # Remove "I-" prefix
                    if current_entity is not None and current_entity["label"] == entity_type:
                        # Extend entity
                        current_entity["end"] = token_end
                        current_entity["token_indices"].append(i)
                    else:
                        # Mismatch: start new entity
                        if current_entity is not None:
                            entities.append(current_entity)
                        current_entity = {
                            "start": token_start,
                            "end": token_end,
                            "label": entity_type,
                            "token_indices": [i],
                        }
            else:
                # Flat label format (e.g., "EMAIL", "SKILL", "LOCATION")
                # Check if same entity type continues
                if current_entity is not None and current_entity["label"] == label:
                    # Extend current entity
                    current_entity["end"] = token_end
                    current_entity["token_indices"].append(i)
                else:
                    # Start new entity
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = {
                        "start": token_start,
                        "end": token_end,
                        "label": label,
                        "token_indices": [i],
                    }
        
        # Add final entity if exists
        if current_entity is not None:
            entities.append(current_entity)
        
        # Extract text and compute confidence
        result = []
        for entity in entities:
            entity_text = text[entity["start"]: entity["end"]]
            confidence = None
            
            if logits is not None and return_confidence:
                # Compute average confidence over entity tokens
                try:
                    # Shape: (num_tokens, num_labels)
                    token_logits = logits[entity["token_indices"]]
                    # Manual softmax implementation (NumPy doesn't have softmax)
                    # Subtract max for numerical stability
                    exp_logits = np.exp(
                        token_logits - np.max(token_logits, axis=-1, keepdims=True))
                    # Shape: (num_tokens, num_labels)
                    token_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    # Get max probability (confidence) for each token
                    token_confidences = np.max(token_probs, axis=-1)  # Shape: (num_tokens,)
                    # Average confidence across all tokens in entity
                    confidence = float(np.mean(token_confidences))
                except Exception as e:
                    logger.warning(f"Failed to compute confidence for entity: {e}")
                    confidence = None
            
            result.append(
                {
                    "text": entity_text,
                    "label": entity["label"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": confidence,
                }
            )
        
        return result
    
    def _process_offset_mapping(
        self,
        offset_mapping: Optional[np.ndarray],
        text: str,
        tokens: List[str],
        attention_mask: np.ndarray,
    ) -> List[Optional[Tuple[int, int]]]:
        """
        Process offset mapping from tokenizer or compute manually.
        
        Args:
            offset_mapping: Token offset mapping from tokenizer (seq_len, 2).
            text: Original text.
            tokens: Token list.
            attention_mask: Attention mask.
        
        Returns:
            List of (start, end) tuples or None for special tokens.
        """
        if offset_mapping is not None:
            # offset_mapping is numpy array of shape (seq_len, 2)
            token_offsets = []
            try:
                # Handle numpy array properly
                if hasattr(offset_mapping, 'shape'):
                    if len(offset_mapping.shape) == 2:
                        # Shape: (seq_len, 2) - iterate by index
                        for i in range(min(len(offset_mapping), len(tokens))):
                            offset = offset_mapping[i]
                            if hasattr(offset, '__len__') and len(offset) >= 2:
                                start, end = int(offset[0]), int(offset[1])
                            elif hasattr(offset, '__getitem__'):
                                try:
                                    start, end = int(offset[0]), int(offset[1])
                                except (IndexError, TypeError):
                                    token_offsets.append(None)
                                    continue
                            else:
                                token_offsets.append(None)
                                continue
                            # (0, 0) means special token (CLS, SEP, PAD) - skip it
                            if start == 0 and end == 0:
                                token_offsets.append(None)
                            else:
                                token_offsets.append((start, end))
                    else:
                        # Unexpected shape - fall back to manual offsets
                        token_offsets = self._get_token_offsets(text, tokens, attention_mask)
                else:
                    # Not a numpy array - try to iterate
                    for i, offset in enumerate(offset_mapping):
                        if i >= len(tokens):
                            break
                        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                            start, end = int(offset[0]), int(offset[1])
                        elif hasattr(offset, '__getitem__'):
                            try:
                                start, end = int(offset[0]), int(offset[1])
                            except (IndexError, TypeError):
                                token_offsets.append(None)
                                continue
                        else:
                            token_offsets.append(None)
                            continue
                        if start == 0 and end == 0:
                            token_offsets.append(None)
                        else:
                            token_offsets.append((start, end))
            except Exception as e:
                logger.warning(
                    f"Failed to process offset_mapping: {e}, falling back to manual offsets")
                token_offsets = self._get_token_offsets(text, tokens, attention_mask)
        else:
            token_offsets = self._get_token_offsets(text, tokens, attention_mask)
        
        return token_offsets
    
    def _get_token_offsets(
        self,
        text: str,
        tokens: List[str],
        attention_mask: np.ndarray,
    ) -> List[Optional[Tuple[int, int]]]:
        """
        Get character offsets for each token in the original text.
        
        Args:
            text: Original text.
            tokens: Token list.
            attention_mask: Attention mask.
        
        Returns:
            List of (start, end) tuples or None for special tokens.
        """
        offsets = []
        char_idx = 0
        
        for i, (token, mask_val) in enumerate(zip(tokens, attention_mask)):
            if mask_val == 0:
                offsets.append(None)
                continue
            
            # Handle special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                offsets.append(None)
                continue
            
            # Remove special prefixes (## for BERT, Ġ for RoBERTa)
            clean_token = token
            if token.startswith("##"):
                clean_token = token[2:]
            elif token.startswith("Ġ"):
                clean_token = token[1:]
            
            # Find token in text
            if clean_token and char_idx < len(text):
                # Try to find the token in the text
                token_lower = clean_token.lower()
                text_lower = text.lower()
                
                # Search for token starting from current position
                found_idx = text_lower.find(token_lower, char_idx)
                if found_idx != -1:
                    start = found_idx
                    end = start + len(clean_token)
                    offsets.append((start, end))
                    char_idx = end
                else:
                    # Fallback: use character position
                    offsets.append(
                        (char_idx, min(char_idx + len(clean_token), len(text))))
                    char_idx += len(clean_token)
            else:
                offsets.append(None)
        
        return offsets

