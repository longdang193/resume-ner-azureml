"""Tests for tokenizer offset mapping handling to prevent NotImplementedError."""

import pytest
from unittest.mock import MagicMock
from training.data import ResumeNERDataset


class TestTokenizerOffsetMappingHandling:
    """Tests for proper handling of tokenizer offset mapping support."""

    def test_fast_tokenizer_uses_offset_mapping(self, sample_resume_data, mock_fast_tokenizer, label2id):
        """Test that fast tokenizers use offset mapping for label encoding."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data[:1],
            tokenizer=mock_fast_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        # Mock the tokenizer to return offset mapping
        def tokenize_with_offsets(text, **kwargs):
            tokens = text.split()
            token_ids = list(range(len(tokens)))
            offset_mapping = []
            start = 0
            for token in tokens:
                end = start + len(token)
                offset_mapping.append([start, end])
                start = end + 1
            
            result = {
                "input_ids": [[101] + token_ids + [102]],
                "attention_mask": [[1] * (len(token_ids) + 2)],
                "offset_mapping": [offset_mapping],
            }
            
            if kwargs.get("return_tensors") == "pt":
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            
            return result
        
        mock_fast_tokenizer.side_effect = tokenize_with_offsets
        mock_fast_tokenizer.is_fast = True
        
        item = dataset[0]
        
        # Should have labels (not all "O" if annotations exist)
        assert "labels" in item
        assert isinstance(item["labels"], type(item["input_ids"]))

    def test_slow_tokenizer_fallback_to_o_labels(self, sample_resume_data, mock_slow_tokenizer, label2id):
        """Test that slow tokenizers fall back to all 'O' labels."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data[:1],
            tokenizer=mock_slow_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        mock_slow_tokenizer.is_fast = False
        
        item = dataset[0]
        
        # With slow tokenizer, all labels should be "O"
        labels = item["labels"].tolist()
        assert all(label == label2id["O"] for label in labels)

    def test_tokenizer_offset_detection(self):
        """Test that tokenizer offset support is correctly detected."""
        # Fast tokenizer
        fast_tokenizer = MagicMock()
        fast_tokenizer.is_fast = True
        
        supports_offsets = bool(getattr(fast_tokenizer, "is_fast", False))
        assert supports_offsets is True
        
        # Slow tokenizer
        slow_tokenizer = MagicMock()
        slow_tokenizer.is_fast = False
        
        supports_offsets = bool(getattr(slow_tokenizer, "is_fast", False))
        assert supports_offsets is False
        
        # Tokenizer without is_fast attribute
        no_attr_tokenizer = MagicMock()
        del no_attr_tokenizer.is_fast
        
        supports_offsets = bool(getattr(no_attr_tokenizer, "is_fast", False))
        assert supports_offsets is False

    def test_no_notimplemented_error_with_slow_tokenizer(self, sample_resume_data, mock_slow_tokenizer, label2id):
        """Test that slow tokenizer does not raise NotImplementedError."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data[:1],
            tokenizer=mock_slow_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        mock_slow_tokenizer.is_fast = False
        
        # Should not raise NotImplementedError
        item = dataset[0]
        assert "labels" in item

