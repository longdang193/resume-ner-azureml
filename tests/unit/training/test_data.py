"""Tests for data loading and processing utilities."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

from training.data import (
    load_dataset,
    normalize_text,
    encode_annotations_to_labels,
    build_label_list,
    ResumeNERDataset,
)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_train_only(self, sample_resume_data, temp_dir):
        """Test loading dataset with only train.json."""
        train_file = temp_dir / "train.json"
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(sample_resume_data, f)
        
        dataset = load_dataset(str(temp_dir))
        
        assert "train" in dataset
        assert "validation" in dataset
        assert dataset["train"] == sample_resume_data
        assert dataset["validation"] == []

    def test_with_validation(self, sample_resume_data, temp_dir):
        """Test loading dataset with both train.json and validation.json."""
        train_data = sample_resume_data[:2]
        val_data = sample_resume_data[2:]
        
        train_file = temp_dir / "train.json"
        val_file = temp_dir / "validation.json"
        
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f)
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f)
        
        dataset = load_dataset(str(temp_dir))
        
        assert dataset["train"] == train_data
        assert dataset["validation"] == val_data

    def test_missing_dataset_path(self):
        """Test that missing dataset path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset path not found"):
            load_dataset("/nonexistent/path")

    def test_missing_train_file(self, temp_dir):
        """Test that missing train.json raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Training file not found"):
            load_dataset(str(temp_dir))

    def test_invalid_json(self, temp_dir):
        """Test that invalid JSON raises an error."""
        train_file = temp_dir / "train.json"
        train_file.write_text("invalid json content {")
        
        with pytest.raises(json.JSONDecodeError):
            load_dataset(str(temp_dir))


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_string_input(self):
        """Test normalization of string input."""
        text = "Hello world"
        result = normalize_text(text)
        assert result == "Hello world"

    def test_list_input(self):
        """Test normalization of list input."""
        text = ["Hello", "world"]
        result = normalize_text(text)
        assert result == "Hello world"

    def test_tuple_input(self):
        """Test normalization of tuple input."""
        text = ("Hello", "world")
        result = normalize_text(text)
        assert result == "Hello world"

    def test_nested_list(self):
        """Test normalization of nested list."""
        text = [["Hello"], ["world", "test"]]
        result = normalize_text(text)
        assert result == "Hello world test"

    def test_dict_input(self):
        """Test normalization of dict input."""
        text = {"key": "value", "number": 123}
        result = normalize_text(text)
        # Should be JSON string representation
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_none_input(self):
        """Test normalization of None input."""
        result = normalize_text(None)
        assert result == ""

    def test_number_input(self):
        """Test normalization of number input."""
        result = normalize_text(123)
        assert result == "123"

    def test_empty_list(self):
        """Test normalization of empty list."""
        result = normalize_text([])
        assert result == ""


class TestEncodeAnnotationsToLabels:
    """Tests for encode_annotations_to_labels function."""

    def test_perfect_alignment(self, label2id):
        """Test encoding with perfect token-annotation alignment."""
        text = "John Doe"
        annotations = [[0, 4, "PERSON"], [5, 8, "PERSON"]]
        offsets = [(0, 4), (5, 8)]  # Token offsets
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        assert labels == [label2id["PERSON"], label2id["PERSON"]]

    def test_no_annotations(self, label2id):
        """Test encoding with no annotations."""
        text = "Hello world"
        annotations = []
        offsets = [(0, 5), (6, 11)]
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        assert labels == [label2id["O"], label2id["O"]]

    def test_partial_overlap(self, label2id):
        """Test encoding with partial token-annotation overlap."""
        text = "John Doe is a software engineer"
        # Annotation spans "John Doe" (0-8) and "software engineer" (14-33)
        annotations = [[0, 8, "PERSON"], [14, 33, "JOB_TITLE"]]
        # Tokens: "John" (0-4), "Doe" (5-8), "is" (9-11), "a" (12-13), 
        # "software" (14-22), "engineer" (23-31)
        offsets = [(0, 4), (5, 8), (9, 11), (12, 13), (14, 22), (23, 31)]
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        # First two tokens overlap with PERSON annotation
        assert labels[0] == label2id["PERSON"]
        assert labels[1] == label2id["PERSON"]
        # Middle tokens are O
        assert labels[2] == label2id["O"]
        assert labels[3] == label2id["O"]
        # Last two tokens overlap with JOB_TITLE
        assert labels[4] == label2id["JOB_TITLE"]
        assert labels[5] == label2id["JOB_TITLE"]

    def test_multiple_annotations_per_token(self, label2id):
        """Test encoding when multiple annotations could apply (first wins)."""
        text = "John"
        # Two overlapping annotations
        annotations = [[0, 4, "PERSON"], [0, 4, "ORG"]]
        offsets = [(0, 4)]
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        # First annotation should win
        assert labels == [label2id["PERSON"]]

    def test_annotation_beyond_text(self, label2id):
        """Test encoding with annotation beyond text boundaries."""
        text = "Hello"
        annotations = [[0, 10, "PERSON"]]  # Annotation extends beyond text
        offsets = [(0, 5)]
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        assert labels == [label2id["PERSON"]]

    def test_token_beyond_annotation(self, label2id):
        """Test encoding with token beyond annotation boundaries."""
        text = "Hello world"
        annotations = [[0, 5, "PERSON"]]  # Only covers "Hello"
        offsets = [(0, 5), (6, 11)]  # "Hello" and "world"
        
        labels = encode_annotations_to_labels(text, annotations, offsets, label2id)
        
        assert labels[0] == label2id["PERSON"]
        assert labels[1] == label2id["O"]


class TestBuildLabelList:
    """Tests for build_label_list function."""

    def test_standard_entity_types(self):
        """Test building label list with standard entity types."""
        data_config = {
            "schema": {
                "entity_types": ["PERSON", "ORG", "JOB_TITLE"],
            },
        }
        
        labels = build_label_list(data_config)
        
        assert labels == ["O", "JOB_TITLE", "ORG", "PERSON"]

    def test_empty_entity_types(self):
        """Test building label list with no entity types."""
        data_config = {
            "schema": {
                "entity_types": [],
            },
        }
        
        labels = build_label_list(data_config)
        
        assert labels == ["O"]

    def test_sorted_order(self):
        """Test that entity types are sorted."""
        data_config = {
            "schema": {
                "entity_types": ["ZEBRA", "APPLE", "BANANA"],
            },
        }
        
        labels = build_label_list(data_config)
        
        assert labels == ["O", "APPLE", "BANANA", "ZEBRA"]


class TestResumeNERDataset:
    """Tests for ResumeNERDataset class."""

    def test_len(self, sample_resume_data, mock_fast_tokenizer, label2id):
        """Test dataset length."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data,
            tokenizer=mock_fast_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        assert len(dataset) == len(sample_resume_data)

    def test_getitem_fast_tokenizer(self, sample_resume_data, mock_fast_tokenizer, label2id):
        """Test __getitem__ with fast tokenizer (with offset mapping)."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data[:1],  # Use first sample only
            tokenizer=mock_fast_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["labels"], torch.Tensor)

    def test_getitem_slow_tokenizer(self, sample_resume_data, mock_slow_tokenizer, label2id):
        """Test __getitem__ with slow tokenizer (without offset mapping)."""
        dataset = ResumeNERDataset(
            samples=sample_resume_data[:1],
            tokenizer=mock_slow_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        # With slow tokenizer, all labels should be "O"
        assert all(label == label2id["O"] for label in item["labels"].tolist())

    def test_empty_text(self, mock_fast_tokenizer, label2id):
        """Test dataset with empty text."""
        samples = [{"text": "", "annotations": []}]
        dataset = ResumeNERDataset(
            samples=samples,
            tokenizer=mock_fast_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item

    def test_no_annotations(self, mock_fast_tokenizer, label2id):
        """Test dataset with no annotations."""
        samples = [{"text": "Hello world", "annotations": []}]
        dataset = ResumeNERDataset(
            samples=samples,
            tokenizer=mock_fast_tokenizer,
            max_length=128,
            label2id=label2id,
        )
        
        item = dataset[0]
        # All labels should be "O"
        assert all(label == label2id["O"] for label in item["labels"].tolist())

    def test_truncation(self, mock_fast_tokenizer, label2id):
        """Test that long sequences are truncated."""
        long_text = " ".join(["word"] * 200)  # Very long text
        samples = [{"text": long_text, "annotations": []}]
        dataset = ResumeNERDataset(
            samples=samples,
            tokenizer=mock_fast_tokenizer,
            max_length=10,  # Very short max length
            label2id=label2id,
        )
        
        item = dataset[0]
        # Should be truncated to max_length (plus special tokens)
        assert item["input_ids"].shape[0] <= 10

