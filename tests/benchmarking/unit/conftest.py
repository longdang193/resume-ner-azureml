"""Shared fixtures for benchmarking unit tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory with model files."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    
    # Create minimal model files (tokenizer and model config)
    (checkpoint_dir / "tokenizer_config.json").write_text('{"vocab_size": 1000}')
    (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
    (checkpoint_dir / "model.safetensors").write_bytes(b"fake_model_data")
    
    return checkpoint_dir


@pytest.fixture
def mock_test_data_file(tmp_path):
    """Create a mock test data JSON file."""
    test_data = tmp_path / "test.json"
    test_data.write_text(json.dumps([
        {"text": "Sample text 1", "labels": ["O", "O", "O"]},
        {"text": "Sample text 2", "labels": ["O", "B-PER", "I-PER"]},
    ]))
    return test_data

