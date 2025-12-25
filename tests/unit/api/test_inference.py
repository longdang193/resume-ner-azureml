"""Unit tests for inference engine."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.api.inference import ONNXInferenceEngine
from src.api.exceptions import InferenceError, ModelNotLoadedError


class TestONNXInferenceEngine:
    """Test cases for ONNXInferenceEngine."""

    @pytest.fixture
    def mock_onnx_path(self, tmp_path):
        """Create a mock ONNX file path."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()
        return onnx_file

    @pytest.fixture
    def mock_checkpoint_dir(self, tmp_path):
        """Create a mock checkpoint directory."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        return checkpoint

    @patch("src.api.inference.AutoTokenizer")
    @patch("src.api.inference.AutoConfig")
    @patch("src.api.inference.ort.InferenceSession")
    def test_load_model_success(
        self,
        mock_session,
        mock_config,
        mock_tokenizer,
        mock_onnx_path,
        mock_checkpoint_dir,
    ):
        """Test successful model loading."""
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_config_instance = MagicMock()
        mock_config_instance.id2label = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}
        mock_config.from_pretrained.return_value = mock_config_instance

        # Create engine
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)

        # Assertions
        assert engine.session is not None
        assert engine.tokenizer is not None
        assert len(engine.id2label) > 0

    def test_load_model_file_not_found(self, tmp_path):
        """Test model loading with missing files."""
        with pytest.raises(FileNotFoundError):
            ONNXInferenceEngine(
                tmp_path / "nonexistent.onnx",
                tmp_path / "nonexistent",
            )

    @patch("src.api.inference.AutoTokenizer")
    @patch("src.api.inference.AutoConfig")
    @patch("src.api.inference.ort.InferenceSession")
    def test_predict_tokens(
        self,
        mock_session,
        mock_config,
        mock_tokenizer,
        mock_onnx_path,
        mock_checkpoint_dir,
    ):
        """Test token prediction."""
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_session_instance.run.return_value = [
            np.random.randn(1, 10, 5)  # (batch, seq_len, num_labels)
        ]

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]]),
            "attention_mask": np.array([[1, 1, 1, 1, 1]]),
        }
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
            "[CLS]", "hello", "world", "[SEP]", "[PAD]"
        ]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_config_instance = MagicMock()
        mock_config_instance.id2label = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}
        mock_config.from_pretrained.return_value = mock_config_instance

        # Create engine and predict
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        logits, tokens, tokenizer_output = engine.predict_tokens("hello world")

        # Assertions
        assert logits is not None
        assert len(tokens) > 0
        assert "input_ids" in tokenizer_output

    def test_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        engine = ONNXInferenceEngine.__new__(ONNXInferenceEngine)
        engine.session = None
        engine.tokenizer = None

        with pytest.raises(ModelNotLoadedError):
            engine.predict_tokens("test")


