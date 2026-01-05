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

    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
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

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_predict_tokens(
        self,
        mock_session,
        mock_config,
        mock_tokenizer,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
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

        # Mock prepare_onnx_inputs to return proper format
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])

        # Create a callable mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
            "[CLS]", "hello", "world", "[SEP]", "[PAD]"
        ]
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_config_instance = MagicMock()
        mock_config_instance.id2label = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}
        mock_config.from_pretrained.return_value = mock_config_instance

        # Set return values for the already-patched functions
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])

        # Create engine and predict
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens("hello world")

        # Assertions
        assert logits is not None
        assert len(tokens) > 0
        assert "input_ids" in tokenizer_output

    def test_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        # Create an instance without calling __init__ by using __new__
        # Then manually set attributes to None to simulate uninitialized state
        engine = object.__new__(ONNXInferenceEngine)
        engine.session = None
        engine.tokenizer = None
        engine._inference_runner = None

        # predict_tokens will fail because _inference_runner is None
        with pytest.raises((AttributeError, ModelNotLoadedError)):
            engine.predict_tokens("test")


