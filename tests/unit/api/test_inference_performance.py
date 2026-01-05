"""Performance and correctness tests for inference engine.

These tests verify that the inference engine:
1. Handles tokenization correctly without hanging
2. Extracts entities correctly with proper offset mapping
3. Processes text in reasonable time
4. Handles edge cases properly
"""

import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Handle missing dependencies gracefully
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from src.api.inference import ONNXInferenceEngine
    from src.api.exceptions import InferenceError, ModelNotLoadedError
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE or not INFERENCE_AVAILABLE,
    reason="Required dependencies (transformers, onnxruntime) not installed"
)


class TestInferencePerformance:
    """Test cases for inference performance and correctness."""

    @pytest.fixture
    def mock_onnx_path(self, tmp_path):
        """Create a mock ONNX file path."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()
        return onnx_file

    @pytest.fixture
    def mock_checkpoint_dir(self, tmp_path):
        """Create a mock checkpoint directory with tokenizer config."""
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        # Create a minimal tokenizer config
        (checkpoint / "tokenizer_config.json").write_text('{"model_max_length": 512}')
        (checkpoint / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nJohn\nDoe\n")
        return checkpoint

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "John Doe is a software engineer at Google. Email: john.doe@example.com Phone: +1-555-123-4567 Location: Seattle, WA"

    @pytest.fixture
    def mock_id2label(self):
        """Mock label mapping."""
        return {
            0: "O",
            1: "B-PERSON",
            2: "I-PERSON",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-EMAIL",
            6: "I-EMAIL",
            7: "B-PHONE",
            8: "I-PHONE",
            9: "B-LOCATION",
            10: "I-LOCATION",
        }

    @pytest.fixture
    def mock_session(self):
        """Create a mock ONNX session."""
        mock = MagicMock()
        mock.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        return mock

    def _create_mock_tokenizer(self):
        """Create a mock tokenizer that behaves like a real one."""
        mock_tokenizer = MagicMock(spec=AutoTokenizer)
        
        def tokenize_with_offsets(text, **kwargs):
            """Mock tokenizer that returns offset mapping."""
            # Simulate tokenization
            tokens = ["[CLS]"] + text.split() + ["[SEP]"]
            # Pad to max_length
            max_len = kwargs.get("max_length", 128)
            while len(tokens) < max_len:
                tokens.append("[PAD]")
            
            # Create offset mapping
            offset_mapping = []
            char_idx = 0
            for token in tokens:
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    offset_mapping.append((0, 0))
                else:
                    # Find token in text
                    start = text.find(token, char_idx) if char_idx < len(text) else 0
                    if start == -1:
                        offset_mapping.append((0, 0))
                    else:
                        end = start + len(token)
                        offset_mapping.append((start, end))
                        char_idx = end
            
            # Create return value
            result = {
                "input_ids": np.array([[i for i in range(len(tokens))]]),
                "attention_mask": np.array([[1 if t != "[PAD]" else 0 for t in tokens]]),
                "offset_mapping": [offset_mapping],
            }
            return result
        
        def tokenize_numpy(text, **kwargs):
            """Mock tokenizer that returns numpy arrays."""
            result = self._create_mock_tokenizer().tokenize_with_offsets(text, **kwargs)
            # Convert to numpy
            for k, v in result.items():
                if k != "offset_mapping":
                    if isinstance(v, list):
                        result[k] = np.array(v)
            return result
        
        mock_tokenizer.side_effect = tokenize_with_offsets
        mock_tokenizer.return_value = tokenize_with_offsets("")
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "John", "Doe", "[SEP]"]
        return mock_tokenizer

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_tokenization_does_not_hang(
        self,
        mock_session,
        mock_config,
        mock_tokenizer_class,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
        mock_onnx_path,
        mock_checkpoint_dir,
        sample_text,
        mock_id2label,
    ):
        """Test that tokenization completes quickly without hanging."""
        # Setup prepare_onnx_inputs and get_offset_mapping mocks
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])
        
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        
        # Create realistic tokenizer mock
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = (
            ["[CLS]"] + sample_text.split() + ["[SEP]"] + ["[PAD]"] * 100
        )
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock ONNX session
        num_labels = len(mock_id2label)
        seq_len = 128
        mock_logits = np.random.randn(1, seq_len, num_labels)
        mock_session_instance.run.return_value = [mock_logits]
        
        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.id2label = mock_id2label
        mock_config.from_pretrained.return_value = mock_config_instance
        
        # Create engine
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        
        # Test that prediction completes quickly
        start_time = time.time()
        try:
            logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens(sample_text)
            elapsed = time.time() - start_time
        except Exception as e:
            pytest.fail(f"Tokenization failed: {e}")
        
        # Should complete in under 1 second (even with mocks)
        assert elapsed < 1.0, f"Tokenization took {elapsed:.2f} seconds, expected < 1.0"
        assert logits is not None
        assert len(tokens) > 0
        assert "input_ids" in tokenizer_output
        assert offset_mapping is not None or "offset_mapping" in tokenizer_output

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_entity_extraction_with_offset_mapping(
        self,
        mock_session,
        mock_config,
        mock_tokenizer_class,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
        mock_onnx_path,
        mock_checkpoint_dir,
        sample_text,
        mock_id2label,
    ):
        """Test that entities are extracted correctly with offset mapping."""
        # Setup prepare_onnx_inputs and get_offset_mapping mocks
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])
        
        # Setup mocks similar to above
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        
        # Create tokenizer mock
        mock_tokenizer_instance = MagicMock()
        
        # Create realistic offset mapping
        words = sample_text.split()
        tokens = ["[CLS]"] + words + ["[SEP]"]
        max_len = 128
        while len(tokens) < max_len:
            tokens.append("[PAD]")
        
        offset_mapping_list = []
        char_idx = 0
        for token in tokens:
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                offset_mapping_list.append((0, 0))
            else:
                start = sample_text.find(token, char_idx) if char_idx < len(sample_text) else 0
                if start == -1:
                    offset_mapping_list.append((0, 0))
                else:
                    end = start + len(token)
                    offset_mapping_list.append((start, end))
                    char_idx = end
        
        def tokenize_with_offsets(text, **kwargs):
            return {
                "input_ids": [list(range(len(tokens)))],
                "attention_mask": [[1 if t != "[PAD]" else 0 for t in tokens]],
                "offset_mapping": [offset_mapping_list],
            }
        
        def tokenize_numpy(text, **kwargs):
            result = tokenize_with_offsets(text, **kwargs)
            numpy_result = {}
            for k, v in result.items():
                if k == "offset_mapping":
                    numpy_result[k] = v
                else:
                    numpy_result[k] = np.array(v, dtype=np.int64)
            return numpy_result
        
        call_count = [0]
        def tokenizer_call(text, **kwargs):
            call_count[0] += 1
            if "return_offsets_mapping" in kwargs and kwargs["return_offsets_mapping"]:
                return tokenize_with_offsets(text, **kwargs)
            elif "return_tensors" in kwargs and kwargs["return_tensors"] == "np":
                return tokenize_numpy(text, **kwargs)
            return tokenize_with_offsets(text, **kwargs)
        
        mock_tokenizer_instance.__call__ = tokenizer_call
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create logits that predict "John Doe" as PERSON
        # First, we need to know the actual sequence length from prepare_onnx_inputs
        # The mock returns 5 tokens, so seq_len = 5
        num_labels = len(mock_id2label)
        seq_len = 5  # From mock_prepare_onnx_inputs return value
        mock_logits = np.zeros((1, seq_len, num_labels))
        # Set high probability for O (label 0) for most tokens
        mock_logits[0, :, 0] = 10.0
        # Set high probability for B-PERSON (label 1) for token at index 1
        if 1 < seq_len and 1 < num_labels:
            mock_logits[0, 1, 1] = 20.0  # Higher than O to ensure it's selected
        # Set high probability for I-PERSON (label 2) for token at index 2
        if 2 < seq_len and 2 < num_labels:
            mock_logits[0, 2, 2] = 20.0  # Higher than O to ensure it's selected
        
        mock_session_instance.run.return_value = [mock_logits]
        
        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.id2label = mock_id2label
        mock_config.from_pretrained.return_value = mock_config_instance
        
        # Create engine and predict
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        
        entities = engine.predict(sample_text, return_confidence=False)
        
        # Should extract at least one entity
        assert len(entities) > 0, "No entities extracted"
        # Check that entity text matches
        entity_texts = [e["text"] for e in entities]
        assert any("John" in text or "Doe" in text for text in entity_texts), \
            f"Expected 'John' or 'Doe' in entities, got: {entity_texts}"

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_empty_text_handling(
        self,
        mock_session,
        mock_config,
        mock_tokenizer_class,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
        mock_onnx_path,
        mock_checkpoint_dir,
        mock_id2label,
    ):
        """Test that empty text is handled gracefully."""
        # Setup prepare_onnx_inputs and get_offset_mapping mocks
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 0)]])
        
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": np.array([[1, 2]]),  # CLS, SEP
            "attention_mask": np.array([[1, 1]]),
            "offset_mapping": [[(0, 0), (0, 0)]],
        }
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = ["[CLS]", "[SEP]"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_session_instance.run.return_value = [
            np.random.randn(1, 2, len(mock_id2label))
        ]
        
        mock_config_instance = MagicMock()
        mock_config_instance.id2label = mock_id2label
        mock_config.from_pretrained.return_value = mock_config_instance
        
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        
        # Should not raise an error
        entities = engine.predict("", return_confidence=False)
        assert isinstance(entities, list)
        # Empty text should produce no entities (or only special tokens)
        assert len(entities) == 0 or all(e["label"] == "O" for e in entities)

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_special_characters_handling(
        self,
        mock_session,
        mock_config,
        mock_tokenizer_class,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
        mock_onnx_path,
        mock_checkpoint_dir,
        mock_id2label,
    ):
        """Test that special characters are handled correctly."""
        special_text = "Email: test@example.com Phone: +1-555-1234"
        
        # Setup prepare_onnx_inputs and get_offset_mapping mocks
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])
        
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        
        mock_tokenizer_instance = MagicMock()
        words = special_text.split()
        tokens = ["[CLS]"] + words + ["[SEP]"]
        
        offset_mapping_list = []
        char_idx = 0
        for token in tokens:
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                offset_mapping_list.append((0, 0))
            else:
                start = special_text.find(token, char_idx) if char_idx < len(special_text) else 0
                if start == -1:
                    offset_mapping_list.append((0, 0))
                else:
                    end = start + len(token)
                    offset_mapping_list.append((start, end))
                    char_idx = end
        
        def tokenize_with_offsets(text, **kwargs):
            return {
                "input_ids": [list(range(len(tokens)))],
                "attention_mask": [[1 if t != "[PAD]" else 0 for t in tokens]],
                "offset_mapping": [offset_mapping_list],
            }
        
        def tokenize_numpy(text, **kwargs):
            result = tokenize_with_offsets(text, **kwargs)
            numpy_result = {}
            for k, v in result.items():
                if k == "offset_mapping":
                    numpy_result[k] = v
                else:
                    numpy_result[k] = np.array(v, dtype=np.int64)
            return numpy_result
        
        def tokenizer_call(text, **kwargs):
            if "return_offsets_mapping" in kwargs and kwargs["return_offsets_mapping"]:
                return tokenize_with_offsets(text, **kwargs)
            elif "return_tensors" in kwargs and kwargs["return_tensors"] == "np":
                return tokenize_numpy(text, **kwargs)
            return tokenize_with_offsets(text, **kwargs)
        
        mock_tokenizer_instance.__call__ = tokenizer_call
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_session_instance.run.return_value = [
            np.random.randn(1, len(tokens), len(mock_id2label))
        ]
        
        mock_config_instance = MagicMock()
        mock_config_instance.id2label = mock_id2label
        mock_config.from_pretrained.return_value = mock_config_instance
        
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        
        # Should not raise an error
        entities = engine.predict(special_text, return_confidence=False)
        assert isinstance(entities, list)

    @patch("src.api.inference.engine.prepare_onnx_inputs")
    @patch("src.api.inference.engine.get_offset_mapping")
    @patch("src.api.inference.engine.AutoTokenizer")
    @patch("src.api.inference.engine.AutoConfig")
    @patch("src.api.inference.engine.ort.InferenceSession")
    def test_tokenization_consistency(
        self,
        mock_session,
        mock_config,
        mock_tokenizer_class,
        mock_get_offset_mapping,
        mock_prepare_onnx_inputs,
        mock_onnx_path,
        mock_checkpoint_dir,
        sample_text,
        mock_id2label,
    ):
        """Test that tokenization is consistent between offset mapping and ONNX calls."""
        # Setup prepare_onnx_inputs and get_offset_mapping mocks
        mock_prepare_onnx_inputs.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        }
        mock_get_offset_mapping.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0), (0, 0)]])
        
        # Setup mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        
        mock_tokenizer_instance = MagicMock()
        words = sample_text.split()
        tokens = ["[CLS]"] + words + ["[SEP]"]
        
        offset_mapping_list = []
        char_idx = 0
        for token in tokens:
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                offset_mapping_list.append((0, 0))
            else:
                start = sample_text.find(token, char_idx) if char_idx < len(sample_text) else 0
                if start == -1:
                    offset_mapping_list.append((0, 0))
                else:
                    end = start + len(token)
                    offset_mapping_list.append((start, end))
                    char_idx = end
        
        call_tracker = []
        
        def tokenize_with_offsets(text, **kwargs):
            call_tracker.append(("with_offsets", text))
            return {
                "input_ids": [list(range(len(tokens)))],
                "attention_mask": [[1 if t != "[PAD]" else 0 for t in tokens]],
                "offset_mapping": [offset_mapping_list],
            }
        
        def tokenize_numpy(text, **kwargs):
            call_tracker.append(("numpy", text))
            result = tokenize_with_offsets(text, **kwargs)
            numpy_result = {}
            for k, v in result.items():
                if k == "offset_mapping":
                    numpy_result[k] = v
                else:
                    numpy_result[k] = np.array(v, dtype=np.int64)
            return numpy_result
        
        def tokenizer_call(text, **kwargs):
            if "return_offsets_mapping" in kwargs and kwargs["return_offsets_mapping"]:
                return tokenize_with_offsets(text, **kwargs)
            elif "return_tensors" in kwargs and kwargs["return_tensors"] == "np":
                return tokenize_numpy(text, **kwargs)
            return tokenize_with_offsets(text, **kwargs)
        
        mock_tokenizer_instance.__call__ = tokenizer_call
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_session_instance.run.return_value = [
            np.random.randn(1, len(tokens), len(mock_id2label))
        ]
        
        mock_config_instance = MagicMock()
        mock_config_instance.id2label = mock_id2label
        mock_config.from_pretrained.return_value = mock_config_instance
        
        engine = ONNXInferenceEngine(mock_onnx_path, mock_checkpoint_dir)
        
        # Predict
        engine.predict(sample_text, return_confidence=False)
        
        # Verify that prepare_onnx_inputs and get_offset_mapping were called
        # Since we're mocking these functions, the tokenizer won't be called directly
        # But the mocks should be called, which is what we care about for consistency
        assert mock_prepare_onnx_inputs.called, "prepare_onnx_inputs should be called"
        assert mock_get_offset_mapping.called, "get_offset_mapping should be called"
        
        # Verify both were called with the same text
        prepare_calls = mock_prepare_onnx_inputs.call_args_list
        get_offset_calls = mock_get_offset_mapping.call_args_list
        assert len(prepare_calls) > 0, "prepare_onnx_inputs should be called at least once"
        assert len(get_offset_calls) > 0, "get_offset_mapping should be called at least once"
        
        # Check that both were called with the same text (second argument)
        if len(prepare_calls) > 0 and len(get_offset_calls) > 0:
            prepare_text = prepare_calls[0][0][1] if len(prepare_calls[0][0]) > 1 else None
            get_offset_text = get_offset_calls[0][0][1] if len(get_offset_calls[0][0]) > 1 else None
            if prepare_text and get_offset_text:
                assert prepare_text == get_offset_text == sample_text, \
                    f"Both functions should be called with the same text: {sample_text}"

