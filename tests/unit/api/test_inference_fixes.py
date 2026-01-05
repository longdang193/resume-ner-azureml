"""Test cases for the specific fixes made to inference engine.

These tests verify:
1. Tokenization doesn't hang (fix for 5+ minute timeout)
2. Offset mapping is correctly extracted
3. Entity extraction works with proper character spans
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Handle missing dependencies gracefully
try:
    from src.api.inference import ONNXInferenceEngine
    from src.api.exceptions import InferenceError
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(
    not INFERENCE_AVAILABLE,
    reason="Required dependencies (onnxruntime, transformers) not installed"
)


class TestInferenceFixes:
    """Test cases for inference engine fixes."""

    @pytest.fixture
    def mock_setup(self, tmp_path):
        """Setup mock ONNX model and checkpoint."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()
        
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        
        return onnx_file, checkpoint

    @pytest.fixture
    def mock_id2label(self):
        """Mock label mapping."""
        return {
            0: "O",
            1: "B-PERSON",
            2: "I-PERSON",
            3: "B-ORG",
            4: "I-ORG",
        }

    def test_tokenization_returns_numpy_arrays(self, mock_setup, mock_id2label):
        """Test that tokenization with return_tensors='np' returns numpy arrays."""
        onnx_path, checkpoint_dir = mock_setup
        
        with patch("src.api.inference.engine.prepare_onnx_inputs") as mock_prepare, \
             patch("src.api.inference.engine.get_offset_mapping") as mock_get_offset, \
             patch("src.api.inference.engine.ort.InferenceSession") as mock_session, \
             patch("src.api.inference.engine.AutoTokenizer") as mock_tokenizer_class, \
             patch("src.api.inference.engine.AutoConfig") as mock_config:
            
            # Setup prepare_onnx_inputs and get_offset_mapping mocks
            mock_prepare.return_value = {
                "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
            }
            mock_get_offset.return_value = np.array([[(0, 0), (0, 4), (5, 8), (0, 0)]])
            
            # Setup session
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.get_inputs.return_value = [
                MagicMock(name="input_ids"),
                MagicMock(name="attention_mask"),
            ]
            
            # Setup tokenizer to return numpy arrays
            mock_tokenizer_instance = MagicMock()
            
            # Track call count to return different results for different calls
            call_count = [0]
            
            def tokenize_call(text, **kwargs):
                call_count[0] += 1
                if kwargs.get("return_offsets_mapping", False):
                    # First call: with offset mapping (returns lists)
                    result = {
                        "input_ids": [[1, 2, 3, 4]],
                        "attention_mask": [[1, 1, 1, 1]],
                        "offset_mapping": [[(0, 0), (0, 4), (5, 8), (0, 0)]],
                    }
                    return result
                elif kwargs.get("return_tensors") == "np":
                    # Second call: with return_tensors="np" (returns numpy arrays)
                    result = {
                        "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
                        "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
                    }
                    return result
                else:
                    # Fallback
                    return {
                        "input_ids": [[1, 2, 3, 4]],
                        "attention_mask": [[1, 1, 1, 1]],
                    }
            
            # Set up the mock to use our custom call function
            mock_tokenizer_instance.side_effect = tokenize_call
            mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
                "[CLS]", "John", "Doe", "[SEP]"
            ]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            # Setup config
            mock_config_instance = MagicMock()
            mock_config_instance.id2label = mock_id2label
            mock_config.from_pretrained.return_value = mock_config_instance
            
            # Setup ONNX output
            mock_session_instance.run.return_value = [
                np.random.randn(1, 4, len(mock_id2label))
            ]
            
            # Create engine and test
            engine = ONNXInferenceEngine(onnx_path, checkpoint_dir)
            
            # Verify tokenization returns numpy arrays
            logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens("John Doe")
            
            # Check that tokenizer_output contains numpy arrays
            assert isinstance(tokenizer_output["input_ids"], np.ndarray)
            assert isinstance(tokenizer_output["attention_mask"], np.ndarray)
            assert offset_mapping is not None
            assert isinstance(offset_mapping, np.ndarray)

    def test_offset_mapping_extraction(self, mock_setup, mock_id2label):
        """Test that offset mapping is correctly extracted from tokenizer output."""
        onnx_path, checkpoint_dir = mock_setup
        
        with patch("src.api.inference.engine.prepare_onnx_inputs") as mock_prepare, \
             patch("src.api.inference.engine.get_offset_mapping") as mock_get_offset, \
             patch("src.api.inference.engine.ort.InferenceSession") as mock_session, \
             patch("src.api.inference.engine.AutoTokenizer") as mock_tokenizer_class, \
             patch("src.api.inference.engine.AutoConfig") as mock_config:
            
            # Setup prepare_onnx_inputs and get_offset_mapping mocks
            expected_offsets = [(0, 0), (0, 4), (5, 8), (0, 0)]  # CLS, John, Doe, SEP
            mock_prepare.return_value = {
                "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
            }
            mock_get_offset.return_value = np.array([expected_offsets])
            
            # Setup mocks
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.get_inputs.return_value = [
                MagicMock(name="input_ids"),
                MagicMock(name="attention_mask"),
            ]
            
            mock_tokenizer_instance = MagicMock()
            
            def tokenize_call(text, **kwargs):
                if kwargs.get("return_offsets_mapping", False):
                    return {
                        "input_ids": [[1, 2, 3, 4]],
                        "attention_mask": [[1, 1, 1, 1]],
                        "offset_mapping": [expected_offsets],
                    }
                else:
                    return {
                        "input_ids": np.array([[1, 2, 3, 4]]),
                        "attention_mask": np.array([[1, 1, 1, 1]]),
                    }
            
            mock_tokenizer_instance.side_effect = tokenize_call
            mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
                "[CLS]", "John", "Doe", "[SEP]"
            ]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_config_instance = MagicMock()
            mock_config_instance.id2label = mock_id2label
            mock_config.from_pretrained.return_value = mock_config_instance
            
            mock_session_instance.run.return_value = [
                np.random.randn(1, 4, len(mock_id2label))
            ]
            
            engine = ONNXInferenceEngine(onnx_path, checkpoint_dir)
            logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens("John Doe")
            
            # Verify offset mapping
            assert offset_mapping is not None
            # offset_mapping is returned from get_offset_mapping which returns a 2D array
            # The shape should be (1, 4) for batch_size=1, seq_len=4
            # We need to check the first (and only) row
            if offset_mapping.ndim >= 2:
                # It's a 2D array, check the first row
                actual_offsets = offset_mapping[0]
                assert len(actual_offsets) == len(expected_offsets), \
                    f"Expected {len(expected_offsets)} offsets, got {len(actual_offsets)}. Shape: {offset_mapping.shape}"
                # Check that special tokens have (0, 0) offset
                assert actual_offsets[0][0] == 0 and actual_offsets[0][1] == 0  # CLS
                assert actual_offsets[3][0] == 0 and actual_offsets[3][1] == 0  # SEP
            else:
                # It's a 1D array
                assert len(offset_mapping) == len(expected_offsets)
                # Check that special tokens have (0, 0) offset
                assert offset_mapping[0][0] == 0 and offset_mapping[0][1] == 0  # CLS
                assert offset_mapping[3][0] == 0 and offset_mapping[3][1] == 0  # SEP

    def test_entity_extraction_with_offsets(self, mock_setup, mock_id2label):
        """Test that entities are extracted correctly using offset mapping."""
        onnx_path, checkpoint_dir = mock_setup
        text = "John Doe works at Google"
        
        with patch("src.api.inference.engine.prepare_onnx_inputs") as mock_prepare, \
             patch("src.api.inference.engine.get_offset_mapping") as mock_get_offset, \
             patch("src.api.inference.engine.ort.InferenceSession") as mock_session, \
             patch("src.api.inference.engine.AutoTokenizer") as mock_tokenizer_class, \
             patch("src.api.inference.engine.AutoConfig") as mock_config:
            
            # Create realistic offset mapping
            # "John Doe works at Google" -> tokens: [CLS, John, Doe, works, at, Google, SEP]
            offset_mapping_list = [
                (0, 0),      # CLS
                (0, 4),      # John
                (5, 8),      # Doe
                (9, 14),     # works
                (15, 17),    # at
                (18, 24),    # Google
                (0, 0),      # SEP
            ]
            
            # Setup prepare_onnx_inputs and get_offset_mapping mocks
            mock_prepare.return_value = {
                "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.int64),
            }
            mock_get_offset.return_value = np.array([offset_mapping_list])
            
            # Setup mocks
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.get_inputs.return_value = [
                MagicMock(name="input_ids"),
                MagicMock(name="attention_mask"),
            ]
            
            mock_tokenizer_instance = MagicMock()
            
            def tokenize_call(text_input, **kwargs):
                if kwargs.get("return_offsets_mapping", False):
                    # First call: return lists with offset_mapping
                    return {
                        "input_ids": [[1, 2, 3, 4, 5, 6, 7]],
                        "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
                        "offset_mapping": [offset_mapping_list],  # List of tuples
                    }
                elif kwargs.get("return_tensors") == "np":
                    # Second call: return numpy arrays
                    return {
                        "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
                        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.int64),
                    }
                else:
                    return {
                        "input_ids": [[1, 2, 3, 4, 5, 6, 7]],
                        "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
                    }
            
            mock_tokenizer_instance.side_effect = tokenize_call
            mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
                "[CLS]", "John", "Doe", "works", "at", "Google", "[SEP]"
            ]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_config_instance = MagicMock()
            mock_config_instance.id2label = mock_id2label
            mock_config.from_pretrained.return_value = mock_config_instance
            
            # Create logits that predict "John Doe" as PERSON and "Google" as ORG
            # Note: logits shape is (batch_size, seq_len, num_labels)
            # After logits[0], we get (seq_len, num_labels)
            num_labels = len(mock_id2label)
            seq_len = 7  # [CLS, John, Doe, works, at, Google, SEP]
            mock_logits = np.zeros((1, seq_len, num_labels))
            mock_logits[0, :, 0] = 10.0  # Default to O for all positions
            mock_logits[0, 1, 1] = 20.0  # B-PERSON for "John" (index 1) - higher than O
            mock_logits[0, 2, 2] = 20.0  # I-PERSON for "Doe" (index 2) - higher than O
            mock_logits[0, 5, 3] = 20.0  # B-ORG for "Google" (index 5) - higher than O
            
            mock_session_instance.run.return_value = [mock_logits]
            
            engine = ONNXInferenceEngine(onnx_path, checkpoint_dir)
            entities = engine.predict(text, return_confidence=False)
            
            # Should extract entities
            # Note: CLS (index 0) and SEP (index 6) have offset (0,0) so they're skipped
            # We should get: "John Doe" (B-PERSON + I-PERSON) and "Google" (B-ORG)
            assert len(entities) >= 2, f"Expected at least 2 entities, got {len(entities)}. " \
                f"Entities: {entities}. " \
                f"Check that offset_mapping is correctly extracted and logits align with tokens."
            
            # Check entity texts
            entity_texts = [e["text"] for e in entities]
            assert any("John" in text or "Doe" in text for text in entity_texts), \
                f"Expected 'John' or 'Doe' in entities, got: {entity_texts}"
            assert any("Google" in text for text in entity_texts), \
                f"Expected 'Google' in entities, got: {entity_texts}"
            
            # Check that entity spans are correct
            for entity in entities:
                assert "start" in entity
                assert "end" in entity
                assert entity["start"] < entity["end"]
                assert entity["text"] == text[entity["start"]:entity["end"]]

    def test_no_hanging_on_special_tokens(self, mock_setup, mock_id2label):
        """Test that special tokens don't cause the code to hang."""
        onnx_path, checkpoint_dir = mock_setup
        
        with patch("src.api.inference.engine.prepare_onnx_inputs") as mock_prepare, \
             patch("src.api.inference.engine.get_offset_mapping") as mock_get_offset, \
             patch("src.api.inference.engine.ort.InferenceSession") as mock_session, \
             patch("src.api.inference.engine.AutoTokenizer") as mock_tokenizer_class, \
             patch("src.api.inference.engine.AutoConfig") as mock_config:
            
            # Setup prepare_onnx_inputs and get_offset_mapping mocks
            mock_prepare.return_value = {
                "input_ids": np.array([[1, 2, 3, 0, 0, 0]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int64),
            }
            mock_get_offset.return_value = np.array([[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]])
            
            # Setup mocks
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.get_inputs.return_value = [
                MagicMock(name="input_ids"),
                MagicMock(name="attention_mask"),
            ]
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.convert_ids_to_tokens.return_value = [
                "[CLS]", "[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"
            ]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_config_instance = MagicMock()
            mock_config_instance.id2label = mock_id2label
            mock_config.from_pretrained.return_value = mock_config_instance
            
            mock_session_instance.run.return_value = [
                np.random.randn(1, 6, len(mock_id2label))
            ]
            
            engine = ONNXInferenceEngine(onnx_path, checkpoint_dir)
            
            # Should complete quickly without hanging
            import time
            start = time.time()
            entities = engine.predict("", return_confidence=False)
            elapsed = time.time() - start
            
            assert elapsed < 1.0, f"Processing took {elapsed:.2f} seconds, expected < 1.0"
            assert isinstance(entities, list)

