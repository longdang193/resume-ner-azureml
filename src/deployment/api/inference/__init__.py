"""ONNX inference engine submodules."""

from .engine import ONNXModelLoader, InferenceRunner
from .decoder import EntityDecoder

# Import ONNXInferenceEngine from parent module (inference.py) for backward compatibility
# Since there's both a module (inference.py) and a package (inference/), Python prioritizes
# the package. The simplest solution: directly define the class here using the same components.
# This avoids the complex import issues with relative imports in inference.py.
try:
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple, Any
    import numpy as np
    
    # Reconstruct ONNXInferenceEngine class using the same components
    # This is equivalent to what's in inference.py but avoids import issues
    class ONNXInferenceEngine:
        """
        ONNX inference engine for NER predictions.

        This is a facade that composes model loading, inference execution, and entity decoding.
        It maintains backward compatibility with the original API.
        """

        def __init__(
            self,
            onnx_path: Path,
            checkpoint_dir: Path,
            providers: Optional[List[str]] = None,
        ):
            """
            Initialize the inference engine.

            Args:
                onnx_path: Path to ONNX model file.
                checkpoint_dir: Path to checkpoint directory containing tokenizer and config.
                providers: ONNX Runtime providers (default: CPUExecutionProvider).
            """
            # Initialize components
            self._model_loader = ONNXModelLoader(
                onnx_path, checkpoint_dir, providers)
            self._inference_runner = InferenceRunner(
                self._model_loader.session,
                self._model_loader.tokenizer,
                self._model_loader.max_length,
            )
            self._decoder = EntityDecoder(self._model_loader.id2label)

            # Expose commonly accessed attributes for backward compatibility
            self.onnx_path = self._model_loader.onnx_path
            self.checkpoint_dir = self._model_loader.checkpoint_dir
            self.providers = self._model_loader.providers
            self.session = self._model_loader.session
            self.tokenizer = self._model_loader.tokenizer
            self.id2label = self._model_loader.id2label
            self.label2id = self._model_loader.label2id
            self.max_length = self._model_loader.max_length

        def predict_tokens(
            self,
            text: str,
            max_length: Optional[int] = None,
        ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Optional[np.ndarray]]:
            """
            Run inference on text and return token predictions.

            Args:
                text: Input text.
                max_length: Maximum sequence length (default: from config).

            Returns:
                Tuple of (logits, tokens, tokenizer_output, offset_mapping).
            """
            return self._inference_runner.predict_tokens(text, max_length)

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
            return self._decoder.decode_entities(
                text,
                logits,
                tokens,
                tokenizer_output,
                offset_mapping,
                return_confidence,
            )

        def predict(
            self,
            text: str,
            max_length: Optional[int] = None,
            return_confidence: bool = True,
        ) -> List[Dict[str, Any]]:
            """
            Complete prediction pipeline: tokenize, infer, decode.

            Args:
                text: Input text.
                max_length: Maximum sequence length.
                return_confidence: Whether to return confidence scores.

            Returns:
                List of entity dictionaries.
            """
            logits, tokens, tokenizer_output, offset_mapping = self.predict_tokens(
                text, max_length)
            entities = self.decode_entities(
                text,
                logits,
                tokens,
                tokenizer_output,
                offset_mapping,
                return_confidence,
            )
            return entities
except Exception as e:
    # If import fails, set to None (will not be exported)
    ONNXInferenceEngine = None

# Also expose ort, AutoTokenizer, and AutoConfig for test compatibility
# This allows tests to patch src.api.inference.ort.InferenceSession, etc.
# Use lazy imports to avoid requiring onnxruntime at module load time
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Expose transformers classes for test compatibility
try:
    from transformers import AutoTokenizer, AutoConfig
except ImportError:
    AutoTokenizer = None
    AutoConfig = None

__all__ = [
    "ONNXModelLoader",
    "InferenceRunner",
    "EntityDecoder",
]

if ONNXInferenceEngine is not None:
    __all__.append("ONNXInferenceEngine")

if ort is not None:
    __all__.append("ort")
if AutoTokenizer is not None:
    __all__.append("AutoTokenizer")
if AutoConfig is not None:
    __all__.append("AutoConfig")

