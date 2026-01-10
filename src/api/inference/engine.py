"""ONNX model loading and inference execution."""

import time
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoConfig

from ..config import APIConfig
from ..exceptions import InferenceError, ModelNotLoadedError
from common.shared.tokenization_utils import (
    prepare_onnx_inputs,
    get_offset_mapping,
)

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Handles loading of ONNX model, tokenizer, and label mappings."""

    def __init__(
        self,
        onnx_path: Path,
        checkpoint_dir: Path,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize model loader.

        Args:
            onnx_path: Path to ONNX model file.
            checkpoint_dir: Path to checkpoint directory.
            providers: ONNX Runtime providers.
        """
        self.onnx_path = Path(onnx_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.providers = providers or APIConfig.ONNX_PROVIDERS

        self.session: Optional["ort.InferenceSession"] = None
        self.tokenizer: Optional["AutoTokenizer"] = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.max_length: int = APIConfig.MAX_SEQUENCE_LENGTH

        self.load()

    def load(self) -> None:
        """Load ONNX model, tokenizer, and label mappings."""
        import onnxruntime as ort
        from transformers import AutoTokenizer, AutoConfig
        
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Load ONNX model with optimized session options to prevent memory leaks
        try:
            # Configure session options to prevent memory accumulation
            sess_options = ort.SessionOptions()
            # Enable memory reuse to prevent accumulation across runs
            sess_options.enable_mem_reuse = True
            # Keep memory pattern enabled for performance
            sess_options.enable_mem_pattern = True
            # Enable CPU memory arena
            sess_options.enable_cpu_mem_arena = True
            
            # Optimize thread settings based on CPU cores for better performance
            import os
            num_cores = os.cpu_count() or 1
            # Use up to 4 threads for intra-op parallelism (good balance for most CPUs)
            # Keep inter-op at 1 to avoid thread contention
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = min(4, max(1, num_cores))
            # Set execution mode to sequential to prevent thread pool issues
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=self.providers,
            )
        except Exception as e:
            raise InferenceError(f"Failed to load ONNX model: {e}") from e

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_dir,
                use_fast=True,
                model_max_length=self.max_length,
            )
        except Exception as e:
            raise InferenceError(f"Failed to load tokenizer: {e}") from e

        # Load label mappings from model config
        try:
            config = AutoConfig.from_pretrained(self.checkpoint_dir)
            if hasattr(config, "id2label") and config.id2label:
                self.id2label = {int(k): v for k, v in config.id2label.items()}
                self.label2id = {v: k for k, v in self.id2label.items()}
            else:
                raise InferenceError(
                    "Label mappings not found in checkpoint config")
        except Exception as e:
            raise InferenceError(f"Failed to load label mappings: {e}") from e


class InferenceRunner:
    """Handles tokenization and ONNX inference execution."""

    def __init__(
        self,
        session: "ort.InferenceSession",
        tokenizer: "AutoTokenizer",
        max_length: int,
    ):
        """
        Initialize inference runner.

        Args:
            session: ONNX Runtime inference session.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
        """
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        if self.session is None or self.tokenizer is None:
            raise ModelNotLoadedError(
                "Model not loaded. Ensure model loader has been initialized.")

        max_len = max_length or self.max_length

        # Get input names from ONNX model
        input_names = {i.name for i in self.session.get_inputs()}
        logger.debug(f"ONNX model requires inputs: {input_names}")

        # Prepare ONNX inputs using shared utilities
        token_start = time.time()
        try:
            logger.info(
                f"Starting tokenization for text length={len(text)}, max_length={max_len}")

            feeds = prepare_onnx_inputs(
                self.tokenizer,
                text,
                max_len,
                input_names,
            )

            logger.info(
                f"ONNX tokenization completed in {time.time() - token_start:.3f}s")
            logger.debug(f"Tokenizer output keys: {list(feeds.keys())}")

            # Get offset mapping separately
            offset_mapping = get_offset_mapping(self.tokenizer, text, max_len)
            if offset_mapping is not None:
                logger.debug(
                    f"Offset mapping obtained: shape={offset_mapping.shape}")

        except Exception as e:
            logger.error(
                f"Tokenization failed after {time.time() - token_start:.3f}s: {e}")
            raise InferenceError(f"Tokenization failed: {e}") from e

        # Run inference with timeout detection
        inference_start = time.time()
        inference_timeout = 25.0  # 25 seconds timeout per inference
        try:
            logger.debug(
                f"Running ONNX inference with feeds: {list(feeds.keys())}")

            # Run inference - ONNX Runtime doesn't support timeout directly,
            # but we can detect if it takes too long
            outputs = self.session.run(None, feeds)

            elapsed = time.time() - inference_start
            if elapsed > inference_timeout:
                logger.error(
                    f"ONNX inference took {elapsed:.3f}s (exceeded {inference_timeout}s timeout)")
                raise InferenceError(
                    f"Inference took too long ({elapsed:.3f}s). "
                    "The model may be stuck or overloaded.")

            logits = outputs[0]  # Shape: (batch_size, seq_len, num_labels)
            logger.debug(
                f"ONNX inference completed in {elapsed:.3f}s, "
                f"logits shape: {logits.shape}")

            # Explicitly delete outputs array to help GC free memory
            # Note: logits is a view into outputs[0], so we extract it first
            logits = np.array(logits)  # Make independent copy
            del outputs

        except InferenceError:
            # Re-raise inference errors as-is
            raise
        except Exception as e:
            elapsed = time.time() - inference_start
            logger.error(
                f"ONNX inference failed after {elapsed:.3f}s: {e}")
            raise InferenceError(f"Inference failed: {e}") from e

        # Get tokenizer output for token decoding
        # We need to get the original tokenizer output (before int64 conversion)
        # Re-tokenize to get the original format, or reconstruct from feeds
        # For efficiency, we'll reconstruct from feeds but keep original types
        tokenizer_output = {}
        for k, v in feeds.items():
            # Keep the int64 format for consistency, but tokenizer can handle it
            tokenizer_output[k] = v

        # Clear feeds after creating tokenizer_output to free memory
        del feeds
        gc.collect()

        # Get tokens for decoding - only convert non-padding tokens for efficiency
        token_decode_start = time.time()
        try:
            input_ids = tokenizer_output["input_ids"][0]
            attention_mask = tokenizer_output.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask[0]
            else:
                attention_mask = np.ones(len(input_ids), dtype=np.int32)

            # Only convert tokens that are not padding (where attention_mask == 1)
            non_padding_indices = np.where(attention_mask == 1)[0]
            if len(non_padding_indices) > 0:
                non_padding_ids = input_ids[non_padding_indices]
                non_padding_tokens = self.tokenizer.convert_ids_to_tokens(
                    non_padding_ids.tolist())
                # Create full token list with padding tokens as empty strings
                tokens = [""] * len(input_ids)
                for idx, token in zip(non_padding_indices, non_padding_tokens):
                    tokens[idx] = token
            else:
                tokens = self.tokenizer.convert_ids_to_tokens(
                    input_ids.tolist())

            logger.info(
                f"Token decoding completed in {time.time() - token_decode_start:.3f}s "
                f"for {len(non_padding_indices)} non-padding tokens")
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            # Fallback: create empty tokens list
            tokens = []
            raise InferenceError(f"Token decoding failed: {e}") from e
        finally:
            # Clean up intermediate arrays
            del input_ids
            del attention_mask
            if 'non_padding_indices' in locals():
                del non_padding_indices
            gc.collect()

        return logits[0], tokens, tokenizer_output, offset_mapping
