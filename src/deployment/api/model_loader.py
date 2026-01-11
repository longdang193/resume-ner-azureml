"""Model loading and initialization."""

from pathlib import Path
from typing import Optional, Dict, Any

from .inference import ONNXInferenceEngine
from .config import APIConfig
from .exceptions import ModelNotLoadedError


# Global model instance
_engine: Optional[ONNXInferenceEngine] = None
_model_info: Optional[Dict[str, Any]] = None


def initialize_model(
    onnx_path: Path,
    checkpoint_dir: Path,
    providers: Optional[list] = None,
) -> None:
    """
    Initialize the global model instance.

    Args:
        onnx_path: Path to ONNX model file.
        checkpoint_dir: Path to checkpoint directory.
        providers: ONNX Runtime providers.
    """
    global _engine, _model_info

    try:
        _engine = ONNXInferenceEngine(onnx_path, checkpoint_dir, providers)
        _model_info = {
            "backbone": _engine.checkpoint_dir.name,  # Approximate
            "entity_types": list(set(_engine.id2label.values())),
            "max_sequence_length": _engine.max_length,
            "version": "0.1.0",
        }
    except Exception as e:
        raise ModelNotLoadedError(f"Failed to initialize model: {e}") from e


def get_engine() -> ONNXInferenceEngine:
    """Get the global inference engine instance."""
    if _engine is None:
        raise ModelNotLoadedError("Model not initialized. Call initialize_model() first.")
    return _engine


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    if _model_info is None:
        raise ModelNotLoadedError("Model not initialized. Call initialize_model() first.")
    return _model_info.copy()


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _engine is not None


