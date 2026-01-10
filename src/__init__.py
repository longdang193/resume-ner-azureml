"""
Source code package for Resume NER training and conversion scripts.

This package contains:
- Training modules (training/)
- API modules (api/)
- Model conversion utilities (conversion/)
- Shared utilities (shared/)
- Platform adapters (platform_adapters/)
- Orchestration utilities (orchestration/)
"""

# Export commonly used items for convenience
# Use relative imports to avoid path issues
# Make imports optional to avoid blocking module execution when only API is needed
try:
    from .training import train_model
    from .training.config import build_training_config
    __all__ = [
        "train_model",
        "build_training_config",
    ]
except ImportError:
    # Fallback: try absolute import if src is in path
    try:
        from src.training import train_model
        from src.training.config import build_training_config
        __all__ = [
            "train_model",
            "build_training_config",
        ]
    except ImportError:
        # Last resort: try direct import (when src is current directory)
        try:
            from training import train_model
            from training.config import build_training_config
            __all__ = [
                "train_model",
                "build_training_config",
            ]
        except ImportError:
            # If all imports fail, just export empty list (e.g., when only API is needed)
            __all__ = []
