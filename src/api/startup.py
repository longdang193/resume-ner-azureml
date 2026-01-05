"""FastAPI startup and shutdown events."""

from fastapi import FastAPI

from .model_loader import initialize_model, is_model_loaded
from .config import APIConfig


def startup_event(app: FastAPI) -> None:
    """Startup event handler."""
    if APIConfig.ONNX_MODEL_PATH and APIConfig.CHECKPOINT_DIR:
        try:
            initialize_model(
                APIConfig.ONNX_MODEL_PATH,
                APIConfig.CHECKPOINT_DIR,
                APIConfig.ONNX_PROVIDERS,
            )
            app.state.model_loaded = True
        except Exception as e:
            app.state.model_loaded = False
            app.state.model_error = str(e)
    else:
        app.state.model_loaded = False
        app.state.model_error = "Model paths not configured"


def shutdown_event(app: FastAPI) -> None:
    """Shutdown event handler."""
    # Cleanup if needed
    app.state.model_loaded = False


