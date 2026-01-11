"""FastAPI service for NER predictions."""

__version__ = "0.1.0"

# Export the FastAPI app
from .app import app

__all__ = ["app", "__version__"]

