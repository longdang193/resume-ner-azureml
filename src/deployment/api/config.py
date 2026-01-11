"""API configuration settings."""

import os
from pathlib import Path
from typing import List, Optional


class APIConfig:
    """Configuration for the FastAPI service."""

    # Model paths
    ONNX_MODEL_PATH: Optional[Path] = None
    CHECKPOINT_DIR: Optional[Path] = None

    # Server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

    # File upload limits
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB default
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))

    # Model inference settings
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    ONNX_PROVIDERS: List[str] = ["CPUExecutionProvider"]  # Can add CUDAExecutionProvider for GPU

    # CORS settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Text extraction settings
    PDF_EXTRACTOR: str = os.getenv("PDF_EXTRACTOR", "pymupdf")  # pymupdf or pdfplumber
    OCR_EXTRACTOR: str = os.getenv("OCR_EXTRACTOR", "easyocr")  # easyocr or pytesseract

    @classmethod
    def set_model_paths(cls, onnx_path: Path, checkpoint_dir: Path) -> None:
        """Set model paths from command line arguments."""
        cls.ONNX_MODEL_PATH = Path(onnx_path)
        cls.CHECKPOINT_DIR = Path(checkpoint_dir)

    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if cls.ONNX_MODEL_PATH is None or not cls.ONNX_MODEL_PATH.exists():
            raise ValueError(f"ONNX model path does not exist: {cls.ONNX_MODEL_PATH}")
        if cls.CHECKPOINT_DIR is None or not cls.CHECKPOINT_DIR.exists():
            raise ValueError(f"Checkpoint directory does not exist: {cls.CHECKPOINT_DIR}")


