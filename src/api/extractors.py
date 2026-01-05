"""Text extraction from PDF and image files."""

import io
import re
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from .config import APIConfig
from .exceptions import (
    TextExtractionError,
    InvalidFileTypeError,
    FileSizeExceededError,
)


def detect_file_type(file_content: bytes, filename: str) -> str:
    """
    Detect file MIME type from content and filename.

    Args:
        file_content: File content bytes.
        filename: Original filename.

    Returns:
        MIME type string.
    """
    # Check PDF magic bytes
    if file_content.startswith(b"%PDF"):
        return "application/pdf"

    # Check image magic bytes
    if file_content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if file_content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if file_content.startswith(b"GIF87a") or file_content.startswith(b"GIF89a"):
        return "image/gif"
    if file_content.startswith(b"BM"):
        return "image/bmp"

    # Fallback to extension
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
        return f"image/{ext[1:]}"

    raise InvalidFileTypeError(f"Unsupported file type: {filename}")


async def validate_file(file: UploadFile, max_size: Optional[int] = None) -> bytes:
    """
    Validate and read file content.

    Args:
        file: Uploaded file.
        max_size: Maximum file size in bytes.

    Returns:
        File content as bytes.

    Raises:
        FileSizeExceededError: If file exceeds size limit.
    """
    max_size = max_size or APIConfig.MAX_FILE_SIZE

    # Read file content
    content = await file.read()

    # Check size
    if len(content) > max_size:
        raise FileSizeExceededError(
            f"File size {len(content)} exceeds maximum {max_size} bytes"
        )

    return content


def extract_text_from_pdf(
    pdf_bytes: bytes,
    extractor: str = "pymupdf",
) -> str:
    """
    Extract text from PDF file.

    Args:
        pdf_bytes: PDF file content as bytes.
        extractor: Extractor to use ("pymupdf" or "pdfplumber").

    Returns:
        Extracted text.

    Raises:
        TextExtractionError: If extraction fails.
    """
    try:
        if extractor == "pymupdf":
            return _extract_pdf_pymupdf(pdf_bytes)
        elif extractor == "pdfplumber":
            return _extract_pdf_pdfplumber(pdf_bytes)
        else:
            raise ValueError(f"Unknown PDF extractor: {extractor}")
    except Exception as e:
        raise TextExtractionError(
            f"Failed to extract text from PDF: {e}") from e


def _extract_pdf_pymupdf(pdf_bytes: bytes) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise TextExtractionError(
            "PyMuPDF not installed. Install with: pip install pymupdf"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    # Join pages and normalize whitespace to avoid tokenization issues
    # Replace all whitespace (including line breaks) with single spaces
    full_text = "\n\n".join(text_parts)
    # Normalize: replace all whitespace sequences with single space
    normalized_text = re.sub(r'\s+', ' ', full_text).strip()
    return normalized_text


def _extract_pdf_pdfplumber(pdf_bytes: bytes) -> str:
    """Extract text using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise TextExtractionError(
            "pdfplumber not installed. Install with: pip install pdfplumber"
        )

    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    # Normalize whitespace to avoid tokenization issues
    full_text = "\n\n".join(text_parts)
    normalized_text = re.sub(r'\s+', ' ', full_text).strip()
    return normalized_text


def extract_text_from_image(
    image_bytes: bytes,
    extractor: str = "easyocr",
) -> str:
    """
    Extract text from image using OCR.

    Args:
        image_bytes: Image file content as bytes.
        extractor: OCR extractor to use ("easyocr" or "pytesseract").

    Returns:
        Extracted text.

    Raises:
        TextExtractionError: If extraction fails.
    """
    try:
        if extractor == "easyocr":
            return _extract_image_easyocr(image_bytes)
        elif extractor == "pytesseract":
            return _extract_image_pytesseract(image_bytes)
        else:
            raise ValueError(f"Unknown OCR extractor: {extractor}")
    except Exception as e:
        raise TextExtractionError(
            f"Failed to extract text from image: {e}") from e


# Cache EasyOCR reader globally to avoid reinitialization overhead
_easyocr_reader = None
_easyocr_lock = None


def _get_easyocr_reader(gpu: bool = False):
    """Get or create cached EasyOCR reader instance."""
    global _easyocr_reader, _easyocr_lock
    
    if _easyocr_reader is None:
        try:
            import easyocr
            import threading
            if _easyocr_lock is None:
                _easyocr_lock = threading.Lock()
            
            with _easyocr_lock:
                # Double-check pattern in case another thread created it
                if _easyocr_reader is None:
                    _easyocr_reader = easyocr.Reader(["en"], gpu=gpu)
        except ImportError:
            raise TextExtractionError(
                "EasyOCR or Pillow not installed. Install with: pip install easyocr pillow"
            )
    
    return _easyocr_reader


def _extract_image_easyocr(image_bytes: bytes) -> str:
    """Extract text using EasyOCR."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise TextExtractionError(
            "EasyOCR or Pillow not installed. Install with: pip install easyocr pillow"
        )

    # Use cached EasyOCR reader to avoid reinitialization overhead (~1-2s per request)
    reader = _get_easyocr_reader(gpu=False)  # Set gpu=True if GPU available

    # Load image and convert to numpy array (EasyOCR requires numpy array, not PIL Image)
    # Optimize: Open directly and convert in one step
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if necessary (EasyOCR works best with RGB) and convert to numpy in one step
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert PIL Image to numpy array (more efficient than separate steps)
    image_array = np.asarray(image)

    # Run OCR - EasyOCR accepts numpy array, bytes, or file path
    results = reader.readtext(image_array)

    # Combine text from all detections
    # Ensure proper UTF-8 encoding to handle special characters (fixes Windows charmap issues)
    text_parts = []
    for result in results:
        text = result[1]
        # Ensure text is properly encoded as UTF-8 string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        elif not isinstance(text, str):
            text = str(text)
        # Replace any problematic characters that can't be encoded
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        text_parts.append(text)
    
    return "\n".join(text_parts)


def _extract_image_pytesseract(image_bytes: bytes) -> str:
    """Extract text using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise TextExtractionError(
            "pytesseract or Pillow not installed. Install with: pip install pytesseract pillow"
        )

    # Load image
    image = Image.open(io.BytesIO(image_bytes))

    # Run OCR
    text = pytesseract.image_to_string(image)

    # Ensure proper UTF-8 encoding to handle special characters (fixes Windows charmap issues)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    elif not isinstance(text, str):
        text = str(text)
    # Replace any problematic characters that can't be encoded
    text = text.encode('utf-8', errors='replace').decode('utf-8')

    return text
