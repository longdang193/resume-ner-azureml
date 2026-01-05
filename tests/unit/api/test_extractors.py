"""Unit tests for text extractors."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.api.extractors import (
    extract_text_from_pdf,
    extract_text_from_image,
    detect_file_type,
    validate_file,
)
from src.api.exceptions import (
    TextExtractionError,
    InvalidFileTypeError,
    FileSizeExceededError,
)


class TestFileTypeDetection:
    """Test file type detection."""

    def test_detect_pdf(self):
        """Test PDF detection."""
        pdf_content = b"%PDF-1.4\n"
        assert detect_file_type(pdf_content, "test.pdf") == "application/pdf"

    def test_detect_png(self):
        """Test PNG detection."""
        png_content = b"\x89PNG\r\n\x1a\n"
        assert detect_file_type(png_content, "test.png") == "image/png"

    def test_detect_jpeg(self):
        """Test JPEG detection."""
        jpeg_content = b"\xff\xd8\xff"
        assert detect_file_type(jpeg_content, "test.jpg") == "image/jpeg"

    def test_invalid_file_type(self):
        """Test invalid file type."""
        with pytest.raises(InvalidFileTypeError):
            detect_file_type(b"unknown content", "test.unknown")


class TestPDFExtraction:
    """Test PDF text extraction."""

    @patch("src.api.extractors._extract_pdf_pymupdf")
    def test_extract_pdf_pymupdf(self, mock_extract):
        """Test PDF extraction with PyMuPDF."""
        pdf_content = b"%PDF-1.4\n"
        
        mock_extract.return_value = "Extracted text"

        text = extract_text_from_pdf(pdf_content, extractor="pymupdf")
        assert text == "Extracted text"
        mock_extract.assert_called_once_with(pdf_content)

    @patch("src.api.extractors._extract_pdf_pdfplumber")
    def test_extract_pdf_pdfplumber(self, mock_extract):
        """Test PDF extraction with pdfplumber."""
        pdf_content = b"%PDF-1.4\n"
        
        mock_extract.return_value = "Extracted text"

        text = extract_text_from_pdf(pdf_content, extractor="pdfplumber")
        assert text == "Extracted text"
        mock_extract.assert_called_once_with(pdf_content)

    def test_extract_pdf_invalid_extractor(self):
        """Test PDF extraction with invalid extractor."""
        from src.api.exceptions import TextExtractionError
        with pytest.raises(TextExtractionError):
            extract_text_from_pdf(b"content", extractor="invalid")


class TestImageExtraction:
    """Test image OCR extraction."""

    @patch("src.api.extractors._extract_image_easyocr")
    def test_extract_image_easyocr(self, mock_extract):
        """Test image extraction with EasyOCR."""
        image_content = b"\x89PNG\r\n\x1a\n"
        
        mock_extract.return_value = "Extracted\ntext"

        text = extract_text_from_image(image_content, extractor="easyocr")
        assert "Extracted" in text
        assert "text" in text
        mock_extract.assert_called_once_with(image_content)

    @patch("src.api.extractors._extract_image_pytesseract")
    def test_extract_image_pytesseract(self, mock_extract):
        """Test image extraction with pytesseract."""
        image_content = b"\x89PNG\r\n\x1a\n"
        
        mock_extract.return_value = "Extracted text"

        text = extract_text_from_image(image_content, extractor="pytesseract")
        assert text == "Extracted text"
        mock_extract.assert_called_once_with(image_content)

    def test_extract_image_invalid_extractor(self):
        """Test image extraction with invalid extractor."""
        from src.api.exceptions import TextExtractionError
        with pytest.raises(TextExtractionError):
            extract_text_from_image(b"content", extractor="invalid")


class TestFileValidation:
    """Test file validation."""

    @pytest.mark.asyncio
    async def test_validate_file_success(self):
        """Test successful file validation."""
        mock_file = MagicMock()
        # Mock async read method
        async def mock_read():
            return b"content"
        mock_file.read = mock_read
        
        content = await validate_file(mock_file, max_size=100)
        assert content == b"content"

    @pytest.mark.asyncio
    async def test_validate_file_size_exceeded(self):
        """Test file size validation."""
        mock_file = MagicMock()
        # Mock async read method
        async def mock_read():
            return b"x" * 200
        mock_file.read = mock_read
        
        with pytest.raises(FileSizeExceededError):
            await validate_file(mock_file, max_size=100)


