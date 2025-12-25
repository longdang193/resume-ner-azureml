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

    @patch("src.api.extractors.fitz")
    def test_extract_pdf_pymupdf(self, mock_fitz):
        """Test PDF extraction with PyMuPDF."""
        pdf_content = b"%PDF-1.4\n"
        
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Extracted text"
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz.open.return_value = mock_doc

        text = extract_text_from_pdf(pdf_content, extractor="pymupdf")
        assert text == "Extracted text"

    @patch("src.api.extractors.pdfplumber")
    def test_extract_pdf_pdfplumber(self, mock_pdfplumber):
        """Test PDF extraction with pdfplumber."""
        pdf_content = b"%PDF-1.4\n"
        
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        text = extract_text_from_pdf(pdf_content, extractor="pdfplumber")
        assert text == "Extracted text"

    def test_extract_pdf_invalid_extractor(self):
        """Test PDF extraction with invalid extractor."""
        with pytest.raises(ValueError):
            extract_text_from_pdf(b"content", extractor="invalid")


class TestImageExtraction:
    """Test image OCR extraction."""

    @patch("src.api.extractors.easyocr")
    @patch("src.api.extractors.Image")
    def test_extract_image_easyocr(self, mock_image, mock_easyocr):
        """Test image extraction with EasyOCR."""
        image_content = b"\x89PNG\r\n\x1a\n"
        
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (None, "Extracted", None),
            (None, "text", None),
        ]
        mock_easyocr.Reader.return_value = mock_reader
        
        mock_img = MagicMock()
        mock_image.open.return_value = mock_img

        text = extract_text_from_image(image_content, extractor="easyocr")
        assert "Extracted" in text
        assert "text" in text

    @patch("src.api.extractors.pytesseract")
    @patch("src.api.extractors.Image")
    def test_extract_image_pytesseract(self, mock_image, mock_pytesseract):
        """Test image extraction with pytesseract."""
        image_content = b"\x89PNG\r\n\x1a\n"
        
        mock_pytesseract.image_to_string.return_value = "Extracted text"
        mock_img = MagicMock()
        mock_image.open.return_value = mock_img

        text = extract_text_from_image(image_content, extractor="pytesseract")
        assert text == "Extracted text"

    def test_extract_image_invalid_extractor(self):
        """Test image extraction with invalid extractor."""
        with pytest.raises(ValueError):
            extract_text_from_image(b"content", extractor="invalid")


class TestFileValidation:
    """Test file validation."""

    @pytest.mark.asyncio
    async def test_validate_file_success(self):
        """Test successful file validation."""
        mock_file = MagicMock()
        mock_file.read.return_value = b"content"
        
        content = await validate_file(mock_file, max_size=100)
        assert content == b"content"

    @pytest.mark.asyncio
    async def test_validate_file_size_exceeded(self):
        """Test file size validation."""
        mock_file = MagicMock()
        mock_file.read.return_value = b"x" * 200
        
        with pytest.raises(FileSizeExceededError):
            await validate_file(mock_file, max_size=100)


