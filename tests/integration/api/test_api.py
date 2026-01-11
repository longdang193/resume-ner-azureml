"""Integration tests for API endpoints."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Lazy import FastAPI components to handle missing python-multipart
try:
    from fastapi.testclient import TestClient
    from src.deployment.api.app import app
    from src.deployment.api.model_loader import initialize_model, is_model_loaded
except (RuntimeError, ImportError) as e:
    if "python-multipart" in str(e) or "multipart" in str(e).lower():
        pytest.skip("python-multipart not available", allow_module_level=True)
    else:
        raise


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_loaded():
    """Mock model as loaded."""
    with patch("src.api.routes.predictions.is_model_loaded", return_value=True):
        with patch("src.api.routes.predictions.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            # Mock predict method for batch predictions
            mock_engine.predict.return_value = [
                {
                    "text": "John Doe",
                    "label": "NAME",
                    "start": 0,
                    "end": 8,
                    "confidence": 0.95,
                }
            ]
            # Mock predict_tokens method for single predictions
            # Returns: logits, tokens, tokenizer_output, offset_mapping
            import numpy as np
            mock_logits = np.array([[0.1, 0.9, 0.1], [0.2, 0.8, 0.2]])  # Example logits
            mock_tokens = ["John", "Doe"]
            mock_tokenizer_output = {"input_ids": [1, 2]}
            mock_offset_mapping = [(0, 4), (5, 8)]
            mock_engine.predict_tokens.return_value = (
                mock_logits,
                mock_tokens,
                mock_tokenizer_output,
                mock_offset_mapping,
            )
            # Mock decode_entities method (returns list of entity dicts)
            mock_engine.decode_entities.return_value = [
                {
                    "text": "John Doe",
                    "label": "NAME",
                    "start": 0,
                    "end": 8,
                    "confidence": 0.95,
                }
            ]
            # Mock id2label for label mapping
            mock_engine.id2label = {0: "O", 1: "NAME", 2: "SKILL"}
            mock_get_engine.return_value = mock_engine
            yield mock_engine


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_info_not_loaded(self, client):
        """Test model info when model not loaded."""
        with patch("src.api.routes.health.is_model_loaded", return_value=False):
            response = client.get("/info")
            assert response.status_code == 503

    def test_model_info_loaded(self, client):
        """Test model info when model loaded."""
        with patch("src.api.routes.health.is_model_loaded", return_value=True):
            with patch("src.api.routes.health.get_model_info") as mock_info:
                mock_info.return_value = {
                    "backbone": "distilroberta",
                    "entity_types": ["SKILL", "NAME"],
                    "max_sequence_length": 512,
                    "version": "0.1.0",
                }
                response = client.get("/info")
                assert response.status_code == 200
                data = response.json()
                assert "backbone" in data
                assert "entity_types" in data


class TestPredictEndpoint:
    """Test prediction endpoints."""

    def test_predict_not_loaded(self, client):
        """Test predict when model not loaded."""
        with patch("src.api.routes.predictions.is_model_loaded", return_value=False):
            response = client.post(
                "/predict",
                json={"text": "John Doe is a software engineer."},
            )
            assert response.status_code == 503

    def test_predict_success(self, client, mock_model_loaded):
        """Test successful prediction."""
        response = client.post(
            "/predict",
            json={"text": "John Doe is a software engineer."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "processing_time_ms" in data
        assert len(data["entities"]) > 0

    def test_predict_batch_success(self, client, mock_model_loaded):
        """Test successful batch prediction."""
        response = client.post(
            "/predict/batch",
            json={"texts": ["Text 1", "Text 2"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_batch_size_exceeded(self, client, mock_model_loaded):
        """Test batch size limit."""
        with patch("src.api.config.APIConfig.MAX_BATCH_SIZE", 1):
            response = client.post(
                "/predict/batch",
                json={"texts": ["Text 1", "Text 2"]},
            )
            assert response.status_code == 400


class TestFileEndpoints:
    """Test file upload endpoints."""

    def test_predict_file_not_loaded(self, client):
        """Test file predict when model not loaded."""
        with patch("src.api.routes.predictions.is_model_loaded", return_value=False):
            response = client.post(
                "/predict/file",
                files={"file": ("test.pdf", b"%PDF-1.4\n", "application/pdf")},
            )
            assert response.status_code == 503

    @patch("src.api.routes.predictions.extract_text_from_pdf")
    def test_predict_file_pdf(self, mock_extract, client, mock_model_loaded):
        """Test PDF file prediction."""
        mock_extract.return_value = "Extracted text from PDF"
        
        response = client.post(
            "/predict/file",
            files={"file": ("test.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "extracted_text" in data

    @patch("src.api.routes.predictions.extract_text_from_image")
    def test_predict_file_image(self, mock_extract, client, mock_model_loaded):
        """Test image file prediction."""
        mock_extract.return_value = "Extracted text from image"
        
        response = client.post(
            "/predict/file",
            files={"file": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data


