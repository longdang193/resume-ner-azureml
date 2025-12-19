"""Tests for model evaluation functionality."""

import torch
from unittest.mock import MagicMock, patch
import pytest

from training.evaluator import extract_predictions_and_labels, evaluate_model


class TestExtractPredictionsAndLabels:
    """Tests for extract_predictions_and_labels function."""

    def test_extract_predictions_and_labels_basic(self):
        """Test basic extraction of predictions and labels."""
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]]])
        labels = torch.tensor([[1, 0, 0]])
        mask = torch.tensor([[1, 1, 1]])
        id2label = {0: "O", 1: "PERSON"}
        
        all_labels, all_preds = extract_predictions_and_labels(
            logits, labels, mask, id2label
        )
        
        assert len(all_labels) == 1
        assert len(all_preds) == 1
        assert all_labels[0] == ["PERSON", "O", "O"]
        # extract_predictions_and_labels returns label strings, not IDs
        assert all_preds[0] == ["PERSON", "O", "O"]

    def test_extract_predictions_and_labels_with_mask(self):
        """Test extraction with attention mask."""
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.9, 0.1], [0.5, 0.5]]])
        labels = torch.tensor([[1, 0, 0, -100]])
        mask = torch.tensor([[1, 1, 1, 0]])
        id2label = {0: "O", 1: "PERSON"}
        
        all_labels, all_preds = extract_predictions_and_labels(
            logits, labels, mask, id2label
        )
        
        assert len(all_labels) == 1
        assert len(all_preds) == 1
        assert len(all_labels[0]) == 3
        assert len(all_preds[0]) == 3

    def test_extract_predictions_and_labels_multiple_samples(self):
        """Test extraction with multiple samples."""
        logits = torch.tensor([
            [[0.1, 0.9], [0.8, 0.2]],
            [[0.9, 0.1], [0.1, 0.9]],
        ])
        labels = torch.tensor([[1, 0], [0, 1]])
        mask = torch.tensor([[1, 1], [1, 1]])
        id2label = {0: "O", 1: "PERSON"}
        
        all_labels, all_preds = extract_predictions_and_labels(
            logits, labels, mask, id2label
        )
        
        assert len(all_labels) == 2
        assert len(all_preds) == 2
        assert all_labels[0] == ["PERSON", "O"]
        assert all_labels[1] == ["O", "PERSON"]

    def test_extract_predictions_and_labels_unknown_label_id(self):
        """Test extraction with unknown label ID."""
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
        labels = torch.tensor([[1, 99]])
        mask = torch.tensor([[1, 1]])
        id2label = {0: "O", 1: "PERSON"}
        
        all_labels, all_preds = extract_predictions_and_labels(
            logits, labels, mask, id2label
        )
        
        assert all_labels[0][1] == "O"
        # extract_predictions_and_labels returns label strings, not IDs
        assert all_preds[0][1] == "O"

    def test_extract_predictions_and_labels_empty_after_mask(self):
        """Test extraction when all tokens are masked."""
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
        labels = torch.tensor([[1, 0]])
        mask = torch.tensor([[0, 0]])
        id2label = {0: "O", 1: "PERSON"}
        
        all_labels, all_preds = extract_predictions_and_labels(
            logits, labels, mask, id2label
        )
        
        assert len(all_labels) == 0
        assert len(all_preds) == 0


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @patch("training.evaluator.compute_metrics")
    def test_evaluate_model_success(self, mock_compute_metrics):
        """Test successful model evaluation."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
        mock_model.return_value = mock_outputs
        
        mock_dataloader = MagicMock()
        mock_batch = {
            "input_ids": torch.zeros(1, 2),
            "attention_mask": torch.ones(1, 2),
            "labels": torch.tensor([[1, 0]]),
        }
        mock_dataloader.__iter__ = lambda self: iter([mock_batch])
        mock_dataloader.__len__ = lambda self: 1
        
        device = torch.device("cpu")
        id2label = {0: "O", 1: "PERSON"}
        
        mock_compute_metrics.return_value = {"macro_f1": 0.85, "loss": 0.5}
        
        metrics = evaluate_model(mock_model, mock_dataloader, device, id2label)
        
        assert metrics == {"macro_f1": 0.85, "loss": 0.5}
        mock_model.eval.assert_called_once()
        mock_compute_metrics.assert_called_once()

    @patch("training.evaluator.compute_metrics")
    def test_evaluate_model_multiple_batches(self, mock_compute_metrics):
        """Test evaluation with multiple batches."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        mock_outputs1 = MagicMock()
        mock_outputs1.loss = torch.tensor(0.5)
        mock_outputs1.logits = torch.tensor([[[0.1, 0.9]]])
        
        mock_outputs2 = MagicMock()
        mock_outputs2.loss = torch.tensor(0.3)
        mock_outputs2.logits = torch.tensor([[[0.8, 0.2]]])
        
        mock_model.side_effect = [mock_outputs1, mock_outputs2]
        
        mock_dataloader = MagicMock()
        mock_batch1 = {
            "input_ids": torch.zeros(1, 1),
            "attention_mask": torch.ones(1, 1),
            "labels": torch.tensor([[1]]),
        }
        mock_batch2 = {
            "input_ids": torch.zeros(1, 1),
            "attention_mask": torch.ones(1, 1),
            "labels": torch.tensor([[0]]),
        }
        mock_dataloader.__iter__ = lambda self: iter([mock_batch1, mock_batch2])
        mock_dataloader.__len__ = lambda self: 2
        
        device = torch.device("cpu")
        id2label = {0: "O", 1: "PERSON"}
        
        mock_compute_metrics.return_value = {"macro_f1": 0.85, "loss": 0.4}
        
        metrics = evaluate_model(mock_model, mock_dataloader, device, id2label)
        
        assert metrics["loss"] == 0.4
        mock_compute_metrics.assert_called_once()

    @patch("training.evaluator.compute_metrics")
    def test_evaluate_model_zero_loss(self, mock_compute_metrics):
        """Test evaluation with zero loss."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(0.0)
        mock_outputs.logits = torch.tensor([[[0.5, 0.5]]])
        mock_model.return_value = mock_outputs
        
        mock_dataloader = MagicMock()
        mock_batch = {
            "input_ids": torch.zeros(1, 1),
            "attention_mask": torch.ones(1, 1),
            "labels": torch.tensor([[0]]),
        }
        mock_dataloader.__iter__ = lambda self: iter([mock_batch])
        mock_dataloader.__len__ = lambda self: 1
        
        device = torch.device("cpu")
        id2label = {0: "O", 1: "PERSON"}
        
        mock_compute_metrics.return_value = {"macro_f1": 1.0, "loss": 0.0}
        
        metrics = evaluate_model(mock_model, mock_dataloader, device, id2label)
        
        assert metrics["loss"] == 0.0
        call_args = mock_compute_metrics.call_args[0]
        assert call_args[2] == 0.0

    @patch("training.evaluator.compute_metrics")
    def test_evaluate_model_empty_dataloader(self, mock_compute_metrics):
        """Test evaluation with empty dataloader."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = lambda self: iter([])
        mock_dataloader.__len__ = lambda self: 0
        
        device = torch.device("cpu")
        id2label = {0: "O", 1: "PERSON"}
        
        mock_compute_metrics.return_value = {"macro_f1": 0.0, "loss": 0.0}
        
        metrics = evaluate_model(mock_model, mock_dataloader, device, id2label)
        
        assert metrics["loss"] == 0.0
        call_args = mock_compute_metrics.call_args[0]
        assert call_args[2] == 0.0

    @patch("training.evaluator.compute_metrics")
    def test_evaluate_model_uses_no_grad(self, mock_compute_metrics):
        """Test that evaluation uses torch.no_grad()."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.tensor([[[0.1, 0.9]]])
        mock_model.return_value = mock_outputs
        
        mock_dataloader = MagicMock()
        mock_batch = {
            "input_ids": torch.zeros(1, 1),
            "attention_mask": torch.ones(1, 1),
            "labels": torch.tensor([[1]]),
        }
        mock_dataloader.__iter__ = lambda self: iter([mock_batch])
        mock_dataloader.__len__ = lambda self: 1
        
        device = torch.device("cpu")
        id2label = {0: "O", 1: "PERSON"}
        
        mock_compute_metrics.return_value = {"macro_f1": 0.85, "loss": 0.5}
        
        with patch("torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)
            
            evaluate_model(mock_model, mock_dataloader, device, id2label)
            
            mock_no_grad.assert_called_once()

