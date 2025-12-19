"""Tests for training metrics computation."""

import pytest
from training.metrics import (
    compute_f1_for_label,
    compute_token_macro_f1,
    compute_metrics,
)


class TestComputeF1ForLabel:
    """Tests for compute_f1_for_label function."""

    def test_perfect_match(self):
        """Test F1 score with perfect predictions."""
        true_labels = ["PERSON", "PERSON", "O", "ORG", "ORG"]
        pred_labels = ["PERSON", "PERSON", "O", "ORG", "ORG"]
        
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 1.0
        
        f1 = compute_f1_for_label("ORG", true_labels, pred_labels)
        assert f1 == 1.0

    def test_no_match(self):
        """Test F1 score when label is never predicted correctly."""
        true_labels = ["PERSON", "PERSON", "O"]
        pred_labels = ["O", "O", "O"]
        
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 0.0

    def test_partial_match(self):
        """Test F1 score with partial matches."""
        # 2 true positives, 1 false positive, 1 false negative
        true_labels = ["PERSON", "PERSON", "O", "ORG"]
        pred_labels = ["PERSON", "O", "PERSON", "ORG"]
        
        # For PERSON: TP=1, FP=1, FN=1
        # Precision = 1/(1+1) = 0.5, Recall = 1/(1+1) = 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 0.5

    def test_empty_lists(self):
        """Test F1 score with empty label lists."""
        f1 = compute_f1_for_label("PERSON", [], [])
        assert f1 == 0.0

    def test_label_not_present(self):
        """Test F1 score when label is not in true or predicted labels."""
        true_labels = ["ORG", "O"]
        pred_labels = ["ORG", "O"]
        
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 0.0

    def test_only_false_positives(self):
        """Test F1 score with only false positives."""
        true_labels = ["O", "O"]
        pred_labels = ["PERSON", "PERSON"]
        
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 0.0

    def test_only_false_negatives(self):
        """Test F1 score with only false negatives."""
        true_labels = ["PERSON", "PERSON"]
        pred_labels = ["O", "O"]
        
        f1 = compute_f1_for_label("PERSON", true_labels, pred_labels)
        assert f1 == 0.0


class TestComputeTokenMacroF1:
    """Tests for compute_token_macro_f1 function."""

    def test_multiple_labels(self):
        """Test macro F1 with multiple labels."""
        all_labels = [
            ["PERSON", "PERSON", "O"],
            ["ORG", "O", "ORG"],
        ]
        all_preds = [
            ["PERSON", "PERSON", "O"],
            ["ORG", "O", "ORG"],
        ]
        
        macro_f1 = compute_token_macro_f1(all_labels, all_preds)
        assert macro_f1 == 1.0

    def test_single_label(self):
        """Test macro F1 with single label type."""
        all_labels = [["PERSON", "PERSON"]]
        all_preds = [["PERSON", "O"]]
        
        # For PERSON: TP=1, FP=0, FN=1
        # Precision = 1/1 = 1.0, Recall = 1/2 = 0.5
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 1.0 / 1.5 ≈ 0.667
        macro_f1 = compute_token_macro_f1(all_labels, all_preds)
        assert abs(macro_f1 - 0.6666666666666666) < 1e-6

    def test_empty_predictions(self):
        """Test macro F1 with empty predictions."""
        all_labels = [["PERSON", "ORG"]]
        all_preds = [[]]
        
        macro_f1 = compute_token_macro_f1(all_labels, all_preds)
        assert macro_f1 == 0.0

    def test_empty_labels(self):
        """Test macro F1 with empty true labels."""
        all_labels = [[]]
        all_preds = [["PERSON"]]
        
        macro_f1 = compute_token_macro_f1(all_labels, all_preds)
        assert macro_f1 == 0.0

    def test_imbalanced_labels(self):
        """Test macro F1 with imbalanced label distribution."""
        all_labels = [
            ["PERSON", "PERSON", "PERSON", "PERSON", "O", "ORG"],
        ]
        all_preds = [
            ["PERSON", "PERSON", "O", "O", "O", "ORG"],
        ]
        
        # PERSON: TP=2, FP=0, FN=2 -> F1 = 2/3 ≈ 0.667
        # ORG: TP=1, FP=0, FN=0 -> F1 = 1.0
        # O: TP=1, FP=2, FN=1 -> Precision=1/3, Recall=1/2 -> F1 = 2/5 = 0.4
        # Macro F1 = (0.667 + 1.0 + 0.4) / 3 ≈ 0.689
        macro_f1 = compute_token_macro_f1(all_labels, all_preds)
        assert macro_f1 > 0.0
        assert macro_f1 < 1.0


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_complete_metrics(self):
        """Test that all expected metrics are computed."""
        all_labels = [
            ["PERSON", "PERSON", "O"],
            ["ORG", "O", "ORG"],
        ]
        all_preds = [
            ["PERSON", "PERSON", "O"],
            ["ORG", "O", "ORG"],
        ]
        avg_loss = 0.5
        
        metrics = compute_metrics(all_labels, all_preds, avg_loss)
        
        assert "macro-f1" in metrics
        assert "macro-f1-span" in metrics
        assert "loss" in metrics
        assert metrics["loss"] == 0.5
        assert metrics["macro-f1"] == 1.0
        assert metrics["macro-f1-span"] == 1.0

    def test_zero_loss(self):
        """Test metrics computation with zero loss."""
        all_labels = [["PERSON", "O"]]
        all_preds = [["PERSON", "O"]]
        
        metrics = compute_metrics(all_labels, all_preds, 0.0)
        
        assert metrics["loss"] == 0.0
        assert metrics["macro-f1"] == 1.0

    def test_all_metrics_are_floats(self):
        """Test that all metrics are returned as floats."""
        all_labels = [["PERSON", "O"]]
        all_preds = [["O", "O"]]
        
        metrics = compute_metrics(all_labels, all_preds, 0.3)
        
        assert isinstance(metrics["macro-f1"], float)
        assert isinstance(metrics["macro-f1-span"], float)
        assert isinstance(metrics["loss"], float)

    def test_empty_labels(self):
        """Test metrics with empty label lists."""
        metrics = compute_metrics([], [], 0.1)
        
        assert metrics["loss"] == 0.1
        assert metrics["macro-f1"] == 0.0
        assert metrics["macro-f1-span"] == 0.0

