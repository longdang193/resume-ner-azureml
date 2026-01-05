"""Metric calculation utilities."""

from typing import Dict, List

from seqeval.metrics import f1_score, classification_report


def compute_f1_for_label(label: str, flat_true: List[str], flat_pred: List[str]) -> float:
    """
    Compute F1 score for a specific label.

    Args:
        label: Label to compute F1 for.
        flat_true: Flat list of true labels.
        flat_pred: Flat list of predicted labels.

    Returns:
        F1 score for the label.
    """
    tp = fp = fn = 0
    for y_t, y_p in zip(flat_true, flat_pred):
        if y_t == label and y_p == label:
            tp += 1
        elif y_t != label and y_p == label:
            fp += 1
        elif y_t == label and y_p != label:
            fn += 1
    
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    
    if precision == 0.0 and recall == 0.0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_token_macro_f1(all_labels: List[List[str]], all_preds: List[List[str]]) -> float:
    """
    Compute macro-averaged F1 score across all token labels.

    Args:
        all_labels: List of true label sequences.
        all_preds: List of predicted label sequences.

    Returns:
        Macro-averaged F1 score.
    """
    flat_true: List[str] = [lab for seq in all_labels for lab in seq]
    flat_pred: List[str] = [lab for seq in all_preds for lab in seq]
    unique_labels = sorted(set(flat_true)) if flat_true else []
    
    if not unique_labels:
        return 0.0
    
    f1_scores = [
        compute_f1_for_label(label, flat_true, flat_pred)
        for label in unique_labels
    ]
    
    return sum(f1_scores) / len(f1_scores)


def compute_per_entity_metrics(
    all_labels: List[List[str]],
    all_preds: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1 for each entity type.

    Args:
        all_labels: List of true label sequences.
        all_preds: List of predicted label sequences.

    Returns:
        Dictionary mapping entity names to their metrics:
        {
            "PERSON": {"precision": 0.85, "recall": 0.90, "f1": 0.875, "support": 100},
            "ORG": {"precision": 0.75, "recall": 0.80, "f1": 0.775, "support": 50},
            ...
        }
    """
    if not all_labels or not all_preds:
        return {}

    try:
        # Use seqeval's classification_report to get per-entity metrics
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        # Filter out aggregate rows and extract per-entity metrics
        per_entity = {}
        aggregate_keys = {"micro avg", "macro avg", "weighted avg"}
        
        for label, metrics in report.items():
            if label not in aggregate_keys and isinstance(metrics, dict):
                per_entity[label] = {
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1-score", 0.0)),
                    "support": int(metrics.get("support", 0)),
                }
        
        return per_entity
    except Exception:
        # Return empty dict if classification_report fails
        return {}


def compute_metrics(
    all_labels: List[List[str]],
    all_preds: List[List[str]],
    avg_loss: float,
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        all_labels: List of true label sequences.
        all_preds: List of predicted label sequences.
        avg_loss: Average loss value.

    Returns:
        Dictionary containing all computed metrics, including per-entity metrics.
    """
    span_f1 = f1_score(all_labels, all_preds) if all_labels else 0.0
    token_macro_f1 = compute_token_macro_f1(all_labels, all_preds)
    per_entity = compute_per_entity_metrics(all_labels, all_preds)
    
    result = {
        "macro-f1": float(token_macro_f1),
        "macro-f1-span": float(span_f1),
        "loss": float(avg_loss),
    }
    
    # Add per-entity metrics if available
    if per_entity:
        result["per_entity"] = per_entity
    
    return result

