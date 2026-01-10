from __future__ import annotations

"""
@meta
name: selection_logic
type: utility
domain: selection
responsibility:
  - Implement selection logic for choosing best configuration
  - Normalize speed scores
  - Apply accuracy thresholds
inputs:
  - Candidate configurations
  - Accuracy thresholds
outputs:
  - Selected best configuration
tags:
  - utility
  - selection
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Selection logic for choosing best configuration across studies."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.shared.logging_utils import get_logger

from .disk_loader import load_benchmark_speed_score, load_best_trial_from_disk
from orchestration.jobs.errors import SelectionError
from hpo.core.study import extract_best_config_from_study

logger = get_logger(__name__)

# Model speed characteristics (parameter count as proxy for inference speed)
MODEL_SPEED_SCORES = {
    "distilbert": 1.0,   # ~66M parameters (baseline)
    "deberta": 2.79,     # ~184M parameters (~2.79x slower)
}

class SelectionLogic:
    """Handles selection logic for best configuration."""

    @staticmethod
    def normalize_speed_scores(candidates: List[Dict[str, Any]]) -> None:
        """
        Normalize speed scores relative to fastest model.

        Args:
            candidates: List of candidate dictionaries with "speed_score" key.
        """
        raw_speed_scores = [c["speed_score"] for c in candidates]
        fastest_speed = min(raw_speed_scores)

        for candidate in candidates:
            # Normalize: fastest model gets 1.0, others are relative multiples
            candidate["speed_score"] = candidate["speed_score"] / fastest_speed

    @staticmethod
    def apply_threshold(
        candidates: List[Dict[str, Any]],
        accuracy_threshold: Optional[float],
        use_relative_threshold: bool,
        min_accuracy_gain: Optional[float],
    ) -> tuple[Dict[str, Any], str]:
        """
        Apply accuracy-speed tradeoff threshold to select best candidate.

        Args:
            candidates: List of candidates sorted by accuracy (descending).
            accuracy_threshold: Threshold for accuracy-speed tradeoff.
            use_relative_threshold: If True, threshold is relative to best accuracy.
            min_accuracy_gain: Minimum accuracy gain to justify slower model.

        Returns:
            Tuple of (selected_candidate, selection_reason).
        """
        if not candidates:
            raise SelectionError("No candidates provided for selection")

        best_candidate = candidates[0]
        best_accuracy = best_candidate["accuracy"]
        selected = best_candidate
        selection_reason = f"Best accuracy ({best_accuracy:.4f})"

        if accuracy_threshold is None or len(candidates) <= 1:
            return selected, selection_reason

        # Determine effective threshold
        if use_relative_threshold:
            effective_threshold = best_accuracy * accuracy_threshold
        else:
            effective_threshold = accuracy_threshold

        # Find fastest candidate within threshold
        faster_candidates = [
            c for c in candidates[1:]
            if c["speed_score"] < best_candidate["speed_score"]
        ]

        for candidate in faster_candidates:
            accuracy_diff = best_accuracy - candidate["accuracy"]

            # Check threshold
            within_threshold = accuracy_diff <= effective_threshold

            # Check minimum gain
            meets_min_gain = True
            if min_accuracy_gain is not None:
                if use_relative_threshold:
                    relative_gain = accuracy_diff / best_accuracy
                    meets_min_gain = relative_gain >= min_accuracy_gain
                else:
                    meets_min_gain = accuracy_diff >= min_accuracy_gain

            if within_threshold and meets_min_gain:
                selected = candidate
                selection_reason = (
                    f"Accuracy within threshold ({accuracy_threshold:.1%} "
                    f"{'relative' if use_relative_threshold else 'absolute'}), "
                    f"preferring faster model ({candidate['backbone']}). "
                    f"Accuracy diff: {accuracy_diff:.4f}"
                )
                break

        return selected, selection_reason

    @staticmethod
    def select_best(
        candidates: List[Dict[str, Any]],
        accuracy_threshold: Optional[float],
        use_relative_threshold: bool,
        min_accuracy_gain: Optional[float],
    ) -> Dict[str, Any]:
        """
        Select best configuration from candidates with accuracy-speed tradeoff.

        Args:
            candidates: List of candidate dictionaries with accuracy and speed_score.
            accuracy_threshold: Threshold for accuracy-speed tradeoff.
            use_relative_threshold: If True, threshold is relative to best accuracy.
            min_accuracy_gain: Minimum accuracy gain to justify slower model.

        Returns:
            Selected candidate dictionary with selection metadata.
        """
        if not candidates:
            raise SelectionError("No candidates provided for selection")

        # Normalize speed scores
        SelectionLogic.normalize_speed_scores(candidates)

        # Sort by accuracy (descending)
        candidates.sort(key=lambda x: x["accuracy"], reverse=True)

        # Apply threshold logic
        selected, selection_reason = SelectionLogic.apply_threshold(
            candidates, accuracy_threshold, use_relative_threshold, min_accuracy_gain
        )

        # Build result with enhanced metadata
        result = selected["config"] if "config" in selected else selected
        best_accuracy = candidates[0]["accuracy"]

        # Add selection criteria
        if isinstance(result, dict) and "selection_criteria" in result:
            result["selection_criteria"]["selection_strategy"] = (
                "accuracy_first_with_threshold" if accuracy_threshold else "accuracy_only"
            )
            result["selection_criteria"]["reason"] = selection_reason

            if accuracy_threshold is not None:
                result["selection_criteria"]["accuracy_threshold"] = accuracy_threshold
                result["selection_criteria"]["use_relative_threshold"] = use_relative_threshold
                result["selection_criteria"]["all_candidates"] = [
                    {
                        "backbone": c["backbone"],
                        "accuracy": c["accuracy"],
                        "speed_score": c["speed_score"],
                        "speed_data_source": c.get("speed_data_source", "parameter_proxy"),
                        "benchmark_latency_ms": c.get("benchmark_latency_ms"),
                    }
                    for c in candidates
                ]

                # Add speed_data_source for selected model
                result["selection_criteria"]["speed_data_source"] = selected.get(
                    "speed_data_source", "parameter_proxy")
                if len(candidates) > 1:
                    result["selection_criteria"]["accuracy_diff_from_best"] = (
                        best_accuracy - selected["accuracy"]
                    )

        return result

