from __future__ import annotations

"""
@meta
name: naming_context
type: utility
domain: naming
responsibility:
  - Define NamingContext dataclass for path generation
  - Create naming contexts with fingerprint-based identity
inputs:
  - Process type and model information
  - Fingerprints and identifiers
outputs:
  - NamingContext objects
tags:
  - utility
  - naming
  - context
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""NamingContext dataclass and factory function."""

import logging
from dataclasses import dataclass
from typing import Optional

from core.normalize import normalize_for_path
from common.shared.platform_detection import detect_platform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NamingContext:
    """
    Complete context for path generation with fingerprint-based identity.

    This context is used both for human-readable naming (run names) and for
    stable, fingerprint-based output paths.

    Attributes:
        process_type: Type of process
            ("hpo", "hpo_refit", "benchmarking", "final_training",
             "conversion", "best_configurations").
        stage: Optional fine-grained stage identifier
            (e.g., "hpo_sweep", "hpo_trial") used for naming/validation.
        model: Model backbone name (e.g., "distilbert").
        environment: Execution platform identifier
            (e.g., "local", "colab", "kaggle", "azure").
        storage_env: Logical storage environment used in outputs paths
            (e.g., "local", "colab", "kaggle", "azureml").
            Defaults to the same value as ``environment``.
        study_name: Human-readable HPO study/sweep name (e.g.,
            "hpo_distilbert_smoke_test_4.3"). Used for UX only.
        spec_fp: Specification fingerprint (platform-independent experiment
            identity) for training/final_training.
        exec_fp: Execution fingerprint (toolchain/runtime identity) for
            training/final_training.
        variant: Variant number for final_training (default 1, increments for
            force_new / retries).
        trial_id: Trial identifier for HPO/benchmarking (v2 format: "trial-{hash8}").
        trial_number: Explicit Optuna trial number (0-indexed integer).
            Prefer this over parsing trial_id for robustness.
        fold_idx: Optional fold index for cross-validation (0-indexed integer).
        parent_training_id: Parent training identifier for conversion (matches
            the final_training directory fragment, e.g.,
            "spec_abc_exec_xyz/v1").
        conv_fp: Conversion fingerprint for conversion variants.
        study_key_hash: Stable HPO study identifier (full hash).
        trial_key_hash: Stable HPO trial identifier (full hash).
        benchmark_config_hash: Optional benchmark configuration hash used to
            distinguish different benchmarking setups.
    """
    process_type: str
    model: str
    environment: str
    stage: Optional[str] = None
    storage_env: Optional[str] = None
    study_name: Optional[str] = None
    spec_fp: Optional[str] = None
    exec_fp: Optional[str] = None
    variant: int = 1
    trial_id: Optional[str] = None
    trial_number: Optional[int] = None
    fold_idx: Optional[int] = None
    parent_training_id: Optional[str] = None
    conv_fp: Optional[str] = None
    study_key_hash: Optional[str] = None
    trial_key_hash: Optional[str] = None
    benchmark_config_hash: Optional[str] = None

    def __post_init__(self):
        """Validate context after initialization."""
        valid_processes = {
            "hpo",
            "hpo_refit",
            "benchmarking",
            "final_training",
            "conversion",
            "best_configurations",
        }
        valid_environments = {"local", "colab", "kaggle", "azure", "azureml"}

        if self.process_type not in valid_processes:
            raise ValueError(
                f"Invalid process_type: {self.process_type}. "
                f"Must be one of {valid_processes}"
            )

        if self.environment not in valid_environments:
            raise ValueError(
                f"Invalid environment: {self.environment}. "
                f"Must be one of {valid_environments}"
            )

        # Default storage_env to environment if not explicitly provided
        object.__setattr__(self, "storage_env",
                           self.storage_env or self.environment)

        if self.variant < 1:
            raise ValueError(f"Variant must be >= 1, got {self.variant}")

        # Validate required fields per process type
        if self.process_type == "final_training":
            if not self.spec_fp or not self.exec_fp:
                raise ValueError(
                    "final_training requires spec_fp and exec_fp"
                )

        if self.process_type == "conversion":
            if not self.parent_training_id or not self.conv_fp:
                raise ValueError(
                    "conversion requires parent_training_id and conv_fp"
                )

        if self.process_type == "best_configurations":
            if not self.spec_fp:
                raise ValueError(
                    "best_configurations requires spec_fp"
                )


def create_naming_context(
    process_type: str,
    model: str,
    spec_fp: Optional[str] = None,
    exec_fp: Optional[str] = None,
    environment: Optional[str] = None,
    stage: Optional[str] = None,
    storage_env: Optional[str] = None,
    study_name: Optional[str] = None,
    variant: int = 1,
    trial_id: Optional[str] = None,
    trial_number: Optional[int] = None,
    fold_idx: Optional[int] = None,
    parent_training_id: Optional[str] = None,
    conv_fp: Optional[str] = None,
    study_key_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
    benchmark_config_hash: Optional[str] = None,
) -> NamingContext:
    """
    Factory function to create NamingContext with auto-detection.

    Args:
        process_type: Type of process (hpo, benchmarking, final_training, conversion).
        model: Model backbone name.
        spec_fp: Specification fingerprint (required for final_training, best_configurations).
        exec_fp: Execution fingerprint (required for final_training).
        environment: Execution environment (auto-detected if None).
        variant: Variant number for final_training (default 1).
        trial_id: Trial identifier for HPO/benchmarking.
        trial_number: Explicit Optuna trial number (0-indexed integer).
            Prefer this over parsing trial_id for robustness.
        parent_training_id: Parent training identifier for conversion.
        conv_fp: Conversion fingerprint (required for conversion).

    Returns:
        NamingContext instance.
    """
    if environment is None:
        environment = detect_platform()

    # Default storage_env to environment if not explicitly provided
    if storage_env is None:
        storage_env = environment

    # Layer B: Ensure trial_id is never None/empty/whitespace for hpo_refit
    if process_type == "hpo_refit":
        if not trial_id or not trial_id.strip():
            # Try to extract trial number from other context if available
            # This is a fallback - callers should provide trial_id
            logger.warning(
                f"[create_naming_context] hpo_refit missing trial_id, "
                f"cannot auto-fill without trial_number. "
                f"Caller should provide trial_id."
            )
            # We'll let it pass through as None and let the assert catch it
            # This ensures we fail fast rather than silently creating "unknown" names

    return NamingContext(
        process_type=process_type,
        model=model,
        environment=environment,
        stage=stage,
        storage_env=storage_env,
        study_name=study_name,
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=variant,
        trial_id=trial_id,
        trial_number=trial_number,
        fold_idx=fold_idx,
        parent_training_id=parent_training_id,
        conv_fp=conv_fp,
        study_key_hash=study_key_hash,
        trial_key_hash=trial_key_hash,
        benchmark_config_hash=benchmark_config_hash,
    )

