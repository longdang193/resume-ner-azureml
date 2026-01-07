"""Centralized naming and path building with fingerprint-based identity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.platform_detection import detect_platform

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
            (legacy name, e.g., "local", "colab", "kaggle", "azure").
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
        trial_id: Legacy trial identifier for HPO/benchmarking
            (e.g., "trial_1_20251229_100000").
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
        object.__setattr__(self, "storage_env", self.storage_env or self.environment)

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


def _get_pattern_key(process_type: str) -> Optional[str]:
    """Map process_type to paths.yaml pattern key."""
    mapping = {
        "hpo": "hpo_v2",
        "hpo_refit": "hpo_v2",  # Use same pattern as hpo, refit is subdirectory
        "benchmarking": "benchmarking_v2",
        "final_training": "final_training_v2",
        "conversion": "conversion_v2",
        "best_configurations": "best_config_v2",
    }
    return mapping.get(process_type)


def _validate_output_path(path: Path, process_type: str) -> None:
    """
    Basic validation of output path (sanity check).

    Since paths are constructed from validated NamingContext and patterns,
    this is primarily a defense-in-depth check for programming errors.

    Args:
        path: Path to validate
        process_type: Process type for error messages

    Raises:
        ValueError: If path is invalid
    """
    # Basic sanity check - paths from build_output_path should never be None/empty
    if not path or not str(path):
        raise ValueError(f"Invalid {process_type} output path: {path}")

    path_str = str(path)
    if not path_str or path_str in (".", ".."):
        raise ValueError(f"Invalid {process_type} output path: {path_str}")

    # Note: Removed version number check - root cause (pip install command) is fixed.
    # Paths from build_output_path always have structure like outputs/category/.../...
    # so they will never be just a version number.


def _build_output_path_fallback(
    root_dir: Path,
    context: NamingContext,
    base_outputs: str = "outputs"
) -> Path:
    """
    Fallback path building logic (hardcoded, used when patterns not available).

    This maintains backward compatibility if paths.yaml patterns are missing.
    """
    base_path = root_dir / base_outputs

    if context.process_type == "hpo":
        if not context.trial_id:
            raise ValueError("HPO requires trial_id")
        final_path = base_path / "hpo" / context.environment / \
            context.model / context.trial_id

    elif context.process_type == "hpo_refit":
        if not context.trial_id:
            raise ValueError("HPO refit requires trial_id")
        # Refit is a subdirectory of the trial: trial_<n>_<ts>/refit/
        # Extract trial base from trial_id (trial_id may include timestamp)
        final_path = base_path / "hpo" / context.environment / \
            context.model / context.trial_id / "refit"

    elif context.process_type == "benchmarking":
        if not context.trial_id:
            raise ValueError("Benchmarking requires trial_id")
        final_path = base_path / "benchmarking" / \
            context.environment / context.model / context.trial_id

    elif context.process_type == "final_training":
        # Format: spec_<spec_fp>_exec_<exec_fp>/v<variant>
        spec_exec_dir = f"spec_{context.spec_fp}_exec_{context.exec_fp}"
        variant_dir = f"v{context.variant}"
        final_path = base_path / "final_training" / context.environment / \
            context.model / spec_exec_dir / variant_dir

    elif context.process_type == "conversion":
        # Format: <parent_training_id>/conv_<conv_fp>
        conv_dir = f"conv_{context.conv_fp}"
        final_path = base_path / "conversion" / context.environment / \
            context.model / context.parent_training_id / conv_dir

    elif context.process_type == "best_configurations":
        # Format: <model>/spec_<spec_fp>
        spec_dir = f"spec_{context.spec_fp}"
        final_path = base_path / "cache" / "best_configurations" / context.model / spec_dir

    else:
        raise ValueError(f"Unknown process_type: {context.process_type}")

    # CRITICAL: Validate final path to prevent creating invalid files like '1.0.0'
    _validate_output_path(final_path, context.process_type)

    return final_path


def build_output_path(
    root_dir: Path,
    context: NamingContext,
    base_outputs: str = "outputs",
    config_dir: Optional[Path] = None
) -> Path:
    """
    Build output path following new centralized structure (v2).

    Paths are generated using patterns from config/paths.yaml (v2 patterns).
    This ensures path structures are configurable and maintainable.

    Path structures (from paths.yaml patterns):
    - HPO v2:
        outputs/hpo/{storage_env}/{model}/study_{study8}/trial_{trial8}/...
      (falls back to legacy trial_id-based layout when study/trial hashes
       are not available)
    - Benchmarking v2:
        outputs/benchmarking/{storage_env}/{model}/study_{study8}/trial_{trial8}/bench_{bench8}/...
      (falls back to legacy trial_id-based layout when hashes are not
       available)
    - Final training:
        outputs/final_training/{storage_env}/{model}/spec_{spec_fp}_exec_{exec_fp}/v{variant}/
    - Conversion:
        outputs/conversion/{storage_env}/{model}/{parent_training_id}/conv_{conv_fp}/
    - Best config:
        outputs/cache/best_configurations/{model}/spec_{spec_fp}/

    Args:
        root_dir: Project root directory.
        context: Naming context with all required information.
        base_outputs: Base outputs directory name (default: "outputs", overridden by config if available).
        config_dir: Configuration directory (default: root_dir / "config").

    Returns:
        Full path to output directory.
    """
    # Determine config directory
    if config_dir is None:
        config_dir = root_dir / "config"

    # Try to load paths config
    try:
        from orchestration.paths import load_paths_config
        paths_config = load_paths_config(config_dir)
    except Exception as e:
        logger.warning(
            f"Failed to load paths.yaml config: {e}. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Get base outputs from config (or use provided/default)
    base_outputs = paths_config.get("base", {}).get("outputs", base_outputs)
    base_path = root_dir / base_outputs

    # Map process_type to output category
    category_map = {
        "hpo": "hpo",
        "hpo_refit": "hpo",  # Refit is part of HPO output structure
        "benchmarking": "benchmarking",
        "final_training": "final_training",
        "conversion": "conversion",
        "best_configurations": "cache",  # Special case
    }

    category = category_map.get(context.process_type)
    if category is None:
        raise ValueError(f"Unknown process_type: {context.process_type}")

    # Get pattern key
    pattern_key = _get_pattern_key(context.process_type)
    if not pattern_key:
        logger.warning(
            f"No pattern key mapping for process_type: {context.process_type}. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Get pattern from config
    patterns = paths_config.get("patterns", {})
    pattern = patterns.get(pattern_key)

    if not pattern:
        logger.warning(
            f"Pattern '{pattern_key}' not found in paths.yaml. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Extract values from context
    # NOTE: storage_env defaults to environment in NamingContext.__post_init__
    values = {
        "environment": context.environment,
        "storage_env": getattr(context, "storage_env", context.environment),
        "model": context.model,
        "spec_fp": context.spec_fp or "",
        "exec_fp": context.exec_fp or "",
        "variant": context.variant,
        "trial_id": context.trial_id or "",
        "parent_training_id": context.parent_training_id or "",
        "conv_fp": context.conv_fp or "",
        # Optional short forms for HPO/benchmark v2 layouts
        "study8": (context.study_key_hash or "")[:8] if context.study_key_hash else "",
        "trial8": (context.trial_key_hash or "")[:8] if context.trial_key_hash else "",
        "bench8": (context.benchmark_config_hash or "")[
            :8
        ]
        if context.benchmark_config_hash
        else "",
    }

    # Resolve pattern by replacing placeholders
    resolved_pattern = pattern
    for key, value in values.items():
        resolved_pattern = resolved_pattern.replace(f"{{{key}}}", str(value))

    # Build final path
    if context.process_type == "best_configurations":
        # Special handling: cache/best_configurations/{model}/spec_{spec_fp}/
        # Pattern is relative to cache/best_configurations, not outputs
        final_path = base_path / "cache" / \
            "best_configurations" / Path(resolved_pattern)
    else:
        # Get output directory from config
        output_dir = paths_config.get("outputs", {}).get(category, category)
        # Handle nested paths (e.g., "spec_abc_exec_xyz/v1")
        # Split by "/" and create path components
        pattern_parts = resolved_pattern.split("/")
        base_output_path = base_path / output_dir / Path(*pattern_parts)

        # For hpo_refit, append "refit" subdirectory
        if context.process_type == "hpo_refit":
            final_path = base_output_path / "refit"
        else:
            final_path = base_output_path

    # CRITICAL: Validate final path to prevent creating invalid files like '1.0.0'
    _validate_output_path(final_path, context.process_type)

    return final_path


def build_parent_training_id(spec_fp: str, exec_fp: str, variant: int = 1) -> str:
    """
    Build parent training identifier for conversion.

    This creates a string identifier that can be used as parent_training_id
    in conversion contexts. The format matches the directory structure.

    Args:
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        variant: Variant number.

    Returns:
        Parent training identifier string.
    """
    return f"spec_{spec_fp}_exec_{exec_fp}/v{variant}"
