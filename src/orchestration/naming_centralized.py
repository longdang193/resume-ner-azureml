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

    Attributes:
        process_type: Type of process (hpo, benchmarking, final_training, conversion).
        model: Model backbone name (e.g., "distilbert").
        environment: Execution environment (local, colab, kaggle, azure).
        spec_fp: Specification fingerprint (platform-independent experiment identity).
        exec_fp: Execution fingerprint (toolchain/runtime identity).
        variant: Variant number for final_training (default 1, increments for force_new).
        trial_id: Trial identifier for HPO/benchmarking (e.g., "trial_1_20251229_100000").
        parent_training_id: Parent training identifier for conversion.
        conv_fp: Conversion fingerprint for conversion variants.
    """
    process_type: str
    model: str
    environment: str
    spec_fp: Optional[str] = None
    exec_fp: Optional[str] = None
    variant: int = 1
    trial_id: Optional[str] = None
    parent_training_id: Optional[str] = None
    conv_fp: Optional[str] = None

    def __post_init__(self):
        """Validate context after initialization."""
        valid_processes = {"hpo", "hpo_refit", "benchmarking",
                           "final_training", "conversion", "best_configurations"}
        valid_environments = {"local", "colab", "kaggle", "azure"}

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
    variant: int = 1,
    trial_id: Optional[str] = None,
    parent_training_id: Optional[str] = None,
    conv_fp: Optional[str] = None
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
        parent_training_id: Parent training identifier for conversion.
        conv_fp: Conversion fingerprint (required for conversion).

    Returns:
        NamingContext instance.
    """
    if environment is None:
        environment = detect_platform()

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
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=variant,
        trial_id=trial_id,
        parent_training_id=parent_training_id,
        conv_fp=conv_fp
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
    - HPO: outputs/hpo/{environment}/{model}/trial_{trial_id}/
    - Benchmarking: outputs/benchmarking/{environment}/{model}/trial_{trial_id}/
    - Final training: outputs/final_training/{environment}/{model}/spec_{spec_fp}_exec_{exec_fp}/v{variant}/
    - Conversion: outputs/conversion/{environment}/{model}/{parent_training_id}/conv_{conv_fp}/
    - Best config: outputs/cache/best_configurations/{model}/spec_{spec_fp}/

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
    values = {
        "environment": context.environment,
        "model": context.model,
        "spec_fp": context.spec_fp or "",
        "exec_fp": context.exec_fp or "",
        "variant": context.variant,
        "trial_id": context.trial_id or "",
        "parent_training_id": context.parent_training_id or "",
        "conv_fp": context.conv_fp or "",
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
