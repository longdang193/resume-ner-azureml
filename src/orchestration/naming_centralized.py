"""Centralized naming and path building with fingerprint-based identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.platform_detection import detect_platform


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
        valid_processes = {"hpo", "benchmarking", "final_training", "conversion", "best_configurations"}
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


def build_output_path(
    root_dir: Path,
    context: NamingContext,
    base_outputs: str = "outputs"
) -> Path:
    """
    Build output path following new centralized structure.
    
    Path structures:
    - HPO: outputs/hpo/<env>/<model>/trial_<n>_<ts>/
    - Benchmarking: outputs/benchmarking/<env>/<model>/trial_<n>_<ts>/
    - Final training: outputs/final_training/<env>/<model>/spec_<spec_fp>_exec_<exec_fp>/v<variant>/
    - Conversion: outputs/conversion/<env>/<model>/<parent_id>/conv_<conv_fp>/
    - Best config: outputs/cache/best_configurations/<model>/spec_<spec_fp>/
    
    Args:
        root_dir: Project root directory.
        context: Naming context with all required information.
        base_outputs: Base outputs directory name (default: "outputs").
    
    Returns:
        Full path to output directory.
    """
    base_path = root_dir / base_outputs
    
    if context.process_type == "hpo":
        if not context.trial_id:
            raise ValueError("HPO requires trial_id")
        return base_path / "hpo" / context.environment / context.model / context.trial_id
    
    elif context.process_type == "benchmarking":
        if not context.trial_id:
            raise ValueError("Benchmarking requires trial_id")
        return base_path / "benchmarking" / context.environment / context.model / context.trial_id
    
    elif context.process_type == "final_training":
        # Format: spec_<spec_fp>_exec_<exec_fp>/v<variant>
        spec_exec_dir = f"spec_{context.spec_fp}_exec_{context.exec_fp}"
        variant_dir = f"v{context.variant}"
        return base_path / "final_training" / context.environment / context.model / spec_exec_dir / variant_dir
    
    elif context.process_type == "conversion":
        # Format: <parent_training_id>/conv_<conv_fp>
        conv_dir = f"conv_{context.conv_fp}"
        return base_path / "conversion" / context.environment / context.model / context.parent_training_id / conv_dir
    
    elif context.process_type == "best_configurations":
        # Format: <model>/spec_<spec_fp>
        spec_dir = f"spec_{context.spec_fp}"
        return base_path / "cache" / "best_configurations" / context.model / spec_dir
    
    else:
        raise ValueError(f"Unknown process_type: {context.process_type}")


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

