"""Training execution infrastructure.

This module provides utilities for executing training as subprocesses.
"""

from .mlflow_setup import (
    create_training_mlflow_run,
    setup_mlflow_tracking_env,
)
from .subprocess_runner import (
    FoldConfig,
    MLflowConfig,
    TrialConfig,
    TrainingOptions,
    build_training_command,
    execute_training_subprocess,
    setup_training_environment,
    verify_training_environment,
)

__all__ = [
    "build_training_command",
    "setup_training_environment",
    "execute_training_subprocess",
    "verify_training_environment",
    "create_training_mlflow_run",
    "setup_mlflow_tracking_env",
    "TrainingOptions",
    "MLflowConfig",
    "FoldConfig",
    "TrialConfig",
]

