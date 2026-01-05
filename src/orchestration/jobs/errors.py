"""Custom exceptions for HPO and orchestration jobs."""


class HPOError(Exception):
    """Base exception for HPO-related errors."""
    pass


class TrialExecutionError(HPOError):
    """Raised when a training trial execution fails."""
    pass


class SelectionError(HPOError):
    """Raised when configuration selection fails."""
    pass


class MLflowTrackingError(HPOError):
    """Raised when MLflow tracking operations fail."""
    pass


class StudyLoadError(HPOError):
    """Raised when loading an Optuna study from checkpoint fails."""
    pass


class MetricsReadError(HPOError):
    """Raised when reading metrics from trial output fails."""
    pass

