"""Training module for Resume NER model."""

from .config import build_training_config
# Import build_label_list separately since it doesn't require torch
from data.loaders import build_label_list

# Lazy imports for functions that require torch
# These will be imported on-demand to avoid requiring torch at module level

__all__ = [
    "build_training_config",
    "load_dataset",
    "build_label_list",
    "ResumeNERDataset",
    "split_train_test",
    "save_split_files",
    "create_model_and_tokenizer",
    "train_model",
    "evaluate_model",
    "compute_metrics",
    "log_metrics",
    "set_seed",
    "resolve_training_checkpoint_path",
    "validate_checkpoint",
]


def __getattr__(name: str):
    """Lazy import for training functions that require torch."""
    if name == "load_dataset":
        from data.loaders import load_dataset
        return load_dataset
    elif name == "ResumeNERDataset":
        from data.loaders import ResumeNERDataset
        return ResumeNERDataset
    elif name == "split_train_test":
        from data.loaders import split_train_test
        return split_train_test
    elif name == "save_split_files":
        from data.loaders import save_split_files
        return save_split_files
    elif name == "create_model_and_tokenizer":
        from .model import create_model_and_tokenizer
        return create_model_and_tokenizer
    elif name == "train_model":
        from .trainer import train_model
        return train_model
    elif name == "evaluate_model":
        from .evaluator import evaluate_model
        return evaluate_model
    elif name == "compute_metrics":
        from .metrics import compute_metrics
        return compute_metrics
    elif name == "log_metrics":
        from .logging import log_metrics
        return log_metrics
    elif name == "set_seed":
        from .utils import set_seed
        return set_seed
    elif name == "resolve_training_checkpoint_path":
        from .checkpoint_loader import resolve_training_checkpoint_path
        return resolve_training_checkpoint_path
    elif name == "validate_checkpoint":
        from .checkpoint_loader import validate_checkpoint
        return validate_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

