"""
Training script for Resume NER model.

Implements a minimal token-classification training/eval loop using transformers.

This module also acts as the central launcher for single- and multi-GPU
training. It decides whether to run in:

- single-process mode (CPU or single GPU), or
- multi-process DDP mode (one process per GPU)

based on `config/train.yaml` and the available hardware. Notebooks and higher-
level orchestration only ever call this entrypoint; they do not manage ranks.
"""

from .cli import parse_training_arguments
from .distributed_launcher import launch_training


def main() -> None:
    """Main training entry point."""
    args = parse_training_arguments()
    launch_training(args)


if __name__ == "__main__":
    main()


















