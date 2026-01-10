"""Command-line argument parsing for training script."""

import argparse

from common.shared.argument_parsing import (
    add_config_dir_argument,
    add_backbone_argument,
    add_training_hyperparameter_arguments,
    add_training_data_arguments,
    add_cross_validation_arguments,
)


def parse_training_arguments() -> argparse.Namespace:
    """Parse command-line arguments for training script."""
    parser = argparse.ArgumentParser(description="Train Resume NER model")
    
    # Data asset argument
    parser.add_argument(
        "--data-asset",
        type=str,
        required=True,
        help="Azure ML data asset path or local dataset path",
    )
    
    # Common arguments
    add_config_dir_argument(parser)
    add_backbone_argument(parser)
    add_training_hyperparameter_arguments(parser)
    add_training_data_arguments(parser)
    add_cross_validation_arguments(parser)
    
    return parser.parse_args()

