"""Stable orchestration identifiers shared across notebooks and scripts.

These are *not* behaviour knobs (those live in YAML), but naming/selection
constants that rarely change and are used in multiple places.
"""

# Stage names (must match keys under `stages:` in experiment config YAML)
STAGE_SMOKE = "smoke"
STAGE_HPO = "hpo"
STAGE_TRAINING = "training"

# Experiment selection (maps to config/experiment/<name>.yaml)
EXPERIMENT_NAME = "resume_ner_baseline"

# Model & registry naming
MODEL_NAME = "resume-ner-onnx"
PROD_STAGE = "prod"

# Job / display names
CONVERSION_JOB_NAME = "model-conversion"

# File and directory naming constants
METRICS_FILENAME = "metrics.json"
BENCHMARK_FILENAME = "benchmark.json"
CHECKPOINT_DIRNAME = "checkpoint"
OUTPUTS_DIRNAME = "outputs"
MLRUNS_DIRNAME = "mlruns"

# Default values (not in configs)
DEFAULT_RANDOM_SEED = 42
DEFAULT_K_FOLDS = 5
