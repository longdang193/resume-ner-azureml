"""Display and run naming (single authority)."""

from .context import (
    NamingContext,
    create_naming_context,
)
from .context_tokens import build_token_values
from .display_policy import (
    format_run_name,
    load_naming_policy,
    parse_parent_training_id,
    validate_naming_policy,
    validate_run_name,
)
from .experiments import (
    build_aml_experiment_name,
    build_mlflow_experiment_name,
    get_stage_config,
)

# MLflow naming modules
from .mlflow.config import (
    get_auto_increment_config,
    get_index_config,
    get_naming_config,
    get_run_finder_config,
    get_tracking_config,
    load_mlflow_config,
)
from .mlflow.hpo_keys import (
    build_hpo_study_family_hash,
    build_hpo_study_family_key,
    build_hpo_study_key,
    build_hpo_study_key_hash,
    build_hpo_trial_key,
    build_hpo_trial_key_hash,
)
from .mlflow.refit_keys import compute_refit_protocol_fp
from .mlflow.run_keys import (
    build_counter_key,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
from .mlflow.run_names import build_mlflow_run_name
from .mlflow.tags import (
    build_mlflow_tags,
    sanitize_tag_value,
)
from .mlflow.tags_registry import (
    TagKeyError,
    TagsRegistry,
    load_tags_registry,
)

__all__ = [
    # Context
    "NamingContext",
    "create_naming_context",
    "build_token_values",
    # Display Policy
    "load_naming_policy",
    "format_run_name",
    "validate_naming_policy",
    "validate_run_name",
    "parse_parent_training_id",
    # Experiments
    "get_stage_config",
    "build_aml_experiment_name",
    "build_mlflow_experiment_name",
    # MLflow Config
    "load_mlflow_config",
    "get_naming_config",
    "get_index_config",
    "get_run_finder_config",
    "get_auto_increment_config",
    "get_tracking_config",
    # MLflow Run Keys
    "build_mlflow_run_key",
    "build_mlflow_run_key_hash",
    "build_counter_key",
    # MLflow Run Names
    "build_mlflow_run_name",
    # MLflow Tags
    "build_mlflow_tags",
    "sanitize_tag_value",
    # MLflow Tags Registry
    "TagKeyError",
    "TagsRegistry",
    "load_tags_registry",
    # MLflow HPO Keys
    "build_hpo_study_key",
    "build_hpo_study_key_hash",
    "build_hpo_study_family_key",
    "build_hpo_study_family_hash",
    "build_hpo_trial_key",
    "build_hpo_trial_key_hash",
    # MLflow Refit Keys
    "compute_refit_protocol_fp",
]

