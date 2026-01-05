from __future__ import annotations

# NOTE:
# Azure-dependent modules (sweeps, training, conversion, runtime, selection)
# can trigger ImportError in environments without the Azure ML SDK (e.g., Colab).
# To keep local utilities usable, we lazily import these modules and ignore
# ImportError so that local_* helpers continue to work.

# Azure ML-dependent imports (optional)
try:
    from .sweep_jobs import (
        create_data_prep_job,
        create_data_version_job,
        create_training_pipeline,
    )
    from .training import (
        build_final_training_config,
        submit_training_job,
        prepare_training_input,
    )
    from .runtime import submit_data_pipeline_job
    from .selection.selection import select_best_configuration as select_production_configuration
    from .conversion_jobs import (
        register_model_and_get_name,
        create_scoring_job,
        register_online_endpoints,
    )
except ImportError:
    # Azure ML SDK not available; skip Azure-specific helpers.
    create_data_prep_job = None
    create_data_version_job = None
    create_training_pipeline = None
    submit_training_job = None
    prepare_training_input = None
    submit_data_pipeline_job = None
    select_production_configuration = None
    register_model_and_get_name = None
    create_scoring_job = None
    register_online_endpoints = None

# Local-only utilities (always available)
from .hpo.local_sweeps import (
    run_local_hpo_sweep,
    translate_search_space_to_optuna,
)
from .local_selection import (
    select_best_configuration_across_studies,
    extract_best_config_from_study,
    load_best_trial_from_disk,
)

__all__ = [
    # Azure helpers (may be None if Azure SDK missing)
    "create_search_space",
    "create_dry_run_sweep_job_for_backbone",
    "create_hpo_sweep_job_for_backbone",
    "validate_sweep_job",
    "build_final_training_config",
    "create_final_training_job",
    "validate_final_training_job",
    "submit_and_wait_for_job",
    "select_best_configuration",
    "get_checkpoint_output_from_training_job",
    "create_conversion_job",
    "validate_conversion_job",
    # Local helpers (always available)
    "run_local_hpo_sweep",
    "translate_search_space_to_optuna",
    "select_best_configuration_across_studies",
    "extract_best_config_from_study",
    "load_best_trial_from_disk",
]
