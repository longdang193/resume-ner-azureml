"""Legacy orchestration.jobs module facade.

This module provides backward compatibility by re-exporting functions from
the new modular structure. All imports from this module are deprecated.

New module locations:
- training_exec/jobs.py: Training job creation
- hpo/execution/azureml/sweeps.py: HPO sweep job creation
- conversion/jobs.py: Conversion job creation
- azureml/jobs.py: Azure ML job utilities
- hpo/exceptions.py: HPO exceptions
- training_exec/tags.py: Training tags
- selection/local_selection_v2.py: Improved selection logic
- tracking/mlflow/trackers/: MLflow trackers
- tracking/mlflow/finder.py: Run finder
"""

import warnings
from typing import Any

# Issue deprecation warning
warnings.warn(
    "Importing from 'orchestration.jobs' is deprecated. "
    "Please import from the new module locations instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Training job creation - moved to training_exec/jobs.py
try:
    from training_exec.jobs import (
        build_final_training_config,
        submit_training_job,
        prepare_training_input,
    )
except ImportError:
    build_final_training_config = None
    submit_training_job = None
    prepare_training_input = None

# HPO sweep job creation - moved to hpo/execution/azureml/sweeps.py
try:
    from hpo.execution.azureml.sweeps import (
        create_search_space,
        create_dry_run_sweep_job_for_backbone,
        create_hpo_sweep_job_for_backbone,
        validate_sweep_job,
    )
except ImportError:
    create_search_space = None
    create_dry_run_sweep_job_for_backbone = None
    create_hpo_sweep_job_for_backbone = None
    validate_sweep_job = None

# Conversion job creation - moved to conversion/jobs.py
try:
    from conversion.azureml import (
        get_checkpoint_output_from_training_job,
        create_conversion_job,
        validate_conversion_job,
    )
except ImportError:
    get_checkpoint_output_from_training_job = None
    create_conversion_job = None
    validate_conversion_job = None

# Azure ML job utilities - moved to azureml/jobs.py
try:
    from azureml.jobs import submit_and_wait_for_job
except ImportError:
    submit_and_wait_for_job = None

# HPO exceptions - moved to hpo/exceptions.py
try:
    from hpo.exceptions import (
        HPOError,
        TrialExecutionError,
        SelectionError,
        MLflowTrackingError,
        StudyLoadError,
        MetricsReadError,
    )
except ImportError:
    HPOError = None
    TrialExecutionError = None
    SelectionError = None
    MLflowTrackingError = None
    StudyLoadError = None
    MetricsReadError = None

# Training tags - moved to training_exec/tags.py
try:
    from training_exec.tags import apply_lineage_tags
except ImportError:
    apply_lineage_tags = None

# Improved selection - moved to selection/local_selection_v2.py
try:
    from selection.local_selection_v2 import (
        find_study_folder_by_config,
        load_best_trial_from_study_folder,
        write_active_study_marker,
        find_trial_checkpoint_by_hash,
    )
except ImportError:
    find_study_folder_by_config = None
    load_best_trial_from_study_folder = None
    write_active_study_marker = None
    find_trial_checkpoint_by_hash = None

# MLflow trackers - moved to tracking/mlflow/trackers/
try:
    from tracking.mlflow.trackers import (
        BaseTracker,
        MLflowSweepTracker,
        MLflowBenchmarkTracker,
        MLflowTrainingTracker,
        MLflowConversionTracker,
    )
except ImportError:
    BaseTracker = None
    MLflowSweepTracker = None
    MLflowBenchmarkTracker = None
    MLflowTrainingTracker = None
    MLflowConversionTracker = None

# Run finder - moved to tracking/mlflow/finder.py
try:
    from tracking.mlflow.finder import (
        find_mlflow_run,
        find_run_by_trial_id,
    )
except ImportError:
    find_mlflow_run = None
    find_run_by_trial_id = None

# Selection module - moved to selection/
# Create proper module objects for backward compatibility
import sys
from types import ModuleType

# Create orchestration.jobs.selection module
if "orchestration.jobs.selection" not in sys.modules:
    selection_module = ModuleType("orchestration.jobs.selection")
    sys.modules["orchestration.jobs.selection"] = selection_module
    
    # Import and assign submodules
    try:
        from selection import mlflow_selection
        selection_module.mlflow_selection = mlflow_selection
        sys.modules["orchestration.jobs.selection.mlflow_selection"] = mlflow_selection
    except ImportError:
        pass
    
    try:
        from selection import artifact_acquisition
        selection_module.artifact_acquisition = artifact_acquisition
        sys.modules["orchestration.jobs.selection.artifact_acquisition"] = artifact_acquisition
    except ImportError:
        pass
    
    try:
        from selection import cache
        selection_module.cache = cache
        sys.modules["orchestration.jobs.selection.cache"] = cache
    except ImportError:
        pass

# HPO trial metrics - moved to hpo/trial/metrics.py
try:
    from hpo.trial.metrics import (
        read_trial_metrics,
        parse_metrics_file,
    )
except ImportError:
    # Create a fake module for backward compatibility
    class _HPOTrialMetricsModule:
        def __getattr__(self, name):
            raise AttributeError(f"module 'orchestration.jobs.hpo.local.trial.metrics' has no attribute '{name}'. "
                               f"Please import from 'hpo.trial.metrics' instead.")
    if "orchestration.jobs.hpo.local.trial.metrics" not in sys.modules:
        sys.modules["orchestration.jobs.hpo.local.trial.metrics"] = _HPOTrialMetricsModule()

# Benchmarking orchestrator - moved to benchmarking/orchestrator.py
if "orchestration.jobs.benchmarking" not in sys.modules:
    benchmarking_module = ModuleType("orchestration.jobs.benchmarking")
    sys.modules["orchestration.jobs.benchmarking"] = benchmarking_module
    
    try:
        from benchmarking.orchestrator import BenchmarkOrchestrator
        # Create a module-like object that contains the class
        orchestrator_module = ModuleType("orchestration.jobs.benchmarking.orchestrator")
        orchestrator_module.BenchmarkOrchestrator = BenchmarkOrchestrator
        # Also add functions that might be imported
        try:
            from benchmarking.orchestrator import run_benchmarking
            orchestrator_module.run_benchmarking = run_benchmarking
        except ImportError:
            pass
        benchmarking_module.orchestrator = orchestrator_module
        sys.modules["orchestration.jobs.benchmarking.orchestrator"] = orchestrator_module
    except ImportError:
        pass

__all__ = [
    # Training
    "build_final_training_config",
    "submit_training_job",
    "prepare_training_input",
    # HPO sweeps
    "create_search_space",
    "create_dry_run_sweep_job_for_backbone",
    "create_hpo_sweep_job_for_backbone",
    "validate_sweep_job",
    # Conversion
    "get_checkpoint_output_from_training_job",
    "create_conversion_job",
    "validate_conversion_job",
    # Azure ML
    "submit_and_wait_for_job",
    # Exceptions
    "HPOError",
    "TrialExecutionError",
    "SelectionError",
    "MLflowTrackingError",
    "StudyLoadError",
    "MetricsReadError",
    # Tags
    "apply_lineage_tags",
    # Selection
    "find_study_folder_by_config",
    "load_best_trial_from_study_folder",
    "write_active_study_marker",
    "find_trial_checkpoint_by_hash",
    # Trackers
    "BaseTracker",
    "MLflowSweepTracker",
    "MLflowBenchmarkTracker",
    "MLflowTrainingTracker",
    "MLflowConversionTracker",
    # Finder
    "find_mlflow_run",
    "find_run_by_trial_id",
]
