"""Orchestrate benchmarking for best HPO trials.

This module provides utilities to run benchmarks on best trial checkpoints
from HPO runs, handling path resolution, checkpoint selection, and backup.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

from shared.logging_utils import get_logger
from orchestration.benchmark_utils import run_benchmarking
from orchestration.path_resolution import (
    resolve_output_path_for_colab,
    validate_path_before_mkdir,
)
from orchestration.naming_centralized import create_naming_context, build_output_path

logger = get_logger(__name__)

# Constants
CHECKPOINT_DIRNAME = "checkpoint"
# BENCHMARK_FILENAME is now loaded from benchmark_config["output"]["filename"]
# Default fallback value
DEFAULT_BENCHMARK_FILENAME = "benchmark.json"


def find_checkpoint_in_trial_dir(trial_dir: Path) -> Optional[Path]:
    """
    Find checkpoint directory in trial directory.

    Prefers:
    1. refit/checkpoint/ (if refit training completed)
    2. cv/foldN/checkpoint/ (best CV fold based on metrics)
    3. checkpoint/ (fallback)

    Args:
        trial_dir: Path to trial directory

    Returns:
        Path to checkpoint directory, or None if not found
    """
    if not trial_dir.exists():
        logger.warning(f"Trial directory does not exist: {trial_dir}")
        return None

    # Log what's actually in the directory for debugging
    try:
        contents = [item.name for item in trial_dir.iterdir()]
        logger.warning(f"Trial directory {trial_dir} contains: {contents}")
    except Exception as e:
        logger.warning(f"Could not list trial directory contents: {e}")

    # 1. Check for refit checkpoint
    refit_checkpoint = trial_dir / "refit" / CHECKPOINT_DIRNAME
    logger.warning(
        f"Checking refit checkpoint at: {refit_checkpoint} (exists: {refit_checkpoint.exists()})")
    if refit_checkpoint.exists():
        logger.warning(f"Found refit checkpoint: {refit_checkpoint}")
        return refit_checkpoint

    # 2. Check for CV fold checkpoints
    cv_dir = trial_dir / "cv"
    logger.warning(
        f"Checking CV directory at: {cv_dir} (exists: {cv_dir.exists()})")
    if cv_dir.exists():
        # Find all fold directories - handle both "fold0" and "fold_0" patterns
        fold_dirs = []
        for item in cv_dir.iterdir():
            if item.is_dir():
                # Match patterns: fold0, fold_0, fold1, fold_1, etc.
                import re
                if re.match(r"fold_?\d+", item.name):
                    fold_dirs.append(item)

        if fold_dirs:
            logger.warning(
                f"Found {len(fold_dirs)} fold directories in {cv_dir}: {[d.name for d in fold_dirs]}")
            # Try to find the best fold by looking for metrics.json
            best_fold = None
            best_score = None

            for fold_dir in fold_dirs:
                metrics_file = fold_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        import json
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                        # Look for macro-f1 or first numeric metric
                        score = metrics.get("macro-f1")
                        if score is None:
                            # Try to find first numeric value
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    score = value
                                    break

                        if score is not None and (best_score is None or score > best_score):
                            best_score = score
                            best_fold = fold_dir
                    except Exception as e:
                        logger.debug(
                            f"Error reading metrics from {metrics_file}: {e}")
                        continue

            # Use best fold if found, otherwise use first fold
            if best_fold:
                checkpoint = best_fold / CHECKPOINT_DIRNAME
                logger.warning(
                    f"Checking best fold {best_fold.name}: checkpoint at {checkpoint} (exists: {checkpoint.exists()})")
                if checkpoint.exists():
                    logger.warning(
                        f"Found checkpoint in best fold: {checkpoint}")
                    return checkpoint

            # Fallback: try all folds in order
            for fold_dir in fold_dirs:
                checkpoint = fold_dir / CHECKPOINT_DIRNAME
                logger.warning(
                    f"Checking fold {fold_dir.name}: checkpoint at {checkpoint} (exists: {checkpoint.exists()})")
                if checkpoint.exists():
                    logger.warning(f"Found checkpoint in fold: {checkpoint}")
                    return checkpoint
        else:
            logger.warning(f"No fold directories found in {cv_dir}")
    else:
        logger.warning(f"CV directory does not exist: {cv_dir}")

    # 3. Fallback: check root checkpoint
    root_checkpoint = trial_dir / CHECKPOINT_DIRNAME
    logger.warning(
        f"Checking root checkpoint at: {root_checkpoint} (exists: {root_checkpoint.exists()})")
    if root_checkpoint.exists():
        logger.warning(f"Found root checkpoint: {root_checkpoint}")
        return root_checkpoint

    return None


def compute_grouping_tags(
    trial_info: Dict[str, Any],
    data_config: dict,
    hpo_config: dict,
    benchmark_config: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Compute grouping tags (study_key_hash, trial_key_hash, study_family_hash) for a trial.

    Args:
        trial_info: Trial information dict containing hyperparameters
        data_config: Data configuration dict
        hpo_config: HPO configuration dict
        benchmark_config: Optional benchmark configuration dict

    Returns:
        Tuple of (study_key_hash, trial_key_hash, study_family_hash)
    """
    study_key_hash = None
    trial_key_hash = None
    study_family_hash = None

    try:
        from orchestration.jobs.tracking.naming.hpo_keys import (
            build_hpo_study_key,
            build_hpo_study_key_hash,
            build_hpo_study_family_key,
            build_hpo_study_family_hash,
            build_hpo_trial_key,
            build_hpo_trial_key_hash,
        )

        # Get hyperparameters from trial_info
        hyperparameters = trial_info.get("hyperparameters", {})
        backbone = trial_info.get("backbone", "unknown")
        backbone_name = backbone.split("-")[0] if "-" in backbone else backbone

        if hyperparameters and data_config and hpo_config:
            # Compute study_key_hash
            study_key = build_hpo_study_key(
                data_config=data_config,
                hpo_config=hpo_config,
                model=backbone_name,
                benchmark_config=benchmark_config,
            )
            study_key_hash = build_hpo_study_key_hash(study_key)

            # Compute study_family_hash (optional, for cross-model comparison)
            study_family_key = build_hpo_study_family_key(
                data_config=data_config,
                hpo_config=hpo_config,
                benchmark_config=benchmark_config,
            )
            study_family_hash = build_hpo_study_family_hash(study_family_key)

            # Compute trial_key_hash
            trial_key = build_hpo_trial_key(
                study_key_hash=study_key_hash,
                hyperparameters=hyperparameters,
            )
            trial_key_hash = build_hpo_trial_key_hash(trial_key)

            logger.debug(
                f"Computed grouping tags: study_key_hash={study_key_hash[:16]}..., "
                f"trial_key_hash={trial_key_hash[:16]}..."
            )
        else:
            missing = []
            if not hyperparameters:
                missing.append("hyperparameters")
            if not data_config:
                missing.append("data_config")
            if not hpo_config:
                missing.append("hpo_config")
            logger.warning(
                f"Cannot compute grouping tags (missing: {', '.join(missing)})")
    except Exception as e:
        logger.warning(
            f"Could not compute grouping tags locally: {e}", exc_info=True)

    return study_key_hash, trial_key_hash, study_family_hash


def benchmark_best_trials(
    best_trials: Dict[str, Dict[str, Any]],
    test_data_path: Path,
    root_dir: Path,
    environment: str,
    data_config: dict,
    hpo_config: dict,
    benchmark_config: Optional[dict] = None,
    benchmark_batch_sizes: List[int] = None,
    benchmark_iterations: int = 100,
    benchmark_warmup: int = 10,
    benchmark_max_length: int = 512,
    benchmark_device: Optional[str] = None,
    benchmark_tracker: Optional[Any] = None,
    backup_enabled: bool = True,
    backup_to_drive: Optional[Callable[[Path, bool], bool]] = None,
    ensure_restored_from_drive: Optional[Callable[[Path, bool], bool]] = None,
) -> Dict[str, Path]:
    """
    Run benchmarking on best trial checkpoints from HPO runs.

    Args:
        best_trials: Dictionary mapping backbone names to trial info dicts
        test_data_path: Path to test data JSON file
        root_dir: Root directory of the project
        environment: Platform environment (local, colab, kaggle)
        data_config: Data configuration dict
        hpo_config: HPO configuration dict
        benchmark_config: Optional benchmark configuration dict
        benchmark_batch_sizes: List of batch sizes to test (default: [1, 8, 16])
        benchmark_iterations: Number of iterations per batch size (default: 100)
        benchmark_warmup: Number of warmup iterations (default: 10)
        benchmark_max_length: Maximum sequence length (default: 512)
        benchmark_device: Device to use (None = auto-detect)
        benchmark_tracker: Optional MLflowBenchmarkTracker instance
        backup_enabled: Whether backup is enabled
        backup_to_drive: Function to backup files to Drive
        ensure_restored_from_drive: Function to restore files from Drive

    Returns:
        Dictionary mapping backbone names to benchmark output paths
    """
    if benchmark_batch_sizes is None:
        benchmark_batch_sizes = [1, 8, 16]

    if not test_data_path or not test_data_path.exists():
        logger.info("Skipping benchmarking (test data not available)")
        return {}

    benchmark_results = {}

    for backbone, trial_info in best_trials.items():
        
        # Use checkpoint_dir from trial_info if available (from load_best_trial_from_disk)
        # This handles new structure: refit/checkpoint/ or cv/foldN/checkpoint/
        if "checkpoint_dir" in trial_info and trial_info["checkpoint_dir"] is not None:
            checkpoint_dir = Path(trial_info["checkpoint_dir"])
        else:
            # Try to find checkpoint in trial_dir
            if "trial_dir" in trial_info and trial_info["trial_dir"] is not None:
                trial_dir = Path(trial_info["trial_dir"])
                
                if not trial_dir.exists():
                    logger.warning(f"[BENCHMARK_BEST_TRIALS] Trial directory does not exist: {trial_dir}")
                
                checkpoint_dir = find_checkpoint_in_trial_dir(trial_dir)
                if checkpoint_dir is None:
                    logger.warning(
                        f"Checkpoint directory not found in {trial_dir} for {backbone} "
                        f"{trial_info.get('trial_name', 'unknown')}. "
                        f"Tried: refit/checkpoint/, cv/foldN/checkpoint/, checkpoint/"
                    )
                    continue
            else:
                logger.warning(
                    f"Trial directory not found for {backbone} {trial_info.get('trial_name', 'unknown')}. "
                    f"Skipping benchmark."
                )
                continue

        backbone_name = backbone.split("-")[0] if "-" in backbone else backbone

        trial_id_raw = trial_info.get(
            "trial_id") or trial_info.get("trial_name", "unknown")
        if trial_id_raw.startswith("trial_"):
            trial_id = trial_id_raw[6:]
        else:
            trial_id = trial_id_raw

        # Compute grouping tags BEFORE building path (needed for hash-based path)
        study_key_hash, trial_key_hash, study_family_hash = compute_grouping_tags(
            trial_info, data_config, hpo_config, benchmark_config
        )

        # Compute benchmark_config_hash if benchmark_config is available
        benchmark_config_hash = None
        if benchmark_config:
            try:
                import hashlib
                import json
                # Normalize and hash benchmark_config for stable identity
                normalized_config = json.dumps(benchmark_config, sort_keys=True, separators=(',', ':'))
                benchmark_config_hash = hashlib.sha256(normalized_config.encode('utf-8')).hexdigest()
            except Exception as e:
                logger.debug(f"Could not compute benchmark_config_hash: {e}")

        # Build benchmarking output path with hashes
        benchmarking_context = create_naming_context(
            process_type="benchmarking",
            model=backbone_name,
            trial_id=trial_id,
            environment=environment,
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
            benchmark_config_hash=benchmark_config_hash,
        )
        benchmarking_path = build_output_path(root_dir, benchmarking_context)
        # Redirect to Drive on Colab for persistence (similar to checkpoints)
        benchmarking_path = resolve_output_path_for_colab(benchmarking_path)
        # Validate path before creating directory
        benchmarking_path = validate_path_before_mkdir(
            benchmarking_path, context="benchmarking directory"
        )
        benchmarking_path.mkdir(parents=True, exist_ok=True)

        # Get output filename from config, with fallback to default
        output_filename = DEFAULT_BENCHMARK_FILENAME
        if benchmark_config and "output" in benchmark_config:
            output_filename = benchmark_config["output"].get("filename", DEFAULT_BENCHMARK_FILENAME)
        
        benchmark_output = benchmarking_path / output_filename

        if not checkpoint_dir.exists():
            logger.warning(
                f"Checkpoint not found for {backbone} {trial_info['trial_name']} at {checkpoint_dir}"
            )
            # Log trial directory contents for debugging
            if "trial_dir" in trial_info and trial_info["trial_dir"]:
                trial_dir = Path(trial_info["trial_dir"])
                if trial_dir.exists():
                    try:
                        contents = [
                            item.name for item in trial_dir.iterdir() if item.is_dir()]
                        logger.debug(
                            f"Trial directory {trial_dir} contains: {contents}")
                    except Exception as e:
                        logger.debug(f"Could not list trial directory: {e}")
            continue

        # Check if benchmark already exists (handle both local and Drive paths)
        if str(benchmark_output).startswith("/content/drive"):
            # File is in Drive - check directly
            if benchmark_output.exists():
                logger.info(
                    f"Benchmark results already exist in Drive - "
                    f"skipping benchmarking for {backbone}"
                )
                benchmark_results[backbone] = benchmark_output
                continue
        else:
            # File is local - check and restore from Drive if needed
            if ensure_restored_from_drive and ensure_restored_from_drive(
                benchmark_output, is_directory=False
            ):
                logger.info(
                    f"Restored benchmark results from Drive - "
                    f"skipping benchmarking for {backbone}"
                )
                benchmark_results[backbone] = benchmark_output
                continue

        logger.info(f"Benchmarking {backbone} ({trial_info['trial_name']})...")

        # Extract parent run IDs from trial_info (may be None if not available)
        hpo_trial_run_id = trial_info.get("trial_run_id")
        hpo_refit_run_id = trial_info.get("refit_run_id") or trial_info.get("run_id")  # MLflow selection uses "run_id" for refit
        hpo_sweep_run_id = trial_info.get("sweep_run_id")

        # If run IDs are missing or invalid (timestamps), try to query MLflow using trial_key_hash
        import re
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        # Validate and potentially look up run IDs from MLflow
        if trial_key_hash:
            # Check if we need to look up trial run
            if not hpo_trial_run_id or not uuid_pattern.match(hpo_trial_run_id):
                try:
                    import mlflow
                    client = mlflow.tracking.MlflowClient()
                    
                    # Try to get experiment ID from benchmark tracker if available
                    experiment_ids_to_search = None
                    if benchmark_tracker and hasattr(benchmark_tracker, 'experiment_name'):
                        try:
                            # Strategy 1: Try to find HPO experiment by study_key_hash (most reliable)
                            if study_key_hash:
                                try:
                                    # Search for any run with this study_key_hash to find the experiment
                                    temp_runs = client.search_runs(
                                        filter_string=f"tags.code.study_key_hash = '{study_key_hash}'",
                                        max_results=1
                                    )
                                    if temp_runs:
                                        experiment_ids_to_search = [temp_runs[0].info.experiment_id]
                                        logger.info(
                                            f"[BENCHMARK] Found HPO experiment via study_key_hash: "
                                            f"experiment_id={experiment_ids_to_search[0]}"
                                        )
                                except Exception as e:
                                    logger.debug(f"Could not find HPO experiment via study_key_hash: {e}")
                            
                            # Strategy 2: Fallback to name-based search
                            if experiment_ids_to_search is None:
                                # Search in HPO experiment (parent of benchmark experiment)
                                # Benchmark experiment name is usually "{name}-benchmark"
                                # HPO experiment name is usually "{name}"
                                hpo_experiment_name = benchmark_tracker.experiment_name.replace("-benchmark", "")
                                hpo_experiment = mlflow.get_experiment_by_name(hpo_experiment_name)
                                if hpo_experiment:
                                    experiment_ids_to_search = [hpo_experiment.experiment_id]
                                    logger.info(
                                        f"[BENCHMARK] Found HPO experiment by name: {hpo_experiment_name} "
                                        f"(experiment_id={hpo_experiment.experiment_id})"
                                    )
                                else:
                                    logger.debug(
                                        f"[BENCHMARK] HPO experiment '{hpo_experiment_name}' not found, searching all experiments"
                                    )
                        except Exception as e:
                            logger.debug(f"Could not find HPO experiment by name: {e}")
                    
                    # Search for trial run by trial_key_hash (CV trial, not refit)
                    # Try multiple filter strategies to find the trial run
                    filter_strings = [
                        f"tags.code.trial_key_hash = '{trial_key_hash}' AND tags.code.stage = 'hpo'",
                        f"tags.code.trial_key_hash = '{trial_key_hash}'",
                    ]
                    runs = None
                    for filter_str in filter_strings:
                        try:
                            # MLflow requires experiment_ids as positional arg - use [] to search all experiments
                            runs = client.search_runs(
                                experiment_ids=experiment_ids_to_search if experiment_ids_to_search is not None else [],
                                filter_string=filter_str,
                                max_results=5  # Get more results to filter manually
                            )
                            
                            if runs:
                                # Filter out refit runs manually (keep only hpo stage, exclude hpo_refit)
                                runs = [r for r in runs if r.data.tags.get("code.stage") == "hpo"]
                                if runs:
                                    break
                        except Exception as e:
                            logger.debug(f"Filter '{filter_str}' failed: {e}")
                            continue
                    
                    if runs:
                        trial_run = runs[0]
                        hpo_trial_run_id = trial_run.info.run_id
                        logger.info(
                            f"[BENCHMARK] Found trial run ID from MLflow: {hpo_trial_run_id[:12]}... "
                            f"(via trial_key_hash={trial_key_hash[:16]}...)"
                        )
                    else:
                        # Try fallback: search by trial_id if available
                        # Extract trial_id from trial_name (e.g., "trial_0_20260105_122905" -> "trial_0_20260105_122905")
                        trial_id = trial_info.get("trial_id") or trial_info.get("trial_name")
                        if trial_id:
                            try:
                                logger.info(
                                    f"[BENCHMARK] Trying fallback search by trial_id={trial_id}..."
                                )
                                fallback_runs = client.search_runs(
                                    experiment_ids=experiment_ids_to_search if experiment_ids_to_search is not None else [],
                                    filter_string=f"tags.code.trial_id = '{trial_id}' AND tags.code.stage = 'hpo'",
                                    max_results=1
                                )
                                if fallback_runs:
                                    trial_run = fallback_runs[0]
                                    hpo_trial_run_id = trial_run.info.run_id
                                    logger.info(
                                        f"[BENCHMARK] Found trial run ID via trial_id fallback: {hpo_trial_run_id[:12]}..."
                                    )
                                else:
                                    logger.warning(
                                        f"[BENCHMARK] Could not find trial run in MLflow for trial_key_hash={trial_key_hash[:16]}... "
                                        f"or trial_id={trial_id} (searched {len(experiment_ids_to_search) if experiment_ids_to_search else 'all'} experiments)"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"[BENCHMARK] Fallback search by trial_id failed: {e}"
                                )
                        else:
                            logger.warning(
                                f"[BENCHMARK] Could not find trial run in MLflow for trial_key_hash={trial_key_hash[:16]}... "
                                f"(searched {len(experiment_ids_to_search) if experiment_ids_to_search else 'all'} experiments, no trial_id for fallback)"
                            )
                except Exception as e:
                    logger.warning(f"Could not query MLflow for trial run ID: {e}", exc_info=True)

            # Check if we need to look up refit run
            if not hpo_refit_run_id or not uuid_pattern.match(hpo_refit_run_id):
                try:
                    import mlflow
                    client = mlflow.tracking.MlflowClient()
                    
                    # Try to get experiment ID from benchmark tracker if available
                    experiment_ids_to_search = None
                    if benchmark_tracker and hasattr(benchmark_tracker, 'experiment_name'):
                        try:
                            hpo_experiment_name = benchmark_tracker.experiment_name.replace("-benchmark", "")
                            hpo_experiment = mlflow.get_experiment_by_name(hpo_experiment_name)
                            if hpo_experiment:
                                experiment_ids_to_search = [hpo_experiment.experiment_id]
                        except Exception:
                            pass
                    
                    # Try to find refit run (same trial_key_hash with refit tag)
                    # MLflow requires experiment_ids as positional arg - use [] to search all experiments
                    refit_runs = client.search_runs(
                        experiment_ids=experiment_ids_to_search if experiment_ids_to_search is not None else [],
                        filter_string=f"tags.code.trial_key_hash = '{trial_key_hash}' AND tags.code.stage = 'hpo_refit'",
                        max_results=1
                    )
                    if refit_runs:
                        refit_run = refit_runs[0]
                        hpo_refit_run_id = refit_run.info.run_id
                        logger.info(
                            f"[BENCHMARK] Found refit run ID from MLflow: {hpo_refit_run_id[:12]}..."
                        )
                    else:
                        logger.debug(
                            f"[BENCHMARK] Could not find refit run in MLflow for trial_key_hash={trial_key_hash[:16]}..."
                        )
                except Exception as e:
                    logger.warning(f"Could not query MLflow for refit run ID: {e}", exc_info=True)

        logger.info(
            f"[BENCHMARK] Final run IDs: trial={hpo_trial_run_id[:12] if hpo_trial_run_id else None}..., "
            f"refit={hpo_refit_run_id[:12] if hpo_refit_run_id else None}..., "
            f"sweep={hpo_sweep_run_id[:12] if hpo_sweep_run_id else None}..."
        )

        success = run_benchmarking(
            checkpoint_dir=checkpoint_dir,
            test_data_path=test_data_path,
            output_path=benchmark_output,
            batch_sizes=benchmark_batch_sizes,
            iterations=benchmark_iterations,
            warmup_iterations=benchmark_warmup,
            max_length=benchmark_max_length,
            device=benchmark_device,
            project_root=root_dir,
            tracker=benchmark_tracker,
            backbone=backbone,
            benchmark_source="hpo_trial",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
            hpo_trial_run_id=hpo_trial_run_id,
            hpo_refit_run_id=hpo_refit_run_id,
            hpo_sweep_run_id=hpo_sweep_run_id,
            benchmark_config_hash=benchmark_config_hash,
        )

        if success:
            benchmark_results[backbone] = benchmark_output
            logger.info(f"Benchmark completed: {benchmark_output}")

            # Note: On Colab, benchmark_output is already in Drive (via resolve_output_path_for_colab)
            # No need to backup again unless it's a local path
            if backup_enabled and backup_to_drive and not str(benchmark_output).startswith("/content/drive"):
                backup_to_drive(benchmark_output, is_directory=False)
                logger.info("Backed up benchmark results to Drive")
            elif str(benchmark_output).startswith("/content/drive"):
                logger.info(
                    "Benchmark results are already in Drive (no backup needed)")
        else:
            logger.error(f"Benchmark failed for {backbone}")

    logger.info(
        f"Benchmarking complete. {len(benchmark_results)}/{len(best_trials)} trials benchmarked."
    )
    return benchmark_results
