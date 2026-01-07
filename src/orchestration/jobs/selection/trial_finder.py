"""Find best trials from HPO studies or disk.

This module provides utilities to locate and extract best trial information
from Optuna studies or from saved outputs on disk.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging_utils import get_logger

from ..hpo.study_extractor import extract_best_config_from_study
from .disk_loader import load_best_trial_from_disk
from orchestration.path_resolution import resolve_hpo_output_dir

logger = get_logger(__name__)


def find_best_trial_in_study_folder(
    study_folder: Path,
    objective_metric: str = "macro-f1",
) -> Optional[Dict[str, Any]]:
    """
    Find best trial in a specific study folder by reading metrics.json files.

    Supports both v2 paths (trial-{hash}) and legacy paths (trial_N).

    Args:
        study_folder: Path to study folder containing trials
        objective_metric: Name of the objective metric to optimize

    Returns:
        Dictionary with best trial info, or None if no trials found
    """
    logger.info(
        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Searching in study folder: {study_folder}")

    if not study_folder.exists():
        logger.warning(
            f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Study folder does not exist: {study_folder}")
        return None

    best_metric = None
    best_trial_dir = None
    best_trial_name = None

    # Collect all trial directories (support both v2 and legacy paths)
    trial_dirs = []
    for item in study_folder.iterdir():
        if item.is_dir() and (
            item.name.startswith("trial-") or
            item.name.startswith("trial_")
        ):
            trial_dirs.append(item)

    logger.info(
        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Found {len(trial_dirs)} trial directories: {[d.name for d in trial_dirs]}")

    if len(trial_dirs) == 0:
        logger.warning(
            f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] No trial directories found in {study_folder}. "
            f"Contents: {[item.name for item in study_folder.iterdir()]}"
        )

    # Find best trial by metrics
    trials_with_metrics = []
    for trial_dir in trial_dirs:
        # Try multiple locations for metrics.json
        # 1. Trial root: trial_dir/metrics.json
        # 2. CV folds: trial_dir/cv/fold0/metrics.json (for CV trials)
        metrics_file = None
        if (trial_dir / "metrics.json").exists():
            metrics_file = trial_dir / "metrics.json"
        elif (trial_dir / "cv").exists():
            # Check first fold for metrics (CV trials aggregate metrics at fold level)
            for fold_dir in (trial_dir / "cv").iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold"):
                    fold_metrics = fold_dir / "metrics.json"
                    if fold_metrics.exists():
                        metrics_file = fold_metrics
                        break

        if not metrics_file or not metrics_file.exists():
            logger.debug(
                f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Trial {trial_dir.name} has no metrics.json (checked root and cv/fold*), skipping")
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            if objective_metric in metrics:
                metric_value = metrics[objective_metric]

                # Skip fold-specific trials (we'll aggregate later if needed)
                if "_fold" in trial_dir.name:
                    logger.debug(
                        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Skipping fold-specific trial: {trial_dir.name}")
                    continue

                trials_with_metrics.append((trial_dir, metric_value))

                if best_metric is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_trial_dir = trial_dir
                    best_trial_name = trial_dir.name
                    logger.info(
                        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] New best trial: {trial_dir.name} "
                        f"({objective_metric}={metric_value:.4f}, exists: {trial_dir.exists()}, "
                        f"metrics_file={metrics_file.name})"
                    )
            else:
                logger.debug(
                    f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Trial {trial_dir.name} has metrics.json but no '{objective_metric}' key. "
                    f"Available metrics: {list(metrics.keys())}"
                )
        except Exception as e:
            logger.warning(
                f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Could not read {metrics_file}: {e}")
            continue

    if len(trials_with_metrics) == 0:
        logger.warning(
            f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] No trials found with {objective_metric} metric. "
            f"Found {len(trial_dirs)} trial directories but none have valid metrics.json with {objective_metric}"
        )

    if best_trial_dir is None:
        logger.warning(
            f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] No trials with {objective_metric} found in {study_folder}")
        return None

    logger.info(
        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Best trial found: {best_trial_dir.name} ({objective_metric}={best_metric:.4f})")

    # Load metrics
    metrics_file = best_trial_dir / "metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Try to read trial_meta.json for run_id
    trial_run_id = None
    trial_meta_path = best_trial_dir / "trial_meta.json"
    if trial_meta_path.exists():
        try:
            import re
            with open(trial_meta_path, "r") as f:
                trial_meta = json.load(f)
            if "run_id" in trial_meta:
                run_id_from_meta = trial_meta["run_id"]
                uuid_pattern = re.compile(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    re.IGNORECASE
                )
                if uuid_pattern.match(run_id_from_meta):
                    trial_run_id = run_id_from_meta
        except Exception as e:
            logger.debug(f"Could not read trial_meta.json: {e}")

    result = {
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "accuracy": best_metric,
        "metrics": metrics,
    }

    if trial_run_id:
        result["trial_run_id"] = trial_run_id

    logger.info(
        f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] Returning trial_dir: {result['trial_dir']} (exists: {best_trial_dir.exists()})")

    # Verify the trial_dir actually exists before returning
    if not best_trial_dir.exists():
        logger.error(
            f"[FIND_BEST_TRIAL_IN_STUDY_FOLDER] ERROR: Selected trial_dir does not exist: {best_trial_dir}. "
            f"Available trials in study_folder: {[d.name for d in trial_dirs]}. "
            f"Study folder path: {study_folder}. "
            f"Study folder exists: {study_folder.exists()}. "
            f"Study folder contents: {[item.name for item in study_folder.iterdir()] if study_folder.exists() else 'N/A'}"
        )
        # Don't return None - return the result anyway so the caller can see what was attempted
        # The caller should handle the non-existent path

    return result


def format_trial_identifier(trial_dir: Path, trial_number: Optional[int] = None) -> str:
    """Format trial identifier using hash-based naming if available, else fallback to directory name.

    Args:
        trial_dir: Path to trial directory
        trial_number: Optional trial number to include in identifier

    Returns:
        Formatted identifier string (e.g., "study-350a79aa, trial-9d4153fb, t1" or "trial_1_20260106_173735")
    """
    trial_meta_path = trial_dir / "trial_meta.json"
    if trial_meta_path.exists():
        try:
            with open(trial_meta_path, "r") as f:
                meta = json.load(f)
            study_key_hash = meta.get("study_key_hash")
            trial_key_hash = meta.get("trial_key_hash")
            meta_trial_number = meta.get("trial_number")

            # Use trial_number from meta if available, else use provided trial_number
            display_trial_number = meta_trial_number if meta_trial_number is not None else trial_number

            if study_key_hash and trial_key_hash:
                if display_trial_number is not None:
                    return f"study-{study_key_hash[:8]}, trial-{trial_key_hash[:8]}, t{display_trial_number}"
                else:
                    return f"study-{study_key_hash[:8]}, trial-{trial_key_hash[:8]}"
            elif display_trial_number is not None:
                return f"t{display_trial_number}"
        except Exception:
            pass

    # Fallback to directory name or trial number
    if trial_number is not None:
        return f"t{trial_number}"
    return trial_dir.name


def find_study_folder_in_backbone_dir(backbone_dir: Path) -> Optional[Path]:
    """
    Find study folder inside backbone directory.

    Supports both v2 paths (study-{study8}/trial-{trial8}) and legacy paths (hpo_{backbone}_*/trial_N).

    Args:
        backbone_dir: Backbone directory containing study folders

    Returns:
        Path to study folder if found, else None
    """
    logger.info(
        f"[FIND_STUDY_FOLDER] Searching in backbone_dir: {backbone_dir}")

    if not backbone_dir.exists():
        logger.warning(
            f"[FIND_STUDY_FOLDER] Backbone directory does not exist: {backbone_dir}")
        return None

    v2_folders = []
    legacy_folders = []

    for item in backbone_dir.iterdir():
        if not item.is_dir():
            continue

        # Check for v2 study folders (study-{hash})
        if item.name.startswith("study-") and len(item.name) > 7:
            # Check if it contains trial folders (v2: trial-{hash} or legacy: trial_N)
            has_trials = any(
                subitem.is_dir() and (
                    subitem.name.startswith(
                        "trial-") or subitem.name.startswith("trial_")
                )
                for subitem in item.iterdir()
            )
            if has_trials:
                v2_folders.append(item)
                logger.info(
                    f"[FIND_STUDY_FOLDER] Found v2 study folder: {item.name}")

        # Check for legacy study folders (hpo_{backbone}_*)
        elif not item.name.startswith("trial_"):
            has_trials = any(
                subitem.is_dir() and subitem.name.startswith("trial_")
                for subitem in item.iterdir()
            )
            if has_trials:
                legacy_folders.append(item)
                logger.info(
                    f"[FIND_STUDY_FOLDER] Found legacy study folder: {item.name}")

    # Prefer v2 folders over legacy
    if v2_folders:
        logger.info(
            f"[FIND_STUDY_FOLDER] Returning v2 study folder: {v2_folders[0].name} (found {len(v2_folders)} v2, {len(legacy_folders)} legacy)")
        return v2_folders[0]
    elif legacy_folders:
        logger.info(
            f"[FIND_STUDY_FOLDER] Returning legacy study folder: {legacy_folders[0].name} (found {len(legacy_folders)} legacy)")
        return legacy_folders[0]

    logger.warning(
        f"[FIND_STUDY_FOLDER] No study folders found in {backbone_dir}")
    return None


def find_best_trial_from_study(
    study: Any,
    backbone_name: str,
    dataset_version: str,
    objective_metric: str,
    hpo_backbone_dir: Path,
    hpo_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find best trial from an Optuna study object.

    Uses study.best_trial (source of truth) and locates the corresponding
    trial directory on disk by matching trial_key_hash.

    Args:
        study: Optuna study object
        backbone_name: Model backbone name
        dataset_version: Dataset version string
        objective_metric: Objective metric name
        hpo_backbone_dir: HPO backbone output directory
        hpo_config: HPO configuration (needed to compute trial_key_hash)
        data_config: Data configuration (needed to compute trial_key_hash)

    Returns:
        Dictionary with best trial info, or None if not found
    """
    if not study or study.best_trial is None:
        return None

    try:
        best_trial_config = extract_best_config_from_study(
            study, backbone_name, dataset_version, objective_metric
        )

        # Use resolve_storage_path to find the correct study folder (same logic as trial_meta_generator)
        from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

        checkpoint_config = hpo_config.get(
            "checkpoint", {}) if hpo_config else {}
        study_name_template = checkpoint_config.get("study_name") or (
            hpo_config.get("study_name") if hpo_config else None)
        study_name = None
        if study_name_template:
            study_name = study_name_template.replace(
                "{backbone}", backbone_name)

        # Prefer v2 study folders first (check before calling resolve_storage_path to avoid creating legacy folders)
        study_folder = find_study_folder_in_backbone_dir(hpo_backbone_dir)
        
        # If no v2 folder found, try to locate legacy folder via resolve_storage_path (read-only)
        if not study_folder:
            actual_storage_path = resolve_storage_path(
                output_dir=hpo_backbone_dir,
                checkpoint_config=checkpoint_config,
                backbone=backbone_name,
                study_name=study_name,
                create_dirs=False,  # Read-only: don't create legacy folders
            )
            if actual_storage_path and actual_storage_path.exists():
                study_folder = actual_storage_path.parent

        if not study_folder:
            logger.debug(f"Study folder not found in {hpo_backbone_dir}")
            return None

        # If study_folder is a legacy folder, check if there's a v2 study folder we should use instead
        # Prefer v2 paths (study-{hash}) over legacy paths (hpo_{backbone}_*)
        legacy_study_folder = study_folder
        if study_folder.name.startswith(f"hpo_{backbone_name}_"):
            logger.info(
                f"[FIND_BEST_TRIAL_FROM_STUDY] Found legacy study folder: {study_folder.name}, checking for v2 folders")
            # Look for v2 study folders in the same backbone_dir
            v2_study_folders = [
                item for item in hpo_backbone_dir.iterdir()
                if item.is_dir() and item.name.startswith("study-") and len(item.name) > 7
            ]
            if v2_study_folders:
                # Use the first v2 study folder found (or could check trial_key_hash match)
                study_folder = v2_study_folders[0]
                logger.info(
                    f"[FIND_BEST_TRIAL_FROM_STUDY] Switching to v2 study folder: {study_folder.name} (found {len(v2_study_folders)} v2 folders)")
            else:
                logger.info(
                    f"[FIND_BEST_TRIAL_FROM_STUDY] No v2 study folders found, using legacy: {legacy_study_folder.name}")

        best_trial = study.best_trial
        best_trial_number = best_trial.number

        # Compute trial_key_hash from Optuna trial hyperparameters
        computed_trial_key_hash = None
        if hpo_config and data_config:
            try:
                from orchestration.jobs.tracking.mlflow_naming import (
                    build_hpo_study_key,
                    build_hpo_study_key_hash,
                    build_hpo_trial_key,
                    build_hpo_trial_key_hash,
                )

                # Compute study_key_hash
                study_key = build_hpo_study_key(
                    data_config=data_config,
                    hpo_config=hpo_config,
                    backbone=backbone_name,
                    benchmark_config=None,  # Not needed for trial lookup
                )
                study_key_hash = build_hpo_study_key_hash(study_key)

                # Extract hyperparameters (excluding metadata fields)
                hyperparameters = {
                    k: v
                    for k, v in best_trial.params.items()
                    if k not in ("backbone", "trial_number")
                }

                # Compute trial_key_hash
                trial_key = build_hpo_trial_key(
                    study_key_hash=study_key_hash,
                    hyperparameters=hyperparameters,
                )
                computed_trial_key_hash = build_hpo_trial_key_hash(trial_key)

                logger.debug(
                    f"Computed trial_key_hash for trial {best_trial_number}: "
                    f"{computed_trial_key_hash[:16]}..."
                )
            except Exception as e:
                logger.debug(f"Could not compute trial_key_hash: {e}")

        best_trial_dir = None

        # Strategy 1: Match by trial_key_hash (most reliable)
        # Support both v2 paths (trial-{hash}) and legacy paths (trial_N)
        if computed_trial_key_hash:
            # Try v2 path lookup first if we have study_key_hash
            study_key_hash = None
            if hpo_config and data_config:
                try:
                    from orchestration.jobs.tracking.mlflow_naming import (
                        build_hpo_study_key,
                        build_hpo_study_key_hash,
                    )
                    study_key = build_hpo_study_key(
                        data_config=data_config,
                        hpo_config=hpo_config,
                        backbone=backbone_name,
                        benchmark_config=None,
                    )
                    study_key_hash = build_hpo_study_key_hash(study_key)
                except Exception:
                    pass

            # Try v2 path lookup using find_trial_by_hash
            if study_key_hash:
                try:
                    from orchestration.paths import find_trial_by_hash
                    from shared.platform_detection import detect_platform
                    # Find project root and config_dir from hpo_backbone_dir
                    # hpo_backbone_dir is typically: outputs/hpo/{storage_env}/{model}
                    # So we need to go up to project root
                    # outputs/hpo/{storage_env}/{model} -> project_root
                    project_root = hpo_backbone_dir.parent.parent.parent
                    config_dir = project_root / "config"
                    storage_env = detect_platform()

                    v2_trial_dir = find_trial_by_hash(
                        root_dir=project_root,
                        config_dir=config_dir,
                        model=backbone_name,
                        storage_env=storage_env,
                        study_key_hash=study_key_hash,
                        trial_key_hash=computed_trial_key_hash,
                    )
                    if v2_trial_dir and v2_trial_dir.exists():
                        best_trial_dir = v2_trial_dir
                        trial_identifier = format_trial_identifier(
                            v2_trial_dir, best_trial_number)
                        logger.info(
                            f"Found trial {trial_identifier} by trial_key_hash match (v2 path) "
                            f"({computed_trial_key_hash[:16]}...)"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not find trial using v2 path lookup: {e}")

            # Fallback: iterate through study_folder looking for trials
            if best_trial_dir is None:
                for trial_dir in study_folder.iterdir():
                    # Support both v2 (trial-{hash}) and legacy (trial_N) naming
                    if not trial_dir.is_dir():
                        continue
                    if not (trial_dir.name.startswith("trial_") or
                            (trial_dir.name.startswith("trial-") and len(trial_dir.name) > 7)):
                        continue

                    trial_meta_path = trial_dir / "trial_meta.json"
                    if not trial_meta_path.exists():
                        continue

                    try:
                        with open(trial_meta_path, "r") as f:
                            meta = json.load(f)

                        # Match by trial_key_hash
                        if meta.get("trial_key_hash") == computed_trial_key_hash:
                            best_trial_dir = trial_dir
                            trial_identifier = format_trial_identifier(
                                trial_dir, best_trial_number)
                            logger.info(
                                f"Found trial {trial_identifier} by trial_key_hash match "
                                f"({computed_trial_key_hash[:16]}...)"
                            )
                            break
                    except Exception as e:
                        logger.debug(
                            f"Error reading trial_meta.json from {trial_meta_path}: {e}")
                        continue

        # Strategy 2: Fallback to trial number + verify checkpoint exists
        if best_trial_dir is None:
            logger.debug(
                f"Trial not found by trial_key_hash, trying trial number {best_trial_number}"
            )
            for trial_dir in study_folder.iterdir():
                if not trial_dir.is_dir():
                    continue
                # Support both v2 (trial-{hash}) and legacy (trial_N) naming
                if not (trial_dir.name.startswith("trial_") or
                        (trial_dir.name.startswith("trial-") and len(trial_dir.name) > 7)):
                    continue

                # Check trial_meta.json for trial_number match (works for both v2 and legacy)
                trial_meta_path = trial_dir / "trial_meta.json"
                if trial_meta_path.exists():
                    try:
                        with open(trial_meta_path, "r") as f:
                            meta = json.load(f)
                        if meta.get("trial_number") == best_trial_number:
                            best_trial_dir = trial_dir
                            trial_identifier = format_trial_identifier(
                                trial_dir, best_trial_number)
                            logger.info(
                                f"Found trial {trial_identifier} by trial_number match in trial_meta.json"
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Error reading trial_meta.json: {e}")
                        continue

                # For legacy paths only, also try matching by name pattern
                match = re.match(r"trial_(\d+)_", trial_dir.name)
                if match and int(match.group(1)) == best_trial_number:
                    # Verify checkpoint exists (prefer refit, else CV, else root)
                    checkpoint_found = False
                    if (trial_dir / "refit" / "checkpoint").exists():
                        checkpoint_found = True
                    elif (trial_dir / "cv").exists():
                        # Check if any fold has checkpoint
                        for fold_dir in (trial_dir / "cv").iterdir():
                            if fold_dir.is_dir() and (fold_dir / "checkpoint").exists():
                                checkpoint_found = True
                                break
                    elif (trial_dir / "checkpoint").exists():
                        checkpoint_found = True

                    if checkpoint_found:
                        best_trial_dir = trial_dir
                        trial_identifier = format_trial_identifier(
                            trial_dir, best_trial_number)
                        logger.info(
                            f"Found trial {trial_identifier} by trial number {best_trial_number} "
                            "(with checkpoint)"
                        )
                        break
                    else:
                        logger.debug(
                            f"Trial {trial_dir.name} matches number but has no checkpoint, skipping"
                        )

        # Construct best_trial_from_disk from study.best_trial
        if best_trial_dir:
            logger.info(
                f"[FIND_BEST_TRIAL_FROM_STUDY] Found trial directory: {best_trial_dir} (exists: {best_trial_dir.exists()})")
            best_trial_from_disk = {
                "trial_name": best_trial_dir.name,
                "trial_dir": str(best_trial_dir),
                "checkpoint_dir": None,  # Will be determined later
                "checkpoint_type": "unknown",
                "accuracy": best_trial_config.get("selection_criteria", {}).get("best_value"),
                "metrics": best_trial_config.get("metrics", {}),
                "hyperparameters": best_trial_config.get("hyperparameters", {}),
            }
        else:
            # Fallback: trial directory not found by hash or number
            # Use find_best_trial_in_study_folder to find best trial by metrics
            logger.warning(
                f"[FIND_BEST_TRIAL_FROM_STUDY] Trial directory for trial {best_trial_number} not found in {study_folder} "
                f"by hash or number. Falling back to find_best_trial_in_study_folder to find best trial by metrics."
            )
            study_folder_contents = [
                item.name for item in study_folder.iterdir() if item.is_dir()]
            logger.info(
                f"[FIND_BEST_TRIAL_FROM_STUDY] Study folder contents: {study_folder_contents}")

            # Use find_best_trial_in_study_folder as fallback
            logger.info(
                f"[FIND_BEST_TRIAL_FROM_STUDY] Calling find_best_trial_in_study_folder with study_folder={study_folder}, objective_metric={objective_metric}")
            best_trial_from_folder = find_best_trial_in_study_folder(
                study_folder,
                objective_metric,
            )

            if best_trial_from_folder:
                logger.info(
                    f"[FIND_BEST_TRIAL_FROM_STUDY] SUCCESS: Found best trial via find_best_trial_in_study_folder: "
                    f"trial_name={best_trial_from_folder.get('trial_name')}, "
                    f"trial_dir={best_trial_from_folder.get('trial_dir')}, "
                    f"exists={Path(best_trial_from_folder.get('trial_dir')).exists()}"
                )
                best_trial_from_disk = {
                    "trial_name": best_trial_from_folder["trial_name"],
                    "trial_dir": best_trial_from_folder["trial_dir"],
                    "checkpoint_dir": None,
                    "checkpoint_type": "unknown",
                    "accuracy": best_trial_from_folder.get("accuracy"),
                    "metrics": best_trial_from_folder.get("metrics", {}),
                    # Use from study if available
                    "hyperparameters": best_trial_config.get("hyperparameters", {}),
                }
            else:
                # Last resort: Try one more time to find ANY trial directory
                # In v2 study folders, trials use hash-based names, so trial_number matching might fail
                # But we should still be able to find them by metrics
                logger.error(
                    f"[FIND_BEST_TRIAL_FROM_STUDY] FAILED: find_best_trial_in_study_folder returned None. "
                    f"Study folder: {study_folder}, contents: {study_folder_contents}. "
                    f"Attempting to find any trial directory as last resort..."
                )

                # Try to find ANY trial directory in the study folder
                any_trial_dir = None
                for item in study_folder.iterdir():
                    if item.is_dir() and (
                        item.name.startswith(
                            "trial-") or item.name.startswith("trial_")
                    ):
                        # Prefer v2 trials if we're in a v2 study folder
                        if study_folder.name.startswith("study-") and item.name.startswith("trial-"):
                            any_trial_dir = item
                            break
                        elif not any_trial_dir:
                            # Keep first trial found as fallback
                            any_trial_dir = item

                if any_trial_dir and any_trial_dir.exists():
                    logger.warning(
                        f"[FIND_BEST_TRIAL_FROM_STUDY] Using found trial directory as fallback: {any_trial_dir.name} "
                        f"(NOTE: This may not be the best trial, but it exists)"
                    )
                    best_trial_from_disk = {
                        "trial_name": any_trial_dir.name,
                        "trial_dir": str(any_trial_dir),
                        "checkpoint_dir": None,
                        "checkpoint_type": "unknown",
                        "accuracy": best_trial_config.get("selection_criteria", {}).get("best_value"),
                        "metrics": best_trial_config.get("metrics", {}),
                        "hyperparameters": best_trial_config.get("hyperparameters", {}),
                    }
                else:
                    # Absolute last resort: construct fallback path (but it likely won't exist)
                    fallback_trial_dir = study_folder / \
                        f"trial_{best_trial_number}"
                    logger.error(
                        f"[FIND_BEST_TRIAL_FROM_STUDY] CRITICAL: Could not find ANY trial in {study_folder}. "
                        f"Study folder contents: {study_folder_contents}. "
                        f"Using constructed fallback path: {fallback_trial_dir} (exists: {fallback_trial_dir.exists()})"
                    )
                    best_trial_from_disk = {
                        "trial_name": f"trial_{best_trial_number}",
                        "trial_dir": str(fallback_trial_dir),
                        "checkpoint_dir": None,
                        "checkpoint_type": "unknown",
                        "accuracy": best_trial_config.get("selection_criteria", {}).get("best_value"),
                        "metrics": best_trial_config.get("metrics", {}),
                        "hyperparameters": best_trial_config.get("hyperparameters", {}),
                    }

        return best_trial_from_disk

    except Exception as e:
        logger.warning(
            f"Could not extract best trial from study for {backbone_name}: {e}", exc_info=True
        )
        return None


def find_best_trials_for_backbones(
    backbone_values: list[str],
    hpo_studies: Optional[Dict[str, Any]],
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Find best trials for multiple backbones.
    """
    best_trials = {}
    objective_metric = hpo_config["objective"]["metric"]
    dataset_version = data_config.get("version", "unknown")

    hpo_output_dir_new = root_dir / "outputs" / "hpo" / environment

    for backbone in backbone_values:
        backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
        logger.info(
            f"Looking for best trial for {backbone} ({backbone_name})...")

        best_trial_info = None
        study = None

        if hpo_studies and backbone_name in hpo_studies:
            study = hpo_studies[backbone_name]

        # ---------- try loading study from disk ----------
        if not study:
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

                checkpoint_config = hpo_config.get("checkpoint", {})
                study_name_template = (
                    checkpoint_config.get("study_name")
                    or hpo_config.get("study_name")
                )

                study_name = (
                    study_name_template.replace("{backbone}", backbone_name)
                    if study_name_template
                    else None
                )

                # Prefer v2 study folders first (check before calling resolve_storage_path to avoid creating legacy folders)
                study_folder = find_study_folder_in_backbone_dir(hpo_backbone_dir)
                
                # If no v2 folder found, try to locate legacy folder via resolve_storage_path (read-only)
                if not study_folder:
                    actual_storage_path = resolve_storage_path(
                        output_dir=hpo_backbone_dir,
                        checkpoint_config=checkpoint_config,
                        backbone=backbone_name,
                        study_name=study_name,
                        create_dirs=False,  # Read-only: don't create legacy folders
                    )
                    if actual_storage_path and actual_storage_path.exists():
                        study_folder = actual_storage_path.parent

                if study_folder:
                    study_db_path = study_folder / "study.db"
                    if study_db_path.exists():
                        try:
                            from orchestration.jobs.hpo.local.optuna.integration import import_optuna
                            optuna, _, _, _ = import_optuna()
                        except ImportError:
                            import optuna

                        try:
                            study = optuna.load_study(
                                study_name=study_folder.name,
                                storage=f"sqlite:///{study_db_path.resolve()}",
                            )
                            logger.debug(
                                f"Loaded study for {backbone_name} from disk"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Could not load study from disk for {backbone_name}: {e}"
                            )

        # ---------- use study ----------
        if study:
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                best_trial_from_disk = find_best_trial_from_study(
                    study,
                    backbone_name,
                    dataset_version,
                    objective_metric,
                    hpo_backbone_dir,
                    hpo_config=hpo_config,
                    data_config=data_config,
                )

                if best_trial_from_disk:
                    trial_dir_path = Path(best_trial_from_disk["trial_dir"])
                    study_name = trial_dir_path.parent.name if trial_dir_path.parent else None
                    trial_number = study.best_trial.number if study.best_trial else None

                    best_trial_info = {
                        "backbone": backbone_name,
                        "trial_name": best_trial_from_disk["trial_name"],
                        "trial_dir": best_trial_from_disk["trial_dir"],
                        "study_name": study_name,
                        "checkpoint_dir": best_trial_from_disk.get(
                            "checkpoint_dir",
                            str(trial_dir_path / "checkpoint"),
                        ),
                        "checkpoint_type": best_trial_from_disk.get(
                            "checkpoint_type", "unknown"
                        ),
                        "accuracy": best_trial_from_disk["accuracy"],
                        "metrics": best_trial_from_disk["metrics"],
                        "hyperparameters": best_trial_from_disk["hyperparameters"],
                    }

                    identifier = format_trial_identifier(
                        trial_dir_path, trial_number)
                    logger.info(
                        f"{backbone}: Best trial is {identifier} "
                        f"({objective_metric}={best_trial_info['accuracy']:.4f})"
                    )

        # ---------- fallback disk search ----------
        if best_trial_info is None:
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

                checkpoint_config = hpo_config.get("checkpoint", {})
                study_name_template = (
                    checkpoint_config.get("study_name")
                    or hpo_config.get("study_name")
                )

                study_name = (
                    study_name_template.replace("{backbone}", backbone_name)
                    if study_name_template
                    else None
                )

                # Prefer v2 study folders first (check before calling resolve_storage_path to avoid creating legacy folders)
                study_folder = find_study_folder_in_backbone_dir(hpo_backbone_dir)
                
                # If no v2 folder found, try to locate legacy folder via resolve_storage_path (read-only)
                if not study_folder:
                    actual_storage_path = resolve_storage_path(
                        output_dir=hpo_backbone_dir,
                        checkpoint_config=checkpoint_config,
                        backbone=backbone_name,
                        study_name=study_name,
                        create_dirs=False,  # Read-only: don't create legacy folders
                    )
                    if actual_storage_path and actual_storage_path.exists():
                        study_folder = actual_storage_path.parent

                if study_folder:
                    best_trial_info = find_best_trial_in_study_folder(
                        study_folder, objective_metric
                    )
                    if best_trial_info:
                        best_trial_info["study_name"] = study_folder.name
                        best_trial_info["backbone"] = backbone_name

            else:
                hpo_output_dir_old = root_dir / "outputs" / "hpo"
                old_backbone_dir = resolve_hpo_output_dir(
                    hpo_output_dir_old / backbone
                )

                if old_backbone_dir.exists():
                    study_folder = find_study_folder_in_backbone_dir(
                        old_backbone_dir)
                    if study_folder:
                        best_trial_info = find_best_trial_in_study_folder(
                            study_folder, objective_metric
                        )
                        if best_trial_info:
                            best_trial_info["study_name"] = study_folder.name
                            best_trial_info["backbone"] = backbone_name
                elif str(old_backbone_dir).startswith("/content/drive"):
                    best_trial_info = load_best_trial_from_disk(
                        old_backbone_dir.parent,
                        backbone,
                        objective_metric,
                    )
                else:
                    best_trial_info = load_best_trial_from_disk(
                        hpo_output_dir_old,
                        backbone,
                        objective_metric,
                    )

        if best_trial_info:
            best_trials[backbone] = best_trial_info
        else:
            logger.warning(f"No best trial found for {backbone}")

    logger.info(
        f"Summary: Found {len(best_trials)} / {len(backbone_values)} best trials"
    )
    return best_trials
