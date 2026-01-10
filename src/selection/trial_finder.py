"""Find best trials from HPO studies or disk.

This module provides utilities to locate and extract best trial information
from Optuna studies or from saved outputs on disk.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from common.shared.logging_utils import get_logger

from hpo.core.study import extract_best_config_from_study
from .disk_loader import load_best_trial_from_disk
from hpo.utils.paths import resolve_hpo_output_dir

logger = get_logger(__name__)


def find_best_trial_in_study_folder(
    study_folder: Path,
    objective_metric: str = "macro-f1",
) -> Optional[Dict[str, Any]]:
    """
    Find best trial in a specific study folder by reading metrics.json files.

    Supports v2 paths (trial-{hash}) only.

    Args:
        study_folder: Path to study folder containing trials
        objective_metric: Name of the objective metric to optimize

    Returns:
        Dictionary with best trial info, or None if no trials found
    """
    if not study_folder.exists():
        logger.warning(
            f"Study folder does not exist: {study_folder}")
        return None

    best_metric = None
    best_trial_dir = None
    best_trial_name = None

    # Collect all v2 trial directories (trial-{hash})
    trial_dirs = []
    for item in study_folder.iterdir():
        if item.is_dir() and item.name.startswith("trial-"):
            trial_dirs.append(item)

    if len(trial_dirs) == 0:
        logger.warning(
            f"No v2 trial directories found in {study_folder}. "
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
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            if objective_metric in metrics:
                metric_value = metrics[objective_metric]

                # Skip fold-specific trials (we'll aggregate later if needed)
                if "_fold" in trial_dir.name:
                    continue

                trials_with_metrics.append((trial_dir, metric_value))

                if best_metric is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_trial_dir = trial_dir
                    best_trial_name = trial_dir.name
        except Exception as e:
            logger.warning(
                f"Could not read {metrics_file}: {e}")
            continue

    if len(trials_with_metrics) == 0:
        logger.warning(
            f"No trials found with {objective_metric} metric. "
            f"Found {len(trial_dirs)} trial directories but none have valid metrics.json with {objective_metric}"
        )

    if best_trial_dir is None:
        logger.warning(
            f"No trials with {objective_metric} found in {study_folder}")
        return None

    # Load metrics - use same logic as above to find metrics.json
    # Try multiple locations for metrics.json
    # 1. Trial root: trial_dir/metrics.json
    # 2. CV folds: trial_dir/cv/fold0/metrics.json (for CV trials)
    metrics_file = None
    if (best_trial_dir / "metrics.json").exists():
        metrics_file = best_trial_dir / "metrics.json"
    elif (best_trial_dir / "cv").exists():
        # Check first fold for metrics (CV trials aggregate metrics at fold level)
        for fold_dir in (best_trial_dir / "cv").iterdir():
            if fold_dir.is_dir() and fold_dir.name.startswith("fold"):
                fold_metrics = fold_dir / "metrics.json"
                if fold_metrics.exists():
                    metrics_file = fold_metrics
                    break

    if not metrics_file or not metrics_file.exists():
        logger.warning(
            f"metrics.json not found in {best_trial_dir} (checked root and CV folds)")
        return None

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
        except Exception:
            pass

    result = {
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "accuracy": best_metric,
        "metrics": metrics,
    }

    if trial_run_id:
        result["trial_run_id"] = trial_run_id

    # Verify the trial_dir actually exists before returning
    if not best_trial_dir.exists():
        logger.error(
            f"Selected trial_dir does not exist: {best_trial_dir}. "
            f"Available trials: {[d.name for d in trial_dirs]}"
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
    Find v2 study folder inside backbone directory.

    Supports v2 paths (study-{study8}/trial-{trial8}) only.

    Args:
        backbone_dir: Backbone directory containing study folders

    Returns:
        Path to study folder if found, else None
    """
    if not backbone_dir.exists():
        logger.warning(
            f"Backbone directory does not exist: {backbone_dir}")
        return None

    v2_folders = []

    for item in backbone_dir.iterdir():
        if not item.is_dir():
            continue

        # Check for v2 study folders (study-{hash})
        if item.name.startswith("study-") and len(item.name) > 7:
            # Check if it contains v2 trial folders (trial-{hash})
            has_trials = any(
                subitem.is_dir() and subitem.name.startswith("trial-")
                for subitem in item.iterdir()
            )
            if has_trials:
                v2_folders.append(item)

    if v2_folders:
        return v2_folders[0]

    logger.warning(
        f"No v2 study folders found in {backbone_dir}")
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
        from hpo.checkpoint.storage import resolve_storage_path

        checkpoint_config = hpo_config.get(
            "checkpoint", {}) if hpo_config else {}
        study_name_template = checkpoint_config.get("study_name") or (
            hpo_config.get("study_name") if hpo_config else None)
        study_name = None
        if study_name_template:
            study_name = study_name_template.replace(
                "{backbone}", backbone_name)

        # Find v2 study folder
        study_folder = find_study_folder_in_backbone_dir(hpo_backbone_dir)

        if not study_folder:
            logger.warning(f"V2 study folder not found in {hpo_backbone_dir}")
            return None

        best_trial = study.best_trial
        best_trial_number = best_trial.number

        # Compute trial_key_hash from Optuna trial hyperparameters
        computed_trial_key_hash = None
        if hpo_config and data_config:
            try:
                from infrastructure.tracking.mlflow.naming import (
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

            except Exception:
                pass

        best_trial_dir = None

        # Strategy 1: Match by trial_key_hash (most reliable)
        # Support v2 paths (trial-{hash}) only
        if computed_trial_key_hash:
            # Try v2 path lookup first if we have study_key_hash
            study_key_hash = None
            if hpo_config and data_config:
                try:
                    from infrastructure.tracking.mlflow.naming import (
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
                    from infrastructure.paths.parse import find_trial_by_hash
                    from common.shared.platform_detection import detect_platform
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
                except Exception:
                    pass

            # Fallback: iterate through study_folder looking for v2 trials
            if best_trial_dir is None:
                for trial_dir in study_folder.iterdir():
                    # Support v2 (trial-{hash}) naming only
                    if not trial_dir.is_dir():
                        continue
                    if not (trial_dir.name.startswith("trial-") and len(trial_dir.name) > 7):
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
                            break
                    except Exception:
                        continue

        # Strategy 2: Fallback to trial number + verify checkpoint exists
        if best_trial_dir is None:
            for trial_dir in study_folder.iterdir():
                if not trial_dir.is_dir():
                    continue
                # Support v2 (trial-{hash}) naming only
                if not (trial_dir.name.startswith("trial-") and len(trial_dir.name) > 7):
                    continue

                # Check trial_meta.json for trial_number match (works for both v2 and legacy)
                trial_meta_path = trial_dir / "trial_meta.json"
                if trial_meta_path.exists():
                    try:
                        with open(trial_meta_path, "r") as f:
                            meta = json.load(f)
                        if meta.get("trial_number") == best_trial_number:
                            best_trial_dir = trial_dir
                            break
                    except Exception:
                        continue

        # Construct best_trial_from_disk from study.best_trial
        if best_trial_dir:
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
                f"Trial directory for trial {best_trial_number} not found in {study_folder} "
                f"by hash or number. Falling back to find_best_trial_in_study_folder to find best trial by metrics."
            )
            study_folder_contents = [
                item.name for item in study_folder.iterdir() if item.is_dir()]

            # Use find_best_trial_in_study_folder as fallback
            best_trial_from_folder = find_best_trial_in_study_folder(
                study_folder,
                objective_metric,
            )

            if best_trial_from_folder:
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
                # Last resort: Try to find ANY v2 trial directory in the study folder
                any_trial_dir = None
                for item in study_folder.iterdir():
                    if item.is_dir() and item.name.startswith("trial-") and len(item.name) > 7:
                        any_trial_dir = item
                        break

                if any_trial_dir and any_trial_dir.exists():
                    logger.warning(
                        f"Using found v2 trial directory as fallback: {any_trial_dir.name} "
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
                    logger.error(
                        f"Could not find ANY v2 trial in {study_folder}. "
                        f"Study folder contents: {study_folder_contents}."
                    )
                    best_trial_from_disk = None

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
                from hpo.checkpoint.storage import resolve_storage_path

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

                # Find v2 study folder
                study_folder = find_study_folder_in_backbone_dir(
                    hpo_backbone_dir)

                if study_folder:
                    study_db_path = study_folder / "study.db"
                    if study_db_path.exists():
                        try:
                            from hpo.core.optuna_integration import import_optuna
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
                from hpo.checkpoint.storage import resolve_storage_path

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

                # Find v2 study folder
                study_folder = find_study_folder_in_backbone_dir(
                    hpo_backbone_dir)

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
