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


def find_study_folder_in_backbone_dir(backbone_dir: Path) -> Optional[Path]:
    """
    Find study folder inside backbone directory.

    In the new structure, trials are inside study folders like:
    backbone_dir/study_name/trial_0/...

    Args:
        backbone_dir: Backbone directory containing study folders

    Returns:
        Path to study folder if found, else None
    """
    if not backbone_dir.exists():
        return None

    for item in backbone_dir.iterdir():
        if item.is_dir() and not item.name.startswith("trial_"):
            has_trials = any(
                subitem.is_dir() and subitem.name.startswith("trial_")
                for subitem in item.iterdir()
            )
            if has_trials:
                return item

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

        actual_storage_path = resolve_storage_path(
            output_dir=hpo_backbone_dir,
            checkpoint_config=checkpoint_config,
            backbone=backbone_name,
            study_name=study_name,
        )

        if actual_storage_path and actual_storage_path.exists():
            study_folder = actual_storage_path.parent
        else:
            # Fallback: use find_study_folder_in_backbone_dir
            study_folder = find_study_folder_in_backbone_dir(hpo_backbone_dir)

        if not study_folder:
            logger.debug(f"Study folder not found in {hpo_backbone_dir}")
            return None

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
        if computed_trial_key_hash:
            for trial_dir in study_folder.iterdir():
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
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
                        logger.info(
                            f"Found trial {trial_dir.name} by trial_key_hash match "
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
                f"Trial not found by trial_key_hash, falling back to trial number {best_trial_number}"
            )
            for trial_dir in study_folder.iterdir():
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue

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
                        logger.info(
                            f"Found trial {trial_dir.name} by trial number {best_trial_number} "
                            "(with checkpoint)"
                        )
                        break
                    else:
                        logger.debug(
                            f"Trial {trial_dir.name} matches number but has no checkpoint, skipping"
                        )

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
            # Fallback: trial directory not found, use best_trial_config
            logger.warning(
                f"Trial directory for trial {best_trial_number} not found in {study_folder}, "
                "using metadata from study"
            )
            best_trial_from_disk = {
                "trial_name": f"trial_{best_trial_number}",
                "trial_dir": str(study_folder / f"trial_{best_trial_number}"),
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

    First tries to use in-memory study objects if available, otherwise
    searches disk for saved trial outputs.

    Args:
        backbone_values: List of backbone model names
        hpo_studies: Optional dictionary of backbone -> Optuna study
        hpo_config: HPO configuration dictionary
        data_config: Data configuration dictionary
        root_dir: Project root directory
        environment: Platform environment (e.g., "colab", "local")

    Returns:
        Dictionary mapping backbone -> best trial info
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

        # Try to use in-memory study if available
        study = None
        if hpo_studies and backbone_name in hpo_studies:
            study = hpo_studies[backbone_name]

        # If study not in memory, try to load from disk
        if not study:
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                # Try to load study from disk
                from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

                checkpoint_config = hpo_config.get("checkpoint", {})
                study_name_template = checkpoint_config.get(
                    "study_name") or hpo_config.get("study_name")
                study_name = None
                if study_name_template:
                    study_name = study_name_template.replace(
                        "{backbone}", backbone_name)

                actual_storage_path = resolve_storage_path(
                    output_dir=hpo_backbone_dir,
                    checkpoint_config=checkpoint_config,
                    backbone=backbone_name,
                    study_name=study_name,
                )

                if actual_storage_path and actual_storage_path.exists():
                    study_folder = actual_storage_path.parent
                else:
                    study_folder = find_study_folder_in_backbone_dir(
                        hpo_backbone_dir)

                if study_folder:
                    study_db_path = study_folder / "study.db"
                    if study_db_path.exists():
                        try:
                            from orchestration.jobs.hpo.local.optuna.integration import import_optuna
                            optuna, _, _, _ = import_optuna()
                        except ImportError:
                            import optuna

                        try:
                            storage_uri = f"sqlite:///{study_db_path.resolve()}"
                            study = optuna.load_study(
                                study_name=study_folder.name, storage=storage_uri)
                            logger.debug(
                                f"Loaded study for {backbone_name} from disk: {study_folder.name}")
                        except Exception as e:
                            logger.debug(
                                f"Could not load study from disk for {backbone_name}: {e}")

        # Use study (from memory or disk) to find best trial
        if study:
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                best_trial_from_disk = find_best_trial_from_study(
                    study, backbone_name, dataset_version, objective_metric, hpo_backbone_dir,
                    hpo_config=hpo_config,
                    data_config=data_config,
                )

                if best_trial_from_disk:
                    # Extract study name from trial_dir path
                    trial_dir_path = Path(best_trial_from_disk["trial_dir"])
                    study_name = trial_dir_path.parent.name if trial_dir_path.parent else None

                    best_trial_info = {
                        "backbone": backbone_name,
                        "trial_name": best_trial_from_disk["trial_name"],
                        "trial_dir": best_trial_from_disk["trial_dir"],
                        "study_name": study_name,  # Add study name
                        "checkpoint_dir": best_trial_from_disk.get(
                            "checkpoint_dir",
                            str(Path(
                                best_trial_from_disk["trial_dir"]) / "checkpoint"),
                        ),
                        "checkpoint_type": best_trial_from_disk.get("checkpoint_type", "unknown"),
                        "accuracy": best_trial_from_disk["accuracy"],
                        "metrics": best_trial_from_disk["metrics"],
                        "hyperparameters": best_trial_from_disk["hyperparameters"],
                    }
                    logger.info(
                        f"{backbone}: Best trial from HPO run is {best_trial_info['trial_name']} "
                        f"({objective_metric}={best_trial_info['accuracy']:.4f})"
                    )

        # Fallback to disk search if study not available or didn't find trial
        if best_trial_info is None:
            logger.debug(
                f"Searching disk for best trial for {backbone_name}...")
            local_path = hpo_output_dir_new / backbone_name
            hpo_backbone_dir = resolve_hpo_output_dir(local_path)

            if hpo_backbone_dir.exists():
                # Use resolve_storage_path to find the correct study folder
                from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

                checkpoint_config = hpo_config.get("checkpoint", {})
                study_name_template = checkpoint_config.get(
                    "study_name") or hpo_config.get("study_name")
                study_name = None
                if study_name_template:
                    study_name = study_name_template.replace(
                        "{backbone}", backbone_name)

                actual_storage_path = resolve_storage_path(
                    output_dir=hpo_backbone_dir,
                    checkpoint_config=checkpoint_config,
                    backbone=backbone_name,
                    study_name=study_name,
                )

                if actual_storage_path and actual_storage_path.exists():
                    study_folder = actual_storage_path.parent
                else:
                    # Fallback: use find_study_folder_in_backbone_dir
                    study_folder = find_study_folder_in_backbone_dir(
                        hpo_backbone_dir)

                if study_folder:
                    best_trial_info = load_best_trial_from_disk(
                        study_folder.parent.parent,
                        f"{backbone_name}/{study_folder.name}",
                        objective_metric,
                    )
                    if best_trial_info:
                        best_trial_info["study_name"] = study_folder.name
                elif str(hpo_backbone_dir).startswith("/content/drive"):
                    drive_hpo_dir = hpo_backbone_dir.parent.parent
                    relative_backbone = f"{environment}/{backbone_name}"
                    best_trial_info = load_best_trial_from_disk(
                        drive_hpo_dir,
                        relative_backbone,
                        objective_metric,
                    )
                    # Try to extract study_name from trial_dir if available
                    if best_trial_info and "trial_dir" in best_trial_info:
                        trial_dir_path = Path(best_trial_info["trial_dir"])
                        if trial_dir_path.parent:
                            best_trial_info["study_name"] = trial_dir_path.parent.name
                else:
                    best_trial_info = load_best_trial_from_disk(
                        hpo_output_dir_new.parent,
                        f"{environment}/{backbone_name}",
                        objective_metric,
                    )
                    # Try to extract study_name from trial_dir if available
                    if best_trial_info and "trial_dir" in best_trial_info:
                        trial_dir_path = Path(best_trial_info["trial_dir"])
                        if trial_dir_path.parent:
                            best_trial_info["study_name"] = trial_dir_path.parent.name
            else:
                # Try old structure
                hpo_output_dir_old = root_dir / "outputs" / "hpo"
                old_backbone_dir = resolve_hpo_output_dir(
                    hpo_output_dir_old / backbone)
                if old_backbone_dir.exists():
                    study_folder = find_study_folder_in_backbone_dir(
                        old_backbone_dir)
                    if study_folder:
                        best_trial_info = load_best_trial_from_disk(
                            study_folder.parent.parent,
                            f"{backbone}/{study_folder.name}",
                            objective_metric,
                        )
                        if best_trial_info:
                            best_trial_info["study_name"] = study_folder.name
                    elif str(old_backbone_dir).startswith("/content/drive"):
                        drive_hpo_dir = old_backbone_dir.parent
                        best_trial_info = load_best_trial_from_disk(
                            drive_hpo_dir,
                            backbone,
                            objective_metric,
                        )
                        # Try to extract study_name from trial_dir if available
                        if best_trial_info and "trial_dir" in best_trial_info:
                            trial_dir_path = Path(best_trial_info["trial_dir"])
                            if trial_dir_path.parent:
                                best_trial_info["study_name"] = trial_dir_path.parent.name
                    else:
                        best_trial_info = load_best_trial_from_disk(
                            hpo_output_dir_old,
                            backbone,
                            objective_metric,
                        )
                        # Try to extract study_name from trial_dir if available
                        if best_trial_info and "trial_dir" in best_trial_info:
                            trial_dir_path = Path(best_trial_info["trial_dir"])
                            if trial_dir_path.parent:
                                best_trial_info["study_name"] = trial_dir_path.parent.name

            if best_trial_info:
                # Extract study_name from trial_dir if not already present
                if "study_name" not in best_trial_info and "trial_dir" in best_trial_info:
                    trial_dir_path = Path(best_trial_info["trial_dir"])
                    if trial_dir_path.exists() or str(trial_dir_path):
                        # Extract study name from path: .../backbone/study_name/trial_...
                        parts = trial_dir_path.parts
                        # Find backbone name in path, study_name should be next
                        for i, part in enumerate(parts):
                            if part == backbone_name and i + 1 < len(parts):
                                best_trial_info["study_name"] = parts[i + 1]
                                break
                        # Fallback: use parent directory name
                        if "study_name" not in best_trial_info:
                            best_trial_info["study_name"] = trial_dir_path.parent.name if trial_dir_path.parent else None

                logger.info(
                    f"Found best trial for {backbone}: {best_trial_info.get('trial_name', 'unknown')} "
                    f"({objective_metric}={best_trial_info.get('accuracy', 0):.4f})"
                )

        if best_trial_info:
            best_trials[backbone] = best_trial_info
        else:
            logger.warning(f"No best trial found for {backbone}")

    logger.info(
        f"Summary: Found {len(best_trials)} best trial(s) out of {len(backbone_values)} backbone(s)"
    )
    return best_trials
