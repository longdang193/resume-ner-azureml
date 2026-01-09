"""Generate missing trial_meta.json files for existing HPO trials.

This module provides utilities to retroactively create trial_meta.json files
for trials that were created before this feature was added.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def extract_trial_info_from_dirname(dirname: str) -> Optional[Dict[str, Any]]:
    """Extract trial_number and run_id from trial directory name."""
    match = re.match(r"trial_(\d+)_(.+)", dirname)
    if match:
        return {"trial_number": int(match.group(1)), "run_id": match.group(2)}
    return None


def generate_missing_trial_meta(
    study_folder: Path,
    backbone: str,
    study_key_hash: Optional[str] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Generate missing trial_meta.json files for a study.

    Args:
        study_folder: Path to study folder containing study.db and trial directories
        backbone: Model backbone name
        study_key_hash: Optional study key hash (will try to fetch from MLflow if not provided)

    Returns:
        Number of trial_meta.json files created
    """
    study_db_path = study_folder / "study.db"
    if not study_db_path.exists():
        logger.info(f"study.db not found: {study_db_path}")
        return 0

    study_name = study_folder.name
    storage_uri = f"sqlite:///{study_db_path.resolve()}"

    # Try to import Optuna
    try:
        from hpo.core.optuna_integration import import_optuna
        optuna, _, _, _ = import_optuna()
    except ImportError:
        import optuna

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        logger.info(f"Loaded study with {len(study.trials)} trials")
    except Exception as e:
        logger.error(f"Could not load study: {e}")
        return 0

    # Try to get study_key_hash from MLflow if not provided
    if not study_key_hash:
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.code.study_name = '{study_name}' OR tags.code.backbone = '{backbone}'",
                    max_results=10,
                )
                for run in runs:
                    study_key_hash = run.data.tags.get("code.study_key_hash")
                    if study_key_hash:
                        break
                if study_key_hash:
                    break
        except Exception:
            pass
    
    # If still not found, try to compute from configs
    if not study_key_hash and hpo_config and data_config:
        try:
            from orchestration.jobs.tracking.mlflow_naming import (
                build_hpo_study_key,
                build_hpo_study_key_hash,
            )
            study_key = build_hpo_study_key(
                data_config=data_config,
                hpo_config=hpo_config,
                backbone=backbone,
                benchmark_config=None,  # Not needed for hash computation
            )
            study_key_hash = build_hpo_study_key_hash(study_key)
            logger.debug(f"Computed study_key_hash from configs: {study_key_hash[:16]}...")
        except Exception as e:
            logger.debug(f"Could not compute study_key_hash from configs: {e}")

    # Find trial directories
    trial_dirs = [d for d in study_folder.iterdir() if d.is_dir() and d.name.startswith("trial_")]
    created_count = 0
    updated_count = 0

    for trial_dir in trial_dirs:
        trial_meta_path = trial_dir / "trial_meta.json"
        
        # Check if file exists and has valid hashes
        should_create = True
        if trial_meta_path.exists():
            try:
                with open(trial_meta_path, "r") as f:
                    existing_meta = json.load(f)
                # Only skip if hashes are already set
                if existing_meta.get("study_key_hash") and existing_meta.get("trial_key_hash"):
                    continue  # Skip - hashes already exist
                # Otherwise, update it (hashes are null)
                logger.info(f"Updating {trial_dir.name}/trial_meta.json (hashes are null)")
                should_create = False
            except Exception:
                # File exists but can't read it, will overwrite
                pass

        trial_info = extract_trial_info_from_dirname(trial_dir.name)
        if not trial_info:
            continue

        trial_number = trial_info["trial_number"]
        run_id = trial_info["run_id"]

        # Find Optuna trial
        optuna_trial = None
        for t in study.trials:
            if t.number == trial_number:
                optuna_trial = t
                break

        if not optuna_trial:
            continue

        # Try to get trial_key_hash from MLflow
        trial_key_hash = None
        if study_key_hash:
            try:
                import mlflow
                from orchestration.jobs.tracking.mlflow_naming import (
                    build_hpo_trial_key,
                    build_hpo_trial_key_hash,
                )
                client = mlflow.tracking.MlflowClient()
                experiments = client.search_experiments()
                for exp in experiments:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        filter_string=f"tags.code.study_key_hash = '{study_key_hash}' AND tags.code.trial_number = '{trial_number}'",
                        max_results=10,
                    )
                    for run in runs:
                        trial_key_hash = run.data.tags.get("code.trial_key_hash")
                        if trial_key_hash:
                            break
                    if trial_key_hash:
                        break

                # If not found, compute it
                if not trial_key_hash:
                    hyperparameters = {
                        k: v
                        for k, v in optuna_trial.params.items()
                        if k not in ("backbone", "trial_number", "run_id")
                    }
                    trial_key = build_hpo_trial_key(study_key_hash, hyperparameters)
                    trial_key_hash = build_hpo_trial_key_hash(trial_key)
            except Exception as e:
                logger.info(f"Could not compute trial_key_hash: {e}")

        # Create or update trial_meta.json
        # Preserve existing fields if updating
        if should_create:
            trial_meta = {
                "study_key_hash": study_key_hash,
                "trial_key_hash": trial_key_hash,
                "trial_number": trial_number,
                "study_name": study_name,
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "note": "Retroactively generated from existing trial data",
            }
        else:
            # Load existing and update hashes
            with open(trial_meta_path, "r") as f:
                trial_meta = json.load(f)
            trial_meta["study_key_hash"] = study_key_hash
            trial_meta["trial_key_hash"] = trial_key_hash
            trial_meta["updated_at"] = datetime.now().isoformat()
            if "note" not in trial_meta:
                trial_meta["note"] = "Updated with computed hashes"

        try:
            with open(trial_meta_path, "w") as f:
                json.dump(trial_meta, f, indent=2)
            action = "Created" if should_create else "Updated"
            logger.info(f"{action} {trial_dir.name}/trial_meta.json")
            if study_key_hash and trial_key_hash:
                logger.info(f"  study_key_hash: {study_key_hash[:16]}...")
                logger.info(f"  trial_key_hash: {trial_key_hash[:16]}...")
            else:
                logger.warning(f"  Warning: Could not compute hashes (study_key_hash={study_key_hash is not None}, trial_key_hash={trial_key_hash is not None})")
            if should_create:
                created_count += 1
            else:
                updated_count += 1
        except Exception as e:
            logger.error(f"Could not create/update {trial_meta_path}: {e}")

    if updated_count > 0:
        logger.debug(f"  Created: {created_count}, Updated: {updated_count}")
    
    return created_count + updated_count  # Return total count (created + updated)


def generate_missing_trial_meta_for_all_studies(
    hpo_studies: Dict[str, Any],
    backbone_values: list,
    root_dir: Path,
    environment: str,
    hpo_config: dict,
    data_config: Optional[Dict[str, Any]] = None,
    backup_enabled: bool = True,
) -> int:
    """
    Generate missing trial_meta.json files for all studies.

    Args:
        hpo_studies: Dictionary mapping backbone names to Optuna study objects
        backbone_values: List of backbone names
        root_dir: Root directory of the project
        environment: Platform environment (local, colab, kaggle)
        hpo_config: HPO configuration dict
        data_config: Data configuration dict (needed to compute study_key_hash)
        backup_enabled: Whether backup is enabled

    Returns:
        Total number of trial_meta.json files created
    """
    if not backup_enabled:
        logger.info("Skipping trial_meta.json generation (BACKUP_ENABLED=False)")
        return 0

    if not hpo_studies:
        logger.info("Skipping trial_meta.json generation (hpo_studies not available)")
        return 0

    logger.info("Generating missing trial_meta.json files...")
    total_created = 0
    total_updated = 0

    from hpo.utils.paths import resolve_hpo_output_dir
    from hpo.checkpoint.storage import resolve_storage_path

    # Process all backbones from backbone_values, not just those in hpo_studies
    # This ensures we process all backbones even if studies are no longer in memory
    processed_backbones = set()
    
    # First, process backbones that have in-memory studies
    for backbone_name, study in (hpo_studies or {}).items():
        if not study:
            continue
        processed_backbones.add(backbone_name)
        
        # Find study folder
        backbone_output_dir = root_dir / "outputs" / "hpo" / environment / backbone_name
        hpo_backbone_dir = resolve_hpo_output_dir(backbone_output_dir)

        if not hpo_backbone_dir.exists():
            logger.debug(f"Skipping {backbone_name}: backbone directory not found")
            continue

        # Resolve storage path to get study folder
        checkpoint_config = hpo_config.get("checkpoint", {})
        study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")
        study_name = None
        if study_name_template:
            study_name = study_name_template.replace("{backbone}", backbone_name)

        actual_storage_path = resolve_storage_path(
            output_dir=backbone_output_dir,
            checkpoint_config=checkpoint_config,
            backbone=backbone_name,
            study_name=study_name,
        )

        if actual_storage_path and actual_storage_path.exists():
            study_folder = actual_storage_path.parent
        else:
            # Fallback: construct from study name
            study_folder = backbone_output_dir / study_name if study_name else backbone_output_dir
            # Try to find existing study folder
            if not study_folder.exists():
                # Look for any study folder in backbone dir
                if backbone_output_dir.exists():
                    study_folders = [
                        d
                        for d in backbone_output_dir.iterdir()
                        if d.is_dir()
                        and not d.name.startswith("trial_")
                        and d.name != ".ipynb_checkpoints"
                    ]
                    if study_folders:
                        study_folder = study_folders[0]

        if study_folder and study_folder.exists():
            logger.info(f"Processing {backbone_name}: {study_folder.name}")
            result = generate_missing_trial_meta(
                study_folder, backbone_name,
                hpo_config=hpo_config,
                data_config=data_config,
            )
            # result is tuple (created, updated) if generate_missing_trial_meta returns it
            # For now, it returns count, but we'll update it to return (created, updated)
            total_created += result
    
    # Then process remaining backbones from backbone_values (even without in-memory studies)
    for backbone_name in backbone_values:
        if backbone_name in processed_backbones:
            continue
        
        # Find study folder
        backbone_output_dir = root_dir / "outputs" / "hpo" / environment / backbone_name
        hpo_backbone_dir = resolve_hpo_output_dir(backbone_output_dir)

        if not hpo_backbone_dir.exists():
            logger.debug(f"Skipping {backbone_name}: backbone directory not found")
            continue

        # Resolve storage path to get study folder
        checkpoint_config = hpo_config.get("checkpoint", {})
        study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")
        study_name = None
        if study_name_template:
            study_name = study_name_template.replace("{backbone}", backbone_name)

        actual_storage_path = resolve_storage_path(
            output_dir=backbone_output_dir,
            checkpoint_config=checkpoint_config,
            backbone=backbone_name,
            study_name=study_name,
        )

        if actual_storage_path and actual_storage_path.exists():
            study_folder = actual_storage_path.parent
        else:
            # Fallback: construct from study name
            study_folder = backbone_output_dir / study_name if study_name else backbone_output_dir
            # Try to find existing study folder
            if not study_folder.exists():
                # Look for any study folder in backbone dir
                if backbone_output_dir.exists():
                    study_folders = [
                        d
                        for d in backbone_output_dir.iterdir()
                        if d.is_dir()
                        and not d.name.startswith("trial_")
                        and d.name != ".ipynb_checkpoints"
                    ]
                    if study_folders:
                        study_folder = study_folders[0]

        if study_folder and study_folder.exists():
            logger.info(f"Processing {backbone_name}: {study_folder.name} (study not in memory, loading from disk)")
            result = generate_missing_trial_meta(
                study_folder, backbone_name,
                hpo_config=hpo_config,
                data_config=data_config,
            )
            total_created += result

    logger.info(f"Total: Created/Updated {total_created} trial_meta.json files")
    return total_created

