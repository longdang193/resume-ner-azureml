"""HPO backup utilities for Colab environments.

Handles backing up study.db and study folders to Google Drive,
with verification of trial_meta.json files.
"""

from pathlib import Path
from typing import Optional, Callable

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def backup_hpo_study_to_drive(
    backbone: str,
    backbone_output_dir: Path,
    checkpoint_config: dict,
    hpo_config: dict,
    backup_to_drive: Callable[[Path, bool], bool],
    backup_enabled: bool = True,
) -> None:
    """
    Backup HPO study.db and study folder to Google Drive.

    Args:
        backbone: Model backbone name
        backbone_output_dir: Base output directory for HPO
        checkpoint_config: Checkpoint configuration dict
        hpo_config: HPO configuration dict
        backup_to_drive: Function to backup files to Drive
        backup_enabled: Whether backup is enabled
    """
    if not backup_enabled:
        return

    # Get study name
    study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")
    study_name = None
    if study_name_template:
        study_name = study_name_template.replace("{backbone}", backbone)

    # Get the actual storage path (may be in Drive due to resolve_checkpoint_path)
    from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

    actual_storage_path = resolve_storage_path(
        output_dir=backbone_output_dir,
        checkpoint_config=checkpoint_config,
        backbone=backbone,
        study_name=study_name,
    )

    # Construct LOCAL path for reference
    storage_path_template = checkpoint_config.get("storage_path", "{backbone}/study.db")
    storage_path_str = storage_path_template.replace("{backbone}", backbone)
    if study_name:
        storage_path_str = storage_path_str.replace("{study_name}", study_name)
    local_checkpoint_path = backbone_output_dir / storage_path_str

    # Backup study.db
    if actual_storage_path and str(actual_storage_path).startswith("/content/drive"):
        # File is already in Drive - no need to backup, just log
        logger.info(f"HPO checkpoint is already in Drive: {actual_storage_path}")
        logger.info(f"  Study name: {study_name}")
    elif actual_storage_path and actual_storage_path.exists():
        # File exists locally - backup it
        backup_to_drive(actual_storage_path, is_directory=False)
        logger.info(f"Backed up HPO checkpoint database to Drive: {actual_storage_path}")

        # Also backup the parent directory (study folder) if it exists locally
        study_folder = actual_storage_path.parent
        if study_folder.exists() and study_folder.is_dir() and not str(study_folder).startswith("/content/drive"):
            backup_to_drive(study_folder, is_directory=True)
            logger.info(f"Backed up HPO study folder to Drive: {study_folder}")
    elif local_checkpoint_path.exists():
        # Fallback: check local path
        backup_to_drive(local_checkpoint_path, is_directory=False)
        logger.info(f"Backed up HPO checkpoint database to Drive: {local_checkpoint_path}")
    else:
        logger.warning(f"HPO checkpoint not found")
        logger.warning(f"  Expected local path: {local_checkpoint_path}")
        logger.warning(f"  Resolved path: {actual_storage_path}")
        logger.warning(f"  Study name: {study_name}")

    # Backup entire study folder (new structure: outputs/hpo/{env}/{model}/{study_name}/...)
    # Check if checkpoint is already in Drive - if so, study folder is also in Drive
    checkpoint_in_drive = actual_storage_path and str(actual_storage_path).startswith("/content/drive")

    # Use Drive path if checkpoint is in Drive, otherwise use local path
    if checkpoint_in_drive and actual_storage_path:
        # Study folder is in Drive - use the Drive path
        study_folder = actual_storage_path.parent
    else:
        # Study folder is local
        study_folder = backbone_output_dir / study_name if study_name else backbone_output_dir

    if study_folder.exists():
        if checkpoint_in_drive:
            # study.db and study folder are already in Drive
            # Verify trial_meta.json files exist in Drive
            logger.info(f"Study folder is in Drive: {study_folder}")
            trial_dirs = [d for d in study_folder.iterdir() if d.is_dir() and d.name.startswith("trial_")]
            trial_meta_count = 0
            for trial_dir in trial_dirs:
                trial_meta_path = trial_dir / "trial_meta.json"
                if trial_meta_path.exists():
                    trial_meta_count += 1
                else:
                    logger.warning(f"Missing trial_meta.json in {trial_dir.name}")
            if trial_meta_count > 0:
                logger.info(f"Found {trial_meta_count} trial_meta.json file(s) in Drive")
            else:
                logger.warning("No trial_meta.json files found in Drive study folder")
        else:
            # study.db is local, backup the entire study folder (includes study.db + all trials + trial_meta.json)
            result = backup_to_drive(study_folder, is_directory=True)
            if result:
                logger.info(f"Backed up entire study folder to Drive: {study_folder.name}")

                # Verify trial_meta.json files were backed up
                trial_dirs = [d for d in study_folder.iterdir() if d.is_dir() and d.name.startswith("trial_")]
                for trial_dir in trial_dirs:
                    trial_meta_path = trial_dir / "trial_meta.json"
                    if trial_meta_path.exists():
                        # Check if it was backed up (Drive path should exist)
                        drive_trial_meta = Path(str(trial_meta_path).replace(str(backbone_output_dir), "/content/drive/MyDrive/resume-ner-azureml/outputs/hpo"))
                        if drive_trial_meta.exists():
                            logger.debug(f"Verified trial_meta.json backed up: {trial_dir.name}/trial_meta.json")
                        else:
                            logger.warning(f"trial_meta.json not found in backup: {trial_dir.name}/trial_meta.json")
            else:
                logger.warning(f"Failed to backup study folder: {study_folder.name}")
    elif checkpoint_in_drive:
        # Checkpoint is in Drive, so study folder is also in Drive (not local)
        # This is expected behavior in Colab - no error needed
        logger.info(f"Study folder is in Drive (checkpoint already backed up): {actual_storage_path.parent if actual_storage_path else 'N/A'}")
    else:
        # Checkpoint is local but study folder doesn't exist - this is an error
        logger.warning(f"Study folder not found: {study_folder}")

