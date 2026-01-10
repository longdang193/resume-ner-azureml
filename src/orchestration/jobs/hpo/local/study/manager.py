"""Optuna study management utilities for HPO.

Handles study creation, resume, and state management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from common.shared.logging_utils import get_logger
from ..optuna.integration import import_optuna as _import_optuna, create_optuna_pruner

logger = get_logger(__name__)


class StudyManager:
    """Manages Optuna study creation, loading, and resume logic."""

    def __init__(
        self,
        backbone: str,
        hpo_config: Dict[str, Any],
        checkpoint_config: Optional[Dict[str, Any]] = None,
        restore_from_drive: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize study manager.

        Args:
            backbone: Model backbone name.
            hpo_config: HPO configuration dictionary.
            checkpoint_config: Optional checkpoint configuration.
            restore_from_drive: Optional function to restore checkpoint from Drive.
            output_dir: Base output directory for checkpoints.
        """
        self.backbone = backbone
        self.hpo_config = hpo_config
        self.checkpoint_config = checkpoint_config
        self.restore_from_drive = restore_from_drive

        # Lazy import optuna
        self.optuna, _, self.RandomSampler, _ = _import_optuna()

        # Extract configuration
        self.objective_metric = hpo_config["objective"]["metric"]
        self.goal = hpo_config["objective"]["goal"]
        self.direction = "maximize" if self.goal == "maximize" else "minimize"

        # Create pruner and sampler
        self.pruner = create_optuna_pruner(hpo_config)
        algorithm = hpo_config["sampling"]["algorithm"].lower()
        if algorithm == "random":
            self.sampler = self.RandomSampler()
        else:
            self.sampler = self.RandomSampler()  # Default to random

        self.checkpoint_enabled = (
            checkpoint_config is not None and checkpoint_config.get(
                "enabled", False)
        )

    def create_or_load_study(
        self,
        output_dir: Path,
        run_id: str,
        v2_study_folder: Optional[Path] = None,
    ) -> Tuple[Any, str, Path, str, bool]:
        """
        Create or load Optuna study with proper resume handling.

        Args:
            output_dir: Base output directory for checkpoints.
            run_id: Unique run ID for study naming.
            v2_study_folder: Optional v2 study folder path (if provided, study.db will be created here instead of legacy folder).

        Returns:
            Tuple of (study, study_name, storage_path, storage_uri, should_resume).
        """
        # Lazy imports to avoid circular dependency
        from ...hpo_helpers import create_study_name, setup_checkpoint_storage
        from ..checkpoint.manager import get_storage_uri

        # Resolve study_name FIRST (needed for {study_name} placeholder in storage_path)
        # Use temporary should_resume=False for initial study_name resolution
        # We'll recalculate should_resume after checking if study exists
        study_name = create_study_name(
            self.backbone,
            run_id,
            should_resume=False,
            checkpoint_config=self.checkpoint_config,
            hpo_config=self.hpo_config,
        )

        # Set up checkpoint storage with resolved study_name
        # If v2_study_folder is provided, use it for study.db location
        if v2_study_folder and self.checkpoint_config and self.checkpoint_config.get("enabled", False):
            # Create study.db directly in v2 folder
            storage_path = v2_study_folder / "study.db"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_uri = get_storage_uri(storage_path)

            # Check if should resume (if study.db already exists in v2 folder)
            should_resume = (
                self.checkpoint_config.get("auto_resume", True)
                and storage_path.exists()
            )

            # If not found in v2 folder, check if legacy folder has it (for migration)
            if not storage_path.exists() and self.restore_from_drive is not None:
                try:
                    restored = self.restore_from_drive(storage_path)
                    if restored:
                        logger.info(
                            f"Restored HPO checkpoint from Drive: {storage_path}")
                except Exception as e:
                    logger.debug(f"Drive backup not found for checkpoint: {e}")
        else:
            # Use legacy storage path resolution
            storage_path, storage_uri, should_resume = setup_checkpoint_storage(
                output_dir,
                self.checkpoint_config,
                self.backbone,
                study_name,
                restore_from_drive=self.restore_from_drive,
            )

        if should_resume:
            return self._load_existing_study(study_name, storage_path, storage_uri)
        else:
            return self._create_new_study(study_name, storage_path, storage_uri)

    def _load_existing_study(
        self, study_name: str, storage_path: Path, storage_uri: str
    ) -> Tuple[Any, str, Path, str, bool]:
        """Load existing study for resume."""
        logger.info(
            f"[HPO] Resuming optimization for {self.backbone} from checkpoint..."
        )
        logger.debug(f"Checkpoint: {storage_path}")
        try:
            study = self.optuna.create_study(
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=study_name,
                storage=storage_uri,
                load_if_exists=True,
            )

            # Check if HPO is already complete
            user_attrs = study.user_attrs if hasattr(
                study, "user_attrs") else {}
            hpo_complete = user_attrs.get(
                "hpo_complete", "false").lower() == "true"
            checkpoint_uploaded = (
                user_attrs.get("checkpoint_uploaded",
                               "false").lower() == "true"
            )

            if hpo_complete and checkpoint_uploaded:
                best_trial_num = user_attrs.get("best_trial_number", "unknown")
                completion_time = user_attrs.get(
                    "completion_timestamp", "unknown")
                logger.info(
                    f"✓ HPO already completed and checkpoint uploaded (best trial: {best_trial_num}, "
                    f"completed: {completion_time}). Skipping HPO execution."
                )
                return study, study_name, storage_path, storage_uri, True

            # Mark any RUNNING trials as FAILED (they were interrupted)
            self._mark_running_trials_as_failed(study)

            # Count completed trials
            completed_trials = len(
                [
                    t
                    for t in study.trials
                    if t.state == self.optuna.trial.TrialState.COMPLETE
                ]
            )
            running_trials = len(
                [
                    t
                    for t in study.trials
                    if t.state == self.optuna.trial.TrialState.RUNNING
                ]
            )
            logger.info(
                f"Loaded {len(study.trials)} existing trials ({completed_trials} completed, "
                f"{running_trials} marked as failed)"
            )
            return study, study_name, storage_path, storage_uri, True
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            logger.info("Creating new study instead...")
            # Use unique study name when resume fails
            study_name = f"hpo_{self.backbone}_{run_id}"
            study = self.optuna.create_study(
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=study_name,
                storage=storage_uri,
                load_if_exists=False,
            )
            return study, study_name, storage_path, storage_uri, False

    def _create_new_study(
        self, study_name: str, storage_path: Path, storage_uri: str
    ) -> Tuple[Any, str, Path, str, bool]:
        """Create new study."""
        if storage_uri:
            logger.info(
                f"[HPO] Starting optimization for {self.backbone} with checkpointing..."
            )
            logger.debug(f"Checkpoint: {storage_path}")
            # When checkpointing is enabled, use load_if_exists=True so we can resume
            # if the study already exists in the database (even if file was just created)
            load_if_exists = self.checkpoint_enabled
        else:
            load_if_exists = False

        study = self.optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=load_if_exists,
        )

        # If checkpointing is enabled and we loaded an existing study, check auto_resume
        should_resume = False
        if self.checkpoint_enabled and load_if_exists and len(study.trials) > 0:
            auto_resume = self.checkpoint_config.get("auto_resume", True)

            if auto_resume:
                # User wants to resume - update should_resume
                should_resume = True

                # Mark any RUNNING trials as FAILED (they were interrupted)
                self._mark_running_trials_as_failed(study)

                completed_trials = len(
                    [
                        t
                        for t in study.trials
                        if t.state == self.optuna.trial.TrialState.COMPLETE
                    ]
                )
                running_trials = len(
                    [
                        t
                        for t in study.trials
                        if t.state == self.optuna.trial.TrialState.RUNNING
                    ]
                )
                logger.info(
                    f"Loaded existing study with {len(study.trials)} trials "
                    f"({completed_trials} completed, {running_trials} marked as failed)"
                )
            else:
                # auto_resume: false but existing study found - require new study_name
                completed_trials = len(
                    [
                        t
                        for t in study.trials
                        if t.state == self.optuna.trial.TrialState.COMPLETE
                    ]
                )
                raise ValueError(
                    f"❌ Found existing study '{study_name}' with {len(study.trials)} trials "
                    f"({completed_trials} completed), but auto_resume=false.\n"
                    f"   To start fresh, you must use a different study_name.\n"
                    f"   Current study_name: '{study_name}'\n"
                    f"   Solution: Add 'study_name: \"hpo_{self.backbone}_new_name\"' to checkpoint config."
                )

        return study, study_name, storage_path, storage_uri, should_resume

    def _mark_running_trials_as_failed(self, study: Any) -> None:
        """Mark any RUNNING trials as FAILED (they were interrupted)."""
        # Can be disabled via config or environment variable
        cleanup_config = self.hpo_config.get("cleanup", {})
        skip_optuna_mark_config = cleanup_config.get(
            "disable_auto_optuna_mark", False)
        skip_optuna_mark_env = (
            os.environ.get("DISABLE_AUTO_OPTUNA_MARK", "").lower() == "true"
        )
        skip_optuna_mark = skip_optuna_mark_config or skip_optuna_mark_env

        running_trials = [
            t for t in study.trials if t.state == self.optuna.trial.TrialState.RUNNING
        ]

        if running_trials:
            if skip_optuna_mark:
                source = (
                    "config" if not skip_optuna_mark_env else "environment variable"
                )
                logger.info(
                    f"Found {len(running_trials)} RUNNING trials from previous session. "
                    f"Skipping automatic marking (via {source})."
                )
            else:
                logger.warning(
                    f"Found {len(running_trials)} RUNNING trials from previous session. "
                    f"Marking them as FAILED (interrupted)."
                )
                for trial in running_trials:
                    study.tell(
                        trial.number, state=self.optuna.trial.TrialState.FAIL)
