"""Checkpoint cleanup utilities for HPO trials.

Manages checkpoint lifecycle: tracking, best trial detection, and cleanup.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.logging_utils import get_logger
from ..optuna.integration import import_optuna as _import_optuna

logger = get_logger(__name__)


class CheckpointCleanupManager:
    """Manages checkpoint cleanup for HPO trials."""

    def __init__(
        self,
        output_base_dir: Path,
        hpo_config: Dict[str, Any],
        run_id: Optional[str] = None,
        fold_splits: Optional[List] = None,
    ):
        """
        Initialize checkpoint cleanup manager.

        Args:
            output_base_dir: Base output directory for all trials.
            hpo_config: HPO configuration dictionary.
            run_id: Optional run ID for directory naming.
            fold_splits: Optional fold splits for CV (to check old structure).
        """
        self.output_base_dir = output_base_dir
        self.hpo_config = hpo_config
        self.run_id = run_id
        self.fold_splits = fold_splits

        # State for tracking best trial
        self.best_trial_id: Optional[int] = None
        self.best_score: Optional[float] = None
        # trial_id -> list of checkpoint paths
        self.checkpoint_map: Dict[int, List[Path]] = {}
        self.completed_trials: List[int] = []

        # Check if cleanup is enabled
        checkpoint_config = hpo_config.get("checkpoint", {})
        self.save_only_best = checkpoint_config.get("save_only_best", False)

    def get_checkpoint_paths(self, trial_num: int) -> List[Path]:
        """
        Get all checkpoint paths for a trial (all folds for CV, single for non-CV, plus refit if exists).

        Args:
            trial_num: Trial number.

        Returns:
            List of checkpoint directory paths.
        """
        run_suffix = f"_{self.run_id}" if self.run_id else ""
        paths = []
        trial_base_dir = self.output_base_dir / \
            f"trial_{trial_num}{run_suffix}"

        # NEW STRUCTURE: Check for refit checkpoint first (preferred)
        refit_checkpoint_dir = trial_base_dir / "refit" / "checkpoint"
        if refit_checkpoint_dir.exists():
            paths.append(refit_checkpoint_dir)

        # NEW STRUCTURE: Check cv/foldN/checkpoint/ directories
        cv_dir = trial_base_dir / "cv"
        if cv_dir.exists():
            for fold_dir in cv_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold"):
                    checkpoint_dir = fold_dir / "checkpoint"
                    if checkpoint_dir.exists():
                        paths.append(checkpoint_dir)

        # OLD STRUCTURE: Check trial_<n>_fold<k>/checkpoint/ directories (backward compatibility)
        # IMPORTANT: Always check old structure for CV, even if new structure exists, because
        # training script may create checkpoints in old structure
        if self.fold_splits is not None:
            for fold_idx in range(len(self.fold_splits)):
                checkpoint_dir = (
                    self.output_base_dir
                    / f"trial_{trial_num}{run_suffix}_fold{fold_idx}"
                    / "checkpoint"
                )
                logger.info(
                    f"Checking old structure checkpoint for trial {trial_num}, fold {fold_idx}: {checkpoint_dir} (exists={checkpoint_dir.exists()})"
                )
                if checkpoint_dir.exists():
                    paths.append(checkpoint_dir)
                    logger.info(
                        f"Found old structure checkpoint for trial {trial_num}, fold {fold_idx}: {checkpoint_dir}"
                    )
        else:
            # Single training: get single checkpoint (if no refit exists)
            if not refit_checkpoint_dir.exists():
                checkpoint_dir = trial_base_dir / "checkpoint"
                if checkpoint_dir.exists():
                    paths.append(checkpoint_dir)
        return paths

    def delete_checkpoint_paths(self, paths: List[Path], trial_num: int) -> None:
        """
        Delete checkpoint paths and log the operation.

        Args:
            paths: List of checkpoint paths to delete.
            trial_num: Trial number for logging.
        """
        for path in paths:
            try:
                shutil.rmtree(path)
                logger.debug(
                    f"Deleted checkpoint for trial {trial_num}: {path}")
            except Exception as e:
                logger.warning(
                    f"Could not delete checkpoint for trial {trial_num} at {path}: {e}"
                )

    def register_trial_checkpoint(self, trial_num: int) -> None:
        """
        Register a trial's checkpoint paths.

        Args:
            trial_num: Trial number.
        """
        if not self.save_only_best:
            return

        trial_checkpoint_paths = self.get_checkpoint_paths(trial_num)
        logger.info(
            f"Registering checkpoints for trial {trial_num}: found {len(trial_checkpoint_paths)} paths: {[str(p) for p in trial_checkpoint_paths]}"
        )
        if trial_checkpoint_paths:
            self.checkpoint_map[trial_num] = trial_checkpoint_paths
        self.completed_trials.append(trial_num)

    def initialize_best_trial_from_study(
        self, trial: Any, metric_value: float
    ) -> Optional[float]:
        """
        Initialize best trial state from existing study (for resume scenarios).

        Args:
            trial: Current Optuna trial.
            metric_value: Current trial's metric value.

        Returns:
            Metric value if this is the first trial, None otherwise.
        """
        if not self.save_only_best:
            return None

        if self.best_trial_id is not None:
            return None  # Already initialized

        # Check if study has existing completed trials (resume scenario)
        optuna_module, _, _, _ = _import_optuna()
        try:
            study = trial.study
            goal = self.hpo_config["objective"]["goal"]

            # Find best completed trial from existing study
            best_existing_trial = None
            best_existing_value = None

            for t in study.trials:
                if (
                    t.state == optuna_module.trial.TrialState.COMPLETE
                    and t.value is not None
                    and t.number != trial.number  # Exclude current trial
                ):
                    # Register existing trial's checkpoint if it exists
                    existing_paths = self.get_checkpoint_paths(t.number)
                    if existing_paths:
                        self.checkpoint_map[t.number] = existing_paths

                    # Track best existing trial
                    if best_existing_value is None:
                        best_existing_trial = t.number
                        best_existing_value = t.value
                    else:
                        if goal == "maximize":
                            if t.value > best_existing_value:
                                best_existing_trial = t.number
                                best_existing_value = t.value
                        else:  # minimize
                            if t.value < best_existing_value:
                                best_existing_trial = t.number
                                best_existing_value = t.value

            # Initialize state with best existing trial if found
            if best_existing_trial is not None:
                self.best_trial_id = best_existing_trial
                self.best_score = best_existing_value
                logger.debug(
                    f"Resumed: Found existing best trial {best_existing_trial} "
                    f"(metric={best_existing_value:.6f}) from "
                    f"{len([t for t in study.trials if t.state == optuna_module.trial.TrialState.COMPLETE])} completed trials"
                )
            else:
                # No existing completed trials - this is truly the first trial
                self.best_trial_id = trial.number
                self.best_score = metric_value
                logger.debug(
                    f"First trial {trial.number} is best (metric={metric_value:.6f})"
                )
                return metric_value
        except (AttributeError, Exception) as e:
            # If we can't access study, assume this is the first trial
            self.best_trial_id = trial.number
            self.best_score = metric_value
            logger.debug(
                f"Could not access study, assuming first trial {trial.number} "
                f"(metric={metric_value:.6f}): {e}"
            )
            return metric_value

        return None

    def handle_trial_completion(
        self, trial: Any, metric_value: float
    ) -> Optional[float]:
        """
        Handle checkpoint cleanup after trial completes.

        Args:
            trial: Completed Optuna trial.
            metric_value: Trial's metric value.

        Returns:
            Metric value (for early return in first trial case).
        """
        if not self.save_only_best:
            return None

        try:
            # Initialize state from existing study (for resume scenarios)
            early_return = self.initialize_best_trial_from_study(
                trial, metric_value)
            if early_return is not None:
                return early_return

            # Check if this is a new best trial
            goal = self.hpo_config["objective"]["goal"]
            is_new_best = False

            if goal == "maximize":
                is_new_best = metric_value > self.best_score
            else:  # minimize
                is_new_best = metric_value < self.best_score

            if is_new_best:
                # New best trial found - delete ALL non-best checkpoints
                old_best = self.best_trial_id
                self.best_trial_id = trial.number
                self.best_score = metric_value

                logger.debug(
                    f"New best trial {trial.number} (metric={metric_value:.6f}, "
                    f"previous best: trial {old_best})"
                )

                # Delete all non-best checkpoints
                for trial_id, checkpoint_paths in list(self.checkpoint_map.items()):
                    if trial_id != self.best_trial_id:
                        self.delete_checkpoint_paths(
                            checkpoint_paths, trial_id)
                        del self.checkpoint_map[trial_id]

                return metric_value

            # Not a new best - delete this trial's checkpoint immediately
            logger.debug(
                f"Trial {trial.number} is not best (metric={metric_value:.6f}, "
                f"best: trial {self.best_trial_id} with {self.best_score:.6f})"
            )

            # Delete this non-best trial's checkpoint
            if trial.number in self.checkpoint_map:
                self.delete_checkpoint_paths(
                    self.checkpoint_map[trial.number], trial.number
                )
                del self.checkpoint_map[trial.number]

        except Exception as e:
            # Don't fail HPO if checkpoint cleanup fails
            logger.warning(
                f"Error during checkpoint cleanup for trial {trial.number}: {e}"
            )

        return None

    def final_cleanup(self) -> None:
        """Final cleanup: delete all non-best checkpoints after HPO completes, preserving both CV and refit checkpoints for best trial."""
        if not self.save_only_best:
            return

        if self.best_trial_id is None:
            return

        # For best trial: preserve BOTH refit and CV checkpoints (user requirement)
        # Re-query checkpoint paths to get latest state (including refit if it was added after trial completion)
        best_trial_num = self.best_trial_id
        run_suffix = f"_{self.run_id}" if self.run_id else ""
        trial_base_dir = self.output_base_dir / \
            f"trial_{best_trial_num}{run_suffix}"

        # Log trial directory structure for debugging
        logger.info(
            f"Final cleanup: checking trial_base_dir={trial_base_dir} (exists={trial_base_dir.exists()})")
        if trial_base_dir.exists():
            subdirs = [d.name for d in trial_base_dir.iterdir() if d.is_dir()]
            logger.info(
                f"Final cleanup: trial_base_dir subdirectories: {subdirs}")
        else:
            logger.warning(
                f"Final cleanup: trial_base_dir does not exist: {trial_base_dir}")

        best_trial_paths = self.get_checkpoint_paths(best_trial_num)
        logger.info(
            f"Final cleanup: found {len(best_trial_paths)} checkpoint paths for best trial {best_trial_num}: {[str(p) for p in best_trial_paths]}")

        # Check for old structure checkpoints if new structure not found
        if not best_trial_paths and self.fold_splits is not None:
            logger.info(
                f"Final cleanup: No checkpoints found in new structure, checking old structure...")
            for fold_idx in range(len(self.fold_splits)):
                old_structure_path = self.output_base_dir / \
                    f"trial_{best_trial_num}{run_suffix}_fold{fold_idx}" / \
                    "checkpoint"
                logger.info(
                    f"Final cleanup: Checking old structure path: {old_structure_path} (exists={old_structure_path.exists()})")

        # Update checkpoint_map with latest paths (in case refit was added)
        if best_trial_paths:
            self.checkpoint_map[best_trial_num] = best_trial_paths

        # Check if best trial has refit checkpoint
        has_refit = any("refit" in str(p) for p in best_trial_paths)
        has_cv = any("cv" in str(p) or "fold" in str(p)
                     for p in best_trial_paths)

        # Delete all non-best checkpoints
        for trial_id, checkpoint_paths in list(self.checkpoint_map.items()):
            if trial_id != self.best_trial_id:
                self.delete_checkpoint_paths(checkpoint_paths, trial_id)
                del self.checkpoint_map[trial_id]

        deleted_count = len(self.completed_trials) - 1

        logger.info(
            f"Final cleanup: kept checkpoints for best trial {self.best_trial_id} "
            f"(metric={self.best_score:.6f}, CV={'yes' if has_cv else 'no'}, refit={'yes' if has_refit else 'no'}), "
            f"deleted {deleted_count} non-best checkpoints"
        )
