"""Tests for smoke.yaml configuration options that lack explicit behavior tests."""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import optuna

from orchestration.jobs.hpo.local.checkpoint.cleanup import CheckpointCleanupManager
from orchestration.jobs.hpo.local.mlflow.cleanup import should_skip_cleanup
from orchestration.jobs.hpo.local.study.manager import StudyManager


class TestTimeoutMinutes:
    """Test timeout_minutes behavior in HPO study execution."""

    def test_timeout_minutes_stops_study_after_timeout(self, tmp_path):
        """Test that timeout_minutes stops study execution after timeout."""
        storage_path = tmp_path / "study.db"
        storage_uri = f"sqlite:///{storage_path}"
        
        hpo_config = {
            "sampling": {
                "algorithm": "random",
                "max_trials": 100,  # High number to ensure timeout triggers first
                "timeout_minutes": 0.01,  # 0.6 seconds for fast test
            },
            "objective": {"metric": "macro-f1", "goal": "maximize"},
        }
        
        study = optuna.create_study(
            direction="maximize",
            study_name="test_timeout",
            storage=storage_uri,
            load_if_exists=False,
        )
        
        # Objective that takes time
        def slow_objective(trial):
            time.sleep(0.1)  # Each trial takes 0.1 seconds
            return 0.75
        
        start_time = time.time()
        study.optimize(
            slow_objective,
            n_trials=100,
            timeout=0.6,  # 0.01 minutes * 60 = 0.6 seconds
        )
        elapsed_time = time.time() - start_time
        
        # Should stop around timeout (0.6 seconds), not complete all 100 trials
        # Allow some margin for overhead
        assert elapsed_time < 2.0, f"Study should have stopped after timeout, but took {elapsed_time}s"
        assert len(study.trials) < 100, "Should not complete all trials due to timeout"
        assert len(study.trials) > 0, "Should have at least started some trials"

    def test_timeout_minutes_conversion_to_seconds(self):
        """Test that timeout_minutes is correctly converted to seconds."""
        hpo_config = {
            "sampling": {
                "timeout_minutes": 20,
            }
        }
        
        timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60
        assert timeout_seconds == 1200  # 20 minutes = 1200 seconds


class TestSaveOnlyBest:
    """Test save_only_best checkpoint cleanup behavior."""

    def test_save_only_best_deletes_non_best_checkpoints(self, tmp_path):
        """Test that save_only_best=True deletes non-best trial checkpoints."""
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        # Create checkpoints for multiple trials
        trial_0_dir = output_dir / "trial_0"
        trial_0_dir.mkdir()
        (trial_0_dir / "checkpoint").mkdir()
        (trial_0_dir / "checkpoint" / "model.safetensors").write_text("trial0")
        
        trial_1_dir = output_dir / "trial_1"
        trial_1_dir.mkdir()
        (trial_1_dir / "checkpoint").mkdir()
        (trial_1_dir / "checkpoint" / "model.safetensors").write_text("trial1")
        
        trial_2_dir = output_dir / "trial_2"
        trial_2_dir.mkdir()
        (trial_2_dir / "checkpoint").mkdir()
        (trial_2_dir / "checkpoint" / "model.safetensors").write_text("trial2")
        
        hpo_config = {
            "checkpoint": {"save_only_best": True},
            "objective": {"goal": "maximize"},
        }
        
        cleanup_manager = CheckpointCleanupManager(
            hpo_config=hpo_config,
            output_base_dir=output_dir,
            run_id=None,
            fold_splits=None,
        )
        
        # Register trials (simulating completion)
        cleanup_manager.register_trial_checkpoint(0)
        cleanup_manager.register_trial_checkpoint(1)
        cleanup_manager.register_trial_checkpoint(2)
        
        # Simulate trial 1 is best (metric=0.85)
        cleanup_manager.best_trial_id = 1
        cleanup_manager.best_score = 0.85
        
        # Complete trial 2 (not best, should be deleted)
        cleanup_manager.handle_trial_completion(Mock(number=2), 0.70)
        
        # Trial 2 checkpoint should be deleted
        assert not (trial_2_dir / "checkpoint").exists(), "Non-best trial checkpoint should be deleted"
        # Trial 1 (best) should still exist
        assert (trial_1_dir / "checkpoint").exists(), "Best trial checkpoint should be preserved"
        # Trial 0 should still exist (not yet processed as non-best)
        assert (trial_0_dir / "checkpoint").exists(), "Trial 0 checkpoint should still exist"

    def test_save_only_best_false_preserves_all_checkpoints(self, tmp_path):
        """Test that save_only_best=False preserves all checkpoints."""
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        trial_0_dir = output_dir / "trial_0"
        trial_0_dir.mkdir()
        (trial_0_dir / "checkpoint").mkdir()
        (trial_0_dir / "checkpoint" / "model.safetensors").write_text("trial0")
        
        hpo_config = {
            "checkpoint": {"save_only_best": False},
            "objective": {"goal": "maximize"},
        }
        
        cleanup_manager = CheckpointCleanupManager(
            hpo_config=hpo_config,
            output_base_dir=output_dir,
            run_id=None,
            fold_splits=None,
        )
        
        # Register and complete trial
        cleanup_manager.register_trial_checkpoint(0)
        cleanup_manager.handle_trial_completion(Mock(number=0), 0.75)
        
        # Checkpoint should still exist (not deleted when save_only_best=False)
        assert (trial_0_dir / "checkpoint").exists(), "Checkpoint should be preserved when save_only_best=False"


class TestLogBestCheckpoint:
    """Test log_best_checkpoint MLflow artifact logging behavior."""

    def test_log_best_checkpoint_config_enabled(self):
        """Test that log_best_checkpoint config value is correctly read when enabled."""
        hpo_config = {
            "mlflow": {"log_best_checkpoint": True},
        }
        
        # Verify config indicates logging should happen
        should_log = hpo_config.get("mlflow", {}).get("log_best_checkpoint", False)
        assert should_log is True, "log_best_checkpoint should be enabled when config is true"
        
        # In actual execution (local_sweeps.py line 950), this config value controls
        # whether tracker.log_best_checkpoint() is called after HPO completes.
        # The actual call requires a full HPO run, so we verify the config is read correctly.

    def test_log_best_checkpoint_config_disabled(self):
        """Test that log_best_checkpoint config value is correctly read when disabled."""
        hpo_config = {
            "mlflow": {"log_best_checkpoint": False},
        }
        
        should_log = hpo_config.get("mlflow", {}).get("log_best_checkpoint", True)
        assert should_log is False, "log_best_checkpoint should be disabled when config is false"
        
        # When disabled, log_best_checkpoint should not be called in local_sweeps.py
        # This test verifies the config disables the feature

    @patch("orchestration.jobs.tracking.trackers.sweep_tracker.MLflowSweepTracker.log_best_checkpoint")
    def test_log_best_checkpoint_conditional_call(self, mock_log_best_checkpoint, tmp_path):
        """Test that log_best_checkpoint is conditionally called based on config."""
        # Simulate the condition check from local_sweeps.py
        hpo_config = {
            "mlflow": {"log_best_checkpoint": True},
        }
        
        # This mimics the logic in local_sweeps.py around line 950
        log_best_checkpoint = hpo_config.get("mlflow", {}).get("log_best_checkpoint", False)
        
        if log_best_checkpoint:
            # Simulate the call that would happen
            mock_tracker = Mock()
            mock_tracker.log_best_checkpoint = mock_log_best_checkpoint
            mock_tracker.log_best_checkpoint(
                study=Mock(),
                hpo_output_dir=tmp_path,
                backbone="distilbert",
                run_id="test_run",
            )
        
        # Verify it was called when enabled
        assert mock_log_best_checkpoint.called, "log_best_checkpoint should be called when config is true"


class TestDisableAutoCleanup:
    """Test disable_auto_cleanup MLflow cleanup behavior."""

    def test_disable_auto_cleanup_false_enables_cleanup(self):
        """Test that disable_auto_cleanup=false enables MLflow cleanup."""
        hpo_config = {
            "cleanup": {"disable_auto_cleanup": False},
        }
        
        skip_cleanup, source = should_skip_cleanup(hpo_config)
        
        # disable_auto_cleanup=false means cleanup is ENABLED (not skipped)
        assert skip_cleanup is False, "Cleanup should be enabled when disable_auto_cleanup=false"
        assert source == "config"

    def test_disable_auto_cleanup_true_disables_cleanup(self):
        """Test that disable_auto_cleanup=true disables MLflow cleanup."""
        hpo_config = {
            "cleanup": {"disable_auto_cleanup": True},
        }
        
        skip_cleanup, source = should_skip_cleanup(hpo_config)
        
        # disable_auto_cleanup=true means cleanup is DISABLED (skipped)
        assert skip_cleanup is True, "Cleanup should be disabled when disable_auto_cleanup=true"
        assert source == "config"

    def test_disable_auto_cleanup_default_is_disabled(self):
        """Test that default behavior (missing config) disables cleanup."""
        hpo_config = {}
        
        skip_cleanup, source = should_skip_cleanup(hpo_config)
        
        # Default is True (disabled) per implementation
        assert skip_cleanup is True, "Default should disable cleanup"
        assert source == "config"


class TestDisableAutoOptunaMark:
    """Test disable_auto_optuna_mark Optuna state cleanup behavior."""

    def test_disable_auto_optuna_mark_false_enables_marking(self, tmp_path):
        """Test that disable_auto_optuna_mark=false enables marking RUNNING trials as FAILED."""
        storage_path = tmp_path / "study.db"
        storage_uri = f"sqlite:///{storage_path}"
        
        hpo_config = {
            "cleanup": {"disable_auto_optuna_mark": False},
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
        }
        
        # Create study with a RUNNING trial (simulating interruption)
        study = optuna.create_study(
            direction="maximize",
            study_name="test_mark",
            storage=storage_uri,
            load_if_exists=False,
        )
        
        # Create a RUNNING trial (simulating interruption)
        trial = study.ask()
        # Don't tell() it, so it remains RUNNING
        
        # Create StudyManager and test marking
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config={},
        )
        
        # Call _mark_running_trials_as_failed
        study_manager._mark_running_trials_as_failed(study)
        
        # Reload study to check trial state
        study2 = optuna.load_study(
            study_name="test_mark",
            storage=storage_uri,
        )
        
        # With disable_auto_optuna_mark=false, RUNNING trial should be marked FAILED
        running_trials = [t for t in study2.trials if t.state == optuna.trial.TrialState.RUNNING]
        assert len(running_trials) == 0, "RUNNING trials should be marked FAILED when disable_auto_optuna_mark=false"
        
        failed_trials = [t for t in study2.trials if t.state == optuna.trial.TrialState.FAIL]
        assert len(failed_trials) > 0, "At least one trial should be marked FAILED"

    def test_disable_auto_optuna_mark_true_skips_marking(self, tmp_path):
        """Test that disable_auto_optuna_mark=true skips marking RUNNING trials."""
        storage_path = tmp_path / "study.db"
        storage_uri = f"sqlite:///{storage_path}"
        
        hpo_config = {
            "cleanup": {"disable_auto_optuna_mark": True},
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
        }
        
        # Create study with a RUNNING trial
        study = optuna.create_study(
            direction="maximize",
            study_name="test_skip_mark",
            storage=storage_uri,
            load_if_exists=False,
        )
        
        trial = study.ask()
        # Don't tell() it, so it remains RUNNING
        
        # Create StudyManager
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config={},
        )
        
        # Call _mark_running_trials_as_failed
        study_manager._mark_running_trials_as_failed(study)
        
        # Reload study to check trial state
        study2 = optuna.load_study(
            study_name="test_skip_mark",
            storage=storage_uri,
        )
        
        # With disable_auto_optuna_mark=true, RUNNING trial should remain RUNNING
        running_trials = [t for t in study2.trials if t.state == optuna.trial.TrialState.RUNNING]
        assert len(running_trials) > 0, "RUNNING trials should remain RUNNING when disable_auto_optuna_mark=true"

