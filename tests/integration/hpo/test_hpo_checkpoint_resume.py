"""Component tests for HPO checkpoint and resume functionality."""

import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import optuna

from orchestration.jobs.hpo.local.study.manager import StudyManager
from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path, get_storage_uri


class TestCheckpointCreation:
    """Test checkpoint file creation and storage path resolution."""

    def test_checkpoint_storage_path_resolution(self, tmp_path):
        """Test that storage_path template is resolved correctly with placeholders."""
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_{backbone}_smoke_test_path_testing_23",
            "storage_path": "{study_name}/study.db",
        }
        
        output_dir = tmp_path / "outputs" / "hpo"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        
        # Verify path was created (parent directory exists, file will be created by Optuna)
        assert storage_path is not None
        assert storage_path.parent.exists()
        assert storage_path.name == "study.db"
        # Verify {study_name} placeholder was replaced
        assert "hpo_distilbert_smoke_test_path_testing_23" in str(storage_path)

    def test_checkpoint_storage_path_with_backbone_placeholder(self, tmp_path):
        """Test that {backbone} placeholder is replaced in storage_path."""
        checkpoint_config = {
            "enabled": True,
            "storage_path": "{backbone}/study.db",
        }
        
        output_dir = tmp_path / "outputs" / "hpo"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
        )
        
        assert storage_path is not None
        assert "distilbert" in str(storage_path)
        assert storage_path.name == "study.db"

    def test_checkpoint_disabled_returns_none(self, tmp_path):
        """Test that disabled checkpointing returns None."""
        checkpoint_config = {
            "enabled": False,
            "storage_path": "{backbone}/study.db",
        }
        
        storage_path = resolve_storage_path(
            output_dir=tmp_path,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
        )
        
        assert storage_path is None

    def test_storage_uri_conversion(self, tmp_path):
        """Test that storage path is converted to Optuna storage URI."""
        storage_path = tmp_path / "study.db"
        storage_path.touch()
        
        storage_uri = get_storage_uri(storage_path)
        
        assert storage_uri is not None
        assert storage_uri.startswith("sqlite:///")
        assert str(storage_path.resolve()) in storage_uri


class TestCheckpointResume:
    """Test resume functionality from existing checkpoints."""

    def test_resume_from_existing_checkpoint(self, tmp_path):
        """Test that study can be resumed from existing checkpoint file."""
        # Create checkpoint file with existing study
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
        }
        
        # Create initial study and add a trial
        study_name = "hpo_distilbert_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        # Create study and add a trial
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        
        # Now resume from checkpoint
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        study2, loaded_study_name, loaded_storage_path, loaded_storage_uri, should_resume = (
            study_manager.create_or_load_study(
                output_dir=output_dir,
                run_id="test_run_123",
            )
        )
        
        # Verify resume happened
        assert should_resume is True
        assert len(study2.trials) == 1
        assert study2.trials[0].value == 0.75
        assert study2.trials[0].state == optuna.trial.TrialState.COMPLETE

    def test_resume_preserves_trials(self, tmp_path):
        """Test that all completed trials are preserved when resuming."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_resume_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
        }
        
        # Create study with multiple trials
        study_name = "hpo_distilbert_resume_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        
        # Add 3 trials
        for i in range(3):
            trial = study1.ask()
            study1.tell(trial, 0.7 + i * 0.05)
        
        # Resume and verify all trials are preserved
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        study2, _, _, _, should_resume = study_manager.create_or_load_study(
            output_dir=output_dir,
            run_id="test_run_456",
        )
        
        assert should_resume is True
        assert len(study2.trials) == 3
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study2.trials)
        # Use pytest.approx for floating point comparison
        assert [t.value for t in study2.trials] == pytest.approx([0.7, 0.75, 0.8])

    def test_resume_marks_running_trials_as_failed(self, tmp_path):
        """Test that RUNNING trials from previous session are marked as FAILED."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_interrupted_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
            "cleanup": {"disable_auto_optuna_mark": False},
        }
        
        # Create study with a RUNNING trial (simulating interruption)
        study_name = "hpo_distilbert_interrupted_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        
        # Add completed trial
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        
        # Add RUNNING trial (simulating interruption)
        trial2 = study1.ask()
        # Don't call tell() - trial remains RUNNING
        
        # Resume and verify RUNNING trial is marked as FAILED
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        study2, _, _, _, should_resume = study_manager.create_or_load_study(
            output_dir=output_dir,
            run_id="test_run_789",
        )
        
        assert should_resume is True
        assert len(study2.trials) == 2
        
        # First trial should be COMPLETE
        assert study2.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study2.trials[0].value == 0.75
        
        # Second trial should be FAILED (was RUNNING, now marked as FAILED)
        assert study2.trials[1].state == optuna.trial.TrialState.FAIL

    def test_resume_with_auto_resume_false_raises_error(self, tmp_path):
        """Test that auto_resume=false raises error when checkpoint exists."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_no_resume_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": False,  # Disable auto resume
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
        }
        
        # Create existing study
        study_name = "hpo_distilbert_no_resume_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        
        # Try to resume with auto_resume=false - should raise error
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        with pytest.raises(ValueError, match="auto_resume=false"):
            study_manager.create_or_load_study(
                output_dir=output_dir,
                run_id="test_run_no_resume",
            )

    def test_resume_continues_trial_numbering(self, tmp_path):
        """Test that resumed study continues trial numbering from where it left off."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_numbering_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
        }
        
        # Create study with 2 trials
        study_name = "hpo_distilbert_numbering_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        
        trial0 = study1.ask()
        study1.tell(trial0, 0.7)
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        
        # Resume and add new trial
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        study2, _, _, _, should_resume = study_manager.create_or_load_study(
            output_dir=output_dir,
            run_id="test_run_numbering",
        )
        
        assert should_resume is True
        assert len(study2.trials) == 2
        
        # Add new trial - should be trial number 2
        trial2 = study2.ask()
        assert trial2.number == 2
        study2.tell(trial2, 0.8)
        
        assert len(study2.trials) == 3
        assert study2.trials[2].number == 2
        assert study2.trials[2].value == 0.8


class TestCheckpointSmokeYaml:
    """Test checkpoint behavior with smoke.yaml parameters."""

    def test_checkpoint_smoke_yaml_study_name_template(self, tmp_path):
        """Test that smoke.yaml study_name template is resolved correctly."""
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_{backbone}_smoke_test_path_testing_23",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
            "save_only_best": True,
        }
        
        output_dir = tmp_path / "outputs" / "hpo"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        
        # Verify study_name template was used
        assert "hpo_distilbert_smoke_test_path_testing_23" in str(storage_path)

    def test_checkpoint_smoke_yaml_storage_path_template(self, tmp_path):
        """Test that smoke.yaml storage_path template ({study_name}/study.db) works."""
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_{backbone}_smoke_test_path_testing_23",
            "storage_path": "{study_name}/study.db",  # smoke.yaml value
            "auto_resume": True,
        }
        
        output_dir = tmp_path / "outputs" / "hpo"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        
        # Verify storage_path template was resolved
        assert storage_path.name == "study.db"
        assert study_name in str(storage_path.parent.name)

    def test_checkpoint_smoke_yaml_auto_resume_true(self, tmp_path):
        """Test that smoke.yaml auto_resume=true allows resume."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_smoke_resume",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,  # smoke.yaml value
        }
        
        hpo_config = {
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "sampling": {"algorithm": "random"},
            "early_termination": {"enabled": False},
        }
        
        # Create existing study
        study_name = "hpo_distilbert_smoke_resume"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        
        # Resume should work with auto_resume=true
        study_manager = StudyManager(
            backbone="distilbert",
            hpo_config=hpo_config,
            checkpoint_config=checkpoint_config,
        )
        
        study2, _, _, _, should_resume = study_manager.create_or_load_study(
            output_dir=output_dir,
            run_id="test_run_smoke",
        )
        
        assert should_resume is True
        assert len(study2.trials) == 1


class TestCheckpointFileIO:
    """Test that checkpoint files are actually created and can be read."""

    def test_checkpoint_file_exists_after_study_creation(self, tmp_path):
        """Test that study.db file exists after study creation."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_file_test",
            "storage_path": "{study_name}/study.db",
        }
        
        study_name = "hpo_distilbert_file_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        
        # Verify file exists
        assert storage_path.exists()
        assert storage_path.is_file()
        
        # Verify it's a valid SQLite database
        conn = sqlite3.connect(str(storage_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Optuna creates tables for study storage
        assert len(tables) > 0

    def test_checkpoint_file_persists_trials(self, tmp_path):
        """Test that trials are persisted to checkpoint file and can be read back."""
        output_dir = tmp_path / "outputs" / "hpo"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_persist_test",
            "storage_path": "{study_name}/study.db",
        }
        
        study_name = "hpo_distilbert_persist_test"
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri = get_storage_uri(storage_path)
        
        # Create study and add trials
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )
        
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        trial2 = study1.ask()
        study1.tell(trial2, 0.80)
        
        # Close study (simulate process exit)
        del study1
        
        # Load study from same file
        study2 = optuna.load_study(
            study_name=study_name,
            storage=storage_uri,
        )
        
        # Verify trials are preserved
        assert len(study2.trials) == 2
        assert study2.trials[0].value == 0.75
        assert study2.trials[1].value == 0.80
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study2.trials)

    def test_checkpoint_file_can_be_moved_and_loaded(self, tmp_path):
        """Test that checkpoint file can be moved to different location and still loaded."""
        output_dir1 = tmp_path / "outputs1" / "hpo"
        output_dir2 = tmp_path / "outputs2" / "hpo"
        
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_move_test",
            "storage_path": "{study_name}/study.db",
        }
        
        study_name = "hpo_distilbert_move_test"
        storage_path1 = resolve_storage_path(
            output_dir=output_dir1,
            checkpoint_config=checkpoint_config,
            backbone="distilbert",
            study_name=study_name,
        )
        storage_uri1 = get_storage_uri(storage_path1)
        
        # Create study and add trial
        study1 = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_uri1,
            load_if_exists=False,
        )
        trial1 = study1.ask()
        study1.tell(trial1, 0.75)
        del study1
        
        # Move checkpoint file
        storage_path2 = output_dir2 / study_name / "study.db"
        storage_path2.parent.mkdir(parents=True, exist_ok=True)
        storage_path2.write_bytes(storage_path1.read_bytes())
        storage_uri2 = get_storage_uri(storage_path2)
        
        # Load from new location
        study2 = optuna.load_study(
            study_name=study_name,
            storage=storage_uri2,
        )
        
        # Verify trial is preserved
        assert len(study2.trials) == 1
        assert study2.trials[0].value == 0.75
