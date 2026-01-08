"""Component tests for path structure validation in HPO workflow."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestration.paths import resolve_output_path, is_v2_path, find_study_by_hash, find_trial_by_hash
from orchestration.naming_centralized import create_naming_context, build_output_path


class TestPathStructureV2:
    """Test v2 path structure (study-{hash}/trial-{hash})."""

    def test_study_folder_naming(self, tmp_path):
        """Test that study folder follows v2 pattern: study-{study8}."""
        study_key_hash = "a" * 64
        study8 = study_key_hash[:8]
        
        # Create study folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        study_folder.mkdir(parents=True)
        
        # Verify naming pattern
        assert study_folder.name.startswith("study-")
        assert len(study_folder.name) == len("study-") + 8
        assert study_folder.name == f"study-{study8}"

    def test_trial_folder_naming(self, tmp_path):
        """Test that trial folder follows v2 pattern: trial-{trial8}."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create trial folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        
        # Verify naming pattern
        assert trial_folder.name.startswith("trial-")
        assert len(trial_folder.name) == len("trial-") + 8
        assert trial_folder.name == f"trial-{trial8}"

    def test_refit_folder_structure(self, tmp_path):
        """Test that refit folder is under trial folder: trial-{trial8}/refit."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create refit folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        refit_folder = trial_folder / "refit"
        refit_folder.mkdir(parents=True)
        
        # Verify structure
        assert refit_folder.parent == trial_folder
        assert refit_folder.name == "refit"
        assert "trial-" in str(refit_folder)
        assert "refit" in str(refit_folder)

    def test_cv_fold_folder_structure(self, tmp_path):
        """Test that CV fold folders are under trial: trial-{trial8}/cv/fold{idx}."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create CV fold folders
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        
        # Create folds (smoke.yaml: n_splits=2)
        for fold_idx in range(2):
            fold_folder = trial_folder / "cv" / f"fold{fold_idx}"
            fold_folder.mkdir(parents=True)
            
            # Verify structure
            assert fold_folder.parent.parent == trial_folder
            assert fold_folder.parent.name == "cv"
            assert fold_folder.name == f"fold{fold_idx}"

    def test_checkpoint_location_in_trial(self, tmp_path):
        """Test that checkpoint is in trial folder: trial-{trial8}/checkpoint."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create checkpoint folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        checkpoint_folder = trial_folder / "checkpoint"
        checkpoint_folder.mkdir(parents=True)
        
        # Verify structure
        assert checkpoint_folder.parent == trial_folder
        assert checkpoint_folder.name == "checkpoint"

    def test_checkpoint_location_in_refit(self, tmp_path):
        """Test that refit checkpoint is in refit folder: trial-{trial8}/refit/checkpoint."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create refit checkpoint folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        refit_folder = trial_folder / "refit"
        refit_checkpoint = refit_folder / "checkpoint"
        refit_checkpoint.mkdir(parents=True)
        
        # Verify structure
        assert refit_checkpoint.parent == refit_folder
        assert refit_checkpoint.parent.parent == trial_folder
        assert refit_checkpoint.name == "checkpoint"

    def test_checkpoint_location_in_cv_fold(self, tmp_path):
        """Test that CV fold checkpoint is in fold folder: trial-{trial8}/cv/fold{idx}/checkpoint."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create CV fold checkpoint
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        fold_folder = trial_folder / "cv" / "fold0"
        fold_checkpoint = fold_folder / "checkpoint"
        fold_checkpoint.mkdir(parents=True)
        
        # Verify structure
        assert fold_checkpoint.parent == fold_folder
        assert fold_checkpoint.parent.parent.name == "cv"
        assert fold_checkpoint.name == "checkpoint"


class TestPathStructureValidation:
    """Test path structure validation functions."""

    def test_is_v2_path_detects_trial_folder(self, tmp_path):
        """Test that is_v2_path correctly detects v2 trial folder (requires both study and trial)."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create v2 trial folder (is_v2_path requires both study and trial in path)
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        
        # Verify v2 detection (requires both study-{hash} and trial-{hash})
        assert is_v2_path(str(trial_folder))
        assert "study-" in str(trial_folder)
        assert "trial-" in str(trial_folder)

    def test_is_v2_path_detects_trial_folder(self, tmp_path):
        """Test that is_v2_path correctly detects v2 trial folder."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create v2 trial folder
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        
        # Verify v2 detection
        assert is_v2_path(str(trial_folder))
        assert "study-" in str(trial_folder)
        assert "trial-" in str(trial_folder)

    def test_find_study_folder(self, tmp_path):
        """Test that find_study_folder locates study folder by study8 hash."""
        study_key_hash = "a" * 64
        study8 = study_key_hash[:8]
        
        # Create study folder
        root_dir = tmp_path
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        hpo_base = resolve_output_path(root_dir, config_dir, "hpo")
        study_folder = hpo_base / "local" / "distilbert" / f"study-{study8}"
        study_folder.mkdir(parents=True)
        
        # Find study folder
        found_study = find_study_by_hash(
            root_dir=root_dir,
            config_dir=config_dir,
            model="distilbert",
            study_key_hash=study_key_hash,
        )
        
        # Verify found folder
        assert found_study is not None
        assert found_study.name == f"study-{study8}"

    def test_find_trial_folder(self, tmp_path):
        """Test that find_trial_folder locates trial folder by study8 and trial8."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create trial folder
        root_dir = tmp_path
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        hpo_base = resolve_output_path(root_dir, config_dir, "hpo")
        study_folder = hpo_base / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        
        # Find trial folder
        found_trial = find_trial_by_hash(
            root_dir=root_dir,
            config_dir=config_dir,
            model="distilbert",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        # Verify found folder
        assert found_trial is not None
        assert found_trial.name == f"trial-{trial8}"
        assert found_trial.parent.name == f"study-{study8}"


class TestPathStructureWithBuildOutputPath:
    """Test path structure using build_output_path function."""

    def test_build_output_path_hpo_sweep(self, tmp_path):
        """Test that build_output_path creates correct HPO sweep base path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_key_hash = "a" * 64
        study8 = study_key_hash[:8]
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_sweep",
            study_key_hash=study_key_hash,
        )
        
        output_path = build_output_path(
            root_dir=tmp_path,
            context=context,
            config_dir=config_dir,
        )
        
        # Verify path structure (hpo_sweep returns base directory, study folder created separately)
        assert "outputs" in str(output_path)
        assert "hpo" in str(output_path)
        assert "local" in str(output_path)
        assert "distilbert" in str(output_path)
        # Note: build_output_path for hpo_sweep returns base dir, study folder is created separately
        # The study folder would be: output_path / f"study-{study8}"

    def test_build_output_path_hpo_trial(self, tmp_path):
        """Test that build_output_path creates correct HPO trial path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_trial",
            trial_id=f"trial-{trial8}",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        output_path = build_output_path(
            root_dir=tmp_path,
            context=context,
            config_dir=config_dir,
        )
        
        # Verify path structure
        assert f"study-{study8}" in str(output_path)
        assert f"trial-{trial8}" in str(output_path)

    def test_build_output_path_hpo_refit(self, tmp_path):
        """Test that build_output_path creates correct HPO refit path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        context = create_naming_context(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            storage_env="local",
            trial_id=f"trial-{trial8}",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        output_path = build_output_path(
            root_dir=tmp_path,
            context=context,
            config_dir=config_dir,
        )
        
        # Verify path structure
        assert f"study-{study8}" in str(output_path)
        assert f"trial-{trial8}" in str(output_path)
        assert "refit" in str(output_path)
        # Verify refit is under trial
        assert str(output_path).endswith("refit") or "refit" in output_path.name


class TestPathStructureFiles:
    """Test that required files exist in path structure."""

    def test_study_db_location(self, tmp_path):
        """Test that study.db is in study folder: study-{study8}/study.db."""
        study_key_hash = "a" * 64
        study8 = study_key_hash[:8]
        
        # Create study folder and study.db
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        study_folder.mkdir(parents=True)
        study_db = study_folder / "study.db"
        study_db.touch()
        
        # Verify location
        assert study_db.exists()
        assert study_db.parent == study_folder
        assert study_db.name == "study.db"

    def test_trial_meta_json_location(self, tmp_path):
        """Test that trial_meta.json is in trial folder: trial-{trial8}/trial_meta.json."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create trial folder and trial_meta.json
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        trial_meta = trial_folder / "trial_meta.json"
        trial_meta.write_text(json.dumps({
            "study_key_hash": study_key_hash,
            "trial_key_hash": trial_key_hash,
            "trial_number": 0,
        }))
        
        # Verify location and content
        assert trial_meta.exists()
        assert trial_meta.parent == trial_folder
        assert trial_meta.name == "trial_meta.json"
        
        # Verify content
        meta_data = json.loads(trial_meta.read_text())
        assert meta_data["study_key_hash"] == study_key_hash
        assert meta_data["trial_key_hash"] == trial_key_hash

    def test_fold_splits_json_location(self, tmp_path):
        """Test that fold_splits.json is in study folder: study-{study8}/fold_splits.json."""
        study_key_hash = "a" * 64
        study8 = study_key_hash[:8]
        
        # Create study folder and fold_splits.json
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        study_folder.mkdir(parents=True)
        fold_splits = study_folder / "fold_splits.json"
        fold_splits.write_text(json.dumps({
            "folds": [[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [0, 1, 2]]],
            "metadata": {"k": 2, "random_seed": 42},
        }))
        
        # Verify location
        assert fold_splits.exists()
        assert fold_splits.parent == study_folder
        assert fold_splits.name == "fold_splits.json"

    def test_metrics_json_location_in_trial(self, tmp_path):
        """Test that metrics.json is in trial folder: trial-{trial8}/metrics.json."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create trial folder and metrics.json
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        metrics_file = trial_folder / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
        
        # Verify location
        assert metrics_file.exists()
        assert metrics_file.parent == trial_folder
        assert metrics_file.name == "metrics.json"

    def test_metrics_json_location_in_refit(self, tmp_path):
        """Test that metrics.json is in refit folder: trial-{trial8}/refit/metrics.json."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create refit folder and metrics.json
        study_folder = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        refit_folder = trial_folder / "refit"
        refit_folder.mkdir(parents=True)
        metrics_file = refit_folder / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.80}))
        
        # Verify location
        assert metrics_file.exists()
        assert metrics_file.parent == refit_folder
        assert metrics_file.name == "metrics.json"


class TestPathStructureSmokeYaml:
    """Test path structure with smoke.yaml configuration."""

    def test_path_structure_matches_smoke_yaml(self, tmp_path):
        """Test that path structure matches smoke.yaml expectations."""
        # smoke.yaml uses v2 paths with study-{hash} and trial-{hash}
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Create complete structure
        root_dir = tmp_path
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        hpo_base = resolve_output_path(root_dir, config_dir, "hpo")
        study_folder = hpo_base / "local" / "distilbert" / f"study-{study8}"
        study_folder.mkdir(parents=True)
        
        # Create study.db (checkpoint)
        study_db = study_folder / "study.db"
        study_db.touch()
        
        # Create fold_splits.json (smoke.yaml: k_fold.enabled=true, n_splits=2)
        fold_splits = study_folder / "fold_splits.json"
        fold_splits.write_text(json.dumps({
            "folds": [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]],
            "metadata": {"k": 2, "random_seed": 42, "shuffle": True, "stratified": True},
        }))
        
        # Create trial folder
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir()
        
        # Create trial_meta.json
        trial_meta = trial_folder / "trial_meta.json"
        trial_meta.write_text(json.dumps({
            "study_key_hash": study_key_hash,
            "trial_key_hash": trial_key_hash,
            "trial_number": 0,
        }))
        
        # Create CV fold folders (smoke.yaml: n_splits=2)
        for fold_idx in range(2):
            fold_folder = trial_folder / "cv" / f"fold{fold_idx}"
            fold_folder.mkdir(parents=True)
            fold_checkpoint = fold_folder / "checkpoint"
            fold_checkpoint.mkdir()
        
        # Create refit folder (smoke.yaml: refit.enabled=true)
        refit_folder = trial_folder / "refit"
        refit_folder.mkdir()
        refit_checkpoint = refit_folder / "checkpoint"
        refit_checkpoint.mkdir()
        refit_metrics = refit_folder / "metrics.json"
        refit_metrics.write_text(json.dumps({"macro-f1": 0.80}))
        
        # Verify complete structure
        assert study_db.exists()
        assert fold_splits.exists()
        assert trial_meta.exists()
        assert len(list((trial_folder / "cv").iterdir())) == 2  # 2 folds
        assert refit_folder.exists()
        assert refit_checkpoint.exists()
        assert refit_metrics.exists()

    def test_path_structure_study8_trial8_format(self, tmp_path):
        """Test that study8 and trial8 are exactly 8 characters (hex)."""
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        # Verify format
        assert len(study8) == 8
        assert len(trial8) == 8
        assert all(c in "0123456789abcdef" for c in study8)
        assert all(c in "0123456789abcdef" for c in trial8)
        
        # Create folders
        study_folder = tmp_path / f"study-{study8}"
        trial_folder = study_folder / f"trial-{trial8}"
        trial_folder.mkdir(parents=True)
        
        # Verify folder names
        assert study_folder.name == f"study-{study8}"
        assert trial_folder.name == f"trial-{trial8}"

