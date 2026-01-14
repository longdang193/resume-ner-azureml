"""Unit tests for HPO variant generation in helpers.py.

Tests create_study_name() with variant support for run.mode=force_new.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from training.hpo.utils.helpers import create_study_name, find_study_variants


class TestCreateStudyNameWithVariants:
    """Test create_study_name() with variant generation."""

    def test_create_study_name_force_new_no_existing(self, tmp_path):
        """Test study name generation with force_new when no variants exist."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={"enabled": True},
            hpo_config={"run": {"mode": "force_new"}},
            root_dir=tmp_path,
            config_dir=config_dir,
        )
        # First variant should be base name (no suffix)
        assert study_name == "hpo_distilbert"

    def test_create_study_name_force_new_with_existing(self, tmp_path):
        """
        Test study name generation with force_new when variants exist.

        In the simplified design, existing variant folders do not affect the
        study_name – users must bump study_name explicitly in config when they
        want a new version.
        """
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create existing variant folders
        hpo_output = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
        hpo_output.mkdir(parents=True)
        (hpo_output / "hpo_distilbert").mkdir()  # Variant 1
        (hpo_output / "hpo_distilbert_v2").mkdir()  # Variant 2
        
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={"enabled": True},
            hpo_config={"run": {"mode": "force_new"}},
            root_dir=tmp_path,
            config_dir=config_dir,
        )
        # Simplified behavior: still use base name (no automatic _vN suffix)
        assert study_name == "hpo_distilbert"

    def test_create_study_name_reuse_if_exists(self, tmp_path):
        """Test study name generation with reuse_if_exists mode."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={"enabled": True},
            hpo_config={"run": {"mode": "reuse_if_exists"}},
            root_dir=tmp_path,
            config_dir=config_dir,
        )
        # Should use base name for resumability
        assert study_name == "hpo_distilbert"

    def test_create_study_name_custom_study_name_force_new(self, tmp_path):
        """Test study name generation with custom study_name and force_new."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create existing variant folders
        hpo_output = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
        hpo_output.mkdir(parents=True)
        (hpo_output / "hpo_distilbert_prod").mkdir()  # Variant 1
        
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={
                "enabled": True,
                "study_name": "hpo_{backbone}_prod",
            },
            hpo_config={"run": {"mode": "force_new"}},
            root_dir=tmp_path,
            config_dir=config_dir,
        )
        # Simplified behavior: use template as-is (no automatic _vN suffix)
        assert study_name == "hpo_distilbert_prod"

    def test_create_study_name_custom_study_name_reuse_if_exists(self, tmp_path):
        """Test study name generation with custom study_name and reuse_if_exists."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={
                "enabled": True,
                "study_name": "hpo_{backbone}_prod",
            },
            hpo_config={"run": {"mode": "reuse_if_exists"}},
            root_dir=tmp_path,
            config_dir=config_dir,
        )
        # Should use base name (no variant suffix)
        assert study_name == "hpo_distilbert_prod"

    def test_create_study_name_checkpoint_disabled_force_new(self):
        """Test study name with checkpoint disabled and force_new."""
        study_name = create_study_name(
            backbone="distilbert",
            run_id="test123",
            should_resume=False,
            checkpoint_config={"enabled": False},
            hpo_config={"run": {"mode": "force_new"}},
            root_dir=None,
            config_dir=None,
        )
        # Should use run_id-based name when checkpointing disabled
        assert study_name == "hpo_distilbert_test123"


class TestFindStudyVariants:
    """Test find_study_variants() helper function."""

    def test_find_study_variants_none(self, tmp_path):
        """Test finding variants when none exist."""
        output_dir = tmp_path / "hpo_output"
        output_dir.mkdir()
        
        variants = find_study_variants(output_dir, "distilbert")
        assert variants == []

    def test_find_study_variants_implicit_variant_1(self, tmp_path):
        """Test finding implicit variant 1."""
        output_dir = tmp_path / "hpo_output"
        output_dir.mkdir()
        (output_dir / "hpo_distilbert").mkdir()
        
        variants = find_study_variants(output_dir, "distilbert")
        assert variants == ["hpo_distilbert"]

    def test_find_study_variants_explicit_variants(self, tmp_path):
        """Test finding explicit variants."""
        output_dir = tmp_path / "hpo_output"
        output_dir.mkdir()
        (output_dir / "hpo_distilbert").mkdir()  # Variant 1
        (output_dir / "hpo_distilbert_v2").mkdir()  # Variant 2
        (output_dir / "hpo_distilbert_v3").mkdir()  # Variant 3
        
        variants = find_study_variants(output_dir, "distilbert")
        assert sorted(variants) == ["hpo_distilbert", "hpo_distilbert_v2", "hpo_distilbert_v3"]

    def test_find_study_variants_ignores_other_folders(self, tmp_path):
        """Test that other folders are ignored."""
        output_dir = tmp_path / "hpo_output"
        output_dir.mkdir()
        (output_dir / "hpo_distilbert").mkdir()
        (output_dir / "other_folder").mkdir()
        (output_dir / "study-abc123").mkdir()  # v2 folder
        
        variants = find_study_variants(output_dir, "distilbert")
        assert variants == ["hpo_distilbert"]

    def test_find_study_variants_different_backbone(self, tmp_path):
        """Test that variants for different backbone are ignored."""
        output_dir = tmp_path / "hpo_output"
        output_dir.mkdir()
        (output_dir / "hpo_distilbert").mkdir()
        (output_dir / "hpo_roberta").mkdir()  # Different backbone
        
        variants = find_study_variants(output_dir, "distilbert")
        assert variants == ["hpo_distilbert"]


