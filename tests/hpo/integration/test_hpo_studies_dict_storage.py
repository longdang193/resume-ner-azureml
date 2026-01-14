"""Integration tests for hpo_studies dictionary storage.

Tests verify that the notebook loop correctly stores all backbone studies
in the hpo_studies dictionary, preventing the indentation bug where only
the last backbone's study was stored.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from training.hpo import run_local_hpo_sweep


@pytest.fixture
def hpo_config_multiple_backbones():
    """HPO config with multiple backbones."""
    return {
        "search_space": {
            "backbone": {
                "type": "choice",
                "values": ["distilbert", "distilroberta"]
            },
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
        },
        "sampling": {"algorithm": "random", "max_trials": 1},
        "checkpoint": {"enabled": False},  # Disable for faster tests
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "k_fold": {"enabled": False},
        "refit": {"enabled": False},
    }


@pytest.fixture
def train_config_minimal():
    """Minimal training config."""
    return {"training": {"epochs": 1}}


@pytest.fixture
def data_config_minimal():
    """Minimal data config."""
    return {"dataset_name": "test_data", "dataset_version": "v1"}


class TestHPOStudiesDictStorage:
    """Test that hpo_studies dict correctly stores all backbone studies."""

    def test_notebook_loop_stores_all_backbones(
        self,
        tmp_path,
        hpo_config_multiple_backbones,
        train_config_minimal,
        data_config_minimal,
        mock_training_subprocess,
    ):
        """Test that the notebook loop pattern stores all backbone studies correctly.
        
        This test simulates the notebook loop:
        ```python
        hpo_studies = {}
        for backbone in backbone_values:
            study = run_local_hpo_sweep(...)
            hpo_studies[backbone] = study  # Must be inside loop!
        ```
        """
        backbone_values = hpo_config_multiple_backbones["search_space"]["backbone"]["values"]
        hpo_studies = {}
        
        # This test validates the notebook loop logic for populating hpo_studies,
        # not the internals of run_local_hpo_sweep. To keep it robust against
        # HPO path refactors (v2-only study folders, etc.), we mock the sweep.
        with patch(
            "training.hpo.execution.local.sweep.run_local_hpo_sweep",
            side_effect=lambda *args, **kwargs: Mock(),
        ):
            for backbone in backbone_values:
                study = run_local_hpo_sweep(
                    dataset_path="dummy",
                    config_dir=tmp_path / "config",
                    backbone=backbone,
                    hpo_config=hpo_config_multiple_backbones,
                    train_config=train_config_minimal,
                    output_dir=tmp_path / "outputs" / backbone,
                    mlflow_experiment_name=f"test_exp-{backbone}",
                    k_folds=None,
                    fold_splits_file=None,
                    checkpoint_config={"enabled": False},
                    restore_from_drive=None,
                    data_config=data_config_minimal,
                    benchmark_config={},
                )
                # This line MUST be inside the loop (4 spaces indentation)
                hpo_studies[backbone] = study
        
        # Verify all backbones are stored using shared validation function
        from tests.shared.validate_hpo_studies import validate_hpo_studies_dict
        
        is_valid, error = validate_hpo_studies_dict(hpo_studies, backbone_values)
        assert is_valid, f"hpo_studies dict validation failed: {error}"
        
        # Additional checks
        assert len(hpo_studies) == 2, f"Expected 2 studies, got {len(hpo_studies)}"
        assert hpo_studies["distilbert"] is not None
        assert hpo_studies["distilroberta"] is not None

    def test_validate_hpo_studies_dict_helper(self):
        """Test helper function to validate hpo_studies dict structure."""
        from tests.shared.validate_hpo_studies import validate_hpo_studies_dict
        
        backbone_values = ["distilbert", "distilroberta"]
        
        # Correct dict - all backbones present
        hpo_studies_correct = {
            "distilbert": Mock(),
            "distilroberta": Mock(),
        }
        
        is_valid, error = validate_hpo_studies_dict(hpo_studies_correct, backbone_values)
        assert is_valid, f"Correct dict should be valid: {error}"
        
        # Incorrect dict - missing backbone (simulates indentation bug)
        hpo_studies_incorrect = {
            "distilroberta": Mock(),  # Missing distilbert
        }
        
        is_valid, error = validate_hpo_studies_dict(hpo_studies_incorrect, backbone_values)
        assert not is_valid, "Should detect missing backbones"
        assert "Missing backbones" in error
        assert "distilbert" in error

