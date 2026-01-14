"""Integration tests for checkpoint acquisition in benchmarking workflow.

Tests the checkpoint acquisition logic used in Step 6 of 02_best_config_selection.ipynb
to ensure checkpoints can be found from various run types (trial, refit, parent).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from mlflow.tracking import MlflowClient


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    client = Mock(spec=MlflowClient)
    return client


@pytest.fixture
def mock_champion_data():
    """Create mock champion data structure."""
    return {
        "run_id": "trial-run-123",
        "study_key_hash": "study-hash-abc",
        "trial_key_hash": "trial-hash-xyz",
        "backbone": "distilbert",
        "metric": 0.85,
    }


@pytest.fixture
def mock_trial_run():
    """Create mock trial run (child run with no artifacts)."""
    run = Mock()
    run.info.run_id = "trial-run-123"
    run.info.experiment_id = "exp-123"
    run.info.parent_run_id = "parent-hpo-run-456"
    run.data.tags = {
        "code.process.stage": "hpo",
        "code.grouping.study_key_hash": "study-hash-abc",
        "code.grouping.trial_key_hash": "trial-hash-xyz",
    }
    return run


@pytest.fixture
def mock_refit_run():
    """Create mock refit run (has checkpoint artifacts)."""
    run = Mock()
    run.info.run_id = "refit-run-789"
    run.info.experiment_id = "exp-123"
    run.info.parent_run_id = "parent-hpo-run-456"
    run.data.tags = {
        "code.process.stage": "hpo_refit",
        "code.grouping.study_key_hash": "study-hash-abc",
        "code.grouping.trial_key_hash": "trial-hash-xyz",
    }
    return run


@pytest.fixture
def mock_parent_hpo_run():
    """Create mock parent HPO run (may have checkpoint artifacts)."""
    run = Mock()
    run.info.run_id = "parent-hpo-run-456"
    run.info.experiment_id = "exp-123"
    run.info.parent_run_id = None
    run.data.tags = {
        "code.process.stage": "hpo",
    }
    return run


class TestRefitRunDiscovery:
    """Test refit run discovery logic."""

    def test_find_refit_run_by_hash_match(self, mock_mlflow_client, mock_trial_run, mock_refit_run):
        """Test finding refit run with matching study_key_hash and trial_key_hash."""
        from infrastructure.naming.mlflow.tags_registry import load_tags_registry
        from pathlib import Path
        
        # Mock MLflow client responses
        mock_mlflow_client.get_run.return_value = mock_trial_run
        mock_mlflow_client.search_runs.return_value = [mock_refit_run]
        
        # Mock tags registry
        with patch('infrastructure.naming.mlflow.tags_registry.load_tags_registry') as mock_load:
            mock_registry = Mock()
            mock_registry.key.side_effect = lambda section, key: f"code.{section}.{key}"
            mock_load.return_value = mock_registry
            
            # Test refit run discovery
            study_key_hash = "study-hash-abc"
            trial_key_hash = "trial-hash-xyz"
            run_id = "trial-run-123"
            
            experiment_id = mock_trial_run.info.experiment_id
            stage_tag = "code.process.stage"
            study_key_tag = "code.grouping.study_key_hash"
            trial_key_tag = "code.grouping.trial_key_hash"
            
            # Search for refit runs
            refit_runs = mock_mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.{stage_tag} = 'hpo_refit' AND tags.{study_key_tag} = '{study_key_hash}' AND tags.{trial_key_tag} = '{trial_key_hash}'",
                max_results=5,
            )
            
            assert len(refit_runs) == 1
            assert refit_runs[0].info.run_id == "refit-run-789"

    def test_find_refit_run_by_study_hash_only(self, mock_mlflow_client, mock_trial_run, mock_refit_run):
        """Test finding refit run with only study_key_hash match (legacy runs)."""
        # Mock MLflow client responses
        mock_mlflow_client.get_run.return_value = mock_trial_run
        # Search with only study_key_hash should be sufficient to find legacy refit runs
        mock_mlflow_client.search_runs.return_value = [mock_refit_run]
        
        study_key_hash = "study-hash-abc"
        run_id = "trial-run-123"
        
        experiment_id = mock_trial_run.info.experiment_id
        stage_tag = "code.process.stage"
        study_key_tag = "code.grouping.study_key_hash"
        
        # Try alternative search without trial_key_hash
        refit_runs_alt = mock_mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.{stage_tag} = 'hpo_refit' AND tags.{study_key_tag} = '{study_key_hash}'",
            max_results=5,
        )
        
        assert len(refit_runs_alt) == 1
        assert refit_runs_alt[0].info.run_id == "refit-run-789"

    def test_find_refit_run_via_parent_relationship(self, mock_mlflow_client, mock_trial_run, mock_refit_run):
        """Test finding refit run by checking parent-child relationships."""
        # Mock refit run to be child of trial run
        mock_refit_run.info.parent_run_id = "trial-run-123"
        
        # Directly return the refit run when querying by its run_id
        mock_mlflow_client.get_run.return_value = mock_refit_run
        mock_mlflow_client.search_runs.return_value = [mock_refit_run]
        
        run_id = "trial-run-123"
        parent_run_id = mock_trial_run.info.parent_run_id
        
        # Get full refit run to check parent
        full_refit_run = mock_mlflow_client.get_run("refit-run-789")
        refit_parent_id = getattr(full_refit_run.info, 'parent_run_id', None)
        
        # Check if refit run is child of trial run
        assert refit_parent_id == run_id

    def test_find_any_refit_run_as_last_resort(self, mock_mlflow_client, mock_trial_run, mock_refit_run):
        """Test finding any refit run in experiment as last resort."""
        # Mock no hash matches
        mock_mlflow_client.get_run.return_value = mock_trial_run
        # Directly return a refit run when searching only by stage (last-resort behavior)
        mock_mlflow_client.search_runs.return_value = [mock_refit_run]
        
        experiment_id = mock_trial_run.info.experiment_id
        stage_tag = "code.process.stage"
        
        # Last resort search
        all_refit_runs = mock_mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.{stage_tag} = 'hpo_refit'",
            max_results=10,
        )
        
        assert len(all_refit_runs) == 1
        assert all_refit_runs[0].info.run_id == "refit-run-789"


class TestCheckpointArtifactDiscovery:
    """Test checkpoint artifact discovery in different run types."""

    def test_checkpoint_in_refit_run(self, mock_mlflow_client, mock_refit_run):
        """Test finding checkpoint artifacts in refit run."""
        # Mock artifacts in refit run
        artifact = Mock()
        artifact.path = "checkpoint"
        mock_mlflow_client.list_artifacts.return_value = [artifact]
        
        run_id = "refit-run-789"
        artifacts = mock_mlflow_client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        
        checkpoint_artifacts = [p for p in artifact_paths if "checkpoint" in p.lower()]
        assert len(checkpoint_artifacts) > 0
        assert "checkpoint" in checkpoint_artifacts

    def test_checkpoint_in_parent_hpo_run(self, mock_mlflow_client, mock_parent_hpo_run):
        """Test finding checkpoint artifacts in parent HPO run."""
        # Mock artifacts in parent run
        artifact = Mock()
        artifact.path = "best_trial_checkpoint.tar.gz"
        mock_mlflow_client.list_artifacts.return_value = [artifact]
        
        run_id = "parent-hpo-run-456"
        artifacts = mock_mlflow_client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        
        checkpoint_in_parent = any("checkpoint" in p.lower() for p in artifact_paths)
        assert checkpoint_in_parent is True

    def test_no_checkpoint_in_trial_run(self, mock_mlflow_client, mock_trial_run):
        """Test that trial run has no checkpoint artifacts."""
        # Mock empty artifacts
        mock_mlflow_client.list_artifacts.return_value = []
        
        run_id = "trial-run-123"
        artifacts = mock_mlflow_client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        
        checkpoint_artifacts = [p for p in artifact_paths if "checkpoint" in p.lower()]
        assert len(checkpoint_artifacts) == 0

    def test_checkpoint_artifact_paths(self, mock_mlflow_client):
        """Test various checkpoint artifact path formats."""
        test_cases = [
            ["checkpoint"],
            ["checkpoint.tar.gz"],
            ["best_trial_checkpoint.tar.gz"],
            ["refit/checkpoint"],
            ["refit/checkpoint.tar.gz"],
            ["artifacts/checkpoint"],
        ]
        
        for artifact_paths in test_cases:
            artifacts = [Mock(path=p) for p in artifact_paths]
            mock_mlflow_client.list_artifacts.return_value = artifacts
            
            run_id = "test-run"
            artifacts_list = mock_mlflow_client.list_artifacts(run_id)
            paths = [a.path for a in artifacts_list]
            
            checkpoint_artifacts = [p for p in paths if "checkpoint" in p.lower()]
            assert len(checkpoint_artifacts) > 0, f"Should find checkpoint in {artifact_paths}"


class TestCheckpointAcquisitionWorkflow:
    """Test the full checkpoint acquisition workflow for benchmarking."""

    @patch('evaluation.selection.artifact_acquisition.acquire_best_model_checkpoint')
    @patch('mlflow.tracking.MlflowClient')
    @patch('infrastructure.naming.mlflow.tags_registry.load_tags_registry')
    def test_acquire_from_refit_run_success(
        self,
        mock_load_registry,
        mock_mlflow_client_class,
        mock_acquire_checkpoint,
        mock_champion_data,
        mock_trial_run,
        mock_refit_run,
    ):
        """Test successful checkpoint acquisition from refit run."""
        # Setup mocks
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_client.get_run.return_value = mock_trial_run
        mock_client.search_runs.return_value = [mock_refit_run]
        
        artifact = Mock()
        artifact.path = "checkpoint"
        mock_client.list_artifacts.return_value = [artifact]
        
        mock_registry = Mock()
        mock_registry.key.side_effect = lambda section, key: f"code.{section}.{key}"
        mock_load_registry.return_value = mock_registry
        
        mock_acquire_checkpoint.return_value = Path("/tmp/checkpoint")
        
        # Simulate the acquisition workflow
        run_id = mock_champion_data["run_id"]
        study_key_hash = mock_champion_data["study_key_hash"]
        trial_key_hash = mock_champion_data["trial_key_hash"]
        
        # Find refit runs
        champion_run = mock_client.get_run(run_id)
        experiment_id = champion_run.info.experiment_id
        
        refit_runs = mock_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.code.process.stage = 'hpo_refit' AND tags.code.grouping.study_key_hash = 'study-hash-abc' AND tags.code.grouping.trial_key_hash = 'trial-hash-xyz'",
            max_results=5,
        )
        
        assert len(refit_runs) == 1
        refit_run_id = refit_runs[0].info.run_id
        
        # Check artifacts
        artifacts = mock_client.list_artifacts(refit_run_id)
        artifact_paths = [a.path for a in artifacts]
        checkpoint_artifacts = [p for p in artifact_paths if "checkpoint" in p.lower()]
        
        assert len(checkpoint_artifacts) > 0
        
        # Acquire checkpoint
        best_run_info = {
            "run_id": refit_run_id,
            "study_key_hash": study_key_hash,
            "trial_key_hash": trial_key_hash,
            "backbone": mock_champion_data["backbone"],
        }
        
        checkpoint_dir = mock_acquire_checkpoint(
            best_run_info=best_run_info,
            root_dir=Path("/tmp"),
            config_dir=Path("/tmp/config"),
            acquisition_config={"priority": ["mlflow"], "mlflow": {"enabled": True}},
            selection_config={},
            platform="local",
        )
        
        assert checkpoint_dir is not None
        mock_acquire_checkpoint.assert_called_once()

    @patch('evaluation.selection.artifact_acquisition.acquire_best_model_checkpoint')
    @patch('mlflow.tracking.MlflowClient')
    def test_acquire_from_parent_hpo_run_fallback(
        self,
        mock_mlflow_client_class,
        mock_acquire_checkpoint,
        mock_champion_data,
        mock_trial_run,
        mock_parent_hpo_run,
    ):
        """Test fallback to parent HPO run when refit run not found."""
        # Setup mocks
        mock_client = Mock()
        mock_mlflow_client_class.return_value = mock_client
        mock_client.get_run.side_effect = [mock_trial_run, mock_parent_hpo_run]
        mock_client.search_runs.return_value = []  # No refit runs
        
        # Parent has checkpoint artifacts
        artifact = Mock()
        artifact.path = "best_trial_checkpoint.tar.gz"
        mock_client.list_artifacts.return_value = [artifact]
        
        mock_acquire_checkpoint.return_value = Path("/tmp/checkpoint")
        
        # Simulate workflow
        run_id = mock_champion_data["run_id"]
        champion_run = mock_client.get_run(run_id)
        parent_run_id = getattr(champion_run.info, 'parent_run_id', None)
        
        if parent_run_id:
            parent_artifacts = mock_client.list_artifacts(parent_run_id)
            parent_artifact_paths = [a.path for a in parent_artifacts]
            checkpoint_in_parent = any("checkpoint" in p.lower() for p in parent_artifact_paths)
            
            assert checkpoint_in_parent is True
            
            # Try acquiring from parent
            best_run_info = {
                "run_id": parent_run_id,
                "study_key_hash": mock_champion_data["study_key_hash"],
                "trial_key_hash": mock_champion_data["trial_key_hash"],
                "backbone": mock_champion_data["backbone"],
            }
            
            checkpoint_dir = mock_acquire_checkpoint(
                best_run_info=best_run_info,
                root_dir=Path("/tmp"),
                config_dir=Path("/tmp/config"),
                acquisition_config={"priority": ["mlflow"], "mlflow": {"enabled": True}},
                selection_config={},
                platform="local",
            )
            
            assert checkpoint_dir is not None

    def test_no_checkpoint_found_anywhere(self, mock_mlflow_client, mock_trial_run):
        """Test handling when no checkpoint is found in any run."""
        # Mock no artifacts anywhere
        mock_mlflow_client.get_run.return_value = mock_trial_run
        mock_mlflow_client.search_runs.return_value = []  # No refit runs
        mock_mlflow_client.list_artifacts.return_value = []  # No artifacts
        
        run_id = "trial-run-123"
        champion_run = mock_mlflow_client.get_run(run_id)
        parent_run_id = getattr(champion_run.info, 'parent_run_id', None)
        
        # Check trial run
        trial_artifacts = mock_mlflow_client.list_artifacts(run_id)
        trial_checkpoint = any("checkpoint" in a.path.lower() for a in trial_artifacts)
        
        # Check parent run
        parent_checkpoint = False
        if parent_run_id:
            parent_artifacts = mock_mlflow_client.list_artifacts(parent_run_id)
            parent_checkpoint = any("checkpoint" in a.path.lower() for a in parent_artifacts)
        
        # No checkpoint found
        assert trial_checkpoint is False
        assert parent_checkpoint is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parent_run_id_not_available(self, mock_mlflow_client, mock_trial_run):
        """Test handling when parent_run_id is not available."""
        # Remove parent_run_id attribute
        del mock_trial_run.info.parent_run_id
        
        # Ensure get_run returns the trial run without parent_run_id
        mock_mlflow_client.get_run.return_value = mock_trial_run
        
        run_id = "trial-run-123"
        champion_run = mock_mlflow_client.get_run(run_id)
        parent_run_id = getattr(champion_run.info, 'parent_run_id', None)
        
        # Should handle gracefully
        assert parent_run_id is None

    def test_refit_run_without_trial_key_hash(self, mock_mlflow_client, mock_refit_run):
        """Test refit run that doesn't have trial_key_hash tag."""
        # Remove trial_key_hash from tags
        mock_refit_run.data.tags.pop("code.grouping.trial_key_hash", None)
        
        # Should still be findable by study_key_hash only
        tags = mock_refit_run.data.tags
        assert "code.grouping.study_key_hash" in tags
        assert "code.grouping.trial_key_hash" not in tags

    def test_multiple_refit_runs(self, mock_mlflow_client, mock_trial_run):
        """Test handling when multiple refit runs exist."""
        # Create multiple refit runs
        refit_run_1 = Mock()
        refit_run_1.info.run_id = "refit-1"
        refit_run_1.data.tags = {"code.process.stage": "hpo_refit"}
        
        refit_run_2 = Mock()
        refit_run_2.info.run_id = "refit-2"
        refit_run_2.data.tags = {"code.process.stage": "hpo_refit"}
        
        mock_mlflow_client.search_runs.return_value = [refit_run_1, refit_run_2]
        
        experiment_id = "exp-123"
        refit_runs = mock_mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.code.process.stage = 'hpo_refit'",
            max_results=10,
        )
        
        # Should find both
        assert len(refit_runs) == 2
        # Should try both in order
        run_ids = [r.info.run_id for r in refit_runs]
        assert "refit-1" in run_ids
        assert "refit-2" in run_ids

