"""Tests for local HPO best configuration selection."""

import pytest
from unittest.mock import MagicMock, patch
from orchestration.jobs.local_selection import (
    extract_best_config_from_study,
    select_best_configuration_across_studies,
)


class TestExtractBestConfigFromStudy:
    """Tests for extract_best_config_from_study function."""

    def test_extract_best_config(self):
        """Test extracting best configuration from study."""
        # Mock Optuna study and trial
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.params = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "dropout": 0.1,
        }
        mock_trial.values = [0.85]  # Best F1 score
        
        mock_study = MagicMock()
        mock_study.best_trial = mock_trial
        mock_study.direction = MagicMock()
        mock_study.direction.name = "maximize"
        mock_study.study_name = "test_study"
        
        config = extract_best_config_from_study(
            mock_study, "distilbert", "v1", "macro-f1"
        )
        
        assert config["backbone"] == "distilbert"
        assert config["trial_name"] == "trial_0"
        assert config["hyperparameters"]["learning_rate"] == 2e-5
        assert config["metrics"]["objective_value"] == 0.85
        assert config["selection_criteria"]["metric"] == "macro-f1"

    def test_empty_study_raises_error(self):
        """Test that empty study raises ValueError."""
        mock_study = MagicMock()
        mock_study.best_trial = None
        mock_study.study_name = "empty_study"
        
        with pytest.raises(ValueError, match="No completed trials"):
            extract_best_config_from_study(mock_study, "distilbert", "v1")

    def test_excludes_backbone_from_params(self):
        """Test that backbone is excluded from hyperparameters."""
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.params = {
            "backbone": "distilbert",
            "learning_rate": 2e-5,
        }
        mock_trial.values = [0.85]
        
        mock_study = MagicMock()
        mock_study.best_trial = mock_trial
        mock_study.direction = MagicMock()
        mock_study.direction.name = "maximize"
        mock_study.study_name = "test_study"
        
        config = extract_best_config_from_study(mock_study, "distilbert", "v1")
        
        assert "backbone" not in config["hyperparameters"]
        assert config["hyperparameters"]["learning_rate"] == 2e-5

    def test_includes_cv_statistics(self):
        """Test that CV statistics are included if available."""
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.params = {"learning_rate": 2e-5}
        mock_trial.values = [0.85]
        mock_trial.user_attrs = {
            "cv_mean": 0.84,
            "cv_std": 0.02,
            "cv_fold_metrics": [0.83, 0.85],
        }
        
        mock_study = MagicMock()
        mock_study.best_trial = mock_trial
        mock_study.direction = MagicMock()
        mock_study.direction.name = "maximize"
        mock_study.study_name = "test_study"
        
        config = extract_best_config_from_study(mock_study, "distilbert", "v1")
        
        assert config["cv_statistics"] is not None
        assert config["cv_statistics"]["cv_mean"] == 0.84
        assert config["cv_statistics"]["cv_std"] == 0.02


class TestSelectBestConfigurationAcrossStudies:
    """Tests for select_best_configuration_across_studies function."""

    @patch("orchestration.jobs.local_selection._import_optuna")
    def test_select_best_across_studies_maximize(self, mock_import_optuna):
        """Test selecting best configuration across multiple studies (maximize)."""
        import optuna
        
        # Create mock studies
        mock_trial1 = MagicMock()
        mock_trial1.number = 0
        mock_trial1.params = {"learning_rate": 2e-5}
        mock_trial1.values = [0.80]
        
        mock_trial2 = MagicMock()
        mock_trial2.number = 0
        mock_trial2.params = {"learning_rate": 3e-5}
        mock_trial2.values = [0.85]  # Better
        
        mock_study1 = MagicMock()
        mock_study1.best_trial = mock_trial1
        
        mock_study2 = MagicMock()
        mock_study2.best_trial = mock_trial2
        
        studies = {
            "distilbert": mock_study1,
            "deberta": mock_study2,
        }
        
        hpo_config = {
            "objective": {
                "metric": "macro-f1",
                "goal": "maximize",
            },
        }
        
        mock_import_optuna.return_value = optuna
        
        best_config = select_best_configuration_across_studies(
            studies, hpo_config, "v1"
        )
        
        assert best_config["backbone"] == "deberta"
        assert best_config["metrics"]["objective_value"] == 0.85

    @patch("orchestration.jobs.local_selection._import_optuna")
    def test_select_best_across_studies_minimize(self, mock_import_optuna):
        """Test selecting best configuration across multiple studies (minimize)."""
        import optuna
        
        mock_trial1 = MagicMock()
        mock_trial1.number = 0
        mock_trial1.params = {"learning_rate": 2e-5}
        mock_trial1.values = [0.5]  # Higher loss
        
        mock_trial2 = MagicMock()
        mock_trial2.number = 0
        mock_trial2.params = {"learning_rate": 3e-5}
        mock_trial2.values = [0.3]  # Lower loss (better)
        
        mock_study1 = MagicMock()
        mock_study1.best_trial = mock_trial1
        
        mock_study2 = MagicMock()
        mock_study2.best_trial = mock_trial2
        
        studies = {
            "distilbert": mock_study1,
            "deberta": mock_study2,
        }
        
        hpo_config = {
            "objective": {
                "metric": "loss",
                "goal": "minimize",
            },
        }
        
        mock_import_optuna.return_value = optuna
        
        best_config = select_best_configuration_across_studies(
            studies, hpo_config, "v1"
        )
        
        assert best_config["backbone"] == "deberta"
        assert best_config["metrics"]["objective_value"] == 0.3

    @patch("orchestration.jobs.local_selection._import_optuna")
    def test_no_valid_trials_raises_error(self, mock_import_optuna):
        """Test that no valid trials raises ValueError."""
        import optuna
        
        mock_study = MagicMock()
        mock_study.best_trial = None
        
        studies = {"distilbert": mock_study}
        
        hpo_config = {
            "objective": {
                "metric": "macro-f1",
                "goal": "maximize",
            },
        }
        
        mock_import_optuna.return_value = optuna
        
        with pytest.raises(ValueError, match="No valid trials found"):
            select_best_configuration_across_studies(studies, hpo_config, "v1")

