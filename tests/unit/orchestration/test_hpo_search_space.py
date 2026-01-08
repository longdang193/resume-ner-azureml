"""Unit tests for HPO search space translation."""

import pytest
from unittest.mock import Mock, MagicMock

from orchestration.jobs.hpo.search_space import (
    translate_search_space_to_optuna,
    SearchSpaceTranslator,
)


class TestSearchSpaceTranslation:
    """Test search space translation to Optuna format."""

    def test_translate_smoke_yaml_search_space(self):
        """Test translation of smoke.yaml search_space section."""
        # Load smoke.yaml search_space
        hpo_config = {
            "search_space": {
                "backbone": {
                    "type": "choice",
                    "values": ["distilbert"],
                },
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
                "batch_size": {
                    "type": "choice",
                    "values": [4],
                },
                "dropout": {
                    "type": "uniform",
                    "min": 0.1,
                    "max": 0.3,
                },
                "weight_decay": {
                    "type": "loguniform",
                    "min": 0.001,
                    "max": 0.1,
                },
            }
        }

        # Create mock Optuna trial
        mock_trial = Mock()
        # suggest_categorical should return different values for backbone and batch_size
        mock_trial.suggest_categorical = Mock(side_effect=["distilbert", 4])
        mock_trial.suggest_float = Mock(side_effect=[
            3e-5,  # learning_rate
            0.2,   # dropout
            0.01,  # weight_decay
        ])

        # Translate search space
        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        # Assertions - verify all parameters are present
        assert "backbone" in params
        assert params["backbone"] == "distilbert"
        assert "learning_rate" in params
        assert "batch_size" in params
        assert params["batch_size"] == 4
        assert "dropout" in params
        assert "weight_decay" in params
        
        # Verify correct number of calls
        assert mock_trial.suggest_categorical.call_count == 2  # backbone, batch_size
        assert mock_trial.suggest_float.call_count == 3  # learning_rate, dropout, weight_decay

        # Verify backbone choice values
        backbone_call = mock_trial.suggest_categorical.call_args_list[0]
        assert backbone_call[0][0] == "backbone"
        assert backbone_call[0][1] == ["distilbert"]

        # Verify batch_size choice
        batch_call = mock_trial.suggest_categorical.call_args_list[1]
        assert batch_call[0][0] == "batch_size"
        assert batch_call[0][1] == [4]

    def test_backbone_choice_values(self):
        """Test backbone choice values = ['distilbert']."""
        hpo_config = {
            "search_space": {
                "backbone": {
                    "type": "choice",
                    "values": ["distilbert"],
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(return_value="distilbert")

        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        assert params["backbone"] == "distilbert"
        mock_trial.suggest_categorical.assert_called_once_with("backbone", ["distilbert"])

    def test_learning_rate_loguniform_range(self):
        """Test learning_rate is loguniform with min=1e-5, max=5e-5."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=3e-5)

        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        assert "learning_rate" in params
        # Verify the call was made with correct parameters
        mock_trial.suggest_float.assert_called_once()
        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "learning_rate"
        assert call_args[0][1] == 1e-5
        assert call_args[0][2] == 5e-5
        assert call_args[1]["log"] is True

    def test_batch_size_choice_values(self):
        """Test batch_size is choice with values=[4]."""
        hpo_config = {
            "search_space": {
                "batch_size": {
                    "type": "choice",
                    "values": [4],
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(return_value=4)

        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        assert params["batch_size"] == 4
        mock_trial.suggest_categorical.assert_called_once_with("batch_size", [4])

    def test_dropout_uniform_range(self):
        """Test dropout is uniform with min=0.1, max=0.3."""
        hpo_config = {
            "search_space": {
                "dropout": {
                    "type": "uniform",
                    "min": 0.1,
                    "max": 0.3,
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=0.2)

        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        assert "dropout" in params
        # Verify the call was made with correct parameters (uniform, no log)
        mock_trial.suggest_float.assert_called_once()
        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "dropout"
        assert call_args[0][1] == 0.1
        assert call_args[0][2] == 0.3
        # uniform doesn't pass log parameter
        assert "log" not in call_args[1] or call_args[1].get("log", False) is False

    def test_weight_decay_loguniform_range(self):
        """Test weight_decay is loguniform with min=0.001, max=0.1."""
        hpo_config = {
            "search_space": {
                "weight_decay": {
                    "type": "loguniform",
                    "min": 0.001,
                    "max": 0.1,
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=0.01)

        params = translate_search_space_to_optuna(hpo_config, mock_trial)

        assert "weight_decay" in params
        # Verify the call was made with correct parameters
        mock_trial.suggest_float.assert_called_once()
        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "weight_decay"
        assert call_args[0][1] == 0.001
        assert call_args[0][2] == 0.1
        assert call_args[1]["log"] is True

    def test_exclude_params(self):
        """Test that excluded parameters are not included in results."""
        hpo_config = {
            "search_space": {
                "backbone": {
                    "type": "choice",
                    "values": ["distilbert"],
                },
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=3e-5)

        # Exclude backbone
        params = translate_search_space_to_optuna(
            hpo_config, mock_trial, exclude_params=["backbone"]
        )

        assert "backbone" not in params
        assert "learning_rate" in params
        mock_trial.suggest_categorical.assert_not_called()
        mock_trial.suggest_float.assert_called_once()

    def test_unsupported_search_space_type(self):
        """Test that unsupported search space types raise ValueError."""
        hpo_config = {
            "search_space": {
                "invalid_param": {
                    "type": "invalid_type",
                    "values": [1, 2, 3],
                },
            }
        }

        mock_trial = Mock()

        with pytest.raises(ValueError, match="Unsupported search space type"):
            translate_search_space_to_optuna(hpo_config, mock_trial)

    def test_search_space_translator_class(self):
        """Test SearchSpaceTranslator class method."""
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=3e-5)

        params = SearchSpaceTranslator.to_optuna(hpo_config, mock_trial)

        assert "learning_rate" in params
        # Verify the call was made with correct parameters
        mock_trial.suggest_float.assert_called_once()
        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "learning_rate"
        assert call_args[0][1] == 1e-5
        assert call_args[0][2] == 5e-5
        assert call_args[1]["log"] is True

