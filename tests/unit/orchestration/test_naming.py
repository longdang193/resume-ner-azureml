"""Tests for experiment naming and stage configuration."""

import pytest
from unittest.mock import MagicMock
from orchestration.naming import get_stage_config, build_aml_experiment_name
from orchestration.config_loader import ExperimentConfig


class TestGetStageConfig:
    """Tests for get_stage_config function."""

    def test_get_existing_stage(self):
        """Test getting configuration for an existing stage."""
        mock_config = MagicMock(spec=ExperimentConfig)
        mock_config.stages = {
            "training": {"epochs": 5, "batch_size": 16},
            "sweep": {"max_trials": 20}
        }
        
        result = get_stage_config(mock_config, "training")
        
        assert result == {"epochs": 5, "batch_size": 16}

    def test_get_missing_stage(self):
        """Test getting configuration for a missing stage returns empty dict."""
        mock_config = MagicMock(spec=ExperimentConfig)
        mock_config.stages = {"training": {"epochs": 5}}
        
        result = get_stage_config(mock_config, "sweep")
        
        assert result == {}

    def test_get_stage_with_empty_config(self):
        """Test getting stage with empty configuration."""
        mock_config = MagicMock(spec=ExperimentConfig)
        mock_config.stages = {"training": {}}
        
        result = get_stage_config(mock_config, "training")
        
        assert result == {}

    def test_get_stage_with_none_config(self):
        """Test getting stage with None configuration."""
        mock_config = MagicMock(spec=ExperimentConfig)
        mock_config.stages = {"training": None}
        
        result = get_stage_config(mock_config, "training")
        
        assert result == {}


class TestBuildAmlExperimentName:
    """Tests for build_aml_experiment_name function."""

    def test_build_name_with_backbone(self):
        """Test building experiment name with backbone."""
        result = build_aml_experiment_name(
            experiment_name="resume-ner",
            stage="training",
            backbone="bert-base-uncased"
        )
        
        assert result == "resume-ner-training-bert-base-uncased"

    def test_build_name_without_backbone(self):
        """Test building experiment name without backbone."""
        result = build_aml_experiment_name(
            experiment_name="resume-ner",
            stage="training",
            backbone=None
        )
        
        assert result == "resume-ner-training"

    def test_build_name_different_stages(self):
        """Test building names for different stages."""
        result1 = build_aml_experiment_name("resume-ner", "sweep", "bert-base-uncased")
        result2 = build_aml_experiment_name("resume-ner", "selection", "bert-base-uncased")
        
        assert result1 == "resume-ner-sweep-bert-base-uncased"
        assert result2 == "resume-ner-selection-bert-base-uncased"

    def test_build_name_with_special_characters(self):
        """Test building name with special characters in backbone."""
        result = build_aml_experiment_name(
            experiment_name="resume-ner",
            stage="training",
            backbone="microsoft/deberta-v3-base"
        )
        
        assert result == "resume-ner-training-microsoft/deberta-v3-base"

    def test_build_name_empty_strings(self):
        """Test building name with empty strings."""
        result = build_aml_experiment_name("", "", "")
        
        # When all parts are empty, join(["", "", ""]) with "-" gives "--"
        # But if backbone is None, it's join(["", ""]) which gives "-"
        # The function only adds backbone if it's not None, so with "" it should be "-"
        assert result == "-"

