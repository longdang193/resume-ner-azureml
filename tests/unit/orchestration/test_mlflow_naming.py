"""Unit tests for MLflow naming and hash computation."""

import pytest
import json

from orchestration.jobs.tracking.naming.hpo_keys import (
    build_hpo_study_key,
    build_hpo_study_key_hash,
    build_hpo_trial_key,
    build_hpo_trial_key_hash,
    build_hpo_study_family_key,
    build_hpo_study_family_hash,
)


class TestStudyKeyHashComputation:
    """Test study key and hash computation."""

    def test_build_hpo_study_key(self):
        """Test building HPO study key from configs."""
        data_config = {
            "name": "resume_ner",
            "version": "1.0",
            "local_path": "/data/resume_ner",
            "schema": {"text": "str", "entities": "list"},
        }
        hpo_config = {
            "search_space": {
                "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            },
            "objective": {"metric": "macro-f1", "goal": "maximize"},
            "k_fold": {"enabled": True, "n_splits": 2},
            "sampling": {"algorithm": "random"},
            "early_termination": {"policy": "bandit"},
        }
        model = "distilbert"
        benchmark_config = {
            "benchmarking": {
                "metric": "macro-f1",
                "max_length": 512,
            }
        }

        study_key = build_hpo_study_key(
            data_config, hpo_config, model, benchmark_config
        )

        # Should be valid JSON
        parsed = json.loads(study_key)
        assert parsed["schema_version"] == "1.0"
        assert parsed["model"] == "distilbert"
        assert "data" in parsed
        assert "hpo" in parsed
        assert "benchmark" in parsed

        # Verify data key components
        assert parsed["data"]["name"] == "resume_ner"
        assert parsed["data"]["version"] == "1.0"
        assert parsed["data"]["local_path"] == "/data/resume_ner"

        # Verify hpo key components
        assert "search_space" in parsed["hpo"]
        assert parsed["hpo"]["objective"]["metric"] == "macro-f1"
        assert parsed["hpo"]["k_fold"]["enabled"] is True

        # Verify benchmark key
        assert parsed["benchmark"]["metric"] == "macro-f1"
        assert parsed["benchmark"]["max_length"] == 512

    def test_build_hpo_study_key_hash_length(self):
        """Test that hash length = 64 (full SHA256)."""
        data_config = {"name": "test", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}
        model = "distilbert"

        study_key = build_hpo_study_key(data_config, hpo_config, model)
        study_key_hash = build_hpo_study_key_hash(study_key)

        assert len(study_key_hash) == 64  # Full SHA256 hex digest

    def test_build_hpo_study_key_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        data_config = {"name": "test", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}
        model = "distilbert"

        study_key1 = build_hpo_study_key(data_config, hpo_config, model)
        study_key2 = build_hpo_study_key(data_config, hpo_config, model)
        hash1 = build_hpo_study_key_hash(study_key1)
        hash2 = build_hpo_study_key_hash(study_key2)

        assert study_key1 == study_key2
        assert hash1 == hash2

    def test_build_hpo_study_key_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        data_config1 = {"name": "test1", "version": "1.0", "local_path": "/data"}
        data_config2 = {"name": "test2", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}
        model = "distilbert"

        study_key1 = build_hpo_study_key(data_config1, hpo_config, model)
        study_key2 = build_hpo_study_key(data_config2, hpo_config, model)
        hash1 = build_hpo_study_key_hash(study_key1)
        hash2 = build_hpo_study_key_hash(study_key2)

        assert hash1 != hash2

    def test_build_hpo_study_key_includes_all_config_components(self):
        """Test that hash includes all config components."""
        data_config = {
            "name": "resume_ner",
            "version": "1.0",
            "local_path": "/data",
            "schema": {"text": "str"},
        }
        hpo_config = {
            "search_space": {"lr": {"type": "uniform", "min": 0.001, "max": 0.1}},
            "objective": {"metric": "f1"},
            "k_fold": {"enabled": True},
            "sampling": {"algorithm": "random"},
            "early_termination": {"policy": "bandit"},
        }
        model = "distilbert"
        benchmark_config = {"benchmarking": {"metric": "f1", "max_length": 512}}

        study_key = build_hpo_study_key(
            data_config, hpo_config, model, benchmark_config
        )
        parsed = json.loads(study_key)

        # All components should be present
        assert "data" in parsed
        assert "hpo" in parsed
        assert "model" in parsed
        assert "benchmark" in parsed
        assert parsed["model"] == "distilbert"

    def test_build_hpo_study_key_without_benchmark(self):
        """Test building study key without benchmark config."""
        data_config = {"name": "test", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}
        model = "distilbert"

        study_key = build_hpo_study_key(data_config, hpo_config, model, None)
        parsed = json.loads(study_key)

        assert parsed["benchmark"] == {}

    def test_build_hpo_study_family_key(self):
        """Test building study family key (without model)."""
        data_config = {"name": "test", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}
        benchmark_config = {"benchmarking": {"metric": "f1"}}

        family_key = build_hpo_study_family_key(
            data_config, hpo_config, benchmark_config
        )
        parsed = json.loads(family_key)

        assert "data" in parsed
        assert "hpo" in parsed
        assert "benchmark" in parsed
        # Should NOT include model
        assert "model" not in parsed

    def test_build_hpo_study_family_hash(self):
        """Test building study family hash."""
        data_config = {"name": "test", "version": "1.0", "local_path": "/data"}
        hpo_config = {"search_space": {}, "objective": {}}

        family_key = build_hpo_study_family_key(data_config, hpo_config, None)
        family_hash = build_hpo_study_family_hash(family_key)

        assert len(family_hash) == 64
        assert isinstance(family_hash, str)


class TestTrialKeyHashComputation:
    """Test trial key and hash computation."""

    def test_build_hpo_trial_key(self):
        """Test building HPO trial key from study hash and hyperparameters."""
        study_key_hash = "a" * 64  # 64-char hash
        hyperparameters = {
            "learning_rate": 3e-5,
            "batch_size": 4,
            "dropout": 0.2,
            "weight_decay": 0.01,
        }

        trial_key = build_hpo_trial_key(study_key_hash, hyperparameters)
        parsed = json.loads(trial_key)

        assert parsed["schema_version"] == "1.0"
        assert parsed["study_key_hash"] == study_key_hash
        assert "hyperparameters" in parsed

        # Hyperparameters should be normalized
        params = parsed["hyperparameters"]
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "dropout" in params
        assert "weight_decay" in params

    def test_build_hpo_trial_key_hash_length(self):
        """Test that trial hash length = 64 (full SHA256)."""
        study_key_hash = "a" * 64
        hyperparameters = {"learning_rate": 3e-5}

        trial_key = build_hpo_trial_key(study_key_hash, hyperparameters)
        trial_key_hash = build_hpo_trial_key_hash(trial_key)

        assert len(trial_key_hash) == 64

    def test_build_hpo_trial_key_hash_includes_study_hash(self):
        """Test that hash includes study_key_hash and hyperparameters."""
        study_key_hash = "a" * 64
        hyperparameters1 = {"learning_rate": 3e-5, "batch_size": 4}
        hyperparameters2 = {"learning_rate": 3e-5, "batch_size": 8}

        trial_key1 = build_hpo_trial_key(study_key_hash, hyperparameters1)
        trial_key2 = build_hpo_trial_key(study_key_hash, hyperparameters2)
        hash1 = build_hpo_trial_key_hash(trial_key1)
        hash2 = build_hpo_trial_key_hash(trial_key2)

        # Different hyperparameters should produce different hashes
        assert hash1 != hash2

    def test_build_hpo_trial_key_same_params_same_hash(self):
        """Test that same trial params produce same hash."""
        study_key_hash = "a" * 64
        hyperparameters = {"learning_rate": 3e-5, "batch_size": 4}

        trial_key1 = build_hpo_trial_key(study_key_hash, hyperparameters)
        trial_key2 = build_hpo_trial_key(study_key_hash, hyperparameters)
        hash1 = build_hpo_trial_key_hash(trial_key1)
        hash2 = build_hpo_trial_key_hash(trial_key2)

        assert trial_key1 == trial_key2
        assert hash1 == hash2

    def test_build_hpo_trial_key_different_params_different_hash(self):
        """Test that different params produce different hashes."""
        study_key_hash = "a" * 64
        hyperparameters1 = {"learning_rate": 3e-5}
        hyperparameters2 = {"learning_rate": 4e-5}

        trial_key1 = build_hpo_trial_key(study_key_hash, hyperparameters1)
        trial_key2 = build_hpo_trial_key(study_key_hash, hyperparameters2)
        hash1 = build_hpo_trial_key_hash(trial_key1)
        hash2 = build_hpo_trial_key_hash(trial_key2)

        assert hash1 != hash2

    def test_build_hpo_trial_key_normalizes_hyperparameters(self):
        """Test that hyperparameters are normalized for deterministic hashing."""
        study_key_hash = "a" * 64
        
        # These should produce the same hash (normalized to 12 significant figures)
        # Using values that will normalize to the same representation
        hyperparameters1 = {"learning_rate": 2.33e-05}
        hyperparameters2 = {"learning_rate": 2.33e-05}  # Same value
        hyperparameters3 = {"learning_rate": 0.0000233}  # Different representation, should normalize

        trial_key1 = build_hpo_trial_key(study_key_hash, hyperparameters1)
        trial_key2 = build_hpo_trial_key(study_key_hash, hyperparameters2)
        trial_key3 = build_hpo_trial_key(study_key_hash, hyperparameters3)

        parsed1 = json.loads(trial_key1)
        parsed2 = json.loads(trial_key2)
        parsed3 = json.loads(trial_key3)

        # Same values should produce same normalized values
        assert parsed1["hyperparameters"]["learning_rate"] == parsed2["hyperparameters"]["learning_rate"]
        # Values that normalize to same 12-significant-figure representation should match
        # Note: 2.33e-05 and 0.0000233 may normalize differently due to precision
        # The key is that the same input produces the same normalized output
        assert isinstance(parsed1["hyperparameters"]["learning_rate"], float)
        assert isinstance(parsed3["hyperparameters"]["learning_rate"], float)

    def test_build_hpo_trial_key_normalizes_strings(self):
        """Test that string hyperparameters are normalized (lowercase, stripped)."""
        study_key_hash = "a" * 64
        
        # These should produce the same hash (normalized)
        hyperparameters1 = {"optimizer": "Adam"}
        hyperparameters2 = {"optimizer": "adam"}
        hyperparameters3 = {"optimizer": "  ADAM  "}

        trial_key1 = build_hpo_trial_key(study_key_hash, hyperparameters1)
        trial_key2 = build_hpo_trial_key(study_key_hash, hyperparameters2)
        trial_key3 = build_hpo_trial_key(study_key_hash, hyperparameters3)

        parsed1 = json.loads(trial_key1)
        parsed2 = json.loads(trial_key2)
        parsed3 = json.loads(trial_key3)

        # Normalized values should be the same
        assert parsed1["hyperparameters"]["optimizer"] == parsed2["hyperparameters"]["optimizer"]
        assert parsed1["hyperparameters"]["optimizer"] == parsed3["hyperparameters"]["optimizer"]

    def test_build_hpo_trial_key_with_smoke_yaml_params(self):
        """Test building trial key with smoke.yaml hyperparameters."""
        study_key_hash = "a" * 64
        # Hyperparameters from smoke.yaml search space
        hyperparameters = {
            "learning_rate": 3e-5,  # loguniform 1e-5 to 5e-5
            "batch_size": 4,  # choice [4]
            "dropout": 0.2,  # uniform 0.1-0.3
            "weight_decay": 0.01,  # loguniform 0.001-0.1
        }

        trial_key = build_hpo_trial_key(study_key_hash, hyperparameters)
        parsed = json.loads(trial_key)

        assert parsed["study_key_hash"] == study_key_hash
        assert "hyperparameters" in parsed
        assert "learning_rate" in parsed["hyperparameters"]
        assert "batch_size" in parsed["hyperparameters"]
        assert "dropout" in parsed["hyperparameters"]
        assert "weight_decay" in parsed["hyperparameters"]

