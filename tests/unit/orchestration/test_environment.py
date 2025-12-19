"""Tests for environment configuration and Azure ML environment management."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from orchestration.environment import (
    EnvironmentConfig,
    build_environment_config,
    load_conda_environment,
    compute_environment_hash,
    get_or_create_environment,
    create_training_environment,
    prepare_environment_image,
    DEFAULT_ENVIRONMENT_NAME,
    DEFAULT_CONDA_RELATIVE_PATH,
    DEFAULT_DOCKER_IMAGE,
    WARMUP_DISPLAY_NAME,
    WARMUP_HISTORY_LIMIT,
)


class TestBuildEnvironmentConfig:
    """Tests for build_environment_config function."""

    def test_build_with_defaults(self, temp_dir):
        """Test building config with all defaults."""
        config_root = temp_dir
        
        config = build_environment_config(config_root)
        
        assert config.name == DEFAULT_ENVIRONMENT_NAME
        assert config.conda_path == config_root / DEFAULT_CONDA_RELATIVE_PATH
        assert config.docker_image == DEFAULT_DOCKER_IMAGE
        assert config.warmup_display_name == WARMUP_DISPLAY_NAME
        assert config.warmup_history_limit == WARMUP_HISTORY_LIMIT

    def test_build_with_custom_settings(self, temp_dir):
        """Test building config with custom settings."""
        config_root = temp_dir
        env_settings = {
            "environment": {
                "name": "custom-env",
                "conda_file": "custom/conda.yaml",
                "docker_image": "custom/image:latest",
                "warmup": {
                    "display_name": "custom-warmup",
                    "history_limit": 5,
                },
            }
        }
        
        config = build_environment_config(config_root, env_settings)
        
        assert config.name == "custom-env"
        assert config.conda_path == config_root / "custom/conda.yaml"
        assert config.docker_image == "custom/image:latest"
        assert config.warmup_display_name == "custom-warmup"
        assert config.warmup_history_limit == 5

    def test_build_with_partial_settings(self, temp_dir):
        """Test building config with partial settings."""
        config_root = temp_dir
        env_settings = {
            "environment": {
                "name": "partial-env",
            }
        }
        
        config = build_environment_config(config_root, env_settings)
        
        assert config.name == "partial-env"
        assert config.conda_path == config_root / DEFAULT_CONDA_RELATIVE_PATH
        assert config.docker_image == DEFAULT_DOCKER_IMAGE
        assert config.warmup_display_name == WARMUP_DISPLAY_NAME
        assert config.warmup_history_limit == WARMUP_HISTORY_LIMIT

    def test_build_with_empty_env_section(self, temp_dir):
        """Test building config with empty environment section."""
        config_root = temp_dir
        env_settings = {"environment": {}}
        
        config = build_environment_config(config_root, env_settings)
        
        assert config.name == DEFAULT_ENVIRONMENT_NAME
        assert config.conda_path == config_root / DEFAULT_CONDA_RELATIVE_PATH

    def test_build_with_none_env_section(self, temp_dir):
        """Test building config with None environment section."""
        config_root = temp_dir
        env_settings = {"environment": None}
        
        config = build_environment_config(config_root, env_settings)
        
        assert config.name == DEFAULT_ENVIRONMENT_NAME


class TestLoadCondaEnvironment:
    """Tests for load_conda_environment function."""

    @patch("orchestration.environment.load_yaml")
    def test_load_conda_environment_success(self, mock_load_yaml, temp_dir):
        """Test successful conda environment loading."""
        conda_path = temp_dir / "conda.yaml"
        expected_deps = {"dependencies": ["python=3.10", "pytorch"]}
        mock_load_yaml.return_value = expected_deps
        
        result = load_conda_environment(conda_path)
        
        assert result == expected_deps
        mock_load_yaml.assert_called_once_with(conda_path)

    @patch("orchestration.environment.load_yaml")
    def test_load_conda_environment_file_not_found(self, mock_load_yaml, temp_dir):
        """Test conda environment loading when file doesn't exist."""
        conda_path = temp_dir / "nonexistent.yaml"
        mock_load_yaml.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            load_conda_environment(conda_path)


class TestComputeEnvironmentHash:
    """Tests for compute_environment_hash function."""

    def test_compute_hash_deterministic(self):
        """Test that hash is deterministic for same inputs."""
        conda_deps = {"dependencies": ["python=3.10"]}
        docker_image = "test/image:latest"
        
        hash1 = compute_environment_hash(conda_deps, docker_image)
        hash2 = compute_environment_hash(conda_deps, docker_image)
        
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_hash_different_inputs(self):
        """Test that hash differs for different inputs."""
        conda_deps1 = {"dependencies": ["python=3.10"]}
        conda_deps2 = {"dependencies": ["python=3.11"]}
        docker_image = "test/image:latest"
        
        hash1 = compute_environment_hash(conda_deps1, docker_image)
        hash2 = compute_environment_hash(conda_deps2, docker_image)
        
        assert hash1 != hash2

    def test_compute_hash_different_docker_images(self):
        """Test that hash differs for different docker images."""
        conda_deps = {"dependencies": ["python=3.10"]}
        docker_image1 = "test/image:latest"
        docker_image2 = "test/image:v2"
        
        hash1 = compute_environment_hash(conda_deps, docker_image1)
        hash2 = compute_environment_hash(conda_deps, docker_image2)
        
        assert hash1 != hash2

    def test_compute_hash_includes_both_inputs(self):
        """Test that hash includes both conda deps and docker image."""
        conda_deps = {"dependencies": ["python=3.10"]}
        docker_image = "test/image:latest"
        
        hash_result = compute_environment_hash(conda_deps, docker_image)
        
        env_spec = {"conda_dependencies": conda_deps, "docker_image": docker_image}
        env_str = json.dumps(env_spec, sort_keys=True)
        import hashlib
        expected_hash = hashlib.sha256(env_str.encode("utf-8")).hexdigest()[:16]
        
        assert hash_result == expected_hash


class TestGetOrCreateEnvironment:
    """Tests for get_or_create_environment function."""

    def test_get_existing_environment(self):
        """Test getting an existing environment."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_client.environments.get.return_value = mock_env
        
        result = get_or_create_environment(
            ml_client=mock_client,
            name="test-env",
            version="v12345678",
            conda_dependencies={"dependencies": ["python=3.10"]},
            docker_image="test/image:latest",
        )
        
        assert result == mock_env
        mock_client.environments.get.assert_called_once_with(name="test-env", version="v12345678")
        mock_client.environments.create_or_update.assert_not_called()

    def test_create_new_environment(self):
        """Test creating a new environment when it doesn't exist."""
        from azure.core.exceptions import ResourceNotFoundError
        
        mock_client = MagicMock()
        mock_client.environments.get.side_effect = ResourceNotFoundError("Not found")
        mock_new_env = MagicMock()
        mock_client.environments.create_or_update.return_value = mock_new_env
        
        result = get_or_create_environment(
            ml_client=mock_client,
            name="test-env",
            version="v12345678",
            conda_dependencies={"dependencies": ["python=3.10"]},
            docker_image="test/image:latest",
        )
        
        assert result == mock_new_env
        mock_client.environments.get.assert_called_once()
        mock_client.environments.create_or_update.assert_called_once()
        
        call_args = mock_client.environments.create_or_update.call_args[0][0]
        assert call_args.name == "test-env"
        assert call_args.version == "v12345678"
        assert call_args.image == "test/image:latest"


class TestCreateTrainingEnvironment:
    """Tests for create_training_environment function."""

    @patch("orchestration.environment.load_conda_environment")
    @patch("orchestration.environment.get_or_create_environment")
    def test_create_training_environment_success(self, mock_get_or_create, mock_load_conda, temp_dir):
        """Test successful training environment creation."""
        mock_env = MagicMock()
        mock_get_or_create.return_value = mock_env
        
        conda_deps = {"dependencies": ["python=3.10"]}
        mock_load_conda.return_value = conda_deps
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=temp_dir / "conda.yaml",
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        mock_client = MagicMock()
        
        result = create_training_environment(mock_client, env_config)
        
        assert result == mock_env
        mock_load_conda.assert_called_once_with(env_config.conda_path)
        mock_get_or_create.assert_called_once()
        
        call_kwargs = mock_get_or_create.call_args[1]
        assert call_kwargs["name"] == "test-env"
        assert call_kwargs["conda_dependencies"] == conda_deps
        assert call_kwargs["docker_image"] == "test/image:latest"
        assert call_kwargs["version"].startswith("v")


class TestPrepareEnvironmentImage:
    """Tests for prepare_environment_image function."""

    def test_prepare_environment_image_existing_completed_job(self):
        """Test skipping warm-up when completed job exists."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_env.version = "v12345678"
        
        mock_job = MagicMock()
        mock_job.status = "Completed"
        mock_job.environment = MagicMock()
        mock_job.environment.name = "test-env"
        mock_job.environment.version = "v12345678"
        
        mock_client.jobs.list.return_value = [mock_job]
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=Path("conda.yaml"),
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        prepare_environment_image(mock_client, mock_env, "compute-cluster", env_config)
        
        mock_client.jobs.list.assert_called_once()
        mock_client.jobs.create_or_update.assert_not_called()

    def test_prepare_environment_image_no_matching_job(self):
        """Test creating warm-up job when no matching job exists."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_env.version = "v12345678"
        
        mock_client.jobs.list.return_value = []
        mock_submitted = MagicMock()
        mock_submitted.name = "warmup-job"
        mock_client.jobs.create_or_update.return_value = mock_submitted
        mock_completed = MagicMock()
        mock_completed.status = "Completed"
        mock_client.jobs.get.return_value = mock_completed
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=Path("conda.yaml"),
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        with patch("orchestration.environment.command") as mock_command:
            mock_warmup_job = MagicMock()
            mock_command.return_value = mock_warmup_job
            
            prepare_environment_image(mock_client, mock_env, "compute-cluster", env_config)
            
            mock_command.assert_called_once()
            mock_client.jobs.create_or_update.assert_called_once_with(mock_warmup_job)
            mock_client.jobs.stream.assert_called_once_with("warmup-job")
            mock_client.jobs.get.assert_called_once_with("warmup-job")

    def test_prepare_environment_image_warmup_failed(self):
        """Test raising error when warm-up job fails."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_env.version = "v12345678"
        
        mock_client.jobs.list.return_value = []
        mock_submitted = MagicMock()
        mock_submitted.name = "warmup-job"
        mock_client.jobs.create_or_update.return_value = mock_submitted
        mock_completed = MagicMock()
        mock_completed.status = "Failed"
        mock_client.jobs.get.return_value = mock_completed
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=Path("conda.yaml"),
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        with patch("orchestration.environment.command") as mock_command:
            mock_warmup_job = MagicMock()
            mock_command.return_value = mock_warmup_job
            
            with pytest.raises(RuntimeError, match="warm.*up job failed"):
                prepare_environment_image(mock_client, mock_env, "compute-cluster", env_config)

    def test_prepare_environment_image_list_jobs_failure(self):
        """Test that list failure doesn't prevent warm-up."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_env.version = "v12345678"
        
        mock_client.jobs.list.side_effect = Exception("List failed")
        mock_submitted = MagicMock()
        mock_submitted.name = "warmup-job"
        mock_client.jobs.create_or_update.return_value = mock_submitted
        mock_completed = MagicMock()
        mock_completed.status = "Completed"
        mock_client.jobs.get.return_value = mock_completed
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=Path("conda.yaml"),
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        with patch("orchestration.environment.command") as mock_command:
            mock_warmup_job = MagicMock()
            mock_command.return_value = mock_warmup_job
            
            prepare_environment_image(mock_client, mock_env, "compute-cluster", env_config)
            
            mock_client.jobs.create_or_update.assert_called_once()

    def test_prepare_environment_image_respects_history_limit(self):
        """Test that history limit is respected when checking jobs."""
        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_env.version = "v12345678"
        
        mock_jobs = [MagicMock() for _ in range(15)]
        for job in mock_jobs:
            job.status = "Completed"
            job.environment = MagicMock()
            job.environment.name = "test-env"
            job.environment.version = "v12345678"
        
        mock_client.jobs.list.return_value = mock_jobs
        
        env_config = EnvironmentConfig(
            name="test-env",
            conda_path=Path("conda.yaml"),
            docker_image="test/image:latest",
            warmup_display_name="warmup",
            warmup_history_limit=10,
        )
        
        prepare_environment_image(mock_client, mock_env, "compute-cluster", env_config)
        
        mock_client.jobs.list.assert_called_once()

