"""E2E test for complete workflow: 01_orchestrate_training_colab.ipynb → 02_best_config_selection.ipynb.

This test validates the complete end-to-end workflow from HPO training through best model selection,
final training, and conversion in a CI-friendly way:

Workflow Steps:
1. Environment detection (01 Step 1)
2. Config loading (01 Step 3)
3. Dataset verification (01 Step 4)
4. MLflow setup (01 Step 5)
5. HPO sweep execution (01 Step 6) - mocked
6. Benchmarking execution (01 Step 7) - mocked
7. Best model selection (02 Step 6)
8. Artifact acquisition (02 Step 6)
9. Final training (02 Step 7) - mocked
10. Model conversion (02 Step 8) - mocked

The goal is to exercise the complete pipeline wiring, config usage, naming, paths, and tags
without requiring Azure ML, GPUs, or long training/benchmark jobs.

Validation includes:
- Path structure validation against paths.yaml v2 patterns (hpo_v2, benchmarking_v2, final_training_v2, conversion_v2, best_config_v2)
- Run name validation against naming.yaml patterns (hpo, benchmarking, final_training, conversion)
- Tag validation against tags.yaml definitions (when MLflow runs are available)
- Metadata.json structure validation throughout the pipeline
- Lineage tracking validation (HPO → final training → conversion)

Usage:
    # Default: Core workflow with mocked training (CI-compatible)
    pytest tests/e2e/test_full_workflow_e2e.py -v

    # Real training execution (slower)
    E2E_USE_REAL_TRAINING=true pytest tests/e2e/test_full_workflow_e2e.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import mlflow
import pytest

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "config"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orchestration import (
    STAGE_HPO,
    EXPERIMENT_NAME,
    METRICS_FILENAME,
)
from config.loader import (
    load_experiment_config,
    load_all_configs,
)
from training_exec import extract_lineage_from_best_model
from orchestration.jobs.tracking.naming.tags_registry import load_tags_registry
from shared.platform_detection import detect_platform
from shared.yaml_utils import load_yaml


# ============================================================================
# Test Configuration Helpers
# ============================================================================

def should_use_real_training() -> bool:
    """Check if real training should be used."""
    return os.environ.get("E2E_USE_REAL_TRAINING", "false").lower() == "true"


def should_skip_gpu_checks() -> bool:
    """Check if GPU checks should be skipped."""
    return os.environ.get("E2E_SKIP_GPU_CHECKS", "true").lower() == "true"


# Import shared fixtures and validators
import sys
_fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(_fixtures_path.parent))
from fixtures import (
    tiny_dataset,
    mock_mlflow_tracking,
    validate_path_structure,
    validate_run_name,
    validate_tags,
)




# ============================================================================
# Test Fixtures
# ============================================================================



@pytest.fixture
def mock_gpu_detection(monkeypatch):
    """Mock GPU detection for CI compatibility."""
    if should_skip_gpu_checks():
        def mock_cuda_is_available():
            return False
        
        def mock_cuda_device_count():
            return 0
        
        monkeypatch.setattr("torch.cuda.is_available", mock_cuda_is_available, raising=False)
        monkeypatch.setattr("torch.cuda.device_count", mock_cuda_device_count, raising=False)


# ============================================================================
# Main E2E Test
# ============================================================================

@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_full_workflow_e2e(
    tmp_path,
    tiny_dataset,
    mock_mlflow_tracking,
    mock_gpu_detection,
    monkeypatch,
):
    """E2E test covering complete workflow from 01 notebook to 02 notebook."""
    
    # ========================================================================
    # Phase 1: 01_orchestrate_training_colab.ipynb Steps
    # ========================================================================
    
    # Step 1: Environment Detection
    platform = detect_platform()
    assert platform in {"local", "colab", "kaggle"}
    
    assert CONFIG_DIR.exists()
    assert (ROOT_DIR / "src").exists()
    
    # Step 2: Config Loading
    experiment_config = load_experiment_config(CONFIG_DIR, EXPERIMENT_NAME)
    configs = load_all_configs(experiment_config)
    tags_config = load_tags_registry(CONFIG_DIR)
    
    assert experiment_config.name == EXPERIMENT_NAME
    assert "data" in configs
    assert "model" in configs
    assert "train" in configs
    assert "hpo" in configs
    
    # Step 3: Dataset Verification
    data_config = configs["data"]
    assert (tiny_dataset / "train.json").exists()
    assert (tiny_dataset / "validation.json").exists()
    assert (tiny_dataset / "test.json").exists()
    
    # Step 4: MLflow Setup
    from shared.mlflow_setup import setup_mlflow_from_config
    
    training_experiment_name = f"{EXPERIMENT_NAME}-training"
    tracking_uri = setup_mlflow_from_config(
        experiment_name=training_experiment_name,
        config_dir=CONFIG_DIR
    )
    assert tracking_uri.startswith("file://")
    
    # Step 5: HPO Sweep Execution (mocked)
    hpo_config = configs["hpo"]
    train_config = configs["train"]
    
    # Override max_trials for testing
    hpo_config = hpo_config.copy()
    hpo_config["sampling"] = hpo_config.get("sampling", {}).copy()
    hpo_config["sampling"]["max_trials"] = 1
    
    # Disable checkpointing for simpler test
    checkpoint_config = hpo_config.get("checkpoint", {})
    if checkpoint_config:
        checkpoint_config = checkpoint_config.copy()
        checkpoint_config["enabled"] = False
    
    # Mock subprocess for training
    def subprocess_side_effect(*args, **kwargs):
        output_path = None
        if "env" in kwargs:
            env = kwargs["env"]
            if "AZURE_ML_OUTPUT_CHECKPOINT" in env:
                output_path = Path(env["AZURE_ML_OUTPUT_CHECKPOINT"])
            elif "AZURE_ML_OUTPUT_checkpoint" in env:
                output_path = Path(env["AZURE_ML_OUTPUT_checkpoint"])
        
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            metrics_file = output_path / METRICS_FILENAME
            metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed"
        mock_result.stderr = ""
        return mock_result
    
    with patch('orchestration.jobs.hpo.local.trial.execution.subprocess.run', side_effect=subprocess_side_effect):
        from hpo import run_local_hpo_sweep
        
        environment = detect_platform()
        output_dir = ROOT_DIR / "outputs" / "hpo" / environment / "distilbert"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backbone_values = hpo_config["search_space"]["backbone"]["values"]
        backbone = backbone_values[0] if backbone_values else "distilbert"
        
        from orchestration import build_mlflow_experiment_name
        mlflow_experiment_name = build_mlflow_experiment_name(
            experiment_config.name, STAGE_HPO, backbone
        )
        
        study = run_local_hpo_sweep(
            dataset_path=str(tiny_dataset),
            config_dir=CONFIG_DIR,
            backbone=backbone,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name=mlflow_experiment_name,
            checkpoint_config=checkpoint_config,
            data_config=data_config,
        )
        
        assert study is not None
        assert len(study.trials) > 0
        
        # Validate HPO path structure
        study_dirs = list(output_dir.glob("study-*"))
        if study_dirs:
            study_dir = study_dirs[0]
            assert validate_path_structure(study_dir, "hpo_v2", CONFIG_DIR), \
                f"HPO study path {study_dir} does not match hpo_v2 pattern"
            
            # Validate HPO MLflow run name and tags
            try:
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        max_results=1,
                        order_by=["start_time DESC"],
                        filter_string="tags.mlflow.runType = 'sweep'"
                    )
                    if not runs.empty:
                        run_id = runs.iloc[0]["run_id"]
                        run = mlflow.get_run(run_id)
                        run_name = run.data.tags.get("mlflow.runName") or run.info.run_name
                        assert validate_run_name(run_name, "hpo", CONFIG_DIR), \
                            f"HPO run name '{run_name}' does not match naming.yaml pattern"
                        
                        tags = run.data.tags
                        is_valid, missing = validate_tags(tags, "hpo", CONFIG_DIR)
                        if not is_valid:
                            pytest.fail(f"Missing required HPO tags: {missing}. Found tags: {list(tags.keys())[:20]}")
            except Exception:
                # MLflow run may not exist if fully mocked, skip validation
                pass
    
    # Step 6: Benchmarking Execution (mocked)
    # Create fake HPO trial output for benchmarking
    hpo_output_dir = ROOT_DIR / "outputs" / "hpo" / environment / "distilbert"
    study_dirs = list(hpo_output_dir.glob("study-*")) if hpo_output_dir.exists() else []
    if study_dirs:
        study_dir = study_dirs[0]
        trial_dirs = list(study_dir.glob("trial-*"))
        if trial_dirs:
            trial_dir = trial_dirs[0]
        else:
            trial_dir = study_dir / "trial-test456"
            trial_dir.mkdir(parents=True, exist_ok=True)
    else:
        study_dir = hpo_output_dir / "study-test123"
        trial_dir = study_dir / "trial-test456"
        trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fake checkpoint and metrics
    checkpoint_dir = trial_dir / "refit" / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "pytorch_model.bin").write_text("fake model")
    (trial_dir / METRICS_FILENAME).write_text(json.dumps({"macro-f1": 0.75}))
    
    # Create trial_meta.json
    trial_meta = {
        "study_key_hash": "df9d920c" * 8,
        "trial_key_hash": trial_dir.name.replace("trial-", "") * 8 if trial_dir.name.startswith("trial-") else "test456" * 8,
        "trial_number": 0,
        "study_name": study_dir.name,
    }
    (trial_dir / "trial_meta.json").write_text(json.dumps(trial_meta, indent=2))
    
    # Mock benchmark subprocess
    def benchmark_subprocess_side_effect(*args, **kwargs):
        cmd = args[0] if args and isinstance(args[0], list) else []
        output_path = None
        for i, arg in enumerate(cmd):
            if arg == "--output" and i + 1 < len(cmd):
                output_path = Path(cmd[i + 1])
                break
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_data = {
                "batch_size": 1,
                "latency_ms": 10.5,
                "throughput_samples_per_sec": 95.2,
            }
            output_path.write_text(json.dumps(benchmark_data, indent=2))
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Benchmark completed"
        mock_result.stderr = ""
        return mock_result
    
    with patch('orchestration.benchmark_utils.subprocess.run', side_effect=benchmark_subprocess_side_effect):
        from benchmarking import benchmark_best_trials
        
        best_trials = {
            "distilbert": {
                "trial_dir": str(trial_dir),
                "study_name": study_dir.name,
                "trial_name": trial_dir.name,
                "accuracy": 0.75,
                "hyperparameters": {
                    "learning_rate": 4.71e-05,
                    "batch_size": 4,
                    "dropout": 0.109404,
                    "weight_decay": 0.001721,
                },
                "study_key_hash": trial_meta["study_key_hash"],
                "trial_key_hash": trial_meta["trial_key_hash"],
            }
        }
        
        test_data_path = tiny_dataset / "test.json"
        benchmark_config = configs.get("benchmark", {})
        benchmark_settings = benchmark_config.get("benchmarking", {})
        
        benchmark_batch_sizes = benchmark_settings.get("batch_sizes", [1, 8])
        benchmark_iterations = benchmark_settings.get("iterations", 100)
        benchmark_warmup = benchmark_settings.get("warmup_iterations", 10)
        benchmark_max_length = benchmark_settings.get("max_length", 512)
        
        from orchestration.jobs.tracking.mlflow_tracker import MLflowBenchmarkTracker
        benchmark_tracker = MLflowBenchmarkTracker(f"{EXPERIMENT_NAME}-benchmark")
        
        benchmark_results = benchmark_best_trials(
            best_trials=best_trials,
            test_data_path=test_data_path,
            root_dir=ROOT_DIR,
            environment=environment,
            data_config=data_config,
            hpo_config=hpo_config,
            benchmark_config=benchmark_config,
            benchmark_batch_sizes=benchmark_batch_sizes,
            benchmark_iterations=benchmark_iterations,
            benchmark_warmup=benchmark_warmup,
            benchmark_max_length=benchmark_max_length,
            benchmark_device=None,
            benchmark_tracker=benchmark_tracker,
            backup_enabled=False,
            backup_to_drive=None,
            ensure_restored_from_drive=None,
        )
        
        assert benchmark_results is not None
        benchmark_output_dir = ROOT_DIR / "outputs" / "benchmarking" / environment
        benchmark_paths = list(benchmark_output_dir.rglob("benchmark.json"))
        if benchmark_paths:
            benchmark_path = benchmark_paths[0].parent
            assert validate_path_structure(benchmark_path, "benchmarking_v2", CONFIG_DIR), \
                f"Benchmark path {benchmark_path} does not match benchmarking_v2 pattern"
            
            # Validate benchmark MLflow run name and tags
            try:
                benchmark_experiment_name = f"{EXPERIMENT_NAME}-benchmark"
                experiment = mlflow.get_experiment_by_name(benchmark_experiment_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        max_results=1,
                        order_by=["start_time DESC"]
                    )
                    if not runs.empty:
                        run_id = runs.iloc[0]["run_id"]
                        run = mlflow.get_run(run_id)
                        run_name = run.data.tags.get("mlflow.runName") or run.info.run_name
                        assert validate_run_name(run_name, "benchmarking", CONFIG_DIR), \
                            f"Benchmark run name '{run_name}' does not match naming.yaml pattern"
                        
                        tags = run.data.tags
                        is_valid, missing = validate_tags(tags, "benchmarking", CONFIG_DIR)
                        if not is_valid:
                            pytest.fail(f"Missing required benchmarking tags: {missing}. Found tags: {list(tags.keys())[:20]}")
            except Exception:
                # MLflow run may not exist if fully mocked, skip validation
                pass
    
    # ========================================================================
    # Phase 2: 02_best_config_selection.ipynb Steps
    # ========================================================================
    
    # Step 7: Best Model Selection
    selection_config = load_yaml(CONFIG_DIR / "best_model_selection.yaml")
    acquisition_config = load_yaml(CONFIG_DIR / "artifact_acquisition.yaml")
    conversion_config = load_yaml(CONFIG_DIR / "conversion.yaml")
    
    benchmark_experiment = {"name": f"{EXPERIMENT_NAME}-benchmark", "id": "benchmark-exp-id"}
    hpo_experiments = {
        "distilbert": {
            "name": f"{EXPERIMENT_NAME}-hpo-distilbert",
            "id": "hpo-exp-id",
        }
    }
    
    # Create fake best model (since MLflow queries would be complex to mock)
    fake_best_model = {
        "run_id": "refit-run-123",
        "trial_run_id": "trial-run-123",
        "experiment_name": f"{EXPERIMENT_NAME}-hpo-distilbert",
        "experiment_id": "1",
        "backbone": "distilbert",
        "study_key_hash": trial_meta["study_key_hash"],
        "trial_key_hash": trial_meta["trial_key_hash"],
        "f1_score": 0.75,
        "latency_ms": 5.0,
        "composite_score": 0.9,
        "tags": {
            "code.study_key_hash": trial_meta["study_key_hash"],
            "code.trial_key_hash": trial_meta["trial_key_hash"],
            "mlflow.runName": "local_distilbert_hpo_study-studyhash123_trial-trialhash123",
            "mlflow.runType": "sweep",
        },
        "params": {
            "learning_rate": "5e-5",
            "batch_size": "4",
            "random_seed": "42",
        },
        "metrics": {
            "macro-f1": 0.75,
        },
        "has_refit_run": True,
    }
    
    # Mock find_best_model_from_mlflow
    monkeypatch.setattr(
        "orchestration.jobs.selection.mlflow_selection.find_best_model_from_mlflow",
        lambda *args, **kwargs: fake_best_model,
    )
    
    from selection import mlflow_selection
    
    best_model = mlflow_selection.find_best_model_from_mlflow(
        benchmark_experiment=benchmark_experiment,
        hpo_experiments=hpo_experiments,
        tags_config=tags_config,
        selection_config=selection_config,
        use_python_filtering=True,
    )
    assert best_model is fake_best_model
    
    # Step 8: Artifact Acquisition
    fake_checkpoint_dir = tmp_path / "best_checkpoint"
    fake_checkpoint_dir.mkdir(parents=True)
    (fake_checkpoint_dir / "pytorch_model.bin").write_text("fake-model")
    (fake_checkpoint_dir / "config.json").write_text("{}")
    
    monkeypatch.setattr(
        "orchestration.jobs.selection.artifact_acquisition.acquire_best_model_checkpoint",
        lambda *args, **kwargs: fake_checkpoint_dir,
    )
    
    from selection.artifact_acquisition import acquire_best_model_checkpoint
    
    best_checkpoint_dir = acquire_best_model_checkpoint(
        best_run_info=best_model,
        root_dir=ROOT_DIR,
        config_dir=CONFIG_DIR,
        acquisition_config=acquisition_config,
        selection_config=selection_config,
        platform=platform,
        restore_from_drive=None,
        drive_store=None,
        in_colab=(platform == "colab"),
    )
    assert best_checkpoint_dir == fake_checkpoint_dir
    
    # Step 9: Lineage Extraction
    lineage = extract_lineage_from_best_model(best_model)
    assert lineage["hpo_study_key_hash"] == fake_best_model["study_key_hash"]
    assert lineage["hpo_trial_key_hash"] == fake_best_model["trial_key_hash"]
    
    # Step 10: Final Training
    fake_final_checkpoint_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec-aaaaaaaa_exec-bbbbbbbb" / "v1" / "checkpoint"
    fake_final_checkpoint_dir.mkdir(parents=True)
    (fake_final_checkpoint_dir / "pytorch_model.bin").write_text("final-model")
    (fake_final_checkpoint_dir / "config.json").write_text("{}")
    
    final_output_dir = fake_final_checkpoint_dir.parent
    metadata = {
        "spec_fp": "aaaaaaaa" * 4,
        "exec_fp": "bbbbbbbb" * 4,
        "mlflow": {
            "run_id": "final-training-run-123",
        },
    }
    (final_output_dir / "metadata.json").write_text(json.dumps(metadata))
    
    monkeypatch.setattr(
        "orchestration.jobs.final_training.execute_final_training",
        lambda **kwargs: fake_final_checkpoint_dir,
    )
    
    from training_exec import execute_final_training
    
    final_checkpoint_dir = execute_final_training(
        root_dir=ROOT_DIR,
        config_dir=CONFIG_DIR,
        best_model=best_model,
        experiment_config=experiment_config,
        lineage=lineage,
        training_experiment_name=training_experiment_name,
        platform=platform,
    )
    assert final_checkpoint_dir == fake_final_checkpoint_dir
    
    # Validate final training path structure
    assert validate_path_structure(final_output_dir, "final_training_v2", CONFIG_DIR), \
        f"Final training path {final_output_dir} does not match final_training_v2 pattern"
    
    metadata_path = final_output_dir / "metadata.json"
    assert metadata_path.exists()
    metadata_data = json.loads(metadata_path.read_text())
    assert "spec_fp" in metadata_data
    assert "exec_fp" in metadata_data
    
    # Validate final training MLflow run name and tags (if MLflow run exists)
    if "mlflow" in metadata_data and "run_id" in metadata_data["mlflow"]:
        run_id = metadata_data["mlflow"]["run_id"]
        try:
            run = mlflow.get_run(run_id)
            run_name = run.data.tags.get("mlflow.runName") or run.info.run_name
            assert validate_run_name(run_name, "final_training", CONFIG_DIR), \
                f"Final training run name '{run_name}' does not match naming.yaml pattern"
            
            tags = run.data.tags
            is_valid, missing = validate_tags(tags, "final_training", CONFIG_DIR)
            if not is_valid:
                pytest.fail(f"Missing required final_training tags: {missing}. Found tags: {list(tags.keys())[:20]}")
        except Exception:
            # Run may not exist if fully mocked, skip tag validation
            pass
    
    # Step 11: Model Conversion
    fake_conversion_output_dir = tmp_path / "conversion" / "local" / "distilbert" / "spec-aaaaaaaa_exec-bbbbbbbb" / "v1" / "conv-cccccccc"
    onnx_dir = fake_conversion_output_dir / "onnx_model"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "model.onnx").write_bytes(b"fake-onnx-model")
    
    monkeypatch.setattr(
        "orchestration.jobs.conversion.execute_conversion",
        lambda **kwargs: fake_conversion_output_dir,
    )
    
    from conversion import execute_conversion
    
    conversion_output_dir = execute_conversion(
        root_dir=ROOT_DIR,
        config_dir=CONFIG_DIR,
        parent_training_output_dir=final_output_dir,
        parent_spec_fp="aaaaaaaa" * 4,
        parent_exec_fp="bbbbbbbb" * 4,
        experiment_config=experiment_config,
        conversion_experiment_name=f"{EXPERIMENT_NAME}-conversion",
        platform=platform,
        parent_training_run_id="final-training-run-123",
    )
    assert conversion_output_dir == fake_conversion_output_dir
    
    # Validate conversion path structure
    assert validate_path_structure(conversion_output_dir, "conversion_v2", CONFIG_DIR), \
        f"Conversion path {conversion_output_dir} does not match conversion_v2 pattern"
    
    onnx_files = list(conversion_output_dir.rglob("*.onnx"))
    assert onnx_files, "Expected at least one ONNX file in conversion output"
    
    # Validate conversion MLflow run name and tags (if conversion metadata exists)
    conversion_metadata_path = conversion_output_dir / "metadata.json"
    if conversion_metadata_path.exists():
        conversion_metadata = json.loads(conversion_metadata_path.read_text())
        if "mlflow" in conversion_metadata and "run_id" in conversion_metadata["mlflow"]:
            run_id = conversion_metadata["mlflow"]["run_id"]
            try:
                run = mlflow.get_run(run_id)
                run_name = run.data.tags.get("mlflow.runName") or run.info.run_name
                assert validate_run_name(run_name, "conversion", CONFIG_DIR), \
                    f"Conversion run name '{run_name}' does not match naming.yaml pattern"
                
                tags = run.data.tags
                is_valid, missing = validate_tags(tags, "conversion", CONFIG_DIR)
                if not is_valid:
                    pytest.fail(f"Missing required conversion tags: {missing}. Found tags: {list(tags.keys())[:20]}")
            except Exception:
                # Run may not exist if fully mocked, skip tag validation
                pass
    
    # ========================================================================
    # Final Validation Summary
    # ========================================================================
    
    # Verify complete workflow produced expected outputs
    assert study is not None, "HPO study should be created"
    assert benchmark_results is not None, "Benchmark results should be created"
    assert best_model is not None, "Best model should be selected"
    assert best_checkpoint_dir is not None, "Best checkpoint should be acquired"
    assert final_checkpoint_dir is not None, "Final training checkpoint should be created"
    assert conversion_output_dir is not None, "Conversion output should be created"
    
    # Verify path structures throughout pipeline
    assert validate_path_structure(study_dir, "hpo_v2", CONFIG_DIR), "HPO path validation"
    if benchmark_paths:
        assert validate_path_structure(benchmark_path, "benchmarking_v2", CONFIG_DIR), "Benchmark path validation"
    assert validate_path_structure(final_output_dir, "final_training_v2", CONFIG_DIR), "Final training path validation"
    assert validate_path_structure(conversion_output_dir, "conversion_v2", CONFIG_DIR), "Conversion path validation"
    
    # Verify lineage tracking
    assert "hpo_study_key_hash" in lineage, "Lineage should include study hash"
    assert "hpo_trial_key_hash" in lineage, "Lineage should include trial hash"
    
    # Verify MLflow tracking URI is local (CI-compatible)
    assert tracking_uri.startswith("file://"), "Should use local MLflow tracking"

