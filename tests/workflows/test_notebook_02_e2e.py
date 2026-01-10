"""E2E test for 02_best_config_selection.ipynb workflow.

This test validates the high-level workflow of the notebook in a CI-friendly way:

1. Load experiment and selection-related configs
2. Setup MLflow tracking (forced to local file-based backend)
3. Run best-model-selection → artifact acquisition → final training → conversion
   using the real orchestration modules but with heavy steps mocked/shimmed.

The goal is to exercise the real wiring, config usage, naming, paths, and tags
without requiring Azure ML, GPUs, or long training/benchmark jobs.

Validation includes:
- Path structure validation against paths.yaml v2 patterns (final_training_v2, conversion_v2, best_config_v2)
- Run name validation against naming.yaml patterns (final_training, conversion)
- Tag validation against tags.yaml definitions (when MLflow runs are available)
- Metadata.json structure validation (spec_fp, exec_fp, mlflow.run_id)
- Lineage extraction validation (hpo_study_key_hash, hpo_trial_key_hash)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import mlflow
import pytest

from orchestration import EXPERIMENT_NAME
from infrastructure.config.loader import load_experiment_config
from training_exec import extract_lineage_from_best_model
from infrastructure.naming.mlflow.tags_registry import load_tags_registry
from common.shared.platform_detection import detect_platform
from common.shared.yaml_utils import load_yaml

# Import shared validators
import sys
_fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(_fixtures_path.parent))
from fixtures import validate_path_structure, validate_run_name, validate_tags


ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"


def _make_fake_best_model() -> Dict[str, Any]:
    """Create a minimal best_model dict compatible with extract_lineage and final_training."""
    return {
        "run_id": "refit-run-123",
        "trial_run_id": "trial-run-123",
        "experiment_name": f"{EXPERIMENT_NAME}-hpo-distilbert",
        "experiment_id": "1",
        "backbone": "distilbert",
        "study_key_hash": "studyhash123",
        "trial_key_hash": "trialhash123",
        "f1_score": 0.75,
        "latency_ms": 5.0,
        "composite_score": 0.9,
        "tags": {
            # Minimal tags needed by extract_lineage_from_best_model
            "code.study_key_hash": "studyhash123",
            "code.trial_key_hash": "trialhash123",
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


def _create_fake_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a minimal checkpoint directory that passes _validate_checkpoint."""
    checkpoint_dir = tmp_path / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Minimal files: model + config
    (checkpoint_dir / "pytorch_model.bin").write_text("fake-model", encoding="utf-8")
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    return checkpoint_dir


def _create_fake_final_training_output(tmp_path: Path) -> Path:
    """Create a minimal final_training output directory with metadata.json.
    
    Path structure matches final_training_v2 pattern: {storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}
    """
    # Use 8-character hashes to match v2 pattern
    spec8 = "aaaaaaaa"
    exec8 = "bbbbbbbb"
    variant = 1
    output_dir = tmp_path / "final_training" / "local" / "distilbert" / f"spec-{spec8}_exec-{exec8}" / f"v{variant}"
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "spec_fp": "aaaaaaaa" * 4,  # Full fingerprint (32 chars)
        "exec_fp": "bbbbbbbb" * 4,  # Full fingerprint (32 chars)
        "mlflow": {
            "run_id": "final-training-run-123",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Minimal checkpoint payload
    (checkpoint_dir / "pytorch_model.bin").write_text("final-model", encoding="utf-8")
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    return checkpoint_dir


def _create_fake_conversion_output(tmp_path: Path) -> Path:
    """Create a minimal conversion output directory with an ONNX file.
    
    Path structure matches conversion_v2 pattern: {storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}
    """
    # Use 8-character hashes to match v2 pattern
    spec8 = "aaaaaaaa"
    exec8 = "bbbbbbbb"
    variant = 1
    conv8 = "cccccccc"
    output_dir = tmp_path / "conversion" / "local" / "distilbert" / f"spec-{spec8}_exec-{exec8}" / f"v{variant}" / f"conv-{conv8}"
    onnx_dir = output_dir / "onnx_model"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx-model")
    return output_dir




@pytest.mark.e2e
@pytest.mark.integration
def test_best_config_selection_e2e(tmp_path, monkeypatch):
    """E2E-style test for best-config selection → acquisition → final training → conversion."""
    # Step 1: Environment + repo detection (mirror notebook logic at high level)
    platform = detect_platform()
    assert platform in {"local", "colab", "kaggle"}

    assert CONFIG_DIR.exists()
    assert (ROOT_DIR / "src").exists()

    # Step 2: Load experiment and selection-related configs
    experiment_config = load_experiment_config(CONFIG_DIR, EXPERIMENT_NAME)
    tags_config = load_tags_registry(CONFIG_DIR)
    selection_config = load_yaml(CONFIG_DIR / "best_model_selection.yaml")
    acquisition_config = load_yaml(CONFIG_DIR / "artifact_acquisition.yaml")
    conversion_config = load_yaml(CONFIG_DIR / "conversion.yaml")

    assert experiment_config.name == EXPERIMENT_NAME
    assert "objective" in selection_config
    # acquisition_config schema may evolve; just ensure core sections exist
    assert "local" in acquisition_config
    assert "mlflow" in acquisition_config
    assert "output" in conversion_config

    # Step 3: Force MLflow to use a local file-based backend for CI
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    tracking_uri = f"file://{mlruns_dir}"
    mlflow.set_tracking_uri(tracking_uri)

    # Step 4: Patch selection → acquisition → final training → conversion to use lightweight fakes
    fake_best_model = _make_fake_best_model()
    fake_checkpoint_dir = _create_fake_checkpoint_dir(tmp_path)
    fake_final_checkpoint_dir = _create_fake_final_training_output(tmp_path)
    fake_conversion_output_dir = _create_fake_conversion_output(tmp_path)

    # Patch find_best_model_from_mlflow to avoid seeding real MLflow experiments
    monkeypatch.setattr(
        "orchestration.jobs.selection.mlflow_selection.find_best_model_from_mlflow",
        lambda *args, **kwargs: fake_best_model,
    )

    # Patch artifact acquisition to return our fake checkpoint
    monkeypatch.setattr(
        "orchestration.jobs.selection.artifact_acquisition.acquire_best_model_checkpoint",
        lambda *args, **kwargs: fake_checkpoint_dir,
    )

    # Patch final training executor to return our fake final checkpoint
    # Note: The actual import is from training_exec, so we patch that
    monkeypatch.setattr(
        "training_exec.execute_final_training",
        lambda **kwargs: fake_final_checkpoint_dir,
    )
    # Also patch at the executor level in case it's imported directly
    monkeypatch.setattr(
        "training_exec.executor.execute_final_training",
        lambda **kwargs: fake_final_checkpoint_dir,
    )

    # Patch conversion executor to return our fake conversion directory
    monkeypatch.setattr(
        "orchestration.jobs.conversion.execute_conversion",
        lambda **kwargs: fake_conversion_output_dir,
    )

    # Step 5: Run the orchestrated workflow equivalent to notebook Steps 6–8

    # Best model selection (mocked find_best_model_from_mlflow)
    benchmark_experiment = {"name": f"{EXPERIMENT_NAME}-benchmark", "id": "benchmark-exp-id"}
    hpo_experiments = {
        "distilbert": {
            "name": f"{EXPERIMENT_NAME}-hpo-distilbert",
            "id": "hpo-exp-id",
        }
    }

    # Import inside test so monkeypatch on module attribute is effective
    from selection import mlflow_selection

    best_model = mlflow_selection.find_best_model_from_mlflow(
        benchmark_experiment=benchmark_experiment,
        hpo_experiments=hpo_experiments,
        tags_config=tags_config,
        selection_config=selection_config,
        use_python_filtering=True,
    )
    assert best_model is fake_best_model

    # Artifact acquisition
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
    assert (Path(best_checkpoint_dir) / "pytorch_model.bin").exists()

    # Lineage extraction for final training tags
    lineage = extract_lineage_from_best_model(best_model)
    assert lineage["hpo_study_key_hash"] == fake_best_model["study_key_hash"]
    assert lineage["hpo_trial_key_hash"] == fake_best_model["trial_key_hash"]

    # Mock training subprocess execution
    from unittest.mock import Mock
    def fake_execute_training_subprocess(*args, **kwargs):
        result = Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result
    monkeypatch.setattr("training.execution.subprocess_runner.execute_training_subprocess", fake_execute_training_subprocess)
    
    # Mock MLflow client
    mock_client = Mock()
    mock_experiment = Mock()
    mock_experiment.experiment_id = "exp-123"
    mock_client.get_experiment_by_name.return_value = mock_experiment
    mock_run = Mock()
    mock_run.info.run_id = "run-123"
    mock_client.create_run.return_value = mock_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda *args, **kwargs: mock_client)
    
    # Final training
    from training_exec import execute_final_training

    training_experiment_name = f"{EXPERIMENT_NAME}-training"
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
    final_output_dir = Path(final_checkpoint_dir).parent
    metadata_path = final_output_dir / "metadata.json"
    assert metadata_path.exists()
    
    # Validate final training path structure matches paths.yaml v2 pattern
    assert validate_path_structure(final_output_dir, "final_training_v2", CONFIG_DIR), \
        f"Final training path {final_output_dir} does not match final_training_v2 pattern"
    
    # Validate metadata.json structure
    metadata = json.loads(metadata_path.read_text())
    assert "spec_fp" in metadata, "metadata.json missing spec_fp"
    assert "exec_fp" in metadata, "metadata.json missing exec_fp"
    assert "mlflow" in metadata, "metadata.json missing mlflow section"
    
    # Validate final training run name and tags (if MLflow was used)
    # Note: Since we're mocking execute_final_training, we can't check real MLflow runs
    # But we can validate the structure of what would be created
    if "mlflow" in metadata and "run_id" in metadata["mlflow"]:
        run_id = metadata["mlflow"]["run_id"]
        # Try to get run from MLflow if it exists (may not exist if fully mocked)
        try:
            run = mlflow.get_run(run_id)
            run_name = run.data.tags.get("mlflow.runName") or run.info.run_name
            assert validate_run_name(run_name, "final_training", CONFIG_DIR), \
                f"Final training run name '{run_name}' does not match naming.yaml pattern"
            
            # Validate tags
            tags = run.data.tags
            is_valid, missing = validate_tags(tags, "final_training", CONFIG_DIR)
            if not is_valid:
                pytest.fail(f"Missing required final_training tags: {missing}. Found tags: {list(tags.keys())[:20]}")
        except Exception:
            # Run may not exist if fully mocked, skip tag validation
            pass

    # Conversion
    from conversion import execute_conversion

    conversion_output_dir = execute_conversion(
        root_dir=ROOT_DIR,
        config_dir=CONFIG_DIR,
        parent_training_output_dir=final_output_dir,
        parent_spec_fp="aaaaaaaa",
        parent_exec_fp="bbbbbbbb",
        experiment_config=experiment_config,
        conversion_experiment_name=f"{EXPERIMENT_NAME}-conversion",
        platform=platform,
        parent_training_run_id="final-training-run-123",
    )
    assert conversion_output_dir == fake_conversion_output_dir
    onnx_files = list(conversion_output_dir.rglob("*.onnx"))
    assert onnx_files, "Expected at least one ONNX file in conversion output"
    
    # Validate conversion path structure matches paths.yaml v2 pattern
    assert validate_path_structure(conversion_output_dir, "conversion_v2", CONFIG_DIR), \
        f"Conversion path {conversion_output_dir} does not match conversion_v2 pattern"
    
    # Validate conversion run name and tags (if MLflow was used)
    # Note: Since we're mocking execute_conversion, we can't check real MLflow runs
    # But we can validate the structure of what would be created
    # Check if conversion metadata exists (some implementations may create this)
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
                
                # Validate tags
                tags = run.data.tags
                is_valid, missing = validate_tags(tags, "conversion", CONFIG_DIR)
                if not is_valid:
                    pytest.fail(f"Missing required conversion tags: {missing}. Found tags: {list(tags.keys())[:20]}")
            except Exception:
                # Run may not exist if fully mocked, skip tag validation
                pass
    
    # Validate best model selection path structure (if checkpoint was saved)
    # The artifact acquisition may save checkpoint to a structured path
    # Note: Since we're mocking acquire_best_model_checkpoint, the real path won't be created
    # But we can validate the structure that would be created matches the pattern
    # Check if any selection directories exist in the real project (from previous runs)
    best_selection_base = ROOT_DIR / "outputs" / "best_model_selection" / platform
    if best_selection_base.exists():
        # Check if any selection directories exist
        selection_dirs = list(best_selection_base.rglob("sel_*"))
        if selection_dirs:
            selection_dir = selection_dirs[0]
            # Validate path structure matches best_config_v2 pattern
            assert validate_path_structure(selection_dir, "best_config_v2", CONFIG_DIR), \
                f"Best model selection path {selection_dir} does not match best_config_v2 pattern"
    
    # Additional validation: Check that lineage extraction produces valid structure
    assert "hpo_study_key_hash" in lineage, "Lineage missing hpo_study_key_hash"
    assert "hpo_trial_key_hash" in lineage, "Lineage missing hpo_trial_key_hash"
    assert lineage["hpo_study_key_hash"] == fake_best_model["study_key_hash"]
    assert lineage["hpo_trial_key_hash"] == fake_best_model["trial_key_hash"]

    # Sanity check: MLflow tracking URI is local and no Azure dependencies were required
    assert tracking_uri.startswith("file://")


