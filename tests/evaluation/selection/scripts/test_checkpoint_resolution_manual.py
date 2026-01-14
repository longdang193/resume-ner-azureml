"""Manual test script for checkpoint resolution in benchmarking workflow.

This script can be run manually to test checkpoint resolution from various sources.
It simulates the workflow from Step 6 of 02_best_config_selection.ipynb.

Usage:
    python -m tests.evaluation.selection.scripts.test_checkpoint_resolution_manual \
        --experiment-name resume_ner_baseline \
        --backbone distilbert
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from mlflow.tracking import MlflowClient
from infrastructure.naming.mlflow.tags_registry import load_tags_registry
from evaluation.selection.artifact_acquisition import acquire_best_model_checkpoint
from common.shared.yaml_utils import load_yaml
from common.shared.platform_detection import detect_platform


def find_refit_runs(
    mlflow_client: MlflowClient,
    tags_registry,
    trial_run_id: str,
    study_key_hash: str,
    trial_key_hash: str,
) -> List[str]:
    """Find refit runs for a given trial run."""
    run_ids = []
    
    try:
        # Get trial run info
        trial_run = mlflow_client.get_run(trial_run_id)
        experiment_id = trial_run.info.experiment_id
        
        stage_tag = tags_registry.key("process", "stage")
        study_key_tag = tags_registry.key("grouping", "study_key_hash")
        trial_key_tag = tags_registry.key("grouping", "trial_key_hash")
        
        # Strategy 1: Search with full hash match
        refit_runs = mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.{stage_tag} = 'hpo_refit' AND "
                f"tags.{study_key_tag} = '{study_key_hash}' AND "
                f"tags.{trial_key_tag} = '{trial_key_hash}'"
            ),
            max_results=5,
        )
        
        if refit_runs:
            print(f"  ✓ Found {len(refit_runs)} refit run(s) with full hash match")
            for refit_run in refit_runs:
                run_ids.append(refit_run.info.run_id)
        else:
            # Strategy 2: Search with study hash only
            refit_runs_alt = mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=(
                    f"tags.{stage_tag} = 'hpo_refit' AND "
                    f"tags.{study_key_tag} = '{study_key_hash}'"
                ),
                max_results=5,
            )
            
            if refit_runs_alt:
                print(f"  ✓ Found {len(refit_runs_alt)} refit run(s) with study hash match")
                parent_run_id = getattr(trial_run.info, 'parent_run_id', None)
                
                for refit_run in refit_runs_alt:
                    refit_run_id = refit_run.info.run_id
                    # Check parent relationship
                    try:
                        full_refit_run = mlflow_client.get_run(refit_run_id)
                        refit_parent_id = getattr(full_refit_run.info, 'parent_run_id', None)
                        if refit_parent_id == trial_run_id or (parent_run_id and refit_parent_id == parent_run_id):
                            if refit_run_id not in run_ids:
                                run_ids.append(refit_run_id)
                    except Exception:
                        # Add anyway if we can't check
                        if refit_run_id not in run_ids:
                            run_ids.append(refit_run_id)
            
            # Strategy 3: Last resort - any refit run in experiment
            if not run_ids:
                all_refit_runs = mlflow_client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.{stage_tag} = 'hpo_refit'",
                    max_results=10,
                )
                if all_refit_runs:
                    print(f"  ⚠ Found {len(all_refit_runs)} refit run(s) in experiment (no hash match)")
                    for refit_run in all_refit_runs:
                        run_ids.append(refit_run.info.run_id)
        
        # Strategy 4: Check if parent is a refit run
        parent_run_id = getattr(trial_run.info, 'parent_run_id', None)
        if parent_run_id:
            try:
                parent_run = mlflow_client.get_run(parent_run_id)
                parent_stage = parent_run.data.tags.get(stage_tag, "")
                if parent_stage == "hpo_refit":
                    if parent_run_id not in run_ids:
                        run_ids.insert(0, parent_run_id)
                        print(f"  ✓ Found parent refit run: {parent_run_id[:12]}...")
            except Exception as e:
                print(f"  ⚠ Could not check parent run: {e}")
    
    except Exception as e:
        print(f"  ⚠ Error searching for refit runs: {e}")
    
    return run_ids


def check_artifacts(mlflow_client: MlflowClient, run_id: str) -> List[str]:
    """Check what artifacts are available in a run."""
    try:
        artifacts = mlflow_client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        return artifact_paths
    except Exception as e:
        print(f"    ⚠ Could not list artifacts: {e}")
        return []


def run_checkpoint_resolution(
    experiment_name: str,
    backbone: str,
    config_dir: Path,
    root_dir: Path,
) -> bool:
    """Test checkpoint resolution for a given backbone."""
    print(f"\n{'='*60}")
    print(f"Testing checkpoint resolution for {backbone}")
    print(f"{'='*60}\n")
    
    # Setup
    mlflow_client = MlflowClient()
    tags_registry = load_tags_registry(config_dir)
    selection_config = load_yaml(config_dir / "best_model_selection.yaml")
    acquisition_config = load_yaml(config_dir / "artifact_acquisition.yaml")
    
    # Find HPO experiment
    hpo_experiment_name = f"{experiment_name}-hpo-{backbone}"
    experiments = mlflow_client.search_experiments()
    hpo_experiment = None
    for exp in experiments:
        if exp.name == hpo_experiment_name:
            hpo_experiment = exp
            break
    
    if not hpo_experiment:
        print(f"  ❌ HPO experiment '{hpo_experiment_name}' not found")
        return False
    
    print(f"  ✓ Found HPO experiment: {hpo_experiment_name}")
    
    # Find champion (best trial run)
    from evaluation.selection.trial_finder import select_champions_for_backbones
    
    hpo_experiments = {backbone: {"name": hpo_experiment_name, "id": hpo_experiment.experiment_id}}
    
    try:
        champions = select_champions_for_backbones(
            hpo_experiments=hpo_experiments,
            selection_config=selection_config,
            tags_config=tags_registry,
            config_dir=config_dir,
            project_root=root_dir,
        )
    except Exception as e:
        print(f"  ❌ Error selecting champions: {e}")
        return False
    
    if backbone not in champions:
        print(f"  ❌ No champion found for {backbone}")
        return False
    
    champion_data = champions[backbone]
    champion = champion_data["champion"]
    
    print(f"  ✓ Found champion:")
    print(f"    - Run ID: {champion['run_id'][:12]}...")
    print(f"    - Study hash: {champion.get('study_key_hash', 'N/A')[:16]}...")
    print(f"    - Trial hash: {champion.get('trial_key_hash', 'N/A')[:16]}...")
    print(f"    - Metric: {champion.get('metric', 'N/A')}")
    
    # Check checkpoint_path from local disk
    checkpoint_path = champion.get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\n  ✓ Checkpoint found on local disk: {checkpoint_path}")
        return True
    
    print(f"\n  ⚠ Checkpoint not found on local disk, trying MLflow...")
    
    # Try to find checkpoint in MLflow
    run_id = champion.get("run_id")
    study_key_hash = champion.get("study_key_hash")
    trial_key_hash = champion.get("trial_key_hash")
    
    if not run_id:
        print(f"  ❌ No run_id in champion data")
        return False
    
    # Build list of run IDs to try
    run_ids_to_try = []
    
    # Find refit runs
    if study_key_hash and trial_key_hash:
        refit_run_ids = find_refit_runs(
            mlflow_client, tags_registry, run_id, study_key_hash, trial_key_hash
        )
        run_ids_to_try.extend(refit_run_ids)
        print(f"  ✓ Found {len(refit_run_ids)} refit run(s) to try")
    
    # Check parent HPO run
    try:
        trial_run = mlflow_client.get_run(run_id)
        parent_run_id = getattr(trial_run.info, 'parent_run_id', None)
        if parent_run_id:
            parent_artifacts = check_artifacts(mlflow_client, parent_run_id)
            checkpoint_in_parent = any("checkpoint" in p.lower() for p in parent_artifacts)
            if checkpoint_in_parent:
                if parent_run_id not in run_ids_to_try:
                    run_ids_to_try.append(parent_run_id)
                print(f"  ✓ Found checkpoint artifacts in parent HPO run")
    except Exception as e:
        print(f"  ⚠ Could not check parent run: {e}")
    
    # Add trial run as last resort
    if run_id not in run_ids_to_try:
        run_ids_to_try.append(run_id)
    
    print(f"\n  Will try {len(run_ids_to_try)} run(s) in order:")
    for i, candidate_run_id in enumerate(run_ids_to_try, 1):
        print(f"    {i}. {candidate_run_id[:12]}...")
    
    # Try each run
    for candidate_run_id in run_ids_to_try:
        print(f"\n  Trying run {candidate_run_id[:12]}...")
        
        # Check artifacts
        artifact_paths = check_artifacts(mlflow_client, candidate_run_id)
        print(f"    Found {len(artifact_paths)} artifact(s)")
        if artifact_paths:
            print(f"    Artifacts: {artifact_paths[:5]}{'...' if len(artifact_paths) > 5 else ''}")
        
        checkpoint_artifacts = [p for p in artifact_paths if "checkpoint" in p.lower()]
        if not checkpoint_artifacts:
            print(f"    ⚠ No checkpoint artifacts found, skipping")
            continue
        
        print(f"    ✓ Found checkpoint artifact(s): {checkpoint_artifacts}")
        
        # Try to acquire checkpoint
        try:
            best_run_info = {
                "run_id": candidate_run_id,
                "study_key_hash": study_key_hash,
                "trial_key_hash": trial_key_hash,
                "backbone": backbone,
            }
            
            # Temporarily set priority to MLflow only
            original_priority = acquisition_config.get("priority", ["local", "mlflow"])
            acquisition_config["priority"] = ["mlflow"]
            
            acquired_checkpoint_dir = acquire_best_model_checkpoint(
                best_run_info=best_run_info,
                root_dir=root_dir,
                config_dir=config_dir,
                acquisition_config=acquisition_config,
                selection_config=selection_config,
                platform=detect_platform(),
                restore_from_drive=None,
                drive_store=None,
                in_colab=False,
            )
            
            # Restore priority
            acquisition_config["priority"] = original_priority
            
            if acquired_checkpoint_dir and Path(acquired_checkpoint_dir).exists():
                print(f"    ✓ Successfully acquired checkpoint: {acquired_checkpoint_dir}")
                return True
            else:
                print(f"    ⚠ Acquisition returned invalid path: {acquired_checkpoint_dir}")
        
        except Exception as e:
            print(f"    ⚠ Error acquiring checkpoint: {e}")
            continue
    
    print(f"\n  ❌ Could not acquire checkpoint from any run")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Test checkpoint resolution for benchmarking workflow"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="resume_ner_baseline",
        help="Experiment name",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Model backbone to test (e.g., distilbert)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Config directory path",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Project root directory",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    root_dir = args.root_dir.resolve()
    config_dir = (root_dir / args.config_dir).resolve() if not args.config_dir.is_absolute() else args.config_dir.resolve()
    
    print(f"Root directory: {root_dir}")
    print(f"Config directory: {config_dir}")
    print(f"Experiment: {args.experiment_name}")
    
    success = run_checkpoint_resolution(
        experiment_name=args.experiment_name,
        backbone=args.backbone,
        config_dir=config_dir,
        root_dir=root_dir,
    )
    
    if success:
        print(f"\n{'='*60}")
        print("✓ Test PASSED - Checkpoint resolution successful")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("❌ Test FAILED - Checkpoint resolution failed")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

