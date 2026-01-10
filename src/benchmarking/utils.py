from __future__ import annotations

"""
@meta
name: benchmarking_utils
type: utility
domain: benchmarking
responsibility:
  - Run inference benchmarks on model checkpoints
  - Execute benchmark subprocess
  - Log benchmark results
inputs:
  - Model checkpoint directory
  - Test data path
outputs:
  - Benchmark results (JSON)
tags:
  - utility
  - benchmarking
ci:
  runnable: true
  needs_gpu: true
  needs_cloud: false
lifecycle:
  status: active
"""

"""Benchmarking utility functions for running inference benchmarks."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

def run_benchmarking(
    checkpoint_dir: Path,
    test_data_path: Path,
    output_path: Path,
    batch_sizes: List[int],
    iterations: int,
    warmup_iterations: int,
    max_length: int = 512,
    device: Optional[str] = None,
    benchmark_script_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    tracker: Optional[Any] = None,
    backbone: Optional[str] = None,
    benchmark_source: str = "final_training",
    study_key_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
    trial_id: Optional[str] = None,
    hpo_trial_run_id: Optional[str] = None,
    hpo_refit_run_id: Optional[str] = None,
    hpo_sweep_run_id: Optional[str] = None,
    benchmark_config_hash: Optional[str] = None,
) -> bool:
    """
    Run benchmarking on a model checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        test_data_path: Path to test data JSON file.
        output_path: Path to output benchmark.json file.
        batch_sizes: List of batch sizes to test.
        iterations: Number of iterations per batch size.
        warmup_iterations: Number of warmup iterations.
        max_length: Maximum sequence length.
        device: Device to use (None = auto-detect).
        benchmark_script_path: Path to benchmark script. If None, will use
            src/benchmarking/cli.py relative to project_root.
        project_root: Project root directory. Required if benchmark_script_path is None.
        tracker: Optional MLflowBenchmarkTracker instance for logging results.
        backbone: Optional model backbone name for tracking.
        benchmark_source: Source of benchmark ("hpo_trial" or "final_training").
        study_key_hash: Optional study key hash from HPO trial (for grouping tags).
        trial_key_hash: Optional trial key hash from HPO trial (for grouping tags).
        hpo_trial_run_id: Optional HPO trial run ID (CV trial run, lineage parent).
        hpo_refit_run_id: Optional HPO refit run ID (refit run, artifact parent).
        hpo_sweep_run_id: Optional HPO sweep run ID (HPO parent, optional).
        benchmark_config_hash: Optional benchmark configuration hash for run naming.

    Returns:
        True if successful, False otherwise.
    """
    # Determine benchmark script path
    if benchmark_script_path is None:
        if project_root is None:
            raise ValueError(
                "Either benchmark_script_path or project_root must be provided")
        # Benchmark CLI is now at src/benchmarking/cli.py
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
    else:
        benchmark_script = benchmark_script_path

    if not benchmark_script.exists():
        print(f"Warning: Benchmark script not found: {benchmark_script}")
        return False

    args = [
        sys.executable,
        "-u",  # Unbuffered output - ensures real-time progress visibility
        str(benchmark_script),
        "--checkpoint",
        str(checkpoint_dir),
        "--test-data",
        str(test_data_path),
        "--batch-sizes",
    ] + [str(bs) for bs in batch_sizes] + [
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup_iterations),
        "--max-length",
        str(max_length),
        "--output",
        str(output_path),
    ]

    if device:
        args.extend(["--device", device])

    cwd = project_root if project_root else checkpoint_dir.parent.parent.parent

    # Run subprocess without capturing output so progress is visible in real-time
    # This is especially important on Colab where users need to see progress
    logger.info(f"Running benchmark script: {' '.join(args)}")
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=False,  # Don't capture - show output in real-time
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            f"Benchmarking failed with return code {result.returncode}")
        return False

    # Log results to MLflow if tracker provided
    if tracker and output_path.exists():
        try:
            # Parse benchmark.json
            with open(output_path, 'r') as f:
                benchmark_data = json.load(f)

            # Extract backbone from checkpoint_dir if not provided
            if not backbone:
                backbone = checkpoint_dir.name

            # Build run name using systematic naming with auto-increment
            from infrastructure.naming import create_naming_context
            from infrastructure.tracking.mlflow.naming import build_mlflow_run_name
            from common.shared.platform_detection import detect_platform
            from pathlib import Path

            # Infer root_dir and config_dir from output_path
            root_dir = project_root if project_root else Path.cwd()
            config_dir = root_dir / "config" if root_dir else None

            # Extract trial_id from checkpoint path if benchmarking an HPO trial
            # Prefer passed trial_id parameter, fallback to path extraction
            extracted_trial_id = trial_id
            logger.debug(
                f"[Benchmark Run Name] Initial trial_id from parameter: {trial_id}, "
                f"benchmark_source={benchmark_source}, checkpoint_dir={checkpoint_dir}"
            )
            if not extracted_trial_id and benchmark_source == "hpo_trial" and checkpoint_dir:
                # Extract trial_id from checkpoint path
                # Handle both old format (trial_1_20251231_161745) and new format (trial-25d03eeb)
                # Also handle refit checkpoints (trial-25d03eeb/refit/checkpoint)
                checkpoint_path = Path(checkpoint_dir)
                current = checkpoint_path
                
                # Walk up the path to find trial directory
                max_levels = 5  # Safety limit
                levels_checked = 0
                while current != current.parent and levels_checked < max_levels:
                    current_name = current.name
                    # Check for both trial_ (old format) and trial- (new format)
                    if current_name.startswith("trial_"):
                        extracted_trial_id = current_name
                        logger.info(
                            f"[Benchmark Run Name] Extracted trial_id from checkpoint path: {extracted_trial_id} (at level {levels_checked})"
                        )
                        break
                    elif current_name.startswith("trial-"):
                        extracted_trial_id = current_name
                        logger.info(
                            f"[Benchmark Run Name] Extracted trial_id from checkpoint path: {extracted_trial_id} (at level {levels_checked})"
                        )
                        break
                    current = current.parent
                    levels_checked += 1
                
                # Fallback: if we have trial_key_hash but no trial_id, construct one
                if not extracted_trial_id and trial_key_hash:
                    # Use first 8 chars of trial_key_hash as trial identifier
                    extracted_trial_id = f"trial-{trial_key_hash[:8]}"
                    logger.info(
                        f"[Benchmark Run Name] Constructed trial_id from trial_key_hash: {extracted_trial_id}"
                    )
            
            # Final fallback: use trial_key_hash if still no trial_id (always check, not just for hpo_trial)
            if not extracted_trial_id and trial_key_hash:
                extracted_trial_id = f"trial-{trial_key_hash[:8]}"
                logger.info(
                    f"[Benchmark Run Name] Final fallback: Constructed trial_id from trial_key_hash: {extracted_trial_id}"
                )

            # Create NamingContext for benchmarking
            # For HPO trial benchmarks, trial_id should be the trial identifier
            # For final training benchmarks, trial_id can be None or a variant identifier
            # Pass study_key_hash and trial_key_hash for proper run naming
            benchmark_context = create_naming_context(
                process_type="benchmarking",
                model=backbone.split(
                    "-")[0] if backbone and "-" in backbone else backbone,
                environment=detect_platform(),
                trial_id=extracted_trial_id,  # Use extracted trial_id if available
                study_key_hash=study_key_hash,  # Pass study_key_hash for grouping
                trial_key_hash=trial_key_hash,  # Pass trial_key_hash for grouping
                benchmark_config_hash=benchmark_config_hash,  # Pass benchmark_config_hash for run naming
            )

            logger.info(
                f"[Benchmark Run Name] Building run name: trial_id={benchmark_context.trial_id}, "
                f"root_dir={root_dir}, config_dir={config_dir}"
            )

            # Build systematic run name with auto-increment
            run_name = build_mlflow_run_name(
                benchmark_context,
                config_dir=config_dir,
                root_dir=root_dir,
                output_dir=output_path.parent if output_path else None,
            )

            logger.info(
                f"[Benchmark Run Name] Generated run name: {run_name}"
            )

            # Start benchmark run and log results
            with tracker.start_benchmark_run(
                run_name=run_name,
                backbone=backbone,
                benchmark_source=benchmark_source,
                context=benchmark_context,
                output_dir=output_path.parent if output_path else None,
                study_key_hash=study_key_hash,
                trial_key_hash=trial_key_hash,
                hpo_trial_run_id=hpo_trial_run_id,
                hpo_refit_run_id=hpo_refit_run_id,
                hpo_sweep_run_id=hpo_sweep_run_id,
            ):
                tracker.log_benchmark_results(
                    batch_sizes=batch_sizes,
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                    max_length=max_length,
                    device=device,
                    benchmark_json_path=output_path,
                    benchmark_data=benchmark_data,
                )
        except Exception as e:
            logger.warning(f"Could not log benchmark results to MLflow: {e}")
            # Don't fail benchmarking if MLflow logging fails

    return True
