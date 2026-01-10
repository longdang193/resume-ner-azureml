"""MLflow-based best model selection from benchmark and training runs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from mlflow.tracking import MlflowClient

from common.shared.logging_utils import get_logger
from infrastructure.naming.mlflow.tags_registry import TagsRegistry

logger = get_logger(__name__)


def find_best_model_from_mlflow(
    benchmark_experiment: Dict[str, str],
    hpo_experiments: Dict[str, Dict[str, str]],
    tags_config: Union[TagsRegistry, Dict[str, Any]],
    selection_config: Dict[str, Any],
    use_python_filtering: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Find best model by joining benchmark runs with training (refit) runs.

    Strategy:
    1. Query benchmark runs with required metrics
    2. Preload ALL refit runs from HPO experiments into in-memory lookup
    3. Join benchmark runs with refit runs using (study_key_hash, trial_key_hash)
    4. Compute normalized composite scores (F1 + latency)
    5. Select best candidate

    Args:
        benchmark_experiment: Dict with 'name' and 'id' of benchmark experiment
        hpo_experiments: Dict mapping backbone -> experiment info (name, id)
        tags_config: TagsRegistry or Dict with tags configuration (for backward compatibility)
        selection_config: Selection configuration
        use_python_filtering: If True, fetch all runs and filter in Python (recommended for AzureML)

    Returns:
        Dict with best run info or None if no matches found
    """
    client = MlflowClient()

    # Validate inputs
    if benchmark_experiment is None:
        error_msg = "benchmark_experiment is None. Make sure benchmark runs have been executed and logged to MLflow."
        logger.error(error_msg)
        return None

    if not hpo_experiments:
        error_msg = "No HPO experiments found. Make sure HPO runs have been executed and logged to MLflow."
        logger.error(error_msg)
        return None

    # Tag keys from config (support both TagsRegistry and dict for backward compatibility)
    # Check if it's a TagsRegistry by checking for the key() method (more robust than isinstance)
    # This handles cases where isinstance might fail due to import/class definition issues
    if hasattr(tags_config, 'key') and callable(getattr(tags_config, 'key', None)):
        # It's a TagsRegistry object (or compatible object with key() method)
        study_key_tag = tags_config.key("grouping", "study_key_hash")
        trial_key_tag = tags_config.key("grouping", "trial_key_hash")
        stage_tag = tags_config.key("process", "stage")
        backbone_tag = tags_config.key("process", "backbone")
    elif isinstance(tags_config, dict):
        # Backward compatibility: support dict access
        study_key_tag = tags_config["grouping"]["study_key_hash"]
        trial_key_tag = tags_config["grouping"]["trial_key_hash"]
        stage_tag = tags_config["process"]["stage"]
        backbone_tag = tags_config["process"]["backbone"]
    else:
        # Fallback: try isinstance check
        if isinstance(tags_config, TagsRegistry):
            study_key_tag = tags_config.key("grouping", "study_key_hash")
            trial_key_tag = tags_config.key("grouping", "trial_key_hash")
            stage_tag = tags_config.key("process", "stage")
            backbone_tag = tags_config.key("process", "backbone")
        else:
            raise TypeError(
                f"tags_config must be TagsRegistry or dict, got {type(tags_config)}. "
                f"Object: {tags_config}"
            )

    # Selection config
    objective_metric = selection_config["objective"]["metric"]
    f1_weight = float(selection_config["scoring"]["f1_weight"])
    latency_weight = float(selection_config["scoring"]["latency_weight"])
    normalize_weights = bool(
        selection_config["scoring"].get("normalize_weights", True))
    required_benchmark_metrics = selection_config["benchmark"]["required_metrics"]

    # Normalize weights if needed
    if normalize_weights:
        total_weight = f1_weight + latency_weight
        if total_weight > 0:
            f1_weight = f1_weight / total_weight
            latency_weight = latency_weight / total_weight

    print(f"🔍 Finding best model from MLflow...")
    print(f"   Benchmark experiment: {benchmark_experiment['name']}")
    print(f"   HPO experiments: {len(hpo_experiments)}")
    print(f"   Objective metric: {objective_metric}")
    print(
        f"   Composite weights: F1={f1_weight:.2f}, Latency={latency_weight:.2f}")

    logger.info(f"Finding best model from MLflow")
    logger.info(f"  Benchmark experiment: {benchmark_experiment['name']}")
    logger.info(f"  HPO experiments: {len(hpo_experiments)}")
    logger.info(f"  Objective metric: {objective_metric}")
    logger.info(
        f"  Composite weights: F1={f1_weight:.2f}, Latency={latency_weight:.2f}")

    # Step 1: Query benchmark runs
    print(f"\n📊 Querying benchmark runs...")
    logger.info("Querying benchmark runs...")

    all_benchmark_runs = client.search_runs(
        experiment_ids=[benchmark_experiment["id"]],
        filter_string="",
        max_results=2000,
    )

    # Filter for finished runs in Python (more reliable than MLflow filter on Azure ML)
    benchmark_runs = [
        run for run in all_benchmark_runs if run.info.status == "FINISHED"]
    logger.info(f"Found {len(benchmark_runs)} finished benchmark runs")

    # Filter benchmark runs with required metrics and grouping tags
    valid_benchmark_runs = []
    for run in benchmark_runs:
        has_required_metrics = all(
            metric in run.data.metrics for metric in required_benchmark_metrics)
        has_grouping_tags = (
            study_key_tag in run.data.tags and trial_key_tag in run.data.tags)

        if has_required_metrics and has_grouping_tags:
            valid_benchmark_runs.append(run)

    print(
        f"   Found {len(valid_benchmark_runs)} benchmark runs with required metrics and grouping tags")
    logger.info(
        f"Found {len(valid_benchmark_runs)} benchmark runs with required metrics and grouping tags")

    if not valid_benchmark_runs:
        logger.warning("No valid benchmark runs found")
        return None

    # Step 2: Preload ALL trial runs (for metrics) and refit runs (for artifacts) from HPO experiments
    print(f"\n🔗 Preloading trial runs (metrics) and refit runs (artifacts) from HPO experiments...")
    logger.info("Preloading trial and refit runs from HPO experiments...")
    trial_lookup: Dict[Tuple[str, str], Any] = {}
    refit_lookup: Dict[Tuple[str, str], Any] = {}

    for backbone, exp_info in hpo_experiments.items():
        all_hpo_runs = client.search_runs(
            experiment_ids=[exp_info["id"]],
            filter_string="",
            max_results=5000,
        )

        # Filter for finished runs in Python
        finished_runs = [
            r for r in all_hpo_runs if r.info.status == "FINISHED"]

        # Filter for trial runs (stage = "hpo") - these have macro-f1 metric
        trial_runs = [
            r for r in finished_runs if r.data.tags.get(stage_tag) == "hpo"]

        # Filter for refit runs (stage = "hpo_refit") - these have checkpoints
        refit_runs = [r for r in finished_runs if r.data.tags.get(
            stage_tag) == "hpo_refit"]

        print(
            f"   {exp_info['name']}: {len(finished_runs)} finished runs, {len(trial_runs)} trial runs, {len(refit_runs)} refit runs")
        logger.debug(
            f"Found {len(trial_runs)} trial runs and {len(refit_runs)} refit runs in {exp_info['name']}")

        # Build trial lookup: (study_key_hash, trial_key_hash) -> trial_run (for metrics)
        for trial_run in trial_runs:
            study_hash = trial_run.data.tags.get(study_key_tag)
            trial_hash = trial_run.data.tags.get(trial_key_tag)

            if not study_hash or not trial_hash:
                continue

            key = (study_hash, trial_hash)
            existing = trial_lookup.get(key)
            if existing is None or trial_run.info.start_time > existing.info.start_time:
                trial_lookup[key] = trial_run

        # Build refit lookup: (study_key_hash, trial_key_hash) -> refit_run (for artifacts)
        for refit_run in refit_runs:
            study_hash = refit_run.data.tags.get(study_key_tag)
            trial_hash = refit_run.data.tags.get(trial_key_tag)

            if not study_hash or not trial_hash:
                continue

            key = (study_hash, trial_hash)
            existing = refit_lookup.get(key)
            if existing is None or refit_run.info.start_time > existing.info.start_time:
                refit_lookup[key] = refit_run

    print(
        f"   Built trial lookup with {len(trial_lookup)} unique (study_hash, trial_hash) pairs")
    print(
        f"   Built refit lookup with {len(refit_lookup)} unique (study_hash, trial_hash) pairs")
    logger.info(
        f"Built trial lookup with {len(trial_lookup)} unique (study_hash, trial_hash) pairs")
    logger.info(
        f"Built refit lookup with {len(refit_lookup)} unique (study_hash, trial_hash) pairs")

    if not trial_lookup:
        logger.warning("No trial runs found in HPO experiments")
        return None

    # Step 3: Join benchmark runs with trial runs (for metrics) and refit runs (for artifacts)
    print(f"\n🔗 Joining benchmark runs with trial runs (metrics) and refit runs (artifacts)...")
    logger.info("Joining benchmark runs with trial runs and refit runs...")
    candidates = []

    for benchmark_run in valid_benchmark_runs:
        study_hash = benchmark_run.data.tags[study_key_tag]
        trial_hash = benchmark_run.data.tags[trial_key_tag]

        # Get latency from benchmark run
        latency_ms = benchmark_run.data.metrics.get("latency_batch_1_ms")
        if latency_ms is None:
            continue

        # Look up matching trial run (for metrics - has macro-f1)
        key = (study_hash, trial_hash)
        trial_run = trial_lookup.get(key)

        if trial_run is None:
            continue

        # Get F1 score from trial run (trial runs have macro-f1, refit runs don't)
        f1_score = trial_run.data.metrics.get(objective_metric)
        if f1_score is None:
            continue

        # Look up matching refit run (for artifacts - has checkpoint)
        refit_run = refit_lookup.get(key)
        artifact_run = refit_run if refit_run is not None else trial_run

        # Get backbone from trial run (prefer params, fallback to tags)
        backbone = (
            trial_run.data.params.get("backbone") or
            trial_run.data.tags.get(backbone_tag) or
            trial_run.data.tags.get("code.model", "unknown")
        )

        candidates.append({
            "benchmark_run": benchmark_run,
            "trial_run": trial_run,
            "artifact_run": artifact_run,
            "refit_run": refit_run,
            "f1_score": float(f1_score),
            "latency_ms": float(latency_ms),
            "backbone": backbone,
            "study_key_hash": study_hash,
            "trial_key_hash": trial_hash,
        })

    print(
        f"   Found {len(candidates)} candidate(s) with both benchmark and training metrics")
    logger.info(
        f"Found {len(candidates)} candidate(s) with both benchmark and training metrics")

    if not candidates:
        logger.warning(
            "No candidates found with both benchmark and training metrics")
        return None

    # Step 4: Compute normalized composite scores
    logger.info("Computing composite scores...")

    f1_scores = [c["f1_score"] for c in candidates]
    latency_scores = [c["latency_ms"] for c in candidates]

    f1_min, f1_max = min(f1_scores), max(f1_scores)
    latency_min, latency_max = min(latency_scores), max(latency_scores)

    f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
    latency_range = latency_max - latency_min if latency_max > latency_min else 1.0

    for candidate in candidates:
        # Normalize F1 (higher is better)
        norm_f1 = (candidate["f1_score"] - f1_min) / \
            f1_range if f1_range > 0 else 0.5

        # Normalize latency (lower is better, so invert)
        norm_latency = 1.0 - \
            ((candidate["latency_ms"] - latency_min) /
             latency_range) if latency_range > 0 else 0.5

        # Composite score
        composite_score = (f1_weight * norm_f1) + \
            (latency_weight * norm_latency)
        candidate["composite_score"] = composite_score
        candidate["norm_f1"] = norm_f1
        candidate["norm_latency"] = norm_latency

    # Step 5: Select best candidate
    best_candidate = max(candidates, key=lambda c: c["composite_score"])

    artifact_run = best_candidate["artifact_run"]
    trial_run = best_candidate["trial_run"]

    logger.info("Best model selected:")
    logger.info(f"  Artifact Run ID: {artifact_run.info.run_id}")
    logger.info(f"  Trial Run ID: {trial_run.info.run_id}")
    logger.info(f"  Backbone: {best_candidate['backbone']}")
    logger.info(f"  F1 Score: {best_candidate['f1_score']:.4f}")
    logger.info(f"  Latency: {best_candidate['latency_ms']:.2f} ms")
    logger.info(f"  Composite Score: {best_candidate['composite_score']:.4f}")

    # Find experiment name for best candidate
    best_experiment_name = None
    for backbone, exp_info in hpo_experiments.items():
        if backbone == best_candidate["backbone"]:
            best_experiment_name = exp_info["name"]
            break

    print(f"\n✅ Best model selected:")
    print(f"   Run ID: {artifact_run.info.run_id}")
    print(f"   Experiment: {best_experiment_name or 'unknown'}")
    print(f"   Backbone: {best_candidate['backbone']}")
    print(f"   F1 Score: {best_candidate['f1_score']:.4f}")
    print(f"   Latency: {best_candidate['latency_ms']:.2f} ms")
    print(f"   Composite Score: {best_candidate['composite_score']:.4f}")

    # Return best run info (use artifact_run for artifacts, trial_run for metrics)
    return {
        "run_id": artifact_run.info.run_id,
        "trial_run_id": trial_run.info.run_id,
        "experiment_name": best_experiment_name or "unknown",
        "experiment_id": artifact_run.info.experiment_id,
        "backbone": best_candidate["backbone"],
        "study_key_hash": best_candidate["study_key_hash"],
        "trial_key_hash": best_candidate["trial_key_hash"],
        "f1_score": best_candidate["f1_score"],
        "latency_ms": best_candidate["latency_ms"],
        "composite_score": best_candidate["composite_score"],
        "tags": artifact_run.data.tags,
        "params": artifact_run.data.params,
        "metrics": trial_run.data.metrics,
        "has_refit_run": best_candidate["refit_run"] is not None,
    }
