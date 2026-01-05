"""Human-readable run name generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from orchestration.naming_centralized import NamingContext
from orchestration.jobs.tracking.config.loader import get_naming_config, get_auto_increment_config
from orchestration.jobs.tracking.naming.run_keys import build_mlflow_run_key, build_mlflow_run_key_hash, build_counter_key


def _strip_env_prefix(trial_id: str, environment: Optional[str]) -> str:
    """
    Strip environment prefix from trial_id if present.

    Args:
        trial_id: Trial ID string (may have env prefix like "local_hpo_...").
        environment: Environment name to strip.

    Returns:
        Trial ID without environment prefix.
    """
    if not environment or not trial_id:
        return trial_id

    prefix = f"{environment}_"
    if trial_id.startswith(prefix):
        return trial_id[len(prefix):]
    return trial_id


def build_mlflow_run_name(
    context: NamingContext,
    config_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Build human-readable run name from context (may be overridden by Azure ML).

    Uses systematic naming with optional auto-increment versioning.

    Args:
        context: NamingContext with process type, model, etc.
        config_dir: Configuration directory (for loading naming config).
        root_dir: Project root directory (for counter storage).
        output_dir: Output directory (for inferring root_dir if not provided).

    Returns:
        Human-readable run name string.
    """
    # Infer root_dir from output_dir if not provided
    if root_dir is None and output_dir is not None:
        root_dir = output_dir.parent.parent if output_dir else None

    # Fallback to current directory if still None
    if root_dir is None:
        root_dir = Path.cwd()

    naming_config = get_naming_config(config_dir)
    run_name_config = naming_config.get("run_name", {})

    if context.process_type == "hpo":
        env_prefix = f"{context.environment}_" if context.environment else ""

        if context.trial_id and context.trial_id.startswith("hpo_"):
            base_without_env = _strip_env_prefix(
                context.trial_id, context.environment
            )

            auto_inc_config = get_auto_increment_config(config_dir, "hpo")
            if (
                auto_inc_config.get("enabled")
                and auto_inc_config.get("processes", {}).get("hpo")
            ):
                try:
                    run_key = build_mlflow_run_key(context)
                    run_key_hash = build_mlflow_run_key_hash(run_key)

                    counter_key = build_counter_key(
                        naming_config.get("project_name", "resume-ner"),
                        "hpo",
                        run_key_hash,
                        context.environment or "",
                    )

                    from orchestration.jobs.tracking.mlflow_index import (
                        reserve_run_name_version,
                    )

                    temp_run_id = f"pending_{datetime.now().isoformat()}"
                    version = reserve_run_name_version(
                        counter_key,
                        temp_run_id,
                        root_dir,
                        config_dir,
                    )
                    return f"{env_prefix}{base_without_env}_{version}"
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Could not reserve version for HPO run name: {e}, using fallback"
                    )
                    return f"{env_prefix}{context.trial_id}"
            else:
                return f"{env_prefix}{context.trial_id}"
        else:
            trial_short = (
                context.trial_id[:20]
                if context.trial_id and len(context.trial_id) > 20
                else (context.trial_id or "unknown")
            )
            return f"{env_prefix}hpo_{context.model}_{trial_short}"

    elif context.process_type == "hpo_refit":
        # Refit run name: hpo_refit_{model}_{trial_id}
        env_prefix = f"{context.environment}_" if context.environment else ""
        trial_short = (
            context.trial_id[:30]
            if context.trial_id and len(context.trial_id) > 30
            else (context.trial_id or "unknown")
        )
        return f"{env_prefix}hpo_refit_{context.model}_{trial_short}"

    elif context.process_type == "benchmarking":
        base_name = f"benchmark_{context.model}"

        auto_inc_config = get_auto_increment_config(
            config_dir, "benchmarking"
        )
        if (
            auto_inc_config.get("enabled")
            and auto_inc_config.get("processes", {}).get("benchmarking")
        ):
            try:
                run_key = build_mlflow_run_key(context)
                run_key_hash = build_mlflow_run_key_hash(run_key)

                counter_key = build_counter_key(
                    naming_config.get("project_name", "resume-ner"),
                    "benchmarking",
                    run_key_hash,
                    context.environment or "",
                )

                from orchestration.jobs.tracking.mlflow_index import (
                    reserve_run_name_version,
                )

                temp_run_id = f"pending_{datetime.now().isoformat()}"
                version = reserve_run_name_version(
                    counter_key,
                    temp_run_id,
                    root_dir,
                    config_dir,
                )

                if context.trial_id:
                    return f"{base_name}_{context.trial_id}_{version}"
                else:
                    return f"{base_name}_{version}"
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Could not reserve version for benchmarking run name: {e}, using fallback"
                )
                trial_short = (
                    context.trial_id[:20]
                    if context.trial_id and len(context.trial_id) > 20
                    else (context.trial_id or "unknown")
                )
                return f"{base_name}_{trial_short}"
        else:
            trial_short = (
                context.trial_id[:20]
                if context.trial_id and len(context.trial_id) > 20
                else (context.trial_id or "unknown")
            )
            return f"{base_name}_{trial_short}"

    elif context.process_type == "final_training":
        shorten_fingerprints = run_name_config.get(
            "shorten_fingerprints", True)
        max_length = run_name_config.get("max_length", 100)

        if shorten_fingerprints:
            spec_short = (
                context.spec_fp[:8]
                if context.spec_fp
                else "unknown"
            )
            exec_short = (
                context.exec_fp[:8]
                if context.exec_fp
                else "unknown"
            )
        else:
            spec_short = context.spec_fp or "unknown"
            exec_short = context.exec_fp or "unknown"

        name = f"final_training_{context.model}_spec_{spec_short}_exec_{exec_short}_v{context.variant}"
        if len(name) > max_length:
            # Truncate model name if needed
            model_max_len = max_length - len(name) + len(context.model)
            model_short = context.model[:model_max_len] if model_max_len > 0 else context.model[:10]
            name = f"final_training_{model_short}_spec_{spec_short}_exec_{exec_short}_v{context.variant}"
        return name

    elif context.process_type == "conversion":
        shorten_fingerprints = run_name_config.get(
            "shorten_fingerprints", True)
        if shorten_fingerprints:
            conv_short = (
                context.conv_fp[:8]
                if context.conv_fp
                else "unknown"
            )
        else:
            conv_short = context.conv_fp or "unknown"
        return f"conversion_{context.model}_{context.parent_training_id}_conv_{conv_short}"

    else:
        return f"{context.process_type}_{context.model}_unknown"

