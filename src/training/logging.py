"""Metric logging utilities."""

import json
from pathlib import Path
from typing import Dict, Optional

from infrastructure.platform.adapters.logging_adapter import LoggingAdapter


def log_metrics(
    output_dir: Path,
    metrics: Dict[str, float],
    logging_adapter: Optional[LoggingAdapter] = None,
) -> None:
    """
    Write metrics to file and log using platform adapter.

    Args:
        output_dir: Directory to write metrics file.
        metrics: Dictionary of metric names to values.
        logging_adapter: Platform-specific logging adapter. If None, uses default.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    # Use provided adapter or create default one
    if logging_adapter is None:
        from infrastructure.platform.adapters import get_platform_adapter
        platform_adapter = get_platform_adapter()
        logging_adapter = platform_adapter.get_logging_adapter()

    logging_adapter.log_metrics(metrics)
