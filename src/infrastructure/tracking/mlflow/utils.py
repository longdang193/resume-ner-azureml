from __future__ import annotations

"""
@meta
name: tracking_mlflow_utils
type: utility
domain: tracking
responsibility:
  - Provide retry logic with exponential backoff for MLflow operations
  - Handle retryable errors gracefully
inputs:
  - Functions to retry
outputs:
  - Function results or exceptions
tags:
  - utility
  - tracking
  - mlflow
  - retry
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow utility functions for retry logic."""
import random
import time
from typing import Any, Callable

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

def retry_with_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    operation_name: str = "operation"
) -> Any:
    """
    Retry a function with exponential backoff and jitter.

    Args:
        func: Function to retry (callable that takes no arguments).
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay in seconds.
        operation_name: Name of operation for logging.

    Returns:
        Result of func() if successful.

    Raises:
        Exception: Original exception if all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Detect retryable errors
            is_retryable = any(
                code in error_str
                for code in ['429', '503', '504', 'timeout', 'connection', 'temporary', 'rate limit']
            )

            if not is_retryable or attempt == max_retries - 1:
                # Not retryable or last attempt - raise original exception
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            total_delay = delay + jitter

            logger.debug(
                f"Retry {attempt + 1}/{max_retries} for {operation_name} "
                f"after {total_delay:.2f}s (error: {str(e)[:100]})"
            )
            time.sleep(total_delay)

    # Should never reach here, but just in case
    raise RuntimeError(f"Retry logic exhausted for {operation_name}")

