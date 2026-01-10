"""Performance measurement utilities."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def measure_time(description: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager to measure execution time.
    
    Args:
        description: Description of the operation being measured.
    
    Yields:
        None
    
    Example:
        >>> with measure_time("Loading model"):
        ...     load_model()
        Loading model completed in 1.234s
    """
    start_time = time.time()
    try:
        yield
    finally:
            elapsed_time = time.time() - start_time
            print(f"{description} completed in {elapsed_time:.3f}s")


def format_time(seconds: float, precision: int = 3) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds.
        precision: Number of decimal places.
    
    Returns:
        Formatted time string.
    
    Example:
        >>> format_time(1.234)
        '1.234s'
    """
    return f"{seconds:.{precision}f}s"


def check_performance_threshold(
    elapsed_time: float,
    threshold: float,
    operation_name: str = "Operation"
) -> bool:
    """
    Check if an operation completed within a performance threshold.
    
    Args:
        elapsed_time: Time taken in seconds.
        threshold: Maximum acceptable time in seconds.
        operation_name: Name of the operation being checked.
    
    Returns:
        True if within threshold, False otherwise.
    """
    if elapsed_time > threshold:
        print(f"⚠ WARNING: {operation_name} took {elapsed_time:.2f} seconds (expected < {threshold:.2f})")
        return False
    else:
        print(f"✓ Performance OK: {elapsed_time:.3f} seconds < {threshold:.2f} seconds")
        return True

