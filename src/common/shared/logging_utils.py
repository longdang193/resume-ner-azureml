"""
@meta
name: shared_logging_utils
type: utility
domain: shared
responsibility:
  - Provide consistent logging utilities across scripts
  - Configure loggers with standardized formatting
inputs:
  - Logger names
outputs:
  - Configured logger instances
tags:
  - utility
  - shared
  - logging
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Shared logging utilities for consistent logging across scripts."""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with standardized formatting.
    
    Args:
        name: Logger name (typically __name__ or script name).
        level: Optional logging level (default: INFO).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        if level is not None:
            logger.setLevel(level)
        elif logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
    
    return logger


def get_script_logger(script_name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger for a script with prefix formatting.
    
    Useful for scripts that need simple, prefixed output (e.g., Azure ML jobs).
    
    Args:
        script_name: Name of the script (e.g., "convert_to_onnx").
        level: Optional logging level (default: INFO).
    
    Returns:
        Configured logger instance with script prefix.
    """
    logger = logging.getLogger(f"script.{script_name}")
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[{script_name}] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        if level is not None:
            logger.setLevel(level)
        elif logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
    
    return logger

