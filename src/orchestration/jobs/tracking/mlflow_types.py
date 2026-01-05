"""Structured types for MLflow run information and lookup reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RunHandle:
    """Structured handle for MLflow run with all identifying information."""
    
    run_id: str
    run_key: str
    run_key_hash: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    artifact_uri: Optional[str] = None
    # Grouping tags for cross-platform aggregation (optional, for HPO parent runs)
    study_key_hash: Optional[str] = None
    study_family_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.tracking_uri:
            raise ValueError("tracking_uri is required")


@dataclass
class RunLookupReport:
    """Report of MLflow run lookup attempt with strategy details."""
    
    found: bool
    run_id: Optional[str] = None
    strategy_used: Optional[str] = None
    strategies_attempted: Optional[List[str]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.strategies_attempted is None:
            self.strategies_attempted = []
