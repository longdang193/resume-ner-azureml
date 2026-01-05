"""Utilities for converting between internal entity representations and API response models."""

from typing import List, Dict, Any
from .models import Entity


def convert_entities_to_response(entities_dict: List[Dict[str, Any]]) -> List[Entity]:
    """
    Convert entity dictionaries to Entity response models.
    
    Args:
        entities_dict: List of entity dictionaries with keys:
            - text: Entity text
            - label: Entity label/type
            - start: Start character offset
            - end: End character offset
            - confidence: Optional confidence score
    
    Returns:
        List of Entity response models.
    """
    return [
        Entity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            confidence=e.get("confidence"),
        )
        for e in entities_dict
    ]

