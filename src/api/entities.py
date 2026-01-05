"""Entity processing utilities."""

from typing import List, Dict, Any


def merge_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent entities of the same type.

    Args:
        entities: List of entity dictionaries.

    Returns:
        Merged entity list.
    """
    if not entities:
        return []

    merged = []
    current = entities[0].copy()

    for entity in entities[1:]:
        if (
            entity["label"] == current["label"]
            and entity["start"] == current["end"]
        ):
            # Merge: extend current entity
            current["end"] = entity["end"]
            current["text"] = current["text"] + entity["text"]
            if "confidence" in current and "confidence" in entity:
                # Average confidence
                current["confidence"] = (
                    (current["confidence"] + entity["confidence"]) / 2
                )
        else:
            # Save current and start new
            merged.append(current)
            current = entity.copy()

    merged.append(current)
    return merged


def filter_entities(
    entities: List[Dict[str, Any]],
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Filter entities by confidence threshold.

    Args:
        entities: List of entity dictionaries.
        min_confidence: Minimum confidence score (0-1).

    Returns:
        Filtered entity list.
    """
    if min_confidence <= 0.0:
        return entities

    filtered = []
    for entity in entities:
        confidence = entity.get("confidence")
        if confidence is None or confidence >= min_confidence:
            filtered.append(entity)

    return filtered


def validate_entity_spans(
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validate and fix entity spans to ensure they are within text bounds.

    Args:
        text: Original text.
        entities: List of entity dictionaries.

    Returns:
        Validated entity list.
    """
    validated = []
    text_len = len(text)

    for entity in entities:
        start = max(0, min(entity["start"], text_len))
        end = max(start, min(entity["end"], text_len))

        if start < end:
            entity_text = text[start:end]
            validated.append(
                {
                    **entity,
                    "start": start,
                    "end": end,
                    "text": entity_text,
                }
            )

    return validated

