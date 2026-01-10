from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json


def save_json(path: str | Path, data: Any) -> None:
    """Persist arbitrary data as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(
    path: str | Path,
    default: Optional[Any] = None,
) -> Any:
    """Load JSON from disk, returning default if the file is missing."""
    path = Path(path)
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


