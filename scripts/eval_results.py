"""Persist evaluation totals atomically between episodes."""
import json
from pathlib import Path


def write_result(path, counts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(counts, indent=2) + "\n")
    temporary.replace(path)
