from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
