"""Public configuration loading and schema checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported configuration schema")
    required = {"datasets", "seeds", "models", "graph_training"}
    if not required <= set(config):
        raise ValueError(f"configuration is missing: {sorted(required - set(config))}")
    return config
