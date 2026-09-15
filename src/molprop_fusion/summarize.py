"""Aggregate case-level outputs without exposing row-level values."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from molprop_fusion.io import atomic_write_json


def summarize_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("benchmark output contains no cases")
    methods: dict[str, Any] = {}
    for method in ("T", "G", "T_plus_G"):
        values = [float(row["methods"][method]["rmse"]) for row in cases]
        methods[method] = {
            "count": len(values),
            "mean_rmse": statistics.fmean(values),
            "sample_sd_rmse": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    deltas = [float(row["contrasts"]["T_plus_G_minus_T"]["rmse"]) for row in cases]
    return {
        "schema_version": 1,
        "workflow": payload["workflow"],
        "dataset": payload["dataset"],
        "seeds": payload["seeds"],
        "source_file_sha256": payload["source_file_sha256"],
        "methods": methods,
        "paired": {
            "mean_rmse_delta": statistics.fmean(deltas),
            "sample_sd_rmse_delta": (statistics.stdev(deltas) if len(deltas) > 1 else 0.0),
            "improved_seed_count": sum(value < 0 for value in deltas),
        },
        "row_level_values_persisted": False,
    }


def summarize_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summarize_benchmark(payload)
    atomic_write_json(output_path, summary)
    return summary
