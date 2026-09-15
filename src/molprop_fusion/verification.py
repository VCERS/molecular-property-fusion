"""Verification of the sanitized, repository-tracked aggregate results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PUBLIC_SOURCE_RESULT_SHA256 = "59a81c1cfc965759284ab74836b7c7c1a954d5247f1d90d15f9d3b46efee4b90"
AQSOL_SOURCE_RESULT_SHA256 = "b028e0018891e41c59785bef0413f8ad1b16b7537462d5ef558dbf9903ec188f"


def verify_results(results_root: Path) -> dict[str, Any]:
    public = json.loads((results_root / "public_benchmarks.json").read_text(encoding="utf-8"))
    aqsol = json.loads((results_root / "aqsoldb_transfer.json").read_text(encoding="utf-8"))
    checks = {
        "public_source_hash": (public["source_result_sha256"] == PUBLIC_SOURCE_RESULT_SHA256),
        "aqsoldb_source_hash": (aqsol["source_result_sha256"] == AQSOL_SOURCE_RESULT_SHA256),
        "public_case_count": public["case_count"] == 30,
        "public_dataset_roster": set(public["datasets"]) == {"esol", "freesolv", "lipophilicity"},
        "aqsoldb_complete_seed_count": aqsol["complete_seed_count"] == 5,
        "aqsoldb_aggregate_only": aqsol["aggregate_only"] is True,
        "no_row_level_values": (
            public["row_level_values_included"] is False
            and aqsol["row_level_values_included"] is False
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"aggregate verification failed: {failed}")
    return {"passed": True, "checks": checks}
