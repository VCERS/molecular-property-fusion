from __future__ import annotations

import hashlib
import json
from pathlib import Path

from molprop_fusion.verification import verify_results

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_aggregates_match_source_hashes() -> None:
    result = verify_results(ROOT / "results")
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_source_manifest_destination_hashes() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        digest = hashlib.sha256((ROOT / row["destination_path"]).read_bytes()).hexdigest()
        assert digest == row["destination_sha256"], row["destination_path"]
        assert len(row["source_commit"]) == 40
        assert len(row["source_sha256"]) == 64
    compt = manifest["third_party"][0]
    digest = hashlib.sha256((ROOT / compt["destination_path"]).read_bytes()).hexdigest()
    assert digest == compt["license_sha256"]
