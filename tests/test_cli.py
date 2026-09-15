from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_module_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "molprop_fusion.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "benchmark" in completed.stdout


def test_config_and_result_entry_points() -> None:
    for arguments in (
        [
            "check-config",
            str(ROOT / "configs/public_benchmarks.yaml"),
            str(ROOT / "configs/aqsoldb_transfer.yaml"),
        ],
        ["verify-results", "--results-root", str(ROOT / "results")],
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "molprop_fusion.cli", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
