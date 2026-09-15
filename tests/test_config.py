from __future__ import annotations

from pathlib import Path

from molprop_fusion.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_configuration() -> None:
    config = load_config(ROOT / "configs/public_benchmarks.yaml")
    assert tuple(config["seeds"]) == (13, 29, 47, 71, 97, 131, 173, 211, 257, 307)
    assert set(config["datasets"]) == {"esol", "freesolv", "lipophilicity"}
    assert config["models"]["fusion"]["alpha"] == 10.0
    assert config["datasets"]["lipophilicity"]["max_atoms"] == 115


def test_aqsoldb_configuration() -> None:
    config = load_config(ROOT / "configs/aqsoldb_transfer.yaml")
    assert tuple(config["seeds"]) == (1, 2, 3, 4, 5)
    assert config["datasets"]["aqsoldb"]["partition_column_template"] == "split_seed_{seed}"
    assert config["outputs"]["persist_row_predictions"] is False


def test_public_configs_contain_no_private_machine_paths() -> None:
    text = "\n".join(path.read_text() for path in (ROOT / "configs").glob("*.yaml"))
    forbidden_values = (
        "/" + "Users/",
        "/" + "home/",
        "hkqai" + "_h100",
        "/" + "mnt/" + "DATA",
    )
    for forbidden in forbidden_values:
        assert forbidden not in text
