"""Cheap synthetic checks of the existing reproduction entry points."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from molprop_fusion.fusion import fit_meta

ROOT = Path(__file__).resolve().parents[1]


def test_existing_meta_numerical_path():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    for count in (4, 5):
        oof, query, targets = (
            rng.normal(size=(18, count)),
            rng.normal(size=(3, count)),
            rng.normal(size=18),
        )
        expected = (
            make_pipeline(StandardScaler(), Ridge(alpha=10.0, fit_intercept=True))
            .fit(oof, targets)
            .predict(query)
        )
        actual, details = fit_meta(oof, targets, query)
        np.testing.assert_array_equal(actual, expected)
        assert details["column_count"] == count


def test_existing_commands_with_small_synthetic_inputs(tmp_path):
    for module in ["torch", "rdkit", "xgboost", "catboost", "lightgbm"]:
        pytest.importorskip(module)
    # No real scientific dataset or historical sealed input is read.
    smiles = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCCO",
        "CCN",
        "CCCN",
        "CC(=O)O",
        "CCOC",
        "c1ccccc1",
        "c1ccncc1",
        "CCCl",
        "CCBr",
        "C1CCCCC1",
        "CC(C)O",
        "O=C=O",
        "CC#N",
    ]
    public = yaml.safe_load((ROOT / "configs/public_benchmarks.yaml").read_text())
    public["models"]["graph"].update(hidden_width=16, layers=1)
    public["datasets"]["esol"]["max_atoms"] = 16
    public_config = tmp_path / "public.yaml"
    public_config.write_text(yaml.safe_dump(public))
    data = tmp_path / "synthetic.csv"
    with data.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "smiles",
                public["datasets"]["esol"]["target_column"],
                "SMILES",
                "Solubility",
                "split_seed_1",
            ]
        )
        for i, smi in enumerate(smiles):
            writer.writerow(
                [smi, i / 3, smi, i / 3, "train" if i < 12 else "validation" if i < 15 else "test"]
            )

    def command(*arguments):
        completed = subprocess.run(
            [sys.executable, "-m", "molprop_fusion.cli", *map(str, arguments)],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return completed

    cache = tmp_path / "historical.npz"
    command(
        "prepare", "--dataset", "esol", "--config", public_config, "--data", data, "--output", cache
    )
    benchmark = tmp_path / "benchmark.json"
    command(
        "benchmark",
        "--config",
        public_config,
        "--dataset",
        "esol",
        "--data",
        data,
        "--feature-cache",
        cache,
        "--output",
        benchmark,
        "--seed",
        13,
        "--device",
        "cpu",
        "--smoke",
    )
    case = json.loads(benchmark.read_text())["cases"][0]
    assert case["fit_counts"] == {"tabular_base": 16, "compt": 4, "meta": 2}
    assert case["meta"]["T_plus_G"]["column_count"] == 5
    assert case["row_level_values_persisted"] is False
    command("summarize", "--input", benchmark, "--output", tmp_path / "summary.json")
    transfer = yaml.safe_load((ROOT / "configs/aqsoldb_transfer.yaml").read_text())
    transfer["models"]["graph"].update(hidden_width=16, layers=1)
    transfer["datasets"]["aqsoldb"]["max_atoms"] = 16
    config = tmp_path / "transfer.yaml"
    config.write_text(yaml.safe_dump(transfer))
    result = tmp_path / "transfer.json"
    command(
        "transfer",
        "--config",
        config,
        "--data",
        data,
        "--output",
        result,
        "--seed",
        1,
        "--device",
        "cpu",
        "--smoke",
    )
    assert json.loads(result.read_text())["summary"]["complete_seed_count"] == 1
    for existing in [
        "prepare",
        "benchmark",
        "transfer",
        "chemprop",
        "summarize",
        "check-config",
        "verify-results",
    ]:
        command(existing, "--help")
