"""Runnable synthetic CSV tutorial. Tiny fits demonstrate software, not accuracy."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from molprop_fusion import load_model

SMILES = [
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
    "CC(=O)N",
    "CCS",
    "CC(C)C",
    "CC(C)N",
    "COC",
    "CCF",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory for tutorial artifacts"
    )
    parser.add_argument("--mode", choices=["tabular", "full"], default="tabular")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Distributable, synthetic labels: no real property or accuracy claim.
    train = args.output / "train.csv"
    with train.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles", "synthetic_property"])
        for i, smiles in enumerate(SMILES):
            writer.writerow([f"molecule-{i}", smiles, i / 3 + np.sin(i)])
    query = args.output / "candidates.csv"
    query.write_text("id,smiles\none,CCO\ntwo,CCCCC\nduplicate,CCO\ninvalid,not-a-smiles\n")
    bundle = args.output / "model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "molprop_fusion.cli",
            "train",
            "--data",
            str(train),
            "--target-column",
            "synthetic_property",
            "--output",
            str(bundle),
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--max-atoms",
            "16",
            "--validation-only",
        ],
        check=True,
    )
    prediction_csv = args.output / "predictions.csv"
    # A new process loads the model and predicts an unlabeled CSV.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "molprop_fusion.cli",
            "predict",
            "--model",
            str(bundle),
            "--data",
            str(query),
            "--output",
            str(prediction_csv),
            "--invalid-policy",
            "report",
        ],
        check=True,
    )
    restored = load_model(bundle)
    expected = restored.predict(["CCO", "CCCCC", "CCO"])
    with prediction_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_allclose([float(row["prediction"]) for row in rows[:3]], expected, atol=1e-6)
    assert rows[3]["prediction"] == "" and rows[3]["prediction_status"] == "error"
    assert expected[0] == expected[2]
    print(f"Tutorial completed ({args.mode}): {prediction_csv}")


if __name__ == "__main__":
    main()
