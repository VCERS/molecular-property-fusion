"""Pinned Chemprop adapter sharing the package's version and ephemeral CSV policy."""

import csv
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from molprop_fusion.chemprop_baseline import CHEMPROP_VERSION


def partitioned_chemprop(smiles, targets, run, timer, config):
    if importlib.metadata.version("chemprop") != CHEMPROP_VERSION:
        raise RuntimeError(f"Chemprop {CHEMPROP_VERSION} is required")
    epochs = 1 if config.get("validation_only") else 50
    with tempfile.TemporaryDirectory(prefix="molprop-expanded-chemprop-") as directory:
        work = Path(directory)
        data = work / "input.csv"
        with data.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles", "target", "split"])
            for role, name in [("train", "train"), ("validation", "val")]:
                writer.writerows(
                    (s, float(y), name) for s, y in zip(smiles[role], targets[role], strict=True)
                )
            writer.writerows((s, "", "test") for s in smiles["test"])
        command = [
            sys.executable,
            "-c",
            "from chemprop.cli.main import main; main()",
            "train",
            "--data-path",
            str(data),
            "--output-dir",
            str(work / "model"),
            "--smiles-columns",
            "smiles",
            "--target-columns",
            "target",
            "--splits-column",
            "split",
            "--task-type",
            "regression",
            "--metrics",
            "rmse",
            "--tracking-metric",
            "rmse",
            "--epochs",
            str(epochs),
            "--patience",
            str(epochs),
            "--batch-size",
            "64",
            "--pytorch-seed",
            str(run),
            "--data-seed",
            str(run),
            "--num-replicates",
            "1",
            "--accelerator",
            "cpu" if config.get("device", "cuda") == "cpu" else "gpu",
            "--devices",
            "1",
            "--num-workers",
            "0",
        ]
        with timer.stage(
            "chemprop_workflow",
            method="Chemprop",
            fit_rows=len(targets["train"]),
            evaluation_rows=len(smiles["test"]),
        ):
            # Set CPU thread limits for the child, without changing the parent environment.
            env = {
                **os.environ,
                "PYTHONHASHSEED": str(run),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            result = subprocess.run(
                command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
            if result.returncode:
                raise RuntimeError("Chemprop failed; row-bearing logs suppressed")
            files = list((work / "model").rglob("test_predictions.csv"))
            if len(files) != 1:
                raise RuntimeError("Chemprop prediction artifact differs")
            with files[0].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if [r["smiles"] for r in rows] != smiles["test"]:
                raise RuntimeError("Chemprop matched row order differs")
            columns = set(rows[0]) - {"smiles"}
            if len(columns) != 1:
                raise RuntimeError("Chemprop prediction columns differ")
            prediction = np.asarray([float(r[next(iter(columns))]) for r in rows], dtype=np.float64)
            if not np.isfinite(prediction).all():
                raise ValueError("nonfinite Chemprop output")
        return prediction
