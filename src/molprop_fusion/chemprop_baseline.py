"""Chemprop v2.3.1 baseline with test labels withheld during training."""

from __future__ import annotations

import csv
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from molprop_fusion.benchmark import (
    load_dataset,
    regression_metrics,
    split_indices,
)
from molprop_fusion.config import load_config
from molprop_fusion.fusion import prediction_sha256
from molprop_fusion.io import atomic_write_json, sha256_file

CHEMPROP_VERSION = "2.3.1"
CHEMPROP_COMMIT = "9a0b47ad19ebf440c1557787e666d442728adaad"


def run_chemprop(
    *,
    config_path: Path,
    dataset: str,
    data_path: Path,
    output_path: Path,
    seed: int,
    accelerator: str,
    devices: int,
    smoke: bool = False,
) -> dict[str, Any]:
    observed = importlib.metadata.version("chemprop")
    if observed != CHEMPROP_VERSION:
        raise RuntimeError(f"Chemprop {CHEMPROP_VERSION} is required; observed {observed}")
    config = load_config(config_path)
    spec = config["datasets"][dataset]
    _, canonical, target = load_dataset(
        data_path,
        smiles_column=str(spec["smiles_column"]),
        target_column=str(spec["target_column"]),
        maximum_rows=30 if smoke else None,
    )
    train, validation, test = split_indices(len(target), seed)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="molprop-fusion-chemprop-") as directory:
        work = Path(directory)
        input_path = work / "data.csv"
        model_root = work / "model"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles", "target", "split"])
            for index in train:
                writer.writerow([canonical[index], target[index], "train"])
            for index in validation:
                writer.writerow([canonical[index], target[index], "val"])
            for index in test:
                writer.writerow([canonical[index], "", "test"])
        epochs = 3 if smoke else int(config["chemprop"]["epochs"])
        command = [
            sys.executable,
            "-c",
            "from chemprop.cli.main import main; main()",
            "train",
            "--data-path",
            str(input_path),
            "--output-dir",
            str(model_root),
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
            str(config["chemprop"]["batch_size"]),
            "--pytorch-seed",
            str(seed),
            "--data-seed",
            str(seed),
            "--num-replicates",
            "1",
            "--accelerator",
            accelerator,
            "--devices",
            str(devices),
            "--num-workers",
            "0",
        ]
        completed = subprocess.run(
            command,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Chemprop failed: {completed.stderr[-4000:]}")
        prediction_files = list(model_root.rglob("test_predictions.csv"))
        if len(prediction_files) != 1:
            raise RuntimeError("Chemprop did not create one test prediction file")
        with prediction_files[0].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(test):
            raise RuntimeError("Chemprop test prediction count differs")
        prediction_column = next((name for name in rows[0] if name != "smiles"), None)
        if prediction_column is None:
            raise RuntimeError("Chemprop prediction column is missing")
        predictions = np.asarray([float(row[prediction_column]) for row in rows], dtype=np.float64)
        checkpoints = list(model_root.rglob("best-epoch=*.ckpt"))
        best_epoch = (
            int(checkpoints[0].name.split("best-epoch=")[1].split("-")[0])
            if len(checkpoints) == 1
            else None
        )
    row_ids = [str(int(index) + 2) for index in test]
    result = {
        "schema_version": 1,
        "workflow": "chemprop_baseline",
        "dataset": dataset,
        "seed": int(seed),
        "smoke": bool(smoke),
        "metrics": regression_metrics(predictions, target[test]),
        "prediction_sha256": prediction_sha256(predictions, row_ids),
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "best_epoch": best_epoch,
        "fit_count": 1,
        "wall_seconds": time.perf_counter() - started,
        "source_file_sha256": sha256_file(data_path),
        "chemprop_version": observed,
        "chemprop_source_commit": CHEMPROP_COMMIT,
        "test_labels_available_to_training_or_selection": False,
        "row_level_values_persisted": False,
    }
    atomic_write_json(output_path, result)
    return result
