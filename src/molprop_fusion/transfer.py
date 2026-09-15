"""AqSolDB transfer evaluation using user-supplied data and partitions."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from molprop_fusion.benchmark import (
    prepare_features,
    read_feature_cache,
    run_case,
    write_feature_cache,
)
from molprop_fusion.config import load_config
from molprop_fusion.features import canonicalize_smiles
from molprop_fusion.io import atomic_write_json, sha256_file


def load_partitioned_dataset(
    path: Path,
    *,
    smiles_column: str,
    target_column: str,
    partition_column: str,
) -> tuple[list[str], np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    smiles: list[str] = []
    targets: list[float] = []
    partitions: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {smiles_column, target_column, partition_column}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"dataset is missing columns: {sorted(required)}")
        for index, row in enumerate(reader):
            partition = row[partition_column].strip().lower()
            if partition not in partitions:
                raise ValueError(f"unknown partition {partition!r}")
            value = float(row[target_column])
            if not math.isfinite(value):
                raise ValueError("dataset contains a non-finite target")
            smiles.append(canonicalize_smiles(row[smiles_column]))
            targets.append(value)
            partitions[partition].append(index)
    if any(not values for values in partitions.values()):
        raise ValueError("train, validation, and test partitions must all be nonempty")
    return (
        smiles,
        np.asarray(targets, dtype=np.float64),
        tuple(
            np.asarray(partitions[name], dtype=np.int64) for name in ("train", "validation", "test")
        ),
    )


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("no transfer cases supplied")
    methods: dict[str, Any] = {}
    for method in ("T", "G", "T_plus_G"):
        methods[method] = {}
        for metric in ("mae", "rmse"):
            values = [float(row["methods"][method][metric]) for row in cases]
            methods[method][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    contrasts: dict[str, Any] = {"T_plus_G_minus_T": {}}
    for metric in ("mae", "rmse"):
        values = [float(row["contrasts"]["T_plus_G_minus_T"][metric]) for row in cases]
        contrasts["T_plus_G_minus_T"][metric] = {
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "negative_seed_count": sum(value < 0 for value in values),
        }
    return {
        "complete_seed_count": len(cases),
        "methods": methods,
        "contrasts": contrasts,
    }


def run_transfer(
    *,
    config_path: Path,
    data_path: Path,
    output_path: Path,
    feature_cache: Path | None,
    device: str,
    n_jobs: int,
    seeds: tuple[int, ...] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = "aqsoldb"
    spec = config["datasets"][dataset]
    chosen = seeds or tuple(int(seed) for seed in config["seeds"])
    cases: list[dict[str, Any]] = []
    cached_features: dict[str, np.ndarray] | dict[str, Any] | None = None
    source_hash = sha256_file(data_path)
    for seed in chosen:
        partition_column = str(spec["partition_column_template"]).format(seed=seed)
        smiles, target, partitions = load_partitioned_dataset(
            data_path,
            smiles_column=str(spec["smiles_column"]),
            target_column=str(spec["target_column"]),
            partition_column=partition_column,
        )
        if cached_features is None:
            if feature_cache and feature_cache.is_file():
                cached_features = read_feature_cache(feature_cache, len(smiles))
            else:
                cached_features = prepare_features(smiles, dataset=dataset)
                if feature_cache:
                    write_feature_cache(feature_cache, cached_features)
        features = cached_features
        cases.append(
            run_case(
                config=config,
                dataset=dataset,
                seed=seed,
                canonical_smiles=smiles,
                target=target,
                features=features,
                device=device,
                n_jobs=n_jobs,
                smoke=smoke,
                partitions=partitions,
            )
        )
    result = {
        "schema_version": 1,
        "workflow": "aqsoldb_transfer",
        "seeds": list(chosen),
        "source_file_sha256": source_hash,
        "summary": summarize_cases(cases),
        "cases": cases,
        "row_level_values_persisted": False,
    }
    atomic_write_json(output_path, result)
    return result
