"""End-to-end public benchmark runner for prediction-level molecular fusion."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from molprop_fusion.accuracy_benchmark import fit_predict
from molprop_fusion.config import load_config
from molprop_fusion.features import (
    canonicalize_smiles,
    chemistry_matrix,
    impute_raw_z,
    z3d_matrix,
)
from molprop_fusion.fusion import fit_meta, grouped_folds, prediction_sha256
from molprop_fusion.io import atomic_write_json, sha256_file


def load_dataset(
    path: Path,
    *,
    smiles_column: str,
    target_column: str,
    maximum_rows: int | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    raw: list[str] = []
    targets: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {smiles_column, target_column} <= set(
            reader.fieldnames
        ):
            raise ValueError("dataset does not contain the configured SMILES and target columns")
        for row in reader:
            if maximum_rows is not None and len(raw) >= maximum_rows:
                break
            value = float(row[target_column])
            if not math.isfinite(value):
                raise ValueError("dataset contains a non-finite target")
            raw.append(row[smiles_column])
            targets.append(value)
    if len(raw) < 10:
        raise ValueError("at least ten rows are required")
    canonical = [canonicalize_smiles(value) for value in raw]
    return raw, canonical, np.asarray(targets, dtype=np.float64)


def split_indices(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recreate Lane K: 80/20 by seed, then split the 20% at random state 42."""
    from sklearn.model_selection import train_test_split

    all_rows = np.arange(row_count, dtype=np.int64)
    train, held = train_test_split(all_rows, test_size=0.2, random_state=int(seed))
    validation, test = train_test_split(held, test_size=0.5, random_state=42)
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(validation, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def prepare_features(canonical_smiles: Sequence[str], *, dataset: str) -> dict[str, Any]:
    chemistry, descriptor_names = chemistry_matrix(canonical_smiles)
    z3d, z3d_mask, statuses = z3d_matrix(canonical_smiles, dataset=dataset)
    return {
        "chemistry": chemistry,
        "z3d": z3d,
        "z3d_mask": z3d_mask,
        "descriptor_names": descriptor_names,
        "z3d_statuses": statuses,
    }


def write_feature_cache(path: Path, features: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        chemistry=np.asarray(features["chemistry"], dtype=np.float32),
        z3d=np.asarray(features["z3d"], dtype=np.float64),
        z3d_mask=np.asarray(features["z3d_mask"], dtype=bool),
    )


def read_feature_cache(path: Path, row_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        output = {
            "chemistry": np.asarray(payload["chemistry"], dtype=np.float32),
            "z3d": np.asarray(payload["z3d"], dtype=np.float64),
            "z3d_mask": np.asarray(payload["z3d_mask"], dtype=bool),
        }
    if output["chemistry"].shape != (row_count, 2265):
        raise ValueError("cached chemistry matrix has an unexpected shape")
    if output["z3d"].shape != (row_count, 317):
        raise ValueError("cached Z3D matrix has an unexpected shape")
    if not np.array_equal(output["z3d_mask"], ~np.isfinite(output["z3d"])):
        raise ValueError("cached Z3D mask disagrees with values")
    return output


def _model(
    config: Mapping[str, Any],
    family: str,
    dataset: str,
    seed: int,
    n_jobs: int,
    *,
    smoke: bool,
) -> Any:
    if family == "xgboost":
        from xgboost import XGBRegressor

        parameters = {
            **config["models"]["xgboost"]["common"],
            **config["models"]["xgboost"]["by_dataset"][dataset],
        }
        if smoke:
            parameters["n_estimators"] = min(5, int(parameters["n_estimators"]))
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            missing=np.nan,
            random_state=seed,
            n_jobs=n_jobs,
            **parameters,
        )
    if family == "catboost":
        from catboost import CatBoostRegressor

        parameters = dict(config["models"]["catboost"])
        if smoke:
            parameters["iterations"] = min(5, int(parameters["iterations"]))
        return CatBoostRegressor(
            random_seed=seed,
            thread_count=n_jobs,
            allow_writing_files=False,
            **parameters,
        )
    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        parameters = dict(config["models"]["lightgbm"])
        if smoke:
            parameters["n_estimators"] = min(5, int(parameters["n_estimators"]))
        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            **parameters,
        )
    if family == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        parameters = dict(config["models"]["extratrees"])
        if smoke:
            parameters["n_estimators"] = min(5, int(parameters["n_estimators"]))
        return ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **parameters)
    raise ValueError(f"unknown tabular family: {family}")


def _xz(
    chemistry: np.ndarray,
    z3d: np.ndarray,
    mask: np.ndarray,
    fit: np.ndarray,
    evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_z, eval_z, _ = impute_raw_z(z3d[fit], z3d[evaluate])
    return (
        np.concatenate([chemistry[fit], fit_z, mask[fit].astype(np.float32)], axis=1),
        np.concatenate(
            [chemistry[evaluate], eval_z, mask[evaluate].astype(np.float32)],
            axis=1,
        ),
    )


def _tabular_oof(
    *,
    config: Mapping[str, Any],
    dataset: str,
    family: str,
    chemistry: np.ndarray,
    z3d: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    evaluation_chemistry: np.ndarray,
    evaluation_z3d: np.ndarray,
    evaluation_mask: np.ndarray,
    seed: int,
    n_jobs: int,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    oof = np.empty(len(target), dtype=np.float64)
    fits = 0
    for fold, (fit, held) in enumerate(grouped_folds(groups, seed)):
        fit_x, held_x = _xz(chemistry, z3d, mask, fit, held)
        oof[held], _ = fit_predict(
            _model(config, family, dataset, seed + fold, n_jobs, smoke=smoke),
            fit_x,
            target[fit],
            held_x,
        )
        fits += 1
    fit_z, eval_z, _ = impute_raw_z(z3d, evaluation_z3d)
    fit_x = np.concatenate([chemistry, fit_z, mask.astype(np.float32)], axis=1)
    evaluation_x = np.concatenate(
        [evaluation_chemistry, eval_z, evaluation_mask.astype(np.float32)], axis=1
    )
    prediction, _ = fit_predict(
        _model(config, family, dataset, seed, n_jobs, smoke=smoke),
        fit_x,
        target,
        evaluation_x,
    )
    return oof, prediction, fits + 1


def _graph_oof(
    *,
    smiles: list[str],
    target: np.ndarray,
    groups: np.ndarray,
    evaluation_smiles: list[str],
    seed: int,
    dataset_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    device: str,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from molprop_fusion.compt_graph import fit_graph_regressor

    model_options = {
        "atom_width": 115,
        "bond_width": 13,
        "hidden_width": int(config["models"]["graph"]["hidden_width"]),
        "layers": int(config["models"]["graph"]["layers"]),
        "heads": int(config["models"]["graph"]["attention_heads"]),
        "dropout": float(config["models"]["graph"]["dropout"]),
        "max_atoms": int(dataset_spec["max_atoms"]),
    }
    training_options = {
        **config["graph_training"],
        "warmup_factor": float(dataset_spec["warmup_factor"]),
        "device": device,
    }
    if smoke:
        training_options["max_epochs"] = 1
        training_options["patience"] = 1
        training_options["warmup_epochs"] = 1
    oof = np.empty(len(target), dtype=np.float64)
    epochs: list[int] = []
    for fold, (fit, held) in enumerate(grouped_folds(groups, seed)):
        result = fit_graph_regressor(
            train_smiles=[smiles[index] for index in fit],
            train_target=target[fit],
            train_groups=groups[fit],
            evaluation_smiles=[smiles[index] for index in held],
            seed=seed + fold,
            model_options=model_options,
            training_options=training_options,
        )
        oof[held] = result.predictions
        epochs.append(result.best_epoch)
    final = fit_graph_regressor(
        train_smiles=smiles,
        train_target=target,
        train_groups=groups,
        evaluation_smiles=evaluation_smiles,
        seed=seed,
        model_options=model_options,
        training_options=training_options,
    )
    epochs.append(final.best_epoch)
    return oof, final.predictions, epochs


def regression_metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def run_case(
    *,
    config: Mapping[str, Any],
    dataset: str,
    seed: int,
    canonical_smiles: list[str],
    target: np.ndarray,
    features: Mapping[str, np.ndarray],
    device: str,
    n_jobs: int,
    smoke: bool = False,
    partitions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_spec = config["datasets"][dataset]
    train, validation, test = partitions or split_indices(len(target), seed)
    del validation  # local fusion intentionally does not use the outer validation set
    chemistry = np.asarray(features["chemistry"])
    z3d = np.asarray(features["z3d"])
    z3d_mask = np.asarray(features["z3d_mask"])
    train_smiles = [canonical_smiles[index] for index in train]
    test_smiles = [canonical_smiles[index] for index in test]
    train_target, test_target = target[train], target[test]
    groups = np.asarray(train_smiles, dtype=object)
    tree_oof: list[np.ndarray] = []
    tree_test: list[np.ndarray] = []
    tabular_fits = 0
    for family in ("xgboost", "catboost", "lightgbm", "extratrees"):
        oof, prediction, fits = _tabular_oof(
            config=config,
            dataset=dataset,
            family=family,
            chemistry=chemistry[train],
            z3d=z3d[train],
            mask=z3d_mask[train],
            target=train_target,
            groups=groups,
            evaluation_chemistry=chemistry[test],
            evaluation_z3d=z3d[test],
            evaluation_mask=z3d_mask[test],
            seed=seed,
            n_jobs=n_jobs,
            smoke=smoke,
        )
        tree_oof.append(oof)
        tree_test.append(prediction)
        tabular_fits += fits
    tree_oof_matrix = np.column_stack(tree_oof)
    tree_test_matrix = np.column_stack(tree_test)
    t_prediction, t_meta = fit_meta(tree_oof_matrix, train_target, tree_test_matrix)
    graph_oof, graph_prediction, best_epochs = _graph_oof(
        smiles=train_smiles,
        target=train_target,
        groups=groups,
        evaluation_smiles=test_smiles,
        seed=seed,
        dataset_spec=dataset_spec,
        config=config,
        device=device,
        smoke=smoke,
    )
    fused_oof = np.column_stack([tree_oof_matrix, graph_oof])
    fused_test = np.column_stack([tree_test_matrix, graph_prediction])
    fused_prediction, fused_meta = fit_meta(fused_oof, train_target, fused_test)
    methods = {
        "T": regression_metrics(t_prediction, test_target),
        "G": regression_metrics(graph_prediction, test_target),
        "T_plus_G": regression_metrics(fused_prediction, test_target),
    }
    row_ids = [str(int(index) + 2) for index in test]
    return {
        "schema_version": 1,
        "dataset": dataset,
        "seed": int(seed),
        "smoke": bool(smoke),
        "methods": methods,
        "contrasts": {
            "T_plus_G_minus_T": {
                metric: methods["T_plus_G"][metric] - methods["T"][metric]
                for metric in ("mae", "rmse")
            }
        },
        "prediction_sha256": {
            "T": prediction_sha256(t_prediction, row_ids),
            "G": prediction_sha256(graph_prediction, row_ids),
            "T_plus_G": prediction_sha256(fused_prediction, row_ids),
        },
        "meta": {"T": t_meta, "T_plus_G": fused_meta},
        "best_epochs": {"G": best_epochs},
        "fit_counts": {
            "tabular_base": tabular_fits,
            "compt": len(best_epochs),
            "meta": 2,
        },
        "row_counts": {"train": len(train), "test": len(test)},
        "wall_seconds": time.perf_counter() - started,
        "row_level_values_persisted": False,
    }


def run_benchmark(
    *,
    config_path: Path,
    dataset: str,
    data_path: Path,
    output_path: Path,
    feature_cache: Path | None,
    device: str,
    n_jobs: int,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if dataset not in config["datasets"]:
        raise ValueError(f"unknown dataset: {dataset}")
    spec = config["datasets"][dataset]
    _, canonical, target = load_dataset(
        data_path,
        smiles_column=str(spec["smiles_column"]),
        target_column=str(spec["target_column"]),
        maximum_rows=30 if smoke else None,
    )
    if feature_cache and feature_cache.is_file():
        features = read_feature_cache(feature_cache, len(canonical))
    else:
        features = prepare_features(canonical, dataset=dataset)
        if feature_cache:
            write_feature_cache(feature_cache, features)
    chosen = tuple(int(value) for value in (seeds or config["seeds"]))
    cases = [
        run_case(
            config=config,
            dataset=dataset,
            seed=seed,
            canonical_smiles=canonical,
            target=target,
            features=features,
            device=device,
            n_jobs=n_jobs,
            smoke=smoke,
        )
        for seed in chosen
    ]
    result = {
        "schema_version": 1,
        "workflow": "public_benchmark",
        "dataset": dataset,
        "seeds": list(chosen),
        "source_file_sha256": sha256_file(data_path),
        "cases": cases,
        "row_level_values_persisted": False,
    }
    atomic_write_json(output_path, result)
    return result
