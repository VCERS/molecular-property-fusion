"""Grouped out-of-fold and Ridge prediction-fusion helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

TABULAR_COLUMNS = (
    "xgboost_C_Zraw_mask",
    "catboost_C_Zraw_mask",
    "lightgbm_C_Zraw_mask",
    "extratrees_C_Zraw_mask",
)
META_ALPHA = 10.0


def ordered_row_sha256(rows: Sequence[str]) -> str:
    if len(rows) != len(set(rows)):
        raise ValueError("duplicate source row")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def prediction_sha256(values: Sequence[float], rows: Sequence[str]) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim != 1 or len(array) != len(rows) or not np.isfinite(array).all():
        raise ValueError("prediction vector is not a finite row-aligned column")
    return hashlib.sha256(array.tobytes() + "\n".join(rows).encode("utf-8")).hexdigest()


def grouped_folds(
    groups: Sequence[str], seed: int, *, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    index = np.arange(len(groups), dtype=np.int64)
    folds = list(
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)).split(
            index, groups=np.asarray(groups, dtype=object)
        )
    )
    validate_grouped_folds(folds, groups, n_splits=n_splits)
    return folds


def validate_grouped_folds(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    groups: Sequence[str],
    *,
    n_splits: int = 3,
) -> None:
    if len(folds) != n_splits:
        raise ValueError(f"exactly {n_splits} OOF folds are required")
    seen: list[int] = []
    group_array = np.asarray(groups, dtype=object)
    all_rows = set(range(len(group_array)))
    for fit, held in folds:
        fit_set, held_set = set(map(int, fit)), set(map(int, held))
        if not fit_set or not held_set or fit_set & held_set or fit_set | held_set != all_rows:
            raise ValueError("incomplete or overlapping OOF fold")
        if set(group_array[fit]) & set(group_array[held]):
            raise ValueError("group overlap in OOF fold")
        seen.extend(held_set)
    if len(seen) != len(all_rows) or set(seen) != all_rows:
        raise ValueError("held folds are not a complete disjoint partition")


def fit_meta(
    oof: np.ndarray, train_target: np.ndarray, evaluation_columns: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if oof.ndim != 2 or evaluation_columns.ndim != 2:
        raise ValueError("meta inputs must be matrices")
    if oof.shape[1] != evaluation_columns.shape[1] or oof.shape[1] not in (4, 5):
        raise ValueError("meta learner accepts four tree columns and optional graph prediction")
    if len(train_target) != len(oof):
        raise ValueError("meta train length mismatch")
    if not np.isfinite(oof).all() or not np.isfinite(evaluation_columns).all():
        raise ValueError("non-finite meta input")
    model = make_pipeline(StandardScaler(), Ridge(alpha=META_ALPHA, fit_intercept=True))
    model.fit(oof, train_target)
    prediction = np.asarray(model.predict(evaluation_columns), dtype=np.float64)
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    return prediction, {
        "alpha": float(ridge.alpha),
        "fit_intercept": bool(ridge.fit_intercept),
        "coefficients_on_scaled_columns": np.asarray(ridge.coef_, dtype=float).tolist(),
        "intercept": float(ridge.intercept_),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "column_count": int(oof.shape[1]),
    }
