"""Pure helpers for the frozen accuracy-first benchmark."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np


def prediction_hash(values: np.ndarray, rows: list[str]) -> str:
    payload = np.ascontiguousarray(values, dtype="<f8").tobytes() + "\n".join(rows).encode()
    return hashlib.sha256(payload).hexdigest()


def normalized_metric(
    pred: np.ndarray, target: np.ndarray, train_target: np.ndarray, name: str
) -> float:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    value = (
        float(np.sqrt(np.mean(error * error))) if name == "rmse" else float(np.mean(np.abs(error)))
    )
    scale = float(np.std(train_target, ddof=0))
    if not scale > 0:
        raise ValueError("training target scale must be positive")
    return value / scale


def fit_predict(
    model: Any, fit_x: np.ndarray, fit_y: np.ndarray, eval_x: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    model.fit(fit_x, fit_y)
    fitted = time.perf_counter()
    pred = np.asarray(model.predict(eval_x), dtype=np.float64)
    ended = time.perf_counter()
    if not np.isfinite(pred).all():
        raise ValueError("non-finite prediction")
    return pred, {"fit_seconds": fitted - start, "inference_seconds": ended - fitted}


def add_cost(total: dict[str, float], part: dict[str, float]) -> None:
    for key in ("fit_seconds", "inference_seconds"):
        total[key] = total.get(key, 0.0) + float(part[key])


def select_winner(rows: dict[str, dict[str, Any]], eligible: list[str], tolerance: float) -> str:
    if not eligible or any(candidate not in rows for candidate in eligible):
        raise ValueError("invalid eligible candidate set")
    best = min(float(rows[c]["macro_normalized_error"]) for c in eligible)
    tied = [c for c in eligible if float(rows[c]["macro_normalized_error"]) <= best + tolerance]
    tied.sort(
        key=lambda c: (
            -int(rows[c]["datasets_better_than_anchor"]),
            -int(rows[c]["cases_better_than_anchor"]),
            float(rows[c]["total_seconds"]),
            c,
        )
    )
    return tied[0]
