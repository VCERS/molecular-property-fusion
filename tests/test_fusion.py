from __future__ import annotations

import numpy as np
import pytest

from molprop_fusion.fusion import (
    fit_meta,
    grouped_folds,
    prediction_sha256,
    validate_grouped_folds,
)


def test_grouped_folds_are_disjoint_and_complete() -> None:
    groups = [f"group-{index}" for index in range(12)]
    folds = grouped_folds(groups, 13)
    validate_grouped_folds(folds, groups)
    held = np.concatenate([row[1] for row in folds])
    assert sorted(held.tolist()) == list(range(12))


def test_grouped_folds_reject_group_leakage() -> None:
    groups = ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="group overlap"):
        validate_grouped_folds(
            [(np.array([0, 2]), np.array([1, 3]))],
            groups,
            n_splits=1,
        )


def test_meta_accepts_tree_and_graph_columns() -> None:
    rng = np.random.default_rng(13)
    target = rng.normal(size=30)
    oof = rng.normal(size=(30, 5))
    evaluation = rng.normal(size=(4, 5))
    prediction, details = fit_meta(oof, target, evaluation)
    assert prediction.shape == (4,)
    assert details["alpha"] == 10.0
    assert details["column_count"] == 5


def test_prediction_hash_is_order_sensitive_and_finite() -> None:
    assert prediction_sha256([1.0, 2.0], ["a", "b"]) != prediction_sha256([1.0, 2.0], ["b", "a"])
    with pytest.raises(ValueError):
        prediction_sha256([np.nan], ["a"])
