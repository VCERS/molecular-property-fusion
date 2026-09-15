from __future__ import annotations

import numpy as np

from molprop_fusion.benchmark import regression_metrics, split_indices


def test_lane_k_split_counts_and_disjointness() -> None:
    train, validation, test = split_indices(1128, 13)
    assert (len(train), len(validation), len(test)) == (902, 113, 113)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert len(set(train) | set(validation) | set(test)) == 1128


def test_native_mae_and_rmse() -> None:
    observed = regression_metrics([0.0, 2.0], [1.0, 0.0])
    assert observed["mae"] == 1.5
    assert observed["rmse"] == np.sqrt(2.5)
