from __future__ import annotations

import numpy as np
import pytest

from molprop_fusion.features import chemistry_matrix, impute_raw_z


def test_fit_only_imputation_and_all_missing_zero() -> None:
    fit = np.array([[1.0, np.nan], [3.0, np.nan]])
    evaluate = np.array([[np.nan, np.nan]])
    fit_out, eval_out, details = impute_raw_z(fit, evaluate)
    assert fit_out.tolist() == [[1.0, 0.0], [3.0, 0.0]]
    assert eval_out.tolist() == [[2.0, 0.0]]
    assert details["all_missing_fit_columns"] == 1


def test_chemistry_feature_shape_in_frozen_rdkit() -> None:
    pytest.importorskip("rdkit")
    matrix, names = chemistry_matrix(["CCO", "c1ccccc1"])
    assert matrix.shape == (2, 2265)
    assert len(names) == 217
