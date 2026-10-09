from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from molprop_fusion import MolecularPropertyFusionRegressor, load_model
from molprop_fusion.estimator import MedianImputer, molecule_seed
from molprop_fusion.features import canonicalize_smiles, z3d_matrix
from molprop_fusion.fusion import grouped_folds
from molprop_fusion.z3d import LabelFreeIdentity, generate_z3d, source_seed

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
]
TARGETS = np.arange(len(SMILES), dtype=float) / 3 + np.sin(np.arange(len(SMILES)))
QUERY = ["CCO", "CCCC", "CCO", "c1ccccc1"]


def _available(mode):
    for dependency in ("rdkit", "xgboost", "catboost", "lightgbm"):
        pytest.importorskip(dependency)
    if mode == "full":
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)


@pytest.fixture(scope="module", params=["tabular", "full"])
def fitted(request, tmp_path_factory):
    _available(request.param)
    model = MolecularPropertyFusionRegressor(
        mode=request.param, device="cpu", max_atoms=16, validation_only=True
    ).fit(SMILES, TARGETS)
    path = tmp_path_factory.mktemp(request.param) / "model"
    model.save(path)
    return model, path


def test_real_fit_bundle_fresh_process_and_batch_order(fitted, tmp_path):
    model, path = fitted
    expected = model.predict(QUERY)
    restored = load_model(path)
    assert restored.device == "cpu"
    np.testing.assert_allclose(restored.predict(QUERY), expected, rtol=1e-6, atol=1e-6)
    # Force regeneration so cached geometry cannot conceal batch-dependent seeds.
    for index, smiles in enumerate(QUERY):
        restored._feature_cache.clear()
        np.testing.assert_allclose(
            restored.predict([smiles]), expected[index : index + 1], atol=1e-6
        )
    restored._feature_cache.clear()
    np.testing.assert_allclose(restored.predict(QUERY[::-1]), expected[::-1], atol=1e-6)
    assert expected[0] == expected[2]
    assert restored.predict([]).shape == (0,)
    code = (
        "import json; from molprop_fusion import load_model; "
        f"m = load_model({str(path)!r}); print(json.dumps(m.predict({QUERY!r}).tolist()))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    np.testing.assert_allclose(json.loads(result.stdout), expected, rtol=1e-6, atol=1e-6)
    # Loaded native LightGBM wrapper must support another save/load cycle.
    again = tmp_path / "again"
    restored.save(again)
    np.testing.assert_allclose(load_model(again).predict(QUERY), expected, atol=1e-6)
    metadata = json.loads((path / "model.json").read_text())
    assert metadata["training_summary"]["oof_columns"] == (5 if model.mode == "full" else 4)
    assert len(metadata["features"]["rdkit_descriptors"]) == 217
    assert len(metadata["features"]["geometry_columns"]) == 317
    assert "targets" not in metadata and "smiles" not in metadata
    if model.mode == "full":
        assert restored.graph_.model_options["max_atoms"] == 16
        assert restored.graph_.model.pos_embed.pe.num_embeddings == 17
        assert restored.graph_.target_scale > 0
        with pytest.raises(ValueError, match="retrain"):
            restored.predict(["C" * 17])


def test_cli_unlabeled_csv_reporting_and_strict_errors(fitted, tmp_path):
    model, path = fitted
    source = tmp_path / "query.csv"
    source.write_text("id,smiles\none,CCO\nbad,not-a-smiles\nthree,CCCC\ndup,CCO\n")
    output = tmp_path / "prediction.csv"
    command = [
        sys.executable,
        "-m",
        "molprop_fusion.cli",
        "predict",
        "--model",
        str(path),
        "--data",
        str(source),
        "--output",
        str(output),
    ]
    strict = subprocess.run(command, capture_output=True, text=True)
    assert strict.returncode == 2
    assert not output.exists()
    reported = subprocess.run(
        [*command, "--invalid-policy", "report"], capture_output=True, text=True
    )
    assert reported.returncode == 0, reported.stderr
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["one", "bad", "three", "dup"]
    assert rows[1]["prediction"] == "" and rows[1]["prediction_status"] == "error"
    assert rows[0]["geometry_status"] == "ok"
    np.testing.assert_allclose(
        [float(rows[i]["prediction"]) for i in [0, 2, 3]],
        model.predict(["CCO", "CCCC", "CCO"]),
        atol=1e-6,
    )


def test_fold_local_imputation_with_real_tree_fits(monkeypatch):
    _available("tabular")
    raw = np.full((len(SMILES), 2582), np.nan)
    raw[:, :2265] = np.arange(len(SMILES))[:, None]
    raw[::2, 2265:] = np.arange(0, len(SMILES), 2)[:, None]
    observed = []
    original = MedianImputer.fit

    def capture(self, values):
        observed.append(values.copy())
        return original(self, values)

    monkeypatch.setattr(MedianImputer, "fit", capture)

    def features(self, canonical):
        self.descriptor_names_ = [str(i) for i in range(217)]
        return raw[:, :2265], raw[:, 2265:], [{"geometry_status": "ok"}] * len(SMILES)

    monkeypatch.setattr(MolecularPropertyFusionRegressor, "_features", features)
    model = MolecularPropertyFusionRegressor(mode="tabular", validation_only=True).fit(
        SMILES, TARGETS
    )
    folds = grouped_folds([canonicalize_smiles(s) for s in SMILES], 42)
    assert len(observed) == 4
    for values, (fit, held) in zip(observed[:3], folds, strict=True):
        np.testing.assert_equal(values, raw[fit])
        assert set(fit).isdisjoint(held)
    np.testing.assert_equal(observed[-1], raw)
    assert model.training_summary_["oof_columns"] == 4


def test_geometry_identity_and_historical_seed_unchanged():
    _available("tabular")
    model = MolecularPropertyFusionRegressor(mode="tabular", feature_seed=7)
    _, z, statuses = model._features(["CCO", "CC", "CCO"])
    model._feature_cache.clear()
    _, reordered, _ = model._features(["CCO", "CCO", "CC"])
    np.testing.assert_equal(reordered, z[[0, 2, 1]])
    assert (
        statuses[0]["geometry_missing_count"] > 0
    )  # absent pharmacophores are ordinary missingness
    assert statuses[0]["geometry_status"] == "ok"
    assert molecule_seed("CCO", 7) != molecule_seed("CCO", 8)
    historical, _, _ = z3d_matrix(["CCO"], dataset="esol")
    row = generate_z3d(
        LabelFreeIdentity("esol", "2", "CCO", ("2",)), {"question": "q", "definition": "d"}
    )
    assert row["seed_u31"] == source_seed("esol", "2")
    np.testing.assert_equal(historical[0], row["values"])
    _, _, failure = model._features(["CC.[Na+]"])
    assert failure[0]["geometry_status"] == "multiple_fragments"
    assert failure[0]["geometry_missing_count"] == 317


@pytest.mark.parametrize(
    "smiles,targets,match",
    [
        (SMILES, [0], "matching"),
        (SMILES, [np.inf] * len(SMILES), "finite"),
        (["CC", "CC", "C"], [1, 2, 3], "three distinct"),
        (["", "CC", "CCC"], [1, 2, 3], "row 1"),
        (["wrong", "CC", "CCC"], [1, 2, 3], "row 1"),
    ],
)
def test_actionable_input_errors(smiles, targets, match):
    _available("tabular")
    with pytest.raises(ValueError, match=match):
        MolecularPropertyFusionRegressor(mode="tabular").fit(smiles, targets)


def test_full_capacity_and_inner_scale_validated_before_features(monkeypatch):
    _available("full")

    def should_not_run(*args):
        pytest.fail("features should not be generated for an invalid training request")

    monkeypatch.setattr(MolecularPropertyFusionRegressor, "_features", should_not_run)
    with pytest.raises(ValueError, match="capacity"):
        MolecularPropertyFusionRegressor(max_atoms=2).fit(SMILES, TARGETS)
    with pytest.raises(ValueError, match="zero variation"):
        MolecularPropertyFusionRegressor().fit(SMILES, np.ones(len(SMILES)))


def test_schema_mismatch_rejected_before_model_deserialization(fitted, tmp_path):
    _, path = fitted
    metadata = json.loads((path / "model.json").read_text())
    metadata["features"]["rdkit_descriptors"].reverse()
    (tmp_path / "model.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        load_model(tmp_path)


def test_import_and_help_do_not_import_graph_or_tree_extras():
    code = (
        "import sys; from molprop_fusion import MolecularPropertyFusionRegressor; "
        "assert not {'torch', 'rdkit', 'xgboost', 'lightgbm', 'catboost'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_fitted_inference_configuration_cannot_resize_or_change_features(fitted):
    _, path = fitted
    model = load_model(path)
    model.max_atoms += 1
    with pytest.raises(ValueError, match="fixed"):
        model.predict(["CCO"])
    model = load_model(path)
    model.feature_seed += 1
    with pytest.raises(ValueError, match="fixed"):
        model.predict(["CCO"])


def test_canonical_duplicates_remain_in_one_fold():
    _available("tabular")
    canonical = [canonicalize_smiles(s) for s in ["CCO", "OCC", *SMILES]]
    folds = grouped_folds(canonical, 42)
    for fit, held in folds:
        assert set(np.asarray(canonical)[fit]).isdisjoint(np.asarray(canonical)[held])


def test_dependency_guidance_is_concise(monkeypatch):
    from molprop_fusion import estimator

    def missing(module):
        raise ImportError("missing")

    monkeypatch.setattr(estimator.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"molecular-property-fusion\[tabular\]"):
        estimator._dependencies("tabular")


def test_torch_hardware_wheel_suffix_does_not_prevent_cpu_loading(fitted, tmp_path):
    import shutil

    model, path = fitted
    if model.mode != "full":
        return
    shutil.copytree(path, tmp_path / "model")
    metadata_path = tmp_path / "model" / "model.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dependency_versions"]["torch"] = (
        metadata["dependency_versions"]["torch"].split("+")[0] + "+cu128"
    )
    metadata_path.write_text(json.dumps(metadata))
    np.testing.assert_allclose(
        load_model(tmp_path / "model", device="cpu").predict(QUERY), model.predict(QUERY), atol=1e-6
    )


@pytest.mark.parametrize("unsupported", ["[H][H]", "*"])
def test_no_usable_heavy_atom_rejected_before_descriptors(fitted, monkeypatch, unsupported):
    from molprop_fusion import estimator

    model, _ = fitted

    def descriptors_must_not_run(*args):
        pytest.fail("unsupported inputs must be rejected before descriptor evaluation")

    monkeypatch.setattr(estimator, "chemistry_matrix", descriptors_must_not_run)
    with pytest.raises(ValueError, match="row 2: unsupported molecule.*no usable heavy atom"):
        model.predict(["CCO", unsupported])
    with pytest.raises(ValueError, match="row 2: unsupported molecule.*no usable heavy atom"):
        MolecularPropertyFusionRegressor(mode=model.mode, device="cpu").fit(
            ["CCO", unsupported, "CC", "CCC"], [1, 2, 3, 4]
        )
    # Historical canonicalization continues to accept these syntactically valid structures.
    assert canonicalize_smiles(unsupported) == unsupported


def test_supported_single_heavy_atom_and_missing_geometry(fitted):
    model, _ = fitted
    predictions, statuses = model.predict(["C", "[Na+]", "CC.[Na+]"], return_status=True)
    assert np.isfinite(predictions).all()
    assert statuses[2]["geometry_status"] == "multiple_fragments"
    assert statuses[2]["geometry_missing_count"] == 317


def test_csv_unsupported_rows_reporting_and_strict(fitted, tmp_path):
    model, path = fitted
    source = tmp_path / "query.csv"
    source.write_text("id,smiles\ngood,CCO\nhydrogen,[H][H]\ndummy,*\nduplicate,OCC\n")
    output = tmp_path / "prediction.csv"
    command = [
        sys.executable,
        "-I",
        "-m",
        "molprop_fusion.cli",
        "predict",
        "--model",
        str(path),
        "--data",
        str(source),
        "--output",
        str(output),
    ]
    strict = subprocess.run(command, capture_output=True, text=True)
    assert strict.returncode == 2
    assert "CSV data row 2: unsupported molecule" in strict.stderr
    assert "no usable heavy atom" in strict.stderr and "Traceback" not in strict.stderr
    assert not output.exists()
    reported = subprocess.run(
        [*command, "--invalid-policy", "report"], capture_output=True, text=True
    )
    assert reported.returncode == 0, reported.stderr
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["good", "hydrogen", "dummy", "duplicate"]
    for index in (1, 2):
        assert rows[index]["prediction"] == ""
        assert rows[index]["prediction_status"] == "error"
        assert f"CSV data row {index + 1}: unsupported molecule" in rows[index]["prediction_error"]
        assert "no usable heavy atom" in rows[index]["prediction_error"]
    assert rows[0]["prediction"] == rows[3]["prediction"]
    np.testing.assert_allclose(float(rows[0]["prediction"]), model.predict(["CCO"])[0], atol=1e-6)
    # Training CLI follows the same validation path.
    source.write_text("smiles,target\nCCO,1\n*,2\nCC,3\nCCC,4\n")
    trained = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "molprop_fusion.cli",
            "train",
            "--mode",
            model.mode,
            "--data",
            str(source),
            "--target-column",
            "target",
            "--output",
            str(tmp_path / "train"),
        ],
        capture_output=True,
        text=True,
    )
    assert trained.returncode == 2
    assert "row 2: unsupported molecule" in trained.stderr and "Traceback" not in trained.stderr
    assert not (tmp_path / "train").exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "model.json",
        "imputation.npy",
        "xgboost.json",
        "catboost.cbm",
        "lightgbm.txt",
        "extratrees.joblib",
        "compt.pt",
    ],
)
def test_missing_bundle_artifacts_are_actionable(fitted, tmp_path, artifact):
    import shutil

    model, path = fitted
    if artifact == "compt.pt" and model.mode == "tabular":
        assert not (path / artifact).exists()
        assert not hasattr(load_model(path), "graph_")
        return
    broken = tmp_path / "broken"
    shutil.copytree(path, broken)
    (broken / artifact).unlink()
    with pytest.raises(FileNotFoundError, match=f"{artifact}.*complete bundle"):
        load_model(broken)
    if artifact in ("model.json", "lightgbm.txt", "compt.pt"):
        source = tmp_path / "query.csv"
        source.write_text("smiles\nCCO\n")
        output = tmp_path / "output.csv"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "molprop_fusion.cli",
                "predict",
                "--model",
                str(broken),
                "--data",
                str(source),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert artifact in result.stderr and "complete bundle" in result.stderr
        assert "Traceback" not in result.stderr
        assert not output.exists()


def test_descriptor_errors_are_not_swallowed(fitted, monkeypatch):
    from molprop_fusion import estimator

    model, _ = fitted
    model._feature_cache.clear()

    def broken_descriptor(*args):
        raise ZeroDivisionError("unexpected descriptor error")

    monkeypatch.setattr(estimator, "chemistry_matrix", broken_descriptor)
    with pytest.raises(ZeroDivisionError, match="unexpected descriptor error"):
        model.predict(["CCO"])
