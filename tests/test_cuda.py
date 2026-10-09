"""CUDA hardware coverage: skipped automatically when CUDA is unavailable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from molprop_fusion import MolecularPropertyFusionRegressor, load_model

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
QUERY = ["CCO", "CCCC", "CCO", "c1ccccc1", "CCN", "C", "CC", "CCC", "CCOC", "CCCl", "CCCO", "CCCN"]


@pytest.fixture(scope="module")
def cuda_bundle(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    for dependency in ("rdkit", "xgboost", "catboost", "lightgbm"):
        pytest.importorskip(dependency)
    torch.set_num_threads(1)
    started = time.perf_counter()
    targets = np.arange(len(SMILES), dtype=float) / 3 + np.sin(np.arange(len(SMILES)))
    model = MolecularPropertyFusionRegressor(
        mode="full",
        device="cuda",
        max_atoms=16,
        validation_only=True,
    ).fit(SMILES, targets)
    assert next(model.graph_.model.parameters()).is_cuda
    assert model.training_summary_["oof_columns"] == 5
    prediction = model.predict(QUERY)
    assert prediction[0] == prediction[2]
    directory = tmp_path_factory.mktemp("cuda") / "model"
    model.save(directory)
    details = {
        "fit_and_save_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(),
    }
    return model, directory, prediction, details


def test_cuda_bundle_preserves_capacity_order_and_batches(cuda_bundle, record_property):
    _, directory, expected, details = cuda_bundle
    for key, value in details.items():
        record_property(key, value)
    restored = load_model(directory, device="cuda")
    np.testing.assert_allclose(restored.predict(QUERY), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(restored.predict(QUERY[::-1]), expected[::-1], rtol=1e-5, atol=1e-5)
    for i, smiles in enumerate(QUERY):
        restored._feature_cache.clear()
        np.testing.assert_allclose(
            restored.predict([smiles]), expected[i : i + 1], rtol=1e-5, atol=1e-5
        )
    assert restored.graph_.model_options["max_atoms"] == 16
    assert restored.graph_.model.pos_embed.pe.num_embeddings == 17
    with pytest.raises(ValueError, match="retrain"):
        restored.predict(["C" * 17])


def test_gpu_trained_bundle_loads_in_fresh_process_with_no_visible_gpu(
    cuda_bundle, tmp_path, record_property
):
    _, directory, expected, _ = cuda_bundle
    code = (
        "import json, torch; from molprop_fusion import load_model; "
        "assert not torch.cuda.is_available(); "
        f"model = load_model({str(directory)!r}); assert model.device == 'cpu'; "
        "assert not next(model.graph_.model.parameters()).is_cuda; "
        f"print(json.dumps(model.predict({QUERY!r}).tolist()))"
    )
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    cpu_prediction = np.asarray(json.loads(completed.stdout))
    record_property(
        "max_gpu_cpu_absolute_difference", float(np.max(np.abs(cpu_prediction - expected)))
    )
    np.testing.assert_allclose(cpu_prediction, expected, rtol=1e-5, atol=1e-5)


def test_native_compt_architecture_on_cuda(cuda_bundle, record_property):
    import torch

    from molprop_fusion.compt_graph import fit_graph_regressor

    targets = np.arange(len(SMILES), dtype=float) / 3 + np.sin(np.arange(len(SMILES)))
    result = fit_graph_regressor(
        train_smiles=SMILES,
        train_target=targets,
        train_groups=SMILES,
        evaluation_smiles=QUERY,
        seed=42,
        model_options={"max_atoms": 100},  # Native width 256, three layers, four heads.
        training_options={
            "device": "cuda",
            "max_epochs": 1,
            "patience": 1,
            "warmup_epochs": 1,
            "batch_size": 8,
            "weight_decay": 0.0,
            "warmup_factor": 0.10,
        },
    )
    assert next(result.predictor.model.parameters()).is_cuda
    assert result.predictor.model.pos_embed.pe.num_embeddings == 101
    record_property("native_graph_fit_seconds", result.wall_seconds)
    record_property("native_graph_peak_cuda_allocated_bytes", result.peak_accelerator_bytes)
    cpu_prediction = result.predictor.predict(QUERY, device="cpu")
    record_property(
        "native_graph_max_gpu_cpu_absolute_difference",
        float(np.max(np.abs(cpu_prediction - result.predictions))),
    )
    np.testing.assert_allclose(cpu_prediction, result.predictions, rtol=1e-5, atol=1e-5)
    assert not next(result.predictor.model.parameters()).is_cuda
    assert torch.cuda.is_available()  # The comparison really used CPU on a GPU-capable host.
