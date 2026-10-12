"""Small software fixtures only: no scientific matrix or upstream target access."""

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from molprop_fusion.expanded_runtime import Timer, attribute_cost
from molprop_fusion.expanded_summary import summarize
from molprop_fusion.expanded_verification import verify_expanded
from molprop_fusion.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SMILES = ["C" * i for i in range(1, 21)] + ["CCO", "OCC", "CCN", "CCCl", "CCBr"]


@pytest.fixture
def expanded_data():
    pytest.importorskip("rdkit")
    from molprop_fusion import expanded_data

    return expanded_data


@pytest.fixture
def torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    return torch


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "fixture.csv"
    with source.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol", "pIC50", "unused"])
        writer.writerows((s, i / 10, "ignored") for i, s in enumerate(SMILES))
    partitions = tmp_path / "bace.csv"
    with partitions.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["source_index", "geometry_source_index", *[f"split_run_{r}" for r in range(1, 6)]]
        )
        # Row 0 deliberately excluded; indices 20/21 are canonical duplicates.
        for i in range(1, 25):
            role = "train" if i < 16 or i in [20, 21] else "validation" if i < 20 else "test"
            writer.writerow([i, 20 if i == 21 else i, *[role] * 5])
    spec = {
        "file": source.name,
        "delimiter": ",",
        "smiles_field": "mol",
        "endpoints": {
            "pIC50": {
                "units": "-log10(IC50/mol_L)",
                "transform": "source_pIC50_identity",
                "primary_metric": "rmse",
            }
        },
        "sha256": sha256_file(source),
        "partition_file_sha256": sha256_file(partitions),
    }
    return tmp_path, spec, partitions


def test_partition_replay_preserves_order_exclusions_duplicates_and_seeds(fixture, expanded_data):
    source, spec, partitions = fixture
    prepared = expanded_data.replay(source, "bace", spec, partitions)
    assert [r["source_index"] for r in prepared["rows"]] == list(range(1, 25))
    assert (
        prepared["rows"][19]["geometry_source_index"]
        == prepared["rows"][20]["geometry_source_index"]
        == 20
    )
    assert all("pIC50" not in r and "target" not in r for r in prepared["rows"])
    assert prepared["memberships"]["1"]["bace:20"] == prepared["memberships"]["1"]["bace:21"]
    partitions.write_text(partitions.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        expanded_data.replay(source, "bace", spec, partitions)


def test_species_adapter_does_not_renumber_source_rows(tmp_path, expanded_data):
    p = tmp_path / "species.tab"
    p.write_text("Drug\tY\tSpecies\nCC\t1\tMus musculus\nCCC\t2\tHomo sapiens\n")
    spec = {
        "file": p.name,
        "sha256": sha256_file(p),
        "delimiter": "\t",
        "smiles_field": "Drug",
        "species": "Homo sapiens",
        "endpoints": {"Y": {}},
    }
    assert list(expanded_data.source_rows(tmp_path, spec))[0][:3] == (1, "CCC", {"Y": 2.0})
    p.write_text(p.read_text() + "\n")
    with pytest.raises(RuntimeError, match="source hash"):
        list(expanded_data.source_rows(tmp_path, spec))


def test_replay_rejects_nonfixed_test_and_parent_overlap(fixture, expanded_data):
    source, spec, partitions = fixture
    text = partitions.read_text()
    partitions.write_text(
        text.replace("22,22,test,test,test,test,test", "22,22,test,train,test,test,test")
    )
    spec["partition_file_sha256"] = sha256_file(partitions)
    with pytest.raises(ValueError, match="fixed"):
        expanded_data.replay(source, "bace", spec, partitions)
    partitions.write_text(
        text.replace("21,20,train,train,train,train,train", "21,20,test,test,test,test,test")
    )
    spec["partition_file_sha256"] = sha256_file(partitions)
    with pytest.raises(ValueError, match="parent overlap"):
        expanded_data.replay(source, "bace", spec, partitions)


def test_whole_group_assignment_preserves_empty_scaffold(expanded_data):
    groups = [""] * 4 + ["a"] * 2 + ["b"] * 3 + ["c"] * 2
    selected = expanded_data.assign_groups(groups, 0.2, 20261009)
    assert selected == expanded_data.assign_groups(groups, 0.2, 20261009)
    assert 0 < sum(g in selected for g in groups) < len(groups)
    assert len({g in selected for g in groups if g == ""}) == 1


def test_metric_directions(expanded_data, torch):
    from molprop_fusion.expanded_benchmark import metrics

    assert (
        metrics(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), "spearman")["spearman"] == -1
    )
    with pytest.raises(ValueError, match="vectors"):
        metrics(np.array([np.nan]), np.array([1.0]), "mae")


def test_nonoverlapping_costs():
    costs = attribute_cost(
        [
            {"method": "tree", "stage": "base_fit", "elapsed_seconds": 2},
            {"method": "G", "stage": "base_fit", "elapsed_seconds": 3},
            {"method": "T_plus_G", "stage": "ridge_fit_scaling", "elapsed_seconds": 1},
            {"method": "Chemprop", "stage": "chemprop_workflow", "elapsed_seconds": 4},
        ]
    )
    assert costs["T_plus_G_component_seconds"] == 6 and costs["Chemprop_component_seconds"] == 4


def test_expanded_aggregates_recompute_and_detect_mutation(tmp_path):
    assert all(verify_expanded(ROOT).values())
    for name in ["configs", "results", "partitions"]:
        shutil.copytree(ROOT / name, tmp_path / name)
    p = tmp_path / "results/expanded_benchmarks.json"
    report = json.loads(p.read_text())
    report["cases"][0]["methods"]["T_plus_G"]["mae"] += 1
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metrics"):
        verify_expanded(tmp_path)


def test_historical_geometry_cache_and_real_tiny_case(fixture, monkeypatch, expanded_data, torch):
    for dependency in ("xgboost", "catboost", "lightgbm"):
        pytest.importorskip(dependency)

    from molprop_fusion import expanded_benchmark as eb
    from molprop_fusion.z3d import LabelFreeIdentity, generate_z3d

    source, spec, partitions = fixture
    prepared = expanded_data.replay(source, "bace", spec, partitions)
    config = yaml.safe_load((ROOT / "configs/expanded_benchmarks.yaml").read_text())
    config["resources"]["tree_workers"] = 1
    config["device"] = "cpu"
    config["validation_only"] = True
    timer = Timer()
    cache = eb.FeatureCache(source / "features", config)
    features, provider, state = cache.matrices("bace", prepared["rows"], timer)
    row = prepared["rows"][0]
    expected = generate_z3d(
        LabelFreeIdentity("bace", str(row["source_index"]), row["canonical_smiles"], ()),
        {"question": "Molecular property regression", "definition": "Software fixture"},
    )
    np.testing.assert_equal(features[1][0], expected["values"].astype(np.float32))
    assert state == "cold"
    warm, _, state = cache.matrices("bace", prepared["rows"], Timer())
    np.testing.assert_equal(features[1], warm[1])
    assert state == "warm"
    np.testing.assert_equal(
        provider(row["canonical_smiles"], 115)[0],
        eb.featurize_smiles(row["canonical_smiles"], 115)[0],
    )
    # Exercise actual four-tree, four CoMPT and Ridge fits. Chemprop CLI contract has its own fixture.
    monkeypatch.setattr(
        eb, "partitioned_chemprop", lambda smiles, targets, *args: np.ones(len(smiles["test"]))
    )
    result = eb.case(config, spec, prepared, "pIC50", 1, source, features, provider, Timer())
    assert result["complete"] and set(result["methods"]) == {"T", "G", "T_plus_G", "Chemprop"}
    assert result["rows"] == {"train": 17, "validation": 4, "test": 3}
    assert result["component_costs"]["incremental_graph_plus_fusion_seconds"] > 0
    summary = summarize([result])
    assert summary["case_count"] == 1


def test_chemprop_adapter_withholds_test_labels_and_deletes_artifacts(tmp_path, monkeypatch, torch):
    from molprop_fusion import expanded_chemprop as cp

    workdirs = []

    def child(command, **kwargs):
        data = Path(command[command.index("--data-path") + 1])
        workdirs.append(data.parent)
        rows = list(csv.DictReader(data.open()))
        assert rows[-1]["split"] == "test" and rows[-1]["target"] == ""
        assert command[command.index("--tracking-metric") + 1] == "rmse"
        assert command[command.index("--epochs") + 1] == "50"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        model = Path(command[command.index("--output-dir") + 1])
        model.mkdir()
        (model / "test_predictions.csv").write_text("smiles,target\nCCN,1.2\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cp.importlib.metadata, "version", lambda _: "2.3.1")
    monkeypatch.setattr(cp.subprocess, "run", child)
    prediction = cp.partitioned_chemprop(
        {"train": ["CC"], "validation": ["CCC"], "test": ["CCN"]},
        {"train": [1.0], "validation": [2.0]},
        1,
        Timer(),
        {"device": "cpu"},
    )
    np.testing.assert_equal(prediction, [1.2])
    assert all(not p.exists() for p in workdirs)


def test_graph_provider_and_timing_hooks_preserve_predictions(expanded_data, torch):
    from molprop_fusion.compt_graph import featurize_smiles, fit_graph_regressor

    smiles = ["C" * i for i in range(1, 13)]
    options = {
        "atom_width": 115,
        "bond_width": 13,
        "hidden_width": 16,
        "layers": 1,
        "heads": 4,
        "dropout": 0.0,
        "max_atoms": 115,
    }
    training = {
        "device": "cpu",
        "weight_decay": 0.0,
        "warmup_factor": 0.15,
        "warmup_epochs": 1,
        "max_epochs": 1,
        "patience": 1,
        "batch_size": 8,
    }
    kwargs = {
        "train_smiles": smiles,
        "train_target": np.arange(12, dtype=float),
        "train_groups": smiles,
        "evaluation_smiles": ["CCN", "CCO"],
        "seed": 1,
        "model_options": options,
        "training_options": training,
    }
    plain = fit_graph_regressor(**kwargs)
    timer = Timer()
    cached = fit_graph_regressor(**kwargs, feature_provider=featurize_smiles, measure=timer.stage)
    np.testing.assert_array_equal(plain.predictions, cached.predictions)
    assert [r["stage"] for r in timer.rows] == ["graph_tensors", "base_fit", "prediction"]
    assert timer.rows[1]["epochs_completed"] == 1


@pytest.mark.parametrize(
    "dataset",
    [
        "ppbr_az",
        "vdss_lombardo",
        "half_life_obach",
        "clearance_microsome_az",
        "clearance_hepatocyte_az",
        "bace",
        "qm9",
    ],
)
def test_all_pinned_dataset_field_adapters(tmp_path, dataset, expanded_data):
    spec = json.loads((ROOT / "configs/expanded_sources.json").read_text())["datasets"][dataset]
    path = tmp_path / spec["file"]
    fields = [spec["smiles_field"], *spec["endpoints"]]
    if spec.get("species"):
        fields.append("Species")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=spec["delimiter"])
        writer.writeheader()
        row = {
            spec["smiles_field"]: "CCO",
            **{f: str(i + 0.25) for i, f in enumerate(spec["endpoints"])},
        }
        if spec.get("species"):
            row["Species"] = spec["species"]
        writer.writerow(row)
    spec["sha256"] = sha256_file(path)
    rows = list(expanded_data.source_rows(tmp_path, spec))
    assert len(rows) == 1 and rows[0][0:2] == (0, "CCO")
    assert set(rows[0][2]) == set(spec["endpoints"])
    assert all(np.isfinite(v) for v in rows[0][2].values())


def test_expanded_prepare_cli_and_prepared_replay_boundary(fixture, expanded_data, torch):
    import subprocess
    import sys

    from molprop_fusion.expanded_workflow import run

    source, spec, partitions = fixture
    manifest = json.loads((ROOT / "configs/expanded_sources.json").read_text())
    manifest["datasets"]["bace"].update(spec, partition_file=partitions.name)
    manifest_path = source / "sources.json"
    manifest_path.write_text(json.dumps(manifest))
    output = source / "prepared.json"
    command = [
        sys.executable,
        "-m",
        "molprop_fusion.cli",
        "expanded-prepare",
        "--dataset",
        "bace",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--config",
        str(ROOT / "configs/expanded_benchmarks.yaml"),
        "--partitions",
        str(source),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["retained_rows"] == 24
    prepared = json.loads(output.read_text())
    prepared["memberships"]["1"]["bace:1"] = "test"
    output.write_text(json.dumps(prepared))
    args = SimpleNamespace(
        config=ROOT / "configs/expanded_benchmarks.yaml",
        manifest=manifest_path,
        dataset="bace",
        endpoint="pIC50",
        run=1,
        source=source,
        partitions=source,
        output=source / "case.json",
        prepared=output,
    )
    with pytest.raises(ValueError, match="prepared snapshot differs"):
        run(args)
