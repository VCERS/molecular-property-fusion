"""Expanded historical numerical path, using the reusable package components."""

import os
import time
from functools import partial
from pathlib import Path

import numpy as np

from molprop_fusion.compt_graph import featurize_smiles, fit_graph_regressor
from molprop_fusion.expanded_chemprop import partitioned_chemprop
from molprop_fusion.expanded_data import chemistry_matrix, source_rows
from molprop_fusion.expanded_runtime import attribute_cost, digest
from molprop_fusion.features import impute_raw_z
from molprop_fusion.fusion import TABULAR_COLUMNS, grouped_folds, validate_grouped_folds
from molprop_fusion.io import sha256_file as file_hash
from molprop_fusion.z3d import LabelFreeIdentity, generate_z3d


def model_profile(config):
    t = config["methods"]["T"]
    return {
        "models": {
            "xgboost": {"common": {}, "by_dataset": {"expanded": t["xgboost"]}},
            **{k: t[k] for k in ["catboost", "lightgbm", "extratrees"]},
        }
    }


class FeatureCache:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.minimum_indices = {}
        self.schema = digest(
            {
                "features": config["features"],
                "z3d_code": file_hash(Path(__file__).with_name("z3d.py")),
                "graph_code": file_hash(Path(__file__).with_name("compt_graph.py")),
            }
        )

    def key(self, dataset, row):
        return digest(
            [
                self.schema,
                dataset,
                self.minimum_indices.get((dataset, row["canonical_smiles"]), row["source_index"]),
                row["canonical_smiles"],
            ]
        )

    def matrices(self, dataset, rows, timer):
        for r in rows:
            key = (dataset, r["canonical_smiles"])
            self.minimum_indices[key] = min(
                self.minimum_indices.get(key, r["geometry_source_index"]),
                r["geometry_source_index"],
            )
        unique = {self.key(dataset, r): r for r in reversed(rows)}
        missing = [
            r for r in unique.values() if not (self.root / (self.key(dataset, r) + ".npz")).exists()
        ]
        state = "warm" if not missing else "cold" if len(missing) == len(unique) else "partial"
        values = {}
        if missing:
            with timer.stage(
                "morgan_rdkit",
                method="features",
                dataset=dataset,
                cache_state=state,
                evaluation_rows=len(missing),
            ):
                chemistry = chemistry_matrix([r["canonical_smiles"] for r in missing])
            with timer.stage(
                "conformer_z3d",
                method="features",
                dataset=dataset,
                cache_state=state,
                evaluation_rows=len(missing),
            ) as timing:
                z = [
                    generate_z3d(
                        LabelFreeIdentity(
                            dataset,
                            str(self.minimum_indices[(dataset, r["canonical_smiles"])]),
                            r["canonical_smiles"],
                            (str(r["source_index"]),),
                        ),
                        {
                            "question": "Public molecular regression",
                            "definition": "Label-free SMILES-derived geometry; no endpoint values.",
                        },
                    )
                    for r in missing
                ]
                timing["geometry_status_counts"] = {
                    s: sum(v["status"] == s for v in z) for s in {v["status"] for v in z}
                }
            with timer.stage(
                "graph_featurization",
                method="features",
                dataset=dataset,
                cache_state=state,
                evaluation_rows=len(missing),
            ):
                graphs = [featurize_smiles(r["canonical_smiles"], 115) for r in missing]
            with timer.stage(
                "cache_write",
                method="features",
                dataset=dataset,
                cache_state=state,
                evaluation_rows=len(missing),
            ):
                for r, c, v, g in zip(missing, chemistry, z, graphs, strict=True):
                    n = r["atom_count"]
                    key = self.key(dataset, r)
                    p = self.root / (key + ".npz")
                    temp = self.root / (key + ".partial.npz")
                    np.savez_compressed(
                        temp,
                        chemistry=c,
                        z=v["values"].astype(np.float32),
                        mask=v["mask"],
                        node=g[0][:n],
                        edge=g[1][:n, :n],
                        distance=g[2][:n, :n],
                    )
                    os.replace(temp, p)
                del graphs, chemistry, z
        with timer.stage(
            "cache_read",
            method="features",
            dataset=dataset,
            cache_state=state,
            evaluation_rows=len(rows),
        ):
            cs, zs, ms = [], [], []
            for r in rows:
                p = self.root / (self.key(dataset, r) + ".npz")
                with np.load(p, allow_pickle=False) as data:
                    cs.append(data["chemistry"])
                    zs.append(data["z"])
                    ms.append(data["mask"])
            values = (np.asarray(cs, np.float32), np.asarray(zs, np.float32), np.asarray(ms, bool))
        lookup = {r["canonical_smiles"]: self.root / (self.key(dataset, r) + ".npz") for r in rows}

        def provider(smiles, capacity):
            if capacity != 115:
                raise ValueError("capacity amendment requires new configuration")
            with np.load(lookup[smiles], allow_pickle=False) as data:
                n = len(data["node"])
                node = np.zeros((115, 116), np.float32)
                edge = np.zeros((115, 115, 13), np.float32)
                distance = np.zeros((115, 115), np.float32)
                node[:n] = data["node"]
                edge[:n, :n] = data["edge"]
                distance[:n, :n] = data["distance"]
            return node, edge, distance

        return values, provider, state


def tabular(config, family, c, z, mask, y, groups, test_c, test_z, test_mask, run, timer):
    from molprop_fusion.benchmark import _model

    splits = grouped_folds(groups, run)
    validate_grouped_folds(splits, groups)
    oof = np.empty(len(y), np.float64)
    final = None
    for fold, (fit, held) in enumerate([*splits, (np.arange(len(y)), None)]):
        label = "final" if held is None else fold
        meta = {
            "method": "tree",
            "learner": family,
            "fold": label,
            "fit_rows": len(fit),
            "evaluation_rows": len(test_c) if held is None else len(held),
            "cpu_workers": config["resources"]["tree_workers"],
        }
        with timer.stage("fold_preprocessing", **meta):
            fit_z, eval_z, _ = impute_raw_z(z[fit], test_z if held is None else z[held])
            fit_x = np.concatenate([c[fit], fit_z, mask[fit].astype(np.float32)], axis=1)
            eval_x = np.concatenate(
                [
                    test_c if held is None else c[held],
                    eval_z,
                    (test_mask if held is None else mask[held]).astype(np.float32),
                ],
                axis=1,
            )
        model = _model(
            model_profile(config),
            family,
            "expanded",
            run if held is None else run + fold,
            config["resources"]["tree_workers"],
            smoke=config.get("validation_only", False),
        )
        with timer.stage("base_fit", **meta):
            model.fit(fit_x, y[fit])
        with timer.stage("prediction", **meta):
            pred = np.asarray(model.predict(eval_x), np.float64)
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite tree output")
        if held is None:
            final = pred
        else:
            oof[held] = pred
        del model, fit_x, eval_x
    return oof, final


def fusion(oof, y, test, method, timer):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if oof.shape[1] != (4 if method == "T" else 5) or test.shape[1] != oof.shape[1]:
        raise ValueError("direct fusion columns differ")
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("nonfinite fusion input")
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, fit_intercept=True))
    metadata = {
        "method": method,
        "fit_rows": len(y),
        "evaluation_rows": len(test),
        "feature_width": oof.shape[1],
    }
    with timer.stage("ridge_fit_scaling", **metadata):
        model.fit(oof, y)
    with timer.stage("prediction", **metadata):
        pred = model.predict(test)
    return pred


def graph(config, smiles, y, groups, test_smiles, run, timer, provider):
    options = {
        "atom_width": 115,
        "bond_width": 13,
        "hidden_width": config["methods"]["G"]["architecture"]["hidden_width"],
        "layers": config["methods"]["G"]["architecture"]["layers"],
        "heads": config["methods"]["G"]["architecture"]["heads"],
        "dropout": config["methods"]["G"]["architecture"]["dropout"],
        "max_atoms": 115,
    }
    training = {
        "device": config.get("device", "cuda"),
        "weight_decay": config["methods"]["G"]["training"]["weight_decay"],
        "warmup_factor": config["methods"]["G"]["training"]["warmup_factor"],
        "warmup_epochs": config["methods"]["G"]["training"]["warmup_epochs"],
        "max_epochs": config["methods"]["G"]["training"]["max_epochs"],
        "patience": config["methods"]["G"]["training"]["patience"],
        "batch_size": config["methods"]["G"]["training"]["batch_size"],
    }
    if config.get("validation_only"):
        options.update(hidden_width=16, layers=1)
        training.update(max_epochs=1, patience=1, warmup_epochs=1, batch_size=8)
    oof = np.empty(len(y), np.float64)
    for fold, (fit, held) in enumerate([*grouped_folds(groups, run), (np.arange(len(y)), None)]):
        eval_smiles = test_smiles if held is None else [smiles[i] for i in held]
        measure = partial(
            timer.stage,
            method="G",
            learner="CoMPT",
            fold="final" if held is None else fold,
            fit_rows=len(fit),
            evaluation_rows=len(eval_smiles),
            cpu_workers=config["resources"]["tree_workers"],
        )
        result = fit_graph_regressor(
            train_smiles=[smiles[i] for i in fit],
            train_target=y[fit],
            train_groups=groups[fit],
            evaluation_smiles=eval_smiles,
            seed=run if held is None else run + fold,
            model_options=options,
            training_options=training,
            feature_provider=provider,
            measure=measure,
        )
        if held is None:
            final = result.predictions
        else:
            oof[held] = result.predictions
    return oof, final


def metrics(pred, truth, primary):
    if pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("matched metric vectors differ")
    delta = pred - truth
    result = {"mae": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta**2)))}
    if primary == "spearman":
        from scipy.stats import spearmanr

        rho = float(spearmanr(pred, truth).statistic)
        result["spearman"] = rho if np.isfinite(rho) else None
    return result


def case(config, spec, prepared, field, run, source, features, provider, timer):
    started = time.monotonic()
    rows = [r for r in prepared["rows"] if field in r["eligible_endpoints"]]
    lookup = {r["row_id"]: i for i, r in enumerate(prepared["rows"])}
    assignment = prepared["memberships"][str(run)]
    partitions = {
        role: [r for r in rows if assignment[r["row_id"]] == role]
        for role in ["train", "validation", "test"]
    }
    indices = {
        role: np.asarray([lookup[r["row_id"]] for r in values])
        for role, values in partitions.items()
    }
    values = {i: labels[field] for i, _, labels, _ in source_rows(source, spec)}
    ys = {
        role: np.asarray([values[r["source_index"]] for r in values_], np.float64)
        for role, values_ in partitions.items()
    }
    if not all(np.isfinite(y).all() for y in ys.values()):
        raise ValueError("eligibility drift")
    cs, zs, ms = features
    train = indices["train"]
    test = indices["test"]
    groups = np.asarray([r["canonical_smiles"] for r in partitions["train"]])
    smiles = {
        role: [r["canonical_smiles"] for r in values_] for role, values_ in partitions.items()
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(smiles[left]) & set(smiles[right]):
            raise ValueError("canonical outer group overlap")
    oofs = []
    tests = []
    for column in TABULAR_COLUMNS:
        oof, pred = tabular(
            config,
            column.split("_")[0],
            cs[train],
            zs[train],
            ms[train],
            ys["train"],
            groups,
            cs[test],
            zs[test],
            ms[test],
            run,
            timer,
        )
        oofs.append(oof)
        tests.append(pred)
    tab_oof = np.column_stack(oofs)
    tab_test = np.column_stack(tests)
    t = fusion(tab_oof, ys["train"], tab_test, "T", timer)
    goof, g = graph(
        config, smiles["train"], ys["train"], groups, smiles["test"], run, timer, provider
    )
    tg = fusion(
        np.column_stack([tab_oof, goof]),
        ys["train"],
        np.column_stack([tab_test, g]),
        "T_plus_G",
        timer,
    )
    cp = partitioned_chemprop(smiles, ys, run, timer, config)
    scores = {
        method: metrics(pred, ys["test"], spec["endpoints"][field]["primary_metric"])
        for method, pred in [("T", t), ("G", g), ("T_plus_G", tg), ("Chemprop", cp)]
    }
    return {
        "dataset": prepared["dataset"],
        "endpoint": field,
        "run": run,
        "complete": True,
        "fit_count": 23,
        "methods": scores,
        "units": spec["endpoints"][field]["units"],
        "transform": spec["endpoints"][field]["transform"],
        "rows": {k: len(v) for k, v in partitions.items()},
        "actual_case_elapsed_seconds": time.monotonic() - started,
        "component_costs": attribute_cost(timer.rows),
        "test_identity_digest": digest([r["row_id"] for r in partitions["test"]]),
    }
