"""Recompute published run aggregates, favorable counts, and component costs."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from molprop_fusion.expanded_runtime import digest
from molprop_fusion.expanded_summary import summarize
from molprop_fusion.io import sha256_file

METHODS = ["T", "G", "T_plus_G", "Chemprop"]


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return all(k in actual and equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == expected


def verify_expanded(root):
    root = Path(root)
    report = json.loads((root / "results/expanded_benchmarks.json").read_text())
    preparation = json.loads((root / "results/expanded_preparation.json").read_text())
    manifest = json.loads((root / "configs/expanded_sources.json").read_text())
    cases = report["cases"]
    expected = {
        (d, field, run)
        for d, spec in manifest["datasets"].items()
        for field in spec["endpoints"]
        for run in range(1, 6)
    }
    keys = [(c["dataset"], c["endpoint"], c["run"]) for c in cases]
    checks = {
        "expanded_matrix": len(keys) == len(set(keys)) == 55 and set(keys) == expected,
        "expanded_fits": all(c["complete"] and c["fit_count"] == 23 for c in cases)
        and sum(c["fit_count"] for c in cases) == report["successful_fit_count"] == 1265,
        "expanded_aggregate_only": report["aggregate_only"]
        and not report["row_level_values_included"],
    }
    rebuilt = summarize(cases)
    checks["expanded_metrics_and_paired_summaries"] = equivalent(report, rebuilt)
    wins = {ref: 0 for ref in ["T", "Chemprop"]}
    adverse = []
    for entry in rebuilt["endpoints"]:
        spec = manifest["datasets"][entry["dataset"]]["endpoints"][entry["endpoint"]]
        group = [
            c
            for c in cases
            if (c["dataset"], c["endpoint"]) == (entry["dataset"], entry["endpoint"])
        ]
        checks[f"{entry['dataset']}_{entry['endpoint']}_metadata"] = all(
            c["units"] == spec["units"]
            and c["transform"] == spec["transform"]
            and set(c["methods"]) == set(METHODS)
            for c in group
        )
        metric = spec["primary_metric"]
        for ref in wins:
            delta = entry["paired_differences"]["T_plus_G_minus_" + ref][metric]["mean"]
            favorable = delta > 0 if metric == "spearman" else delta < 0
            wins[ref] += favorable
            if ref == "T" and not favorable:
                adverse.append(entry["dataset"])
        retained_entry = next(
            e
            for e in report["endpoints"]
            if (e["dataset"], e["endpoint"]) == (entry["dataset"], entry["endpoint"])
        )
        checks[f"{entry['dataset']}_{entry['endpoint']}_costs"] = all(
            equivalent(
                retained_entry["costs"][k],
                {
                    "mean": mean([c["component_costs"][k] for c in group]),
                    "sample_sd": stdev([c["component_costs"][k] for c in group]),
                },
            )
            for k in retained_entry["costs"]
        )
    checks["expanded_favorable_means"] = (
        wins == report["T_plus_G_favorable_endpoint_means_primary"] == {"T": 7, "Chemprop": 10}
    )
    checks["expanded_adverse_T"] = set(adverse) == {
        "vdss_lombardo",
        "half_life_obach",
        "clearance_microsome_az",
        "bace",
    }
    checks["expanded_cost_totals"] = all(
        math.isclose(sum(c["component_costs"][k] for c in cases), v, rel_tol=1e-12)
        for k, v in report["cost_totals_seconds"].items()
    )
    checks["expanded_case_elapsed"] = math.isclose(
        sum(c["actual_case_elapsed_seconds"] for c in cases),
        report["cases_elapsed_seconds"],
        rel_tol=1e-12,
    )
    checks["expanded_workflow_elapsed"] = (
        math.isclose(
            report["workflow_elapsed_seconds"],
            report["cases_elapsed_seconds"]
            + report["shared_features_elapsed_seconds"]
            + report["other_workflow_seconds"],
            rel_tol=1e-12,
        )
        and round(report["workflow_elapsed_seconds"] / 3600, 3) == 27.728
    )
    checks["expanded_component_increment"] = all(
        math.isclose(
            c["component_costs"]["T_plus_G_component_seconds"],
            c["component_costs"]["T_component_seconds"]
            + c["component_costs"]["incremental_graph_plus_fusion_seconds"],
            rel_tol=1e-12,
        )
        for c in cases
    )
    for d, spec in manifest["datasets"].items():
        path = root / "partitions" / spec["partition_file"]
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        checks[d + "_partition_hash"] = sha256_file(path) == spec["partition_file_sha256"]
        checks[d + "_retained_count"] = len(records) == preparation["datasets"][d]["retained"]
        for run in range(1, 6):
            counts = {
                role: sum(r[f"split_run_{run}"] == role for r in records)
                for role in ["train", "validation", "test"]
            }
            tests = [f"{d}:{r['source_index']}" for r in records if r[f"split_run_{run}"] == "test"]
            checks[f"{d}_run_{run}_assignments"] = counts == preparation["datasets"][d]["runs"][
                str(run)
            ] and all(
                c["rows"] == counts and c["test_identity_digest"] == digest(tests)
                for c in cases
                if c["dataset"] == d and c["run"] == run
            )
    qm9 = {e["endpoint"]: e for e in rebuilt["endpoints"] if e["dataset"] == "qm9"}
    checks["expanded_qm9_adverse"] = (
        qm9["mu"]["methods"]["T_plus_G"]["mae"]["mean"]
        > qm9["mu"]["methods"]["Chemprop"]["mae"]["mean"]
        and qm9["lumo"]["methods"]["T_plus_G"]["mae"]["mean"]
        < qm9["lumo"]["methods"]["Chemprop"]["mae"]["mean"]
        and qm9["lumo"]["methods"]["T_plus_G"]["rmse"]["mean"]
        > qm9["lumo"]["methods"]["Chemprop"]["rmse"]["mean"]
    )
    if not all(checks.values()):
        raise ValueError(f"expanded verification failed: {[k for k, v in checks.items() if not v]}")
    return checks
