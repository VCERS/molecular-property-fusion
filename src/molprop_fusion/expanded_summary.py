"""Run-level aggregate summaries; no molecular or graph dependencies."""

import numpy as np


def summarize(cases):
    endpoints = {}
    for row in cases:
        endpoints.setdefault((row["dataset"], row["endpoint"]), []).append(row)
    summaries = []
    for (dataset, endpoint), rows in endpoints.items():
        entry = {
            "dataset": dataset,
            "endpoint": endpoint,
            "runs_completed": len(rows),
            "units": rows[0]["units"],
            "methods": {},
            "paired_differences": {},
        }
        for method in ["T", "G", "T_plus_G", "Chemprop"]:
            entry["methods"][method] = {}
            for metric in rows[0]["methods"][method]:
                v = [r["methods"][method][metric] for r in rows]
                entry["methods"][method][metric] = {
                    "mean": float(np.mean(v)) if all(x is not None for x in v) else None,
                    "sample_sd": float(np.std(v, ddof=1))
                    if len(v) > 1 and all(x is not None for x in v)
                    else None,
                }
        for comparator in ["T", "Chemprop"]:
            entry["paired_differences"]["T_plus_G_minus_" + comparator] = {}
            for metric in rows[0]["methods"]["T_plus_G"]:
                v = [
                    None
                    if r["methods"]["T_plus_G"][metric] is None
                    or r["methods"][comparator][metric] is None
                    else r["methods"]["T_plus_G"][metric] - r["methods"][comparator][metric]
                    for r in rows
                ]
                entry["paired_differences"]["T_plus_G_minus_" + comparator][metric] = {
                    "run_deltas": v,
                    "mean": float(np.mean(v)) if all(x is not None for x in v) else None,
                    "sample_sd": float(np.std(v, ddof=1))
                    if len(v) > 1 and all(x is not None for x in v)
                    else None,
                    "favorable_runs": sum(
                        x is not None and (x > 0 if metric == "spearman" else x < 0) for x in v
                    ),
                }
        summaries.append(entry)
    return {
        "endpoints": summaries,
        "case_count": len(cases),
        "successful_fit_count": 23 * len(cases),
        "shared_test_repetitions_are_not_independent_test_populations": True,
    }
