"""Nonoverlapping component clocks, separate from actual workflow elapsed time."""

import hashlib
import json
import time
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Timer:
    def __init__(self):
        self.rows = []

    @contextmanager
    def stage(self, stage, **metadata):
        # Clock the completed operation, including asynchronous accelerator work.
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        row = {"stage": stage, **metadata}
        yield row
        if cuda:
            torch.cuda.synchronize()
        row["elapsed_seconds"] = time.perf_counter() - start
        self.rows.append(row)


def attribute_cost(rows):
    def cost(method, stage=None):
        return sum(
            r["elapsed_seconds"]
            for r in rows
            if r.get("method") in method and (stage is None or r["stage"] == stage)
        )

    tree = cost({"tree", "T"})
    graph = cost({"G"})
    fusion = cost({"T_plus_G"})
    return {
        "T_component_seconds": tree,
        "incremental_graph_plus_fusion_seconds": graph + fusion,
        "T_plus_G_component_seconds": tree + graph + fusion,
        "Chemprop_component_seconds": cost({"Chemprop"}),
        "shared_feature_component_seconds": cost({"features"}),
        "T_input_feature_seconds": sum(
            r["elapsed_seconds"]
            for r in rows
            if r.get("method") == "features" and r["stage"] in {"morgan_rdkit", "conformer_z3d"}
        ),
        "incremental_graph_feature_seconds": sum(
            r["elapsed_seconds"]
            for r in rows
            if r.get("method") == "features" and r["stage"] == "graph_featurization"
        ),
        "shared_cache_io_seconds": sum(
            r["elapsed_seconds"]
            for r in rows
            if r.get("method") == "features" and r["stage"] in {"cache_read", "cache_write"}
        ),
        "interpretation": "sum of nonoverlapping stage durations; actual workflow elapsed recorded separately; shared features charged once",
    }
