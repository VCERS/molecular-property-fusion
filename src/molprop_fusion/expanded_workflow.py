"""Public expanded-paper commands, independent of private storage and execution gates."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import yaml

from molprop_fusion.io import atomic_write_json, sha256_file


def load_inputs(config_path, manifest_path):
    config = yaml.safe_load(Path(config_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if config.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("unsupported expanded schema")
    if config["runs"] != [1, 2, 3, 4, 5] or len(manifest["datasets"]) != 7:
        raise ValueError("expanded roster or runs differ")
    return config, manifest


def prepare(args):
    from molprop_fusion.expanded_data import replay

    config, manifest = load_inputs(args.config, args.manifest)
    spec = manifest["datasets"][args.dataset]
    prepared = replay(
        args.source, args.dataset, spec, args.partitions / spec["partition_file"], config["runs"]
    )
    # Refuse to overwrite a prepared snapshot.
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_write_json(args.output, prepared)
    return {
        "dataset": args.dataset,
        "retained_rows": len(prepared["rows"]),
        "source_hash": prepared["source_hash"],
        "partition_hash": prepared["partition_hash"],
    }


def run(args):
    from molprop_fusion.expanded_benchmark import FeatureCache, case
    from molprop_fusion.expanded_data import replay
    from molprop_fusion.expanded_runtime import Timer

    config, manifest = load_inputs(args.config, args.manifest)
    spec = manifest["datasets"][args.dataset]
    if args.endpoint not in spec["endpoints"] or args.run not in config["runs"]:
        raise ValueError("unknown endpoint or run")
    if args.output.exists():
        raise FileExistsError(args.output)
    # Verify each supplied prepared snapshot against source/partition replay, not shape alone.
    prepared = json.loads(args.prepared.read_text())
    expected = replay(
        args.source, args.dataset, spec, args.partitions / spec["partition_file"], config["runs"]
    )
    if prepared != expected:
        raise ValueError("prepared snapshot differs from recorded partition replay")
    observed = {
        k: importlib.metadata.version(k.replace("_", "-"))
        for k in config["environment"]
        if k != "python"
    }
    if not args.validation_only and (
        observed != {k: str(v) for k, v in config["environment"].items() if k != "python"}
        or platform.python_version() != config["environment"]["python"]
    ):
        raise RuntimeError(
            "historical fitting requires the exact environment in expanded_benchmarks.yaml"
        )
    config["device"] = args.device
    config["resources"]["tree_workers"] = args.n_jobs
    config["validation_only"] = args.validation_only
    started = time.perf_counter()
    feature_timer = Timer()
    features, provider, state = FeatureCache(args.feature_cache, config).matrices(
        args.dataset, prepared["rows"], feature_timer
    )
    feature_elapsed = time.perf_counter() - started
    result = case(
        config, spec, prepared, args.endpoint, args.run, args.source, features, provider, Timer()
    )
    result.update(
        validation_only=args.validation_only,
        feature_cache_state=state,
        source_file_sha256=sha256_file(args.source / spec["file"]),
        partition_file_sha256=spec["partition_file_sha256"],
        shared_feature_elapsed_seconds=feature_elapsed,
        workflow_elapsed_seconds=time.perf_counter() - started,
        row_level_values_persisted=False,
        dependency_versions=observed,
        configuration_sha256=sha256_file(args.config),
    )
    atomic_write_json(args.output, result)
    return result


def summarize_files(paths, output):
    from molprop_fusion.expanded_summary import summarize

    cases = [json.loads(p.read_text()) for p in paths]
    keys = [(c["dataset"], c["endpoint"], c["run"]) for c in cases]
    if len(keys) != len(set(keys)) or any(not c["complete"] for c in cases):
        raise ValueError("duplicate or incomplete cases")
    modes = {c.get("validation_only", False) for c in cases}
    if len(modes) != 1:
        raise ValueError("cannot combine validation and scientific cases")
    result = summarize(cases)
    result["validation_only"] = modes.pop()
    # Shared features occur per invocation: cold/warm clocks must not be conflated.
    result["cases_elapsed_seconds"] = sum(c["actual_case_elapsed_seconds"] for c in cases)
    result["invocation_elapsed_seconds_sum"] = sum(c["workflow_elapsed_seconds"] for c in cases)
    atomic_write_json(output, result)
    return result
