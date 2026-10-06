"""Command-line interface for the public molecular-property fusion package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molprop-fusion")
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fit a reusable model from a labeled CSV")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--smiles-column", default="smiles")
    train.add_argument("--target-column", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--mode", choices=["full", "tabular"], default="full")
    train.add_argument("--device", default="auto")
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--feature-seed", type=int, default=42)
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--max-atoms", type=int, default=100)
    train.add_argument(
        "--validation-only", action="store_true", help="tiny software check; not an accuracy preset"
    )

    predict = subparsers.add_parser("predict", help="predict an unlabeled CSV in row order")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--smiles-column", default="smiles")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--invalid-policy", choices=["raise", "report"], default="raise")

    prepare = subparsers.add_parser("prepare", help="build public molecular features")
    prepare.add_argument("--config", type=Path, default=Path("configs/public_benchmarks.yaml"))
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--maximum-rows", type=int)

    benchmark = subparsers.add_parser("benchmark", help="run T, G, and T+G")
    benchmark.add_argument("--config", type=Path, default=Path("configs/public_benchmarks.yaml"))
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--feature-cache", type=Path)
    benchmark.add_argument("--seed", type=int, action="append")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--n-jobs", type=int, default=1)
    benchmark.add_argument("--smoke", action="store_true")

    transfer = subparsers.add_parser(
        "transfer", help="run the user-supplied AqSolDB transfer partitions"
    )
    transfer.add_argument("--config", type=Path, default=Path("configs/aqsoldb_transfer.yaml"))
    transfer.add_argument("--data", type=Path, required=True)
    transfer.add_argument("--output", type=Path, required=True)
    transfer.add_argument("--feature-cache", type=Path)
    transfer.add_argument("--seed", type=int, action="append")
    transfer.add_argument("--device", default="auto")
    transfer.add_argument("--n-jobs", type=int, default=1)
    transfer.add_argument("--smoke", action="store_true")

    chemprop = subparsers.add_parser("chemprop", help="run the pinned Chemprop baseline")
    chemprop.add_argument("--config", type=Path, default=Path("configs/public_benchmarks.yaml"))
    chemprop.add_argument("--dataset", required=True)
    chemprop.add_argument("--data", type=Path, required=True)
    chemprop.add_argument("--output", type=Path, required=True)
    chemprop.add_argument("--seed", type=int, required=True)
    chemprop.add_argument("--accelerator", default="gpu")
    chemprop.add_argument("--devices", type=int, default=1)
    chemprop.add_argument("--smoke", action="store_true")

    summarize = subparsers.add_parser("summarize", help="summarize case-level output")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    check = subparsers.add_parser("check-config", help="parse and validate a public config")
    check.add_argument("paths", nargs="+", type=Path)

    verify = subparsers.add_parser("verify-results", help="verify tracked sanitized aggregates")
    verify.add_argument("--results-root", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command in ("train", "predict"):
        from molprop_fusion.user_csv import run_predict, run_train

        try:
            result = run_train(args) if args.command == "train" else run_predict(args)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            _parser().error(str(error))
    elif args.command == "prepare":
        from molprop_fusion.benchmark import load_dataset, prepare_features, write_feature_cache
        from molprop_fusion.config import load_config

        config = load_config(args.config)
        spec = config["datasets"][args.dataset]
        _, canonical, _ = load_dataset(
            args.data,
            smiles_column=str(spec["smiles_column"]),
            target_column=str(spec["target_column"]),
            maximum_rows=args.maximum_rows,
        )
        features = prepare_features(canonical, dataset=args.dataset)
        write_feature_cache(args.output, features)
        result = {
            "rows": len(canonical),
            "chemistry_columns": int(features["chemistry"].shape[1]),
            "z3d_columns": int(features["z3d"].shape[1]),
        }
    elif args.command == "benchmark":
        from molprop_fusion.benchmark import run_benchmark

        result = run_benchmark(
            config_path=args.config,
            dataset=args.dataset,
            data_path=args.data,
            output_path=args.output,
            feature_cache=args.feature_cache,
            device=args.device,
            n_jobs=args.n_jobs,
            seeds=args.seed,
            smoke=args.smoke,
        )
    elif args.command == "transfer":
        from molprop_fusion.transfer import run_transfer

        result = run_transfer(
            config_path=args.config,
            data_path=args.data,
            output_path=args.output,
            feature_cache=args.feature_cache,
            device=args.device,
            n_jobs=args.n_jobs,
            seeds=tuple(args.seed) if args.seed else None,
            smoke=args.smoke,
        )
    elif args.command == "chemprop":
        from molprop_fusion.chemprop_baseline import run_chemprop

        result = run_chemprop(
            config_path=args.config,
            dataset=args.dataset,
            data_path=args.data,
            output_path=args.output,
            seed=args.seed,
            accelerator=args.accelerator,
            devices=args.devices,
            smoke=args.smoke,
        )
    elif args.command == "summarize":
        from molprop_fusion.summarize import summarize_file

        result = summarize_file(args.input, args.output)
    elif args.command == "check-config":
        from molprop_fusion.config import load_config

        result = {
            "configs": [
                {"path": str(path), "study_id": load_config(path).get("study_id")}
                for path in args.paths
            ]
        }
    else:
        from molprop_fusion.verification import verify_results

        result = verify_results(args.results_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
