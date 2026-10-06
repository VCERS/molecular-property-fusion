"""Custom CSV train/predict commands; row outputs are ordinary user artifacts."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from molprop_fusion.estimator import MolecularPropertyFusionRegressor, load_model
from molprop_fusion.features import canonicalize_smiles


def _read(path: Path, required: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or not required <= set(fields):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV has duplicate column names")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError("CSV contains a row with the wrong number of fields")
        return fields, rows


def run_train(args: Any) -> dict[str, Any]:
    _, rows = _read(args.data, {args.smiles_column, args.target_column})
    if args.output.exists():
        raise FileExistsError("model output already exists; choose a new directory")
    try:
        targets = [float(row[args.target_column]) for row in rows]
    except ValueError as error:
        raise ValueError(f"column {args.target_column!r} must contain numeric targets") from error
    model = MolecularPropertyFusionRegressor(
        mode=args.mode,
        device=args.device,
        random_state=args.random_state,
        feature_seed=args.feature_seed,
        n_jobs=args.n_jobs,
        max_atoms=args.max_atoms,
        validation_only=args.validation_only,
        verbose=True,
    ).fit([row[args.smiles_column] for row in rows], targets)
    model.save(args.output)
    return {"model": str(args.output), "mode": model.mode, "training": model.training_summary_}


def run_predict(args: Any) -> dict[str, Any]:
    fields, rows = _read(args.data, {args.smiles_column})
    added = [
        "prediction",
        "prediction_status",
        "prediction_error",
        "geometry_status",
        "geometry_missing_count",
        "conformer_count",
    ]
    if set(added) & set(fields):
        raise ValueError("input CSV already contains prediction/status output columns; rename them")
    model = load_model(args.model, device=args.device)
    model.verbose = True
    indices, canonical = [], []
    for index, row in enumerate(rows):
        row.update(dict.fromkeys(added, ""))
        try:
            smiles = canonicalize_smiles(row[args.smiles_column])
            model._capacity([smiles])
        except ValueError as error:
            if args.invalid_policy == "raise":
                raise ValueError(f"CSV data row {index + 1}: {error}") from error
            row["prediction_status"] = "error"
            row["prediction_error"] = str(error)
            continue
        indices.append(index)
        canonical.append(smiles)
    if canonical:
        predictions, statuses = model.predict(canonical, return_status=True)
        for index, prediction, status in zip(indices, predictions, statuses, strict=True):
            rows[index].update(status)
            rows[index]["prediction"] = format(float(prediction), ".17g")
            rows[index]["prediction_status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=args.output.parent, prefix=".predictions-")
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fields, *added])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "output": str(args.output),
        "rows": len(rows),
        "predicted": len(indices),
        "errors": len(rows) - len(indices),
    }
