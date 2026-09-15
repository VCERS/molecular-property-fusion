"""Molecular feature construction used by the public fusion workflows."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from molprop_fusion.z3d import (
    RAW_COLUMNS,
    LabelFreeIdentity,
    generate_z3d,
    group_id_for_rows,
)


def impute_raw_z(
    fit_values: np.ndarray, evaluation_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit-only median imputation, with zero for entirely missing columns."""
    fit = np.asarray(fit_values, dtype=np.float64).copy()
    evaluation = np.asarray(evaluation_values, dtype=np.float64).copy()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        medians = np.nanmedian(fit, axis=0)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    fit_missing = ~np.isfinite(fit)
    evaluation_missing = ~np.isfinite(evaluation)
    fit[fit_missing] = np.take(medians, np.where(fit_missing)[1])
    evaluation[evaluation_missing] = np.take(medians, np.where(evaluation_missing)[1])
    return (
        fit.astype(np.float32),
        evaluation.astype(np.float32),
        {
            "all_missing_fit_columns": int(all_missing.sum()),
            "median_sha256": hashlib.sha256(np.ascontiguousarray(medians).tobytes()).hexdigest(),
        },
    )


def canonicalize_smiles(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def chemistry_matrix(smiles_rows: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Return Morgan-2048 plus the frozen 217-descriptor RDKit matrix."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    names = [name for name, _ in Descriptors._descList]
    functions = [function for _, function in Descriptors._descList]
    if len(names) != 217:
        raise ValueError("RDKit descriptor registry differs from the frozen environment")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    output: list[np.ndarray] = []
    for smiles in smiles_rows:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        fingerprint = generator.GetFingerprint(molecule)
        morgan = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, morgan)
        descriptors = np.asarray(
            [float(function(molecule)) for function in functions], dtype=np.float32
        )
        descriptors[~np.isfinite(descriptors)] = np.nan
        output.append(np.concatenate([morgan, descriptors]))
    chemistry = np.asarray(output, dtype=np.float32)
    if chemistry.shape != (len(smiles_rows), 2265):
        raise ValueError("chemistry feature matrix has an unexpected shape")
    return chemistry, names


def z3d_matrix(
    canonical_smiles: Sequence[str], *, dataset: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate the frozen 317-value conformer summary and missingness mask."""
    values = np.full((len(canonical_smiles), len(RAW_COLUMNS)), np.nan, dtype=np.float64)
    masks = np.ones((len(canonical_smiles), len(RAW_COLUMNS)), dtype=bool)
    statuses = [""] * len(canonical_smiles)
    task = {"question": "Molecular property regression", "definition": "Public workflow"}
    groups: dict[str, list[int]] = {}
    for index, smiles in enumerate(canonical_smiles):
        groups.setdefault(smiles, []).append(index)
    for smiles, indices in groups.items():
        source_rows = tuple(str(index + 2) for index in indices)
        row = generate_z3d(
            LabelFreeIdentity(
                dataset=dataset,
                stable_group_id=group_id_for_rows(source_rows),
                canonical_smiles=smiles,
                source_rows=source_rows,
            ),
            task,
        )
        for index in indices:
            values[index] = np.asarray(row["values"], dtype=np.float64)
            masks[index] = np.asarray(row["mask"], dtype=bool)
            statuses[index] = str(row["status"])
    if values.shape != (len(canonical_smiles), len(RAW_COLUMNS)):
        raise ValueError("Z3D feature matrix has an unexpected shape")
    if not np.array_equal(masks, ~np.isfinite(values)):
        raise ValueError("Z3D values and missingness mask disagree")
    return values, masks, statuses
