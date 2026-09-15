"""Deterministic, label-free ETKDGv3 source generation for the Z3D pilot."""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import numpy as np

DESCRIPTOR_NAMES = (
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "RadiusOfGyration",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "PBF",
)
SUMMARY_NAMES = ("mean", "std_ddof0", "min", "max")
FEATURE_FAMILIES = (
    "Donor",
    "Acceptor",
    "Aromatic",
    "Hydrophobe",
    "PosIonizable",
    "NegIonizable",
)
FAMILY_PAIRS = tuple(itertools.combinations_with_replacement(FEATURE_FAMILIES, 2))
DISTANCE_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, math.inf)
HISTOGRAM_SUMMARIES = ("mean", "max")
PROTOCOL_ID = "new-chemistry-information-bayesian-fusion-v0"


def raw_column_names() -> tuple[str, ...]:
    columns = []
    for descriptor in DESCRIPTOR_NAMES:
        columns.extend(f"shape.{descriptor}.{summary}" for summary in SUMMARY_NAMES)
    for left, right in FAMILY_PAIRS:
        pair = f"{left}--{right}"
        for bin_index in range(len(DISTANCE_BINS) - 1):
            columns.extend(
                f"pharmacophore.{pair}.bin_{bin_index}.{summary}" for summary in HISTOGRAM_SUMMARIES
            )
    columns.extend(("status.conformer_count", "status.retry_used"))
    columns.extend(
        f"pharmacophore.{left}--{right}.minimum_distance_median" for left, right in FAMILY_PAIRS
    )
    columns.extend(
        (
            "pharmacophore.Donor--Acceptor.contact_fraction_lt_3p5",
            "diversity.heavy_atom_best_rms_median",
        )
    )
    if len(columns) != 317 or len(columns) != len(set(columns)):
        raise AssertionError(f"Unexpected Z3D schema size: {len(columns)}")
    return tuple(columns)


RAW_COLUMNS = raw_column_names()


@dataclass(frozen=True)
class LabelFreeIdentity:
    dataset: str
    stable_group_id: str
    canonical_smiles: str
    source_rows: tuple[str, ...]


def stable_id_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def group_id_for_rows(values: Iterable[str]) -> str:
    return min((str(value) for value in values), key=stable_id_key)


def source_seed(dataset: str, stable_group_id: str) -> int:
    material = f"{PROTOCOL_ID}|z3d|{dataset}|{stable_group_id}".encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)
    return value or 1


def retry_seed(seed: int) -> int:
    value = (seed ^ 0x5A17) % (2**31 - 1)
    return value or 1


def pilot_rank(dataset: str, stable_group_id: str) -> bytes:
    material = f"{PROTOCOL_ID}|pilot|{dataset}|{stable_group_id}".encode()
    return hashlib.sha256(material).digest()


def _rdkit() -> dict[str, Any]:
    try:
        from rdkit import Chem, RDConfig
        from rdkit import __version__ as rdkit_version
        from rdkit.Chem import (
            AllChem,
            ChemicalFeatures,
            Descriptors3D,
            Lipinski,
            rdMolAlign,
        )
    except ImportError as error:  # pragma: no cover - exercised on experiment host
        raise RuntimeError("RDKit is required for Z3D source generation") from error
    return {
        "Chem": Chem,
        "RDConfig": RDConfig,
        "rdkit_version": rdkit_version,
        "AllChem": AllChem,
        "ChemicalFeatures": ChemicalFeatures,
        "Descriptors3D": Descriptors3D,
        "Lipinski": Lipinski,
        "rdMolAlign": rdMolAlign,
    }


def label_free_stress_values(smiles: str) -> tuple[int, int]:
    rdkit = _rdkit()
    mol = rdkit["Chem"].MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return (-1, -1)
    return (int(mol.GetNumHeavyAtoms()), int(rdkit["Lipinski"].NumRotatableBonds(mol)))


def choose_pilot(identities: Sequence[LabelFreeIdentity], n: int = 64) -> list[LabelFreeIdentity]:
    if not identities:
        raise ValueError("Cannot select a pilot from an empty identity list")
    dataset = identities[0].dataset
    if any(row.dataset != dataset for row in identities):
        raise ValueError("Pilot selection must be dataset-local")
    annotated = [(row, *label_free_stress_values(row.canonical_smiles)) for row in identities]
    heavy = sorted(annotated, key=lambda item: (-item[1], stable_id_key(item[0].stable_group_id)))
    rotors = sorted(annotated, key=lambda item: (-item[2], stable_id_key(item[0].stable_group_id)))
    hashed = sorted(
        annotated,
        key=lambda item: (
            pilot_rank(dataset, item[0].stable_group_id),
            stable_id_key(item[0].stable_group_id),
        ),
    )
    selected: list[LabelFreeIdentity] = []
    seen: set[str] = set()
    for sequence in (heavy[:8], rotors[:8], hashed):
        for row, _, _ in sequence:
            if row.canonical_smiles in seen:
                continue
            selected.append(row)
            seen.add(row.canonical_smiles)
            if len(selected) == min(n, len(identities)):
                return selected
    raise AssertionError("Pilot selection did not reach the requested size")


def _embed_parameters(*, seed: int, retry: bool) -> Any:
    rdkit = _rdkit()
    params = rdkit["AllChem"].ETKDGv3()
    params.randomSeed = retry_seed(seed) if retry else seed
    params.enforceChirality = True
    params.useRandomCoords = retry
    params.pruneRmsThresh = 0.5
    params.onlyHeavyAtomsForRMS = True
    params.useSymmetryForPruning = True
    params.numThreads = 1
    params.maxIterations = 2000 if retry else 1000
    return params


def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=0)),
        float(np.min(values)),
        float(np.max(values)),
    )


def _pair_distances(
    positions: Mapping[str, Sequence[np.ndarray]], left: str, right: str
) -> list[float]:
    left_positions = positions[left]
    right_positions = positions[right]
    distances = []
    for left_index, left_position in enumerate(left_positions):
        for right_index, right_position in enumerate(right_positions):
            if left == right and left_index >= right_index:
                continue
            distances.append(float(np.linalg.norm(left_position - right_position)))
    return distances


def _feature_positions(factory: Any, mol: Any, conf_id: int) -> dict[str, list[np.ndarray]]:
    positions = {family: [] for family in FEATURE_FAMILIES}
    for feature in factory.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family in positions:
            positions[family].append(np.asarray(feature.GetPos(conf_id), dtype=np.float64))
    return positions


def _source_hash(values: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    rounded = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)
    if rounded == Decimal("-0.000"):
        rounded = Decimal("0.000")
    return f"{rounded:.3f}"


def render_dossier(
    *,
    canonical_smiles: str,
    task: Mapping[str, str],
    status: str,
    values: np.ndarray,
) -> str:
    index = {name: position for position, name in enumerate(RAW_COLUMNS)}

    def get(name: str) -> float | None:
        value = float(values[index[name]])
        return value if math.isfinite(value) else None

    if status == "ok":
        conformers = str(int(get("status.conformer_count") or 0))
        retry = "yes" if get("status.retry_used") == 1.0 else "no"
    else:
        conformers = "NA"
        retry = "NA"

    def shape_line(label: str, descriptor: str, unit: str = "") -> str:
        mean = format_float(get(f"shape.{descriptor}.mean"))
        minimum = format_float(get(f"shape.{descriptor}.min"))
        maximum = format_float(get(f"shape.{descriptor}.max"))
        return f"{label}{unit}: {mean}; {minimum}..{maximum}"

    pair_lines = []
    for left, right in FAMILY_PAIRS:
        name = f"pharmacophore.{left}--{right}.minimum_distance_median"
        pair_lines.append(f"{left}--{right}: {format_float(get(name))}")
    source_status = "OK" if status == "ok" else "SOURCE_MISSING"
    lines = [
        "<MOLECULE>",
        f"SMILES: {canonical_smiles}",
        "</MOLECULE>",
        '<NEW_CHEMISTRY_SOURCE type="computed_3d_etkdgv3">',
        f"Source status: {source_status}",
        f"Conformers retained: {conformers}",
        f"Retry used: {retry}",
        shape_line("Radius of gyration mean/range", "RadiusOfGyration", " (angstrom)"),
        shape_line("NPR1 mean/range", "NPR1"),
        shape_line("NPR2 mean/range", "NPR2"),
        shape_line("Asphericity mean/range", "Asphericity"),
        shape_line("Plane-of-best-fit mean/range", "PBF"),
        "Conformer heavy-atom RMS diversity median (angstrom): "
        + format_float(get("diversity.heavy_atom_best_rms_median")),
        "Intramolecular donor-acceptor contact fraction below 3.5 angstrom: "
        + format_float(get("pharmacophore.Donor--Acceptor.contact_fraction_lt_3p5")),
        "3D pharmacophore minimum-distance medians (angstrom):",
        *pair_lines,
        "</NEW_CHEMISTRY_SOURCE>",
        "<TASK>",
        task["question"],
        task["definition"],
        "</TASK>",
        "<OUTPUT>",
        "Return one signed decimal number with exactly two digits after the decimal point.",
        "</OUTPUT>",
        "<ANSWER>",
    ]
    return "\n".join(lines)


def generate_z3d(identity: LabelFreeIdentity, task: Mapping[str, str]) -> dict[str, Any]:
    rdkit = _rdkit()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    status = "ok"
    exception = None
    retry_used = False
    values = np.full(len(RAW_COLUMNS), np.nan, dtype=np.float64)
    seed = source_seed(identity.dataset, identity.stable_group_id)
    conformer_count = 0
    try:
        mol = rdkit["Chem"].MolFromSmiles(identity.canonical_smiles, sanitize=True)
        if mol is None:
            status = "sanitize_failed"
        elif len(rdkit["Chem"].GetMolFrags(mol)) != 1:
            status = "multiple_fragments"
        elif mol.GetNumHeavyAtoms() == 0:
            status = "no_heavy_atom"
        elif sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) != 0:
            status = "radical_unsupported"
        else:
            mol_h = rdkit["Chem"].AddHs(mol)
            conf_ids = list(
                rdkit["AllChem"].EmbedMultipleConfs(
                    mol_h,
                    numConfs=16,
                    params=_embed_parameters(seed=seed, retry=False),
                )
            )
            if not conf_ids:
                retry_used = True
                conf_ids = list(
                    rdkit["AllChem"].EmbedMultipleConfs(
                        mol_h,
                        numConfs=16,
                        params=_embed_parameters(seed=seed, retry=True),
                    )
                )
            if not conf_ids:
                status = "embed_failed"
            else:
                conformer_count = len(conf_ids)
                fdef_path = Path(rdkit["RDConfig"].RDDataDir) / "BaseFeatures.fdef"
                factory = rdkit["ChemicalFeatures"].BuildFeatureFactory(str(fdef_path))
                cursor = 0
                for descriptor_name in DESCRIPTOR_NAMES:
                    function = getattr(rdkit["Descriptors3D"], descriptor_name)
                    descriptor_values = np.asarray(
                        [float(function(mol_h, confId=conf_id)) for conf_id in conf_ids],
                        dtype=np.float64,
                    )
                    values[cursor : cursor + 4] = _summary(descriptor_values)
                    cursor += 4

                histograms = {
                    pair: np.zeros((len(conf_ids), len(DISTANCE_BINS) - 1), dtype=np.float64)
                    for pair in FAMILY_PAIRS
                }
                minima: dict[tuple[str, str], list[float]] = {pair: [] for pair in FAMILY_PAIRS}
                contact_indicators = []
                for conf_index, conf_id in enumerate(conf_ids):
                    positions = _feature_positions(factory, mol_h, conf_id)
                    for pair in FAMILY_PAIRS:
                        distances = _pair_distances(positions, *pair)
                        if distances:
                            minima[pair].append(min(distances))
                            histograms[pair][conf_index] = np.histogram(
                                distances, bins=DISTANCE_BINS
                            )[0]
                    donor_acceptor = _pair_distances(positions, "Donor", "Acceptor")
                    contact_indicators.append(
                        1.0 if donor_acceptor and min(donor_acceptor) < 3.5 else 0.0
                    )
                for pair in FAMILY_PAIRS:
                    for bin_index in range(len(DISTANCE_BINS) - 1):
                        column = histograms[pair][:, bin_index]
                        values[cursor] = float(np.mean(column))
                        values[cursor + 1] = float(np.max(column))
                        cursor += 2
                values[cursor] = float(conformer_count)
                values[cursor + 1] = float(retry_used)
                cursor += 2
                for pair in FAMILY_PAIRS:
                    if minima[pair]:
                        values[cursor] = float(np.median(minima[pair]))
                    cursor += 1
                values[cursor] = float(np.mean(contact_indicators))
                cursor += 1
                if conformer_count < 2:
                    values[cursor] = 0.0
                else:
                    heavy = rdkit["Chem"].RemoveHs(mol_h)
                    rms_values = [
                        float(rdkit["rdMolAlign"].GetBestRMS(heavy, heavy, prbId=left, refId=right))
                        for left, right in itertools.combinations(conf_ids, 2)
                    ]
                    values[cursor] = float(np.median(rms_values))
                cursor += 1
                if cursor != len(RAW_COLUMNS):
                    raise AssertionError(f"Z3D cursor mismatch: {cursor}")
                required = values[: 40 + 252 + 2]
                if not np.all(np.isfinite(required)):
                    status = "nonfinite_required_descriptor"
    except Exception as error:  # noqa: BLE001 - every source failure is ledgered
        status = "exception"
        exception = {"type": type(error).__name__, "message": str(error)}

    if status != "ok":
        values[:] = np.nan
    mask = ~np.isfinite(values)
    dossier = render_dossier(
        canonical_smiles=identity.canonical_smiles,
        task=task,
        status=status,
        values=values,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "dataset": identity.dataset,
        "stable_group_id": identity.stable_group_id,
        "source_rows": list(identity.source_rows),
        "canonical_smiles": identity.canonical_smiles,
        "status": status,
        "exception": exception,
        "seed_u31": seed,
        "retry_used": retry_used,
        "conformer_count": conformer_count,
        "values": values,
        "mask": mask,
        "source_sha256": _source_hash(values, mask),
        "text": dossier,
        "text_sha256": hashlib.sha256(dossier.encode()).hexdigest(),
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "peak_rss_kb_upper_bound": int(usage.ru_maxrss),
        "pid": os.getpid(),
    }
