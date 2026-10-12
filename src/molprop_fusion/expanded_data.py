"""Pinned expanded CSV adapters and replay of published source-index assignments."""

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from molprop_fusion.io import sha256_file as file_hash

RDLogger.DisableLog("rdApp.*")


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.Uncharger().uncharge(
            rdMolStandardize.LargestFragmentChooser().choose(cleaned)
        )
        key = Chem.MolToInchiKey(cleaned)
        parent_key = Chem.MolToInchiKey(parent)
        if not key or not parent_key:
            return None
        return {
            "canonical_smiles": Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True),
            "exact_inchikey": key,
            "parent_connectivity_key": parent_key[:14],
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
            "atom_count": cleaned.GetNumAtoms(),
            "fragment_count": len(Chem.GetMolFrags(cleaned)),
        }
    except (ValueError, RuntimeError):
        return None


def source_rows(root, spec):
    path = Path(root) / spec["file"]
    if file_hash(path) != spec["sha256"]:
        raise RuntimeError("public source hash differs: " + spec["file"])
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=spec["delimiter"])
        if not {spec["smiles_field"], *spec["endpoints"]} <= set(reader.fieldnames or []):
            raise ValueError("source fields differ")
        for i, row in enumerate(reader):
            if spec.get("species") and row.get("Species") != spec["species"]:
                continue
            # Only agreed endpoints are converted. Auxiliary fields never become inputs.
            labels = {}
            for field in spec["endpoints"]:
                try:
                    labels[field] = float(row[field])
                except (TypeError, ValueError):
                    labels[field] = float("nan")
            yield i, row[spec["smiles_field"]], labels, row.get("mol_id", "")


def chemistry_matrix(smiles):
    names = [n for n, _ in Descriptors._descList]
    registry = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    if (
        len(names) != 217
        or registry != "7488424ba33defb325a509d6ff50a39fe131f6135839c54bed162051df3fbdb1"
    ):
        raise RuntimeError("descriptor registry differs")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    out = np.empty((len(smiles), 2265), np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), out[i, :2048])
        out[i, 2048:] = [float(fn(mol)) for _, fn in Descriptors._descList]
    out[~np.isfinite(out)] = np.nan
    return out


def assign_groups(groups, fraction, seed):
    """Assign whole groups nearest to a fraction, preserving empty scaffold as a group."""
    count = Counter(groups)
    ordered = sorted(count, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).digest())
    goal = len(groups) * fraction
    selected = set()
    n = 0
    for g in ordered:
        if abs(n + count[g] - goal) < abs(n - goal):
            selected.add(g)
            n += count[g]
    if not selected or len(selected) == len(count):
        raise ValueError("structural split infeasible")
    return selected


def structure_clusters(rows, distance):
    from rdkit.ML.Cluster import Butina

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(r["canonical_smiles"])) for r in rows
    ]
    # Bounded to the chosen population, never all 134k squared distances.
    distances = []
    for i in range(1, len(rows)):
        distances.extend(
            1 - x for x in DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        )
    clusters = Butina.ClusterData(distances, len(rows), distance, isDistData=True, reordering=False)
    result = [""] * len(rows)
    for cluster in clusters:
        key = min(rows[i]["parent_connectivity_key"] for i in cluster)
        for i in cluster:
            result[i] = key
    return result


def official_test_keys(source, manifest, dataset):
    spec = manifest["admet_archive"]
    path = Path(source) / spec["file"]
    if file_hash(path) != spec["sha256"]:
        raise RuntimeError("official archive differs")
    keys = {}
    archive_count = {}
    with zipfile.ZipFile(path) as z:
        for role in ["test", "train_val"]:
            name = f"admet_group/{dataset}/{role}.csv"
            with z.open(name) as raw:
                rows = list(csv.DictReader(io.TextIOWrapper(raw)))
            archive_count[role] = len(rows)
            for r in rows:
                value = identity(r["Drug"])
                if value:
                    keys.setdefault(value["canonical_smiles"], set()).add(role)
    return keys, archive_count


def uncharacterized_ids(source, manifest):
    spec = manifest["qm9_uncharacterized"]
    path = Path(source) / spec["file"]
    if file_hash(path) != spec["sha256"]:
        raise RuntimeError("QM9 consistency list differs")
    return {
        int(m.group(1))
        for line in path.read_text().splitlines()
        if (m := re.match(r"^\s*(\d+)\s", line))
    }


def replay(source, dataset, spec, partition_path, runs=(1, 2, 3, 4, 5)):
    """Replay retained order/exclusions from integer-only study assignments.

    Omitted source rows stay omitted, including historical-overlap exclusions.
    Source values are read only in memory; returned prepared rows contain no labels.
    """
    if file_hash(partition_path) != spec["partition_file_sha256"]:
        raise ValueError("partition manifest hash differs")
    with Path(partition_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_index", "geometry_source_index", *[f"split_run_{r}" for r in runs]}
        if set(reader.fieldnames or []) != required:
            raise ValueError("partition fields differ")
        selected = list(reader)
    indices = [int(r["source_index"]) for r in selected]
    if len(indices) != len(set(indices)) or not indices or min(indices) < 0:
        raise ValueError("invalid or repeated source index")
    requested = set(indices)
    # These upstream public labels establish eligibility, never training selection.
    available = {i: (s, labels) for i, s, labels, _ in source_rows(source, spec) if i in requested}
    if set(available) != requested:
        raise ValueError("selected source rows are missing or species differs")
    rows = []
    memberships = {str(run): {} for run in runs}
    geometry = {}
    for record, index in zip(selected, indices, strict=True):
        smiles, labels = available[index]
        value = identity(smiles)
        if value is None or value["atom_count"] > 115:
            raise ValueError("retained structure no longer supported")
        if not all(np.isfinite(v) for v in labels.values()):
            raise ValueError("retained endpoint eligibility differs")
        seed_index = int(record["geometry_source_index"])
        canonical = value["canonical_smiles"]
        if canonical in geometry and geometry[canonical] != seed_index:
            raise ValueError("duplicate geometry seeds differ")
        geometry[canonical] = seed_index
        row_id = f"{dataset}:{index}"
        roles = [record[f"split_run_{run}"] for run in runs]
        if any(role not in {"train", "validation", "test"} for role in roles):
            raise ValueError("unknown partition role")
        if len({role == "test" for role in roles}) != 1:
            raise ValueError("test population must be fixed")
        rows.append(
            {
                **value,
                "source_index": index,
                "geometry_source_index": seed_index,
                "row_id": row_id,
                "eligible_endpoints": list(spec["endpoints"]),
                "fixed_test": roles[0] == "test",
            }
        )
        for run, role in zip(runs, roles, strict=True):
            memberships[str(run)][row_id] = role
    minima = {}
    for row in rows:
        s = row["canonical_smiles"]
        minima[s] = min(minima.get(s, row["source_index"]), row["source_index"])
    if geometry != minima:
        raise ValueError("geometry source index must equal minimum retained canonical source index")
    for assignment in memberships.values():
        parents = {
            role: {r["parent_connectivity_key"] for r in rows if assignment[r["row_id"]] == role}
            for role in ["train", "validation", "test"]
        }
        if any(not values for values in parents.values()):
            raise ValueError("empty partition")
        if any(
            parents[a] & parents[b]
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
        ):
            raise ValueError("parent overlap between partitions")
    return {
        "dataset": dataset,
        "rows": rows,
        "memberships": memberships,
        "source_hash": spec["sha256"],
        "partition_hash": spec["partition_file_sha256"],
    }
