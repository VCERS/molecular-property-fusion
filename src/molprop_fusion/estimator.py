"""Small fitted estimator for custom molecular regression tasks."""

from __future__ import annotations

import importlib
import json
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from molprop_fusion.features import canonicalize_smiles, chemistry_matrix
from molprop_fusion.fusion import fit_meta_model, grouped_folds
from molprop_fusion.z3d import RAW_COLUMNS, LabelFreeIdentity, generate_z3d

FEATURE_POLICY = "canonical-etkdgv3-v1"
DEFAULT_MAX_ATOMS = 115
FAMILIES = ("xgboost", "catboost", "lightgbm", "extratrees")
BUNDLE_SCHEMA = 1


def _dependencies(mode: str) -> dict[str, str]:
    modules = ["rdkit", "xgboost", "catboost", "lightgbm"]
    if mode == "full":
        modules.append("torch")
    try:
        for module in modules:
            importlib.import_module(module)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"Missing or unusable dependency {module} ({type(error).__name__}). "
            f"Install molecular-property-fusion[{mode}] "
            "in Python 3.11; on macOS tree libraries may also require libomp."
        ) from error
    return {name: version(name) for name in ["numpy", "scikit-learn", "joblib", *modules]}


def _smiles_list(smiles: Sequence[str]) -> list[str]:
    if isinstance(smiles, (str, bytes)):
        raise ValueError("supply a sequence of SMILES, for example ['CCO']")
    try:
        return list(smiles)
    except TypeError as error:
        raise ValueError("supply a sequence of SMILES") from error


def canonicalize_user_smiles(smiles: str) -> str:
    """Validate custom inputs without changing historical paper canonicalization."""
    from rdkit import Chem

    canonical = canonicalize_smiles(smiles)
    if not any(atom.GetAtomicNum() > 1 for atom in Chem.MolFromSmiles(canonical).GetAtoms()):
        raise ValueError(
            f"unsupported molecule {smiles!r}: no usable heavy atom; "
            "supply a structure containing at least one real non-hydrogen atom "
            "(atomic number greater than 1)"
        )
    return canonical


def _canonical(smiles: Sequence[str]) -> list[str]:
    result = []
    for index, value in enumerate(smiles):
        try:
            result.append(canonicalize_user_smiles(value))
        except ValueError as error:
            raise ValueError(f"row {index + 1}: {error}") from error
    return result


def molecule_seed(smiles: str, feature_seed: int) -> int:
    """Canonical identity, explicit seed, and version; no row or dataset identity."""
    import hashlib

    material = f"{FEATURE_POLICY}|{feature_seed}|{smiles}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1) or 1


class MedianImputer:
    """Fold-local medians, zero for entirely missing columns."""

    def fit(self, values: np.ndarray) -> MedianImputer:
        values = np.where(np.isfinite(values), values, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            self.medians = np.nanmedian(values, axis=0)
        self.medians[~np.isfinite(self.medians)] = 0.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(values), values, self.medians).astype(np.float32)


def _meta_state(model: Any) -> dict[str, Any]:
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "alpha": float(ridge.alpha),
    }


def _meta_predict(state: dict[str, Any], columns: np.ndarray) -> np.ndarray:
    return ((columns - np.asarray(state["mean"])) / np.asarray(state["scale"])) @ np.asarray(
        state["coefficients"]
    ) + state["intercept"]


class MolecularPropertyFusionRegressor:
    """Four-tree Ridge fusion, optionally with CoMPT. See docs/api.md.

    Model defaults follow the AqSolDB/expanded-benchmark configuration.
    User geometry seeds remain molecule-based rather than dataset/row-based.
    ``validation_only=True`` uses deliberately tiny fits for software checks.
    """

    def __init__(
        self,
        *,
        mode: str = "full",
        device: str = "auto",
        random_state: int = 42,
        feature_seed: int = 42,
        n_jobs: int = 1,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        validation_only: bool = False,
        verbose: bool = False,
    ):
        if mode not in ("full", "tabular"):
            raise ValueError("mode must be 'full' or 'tabular'")
        if n_jobs < 1 or max_atoms < 1:
            raise ValueError("n_jobs and max_atoms must be positive")
        if not 0 <= random_state < 2**31 - 3:
            raise ValueError("random_state must be between 0 and 2147483643")
        self.mode, self.device = mode, device
        self.random_state, self.feature_seed = int(random_state), int(feature_seed)
        self.n_jobs, self.max_atoms = int(n_jobs), int(max_atoms)
        self.validation_only, self.verbose = bool(validation_only), bool(verbose)
        self._feature_cache: dict[
            tuple[str, int, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]
        ] = {}
        self._fitted = False

    def _progress(self, message: str) -> None:
        if self.verbose:
            import sys

            print(message, file=sys.stderr, flush=True)

    def _capacity(self, canonical: Sequence[str]) -> None:
        if self.mode != "full":
            return
        from rdkit import Chem

        for index, smiles in enumerate(canonical):
            atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
            if atoms > self.max_atoms:
                raise ValueError(
                    f"row {index + 1}: {atoms} atoms exceed trained graph capacity {self.max_atoms}; "
                    "retrain with a larger max_atoms or use a separately trained tabular model. "
                    "Inference cannot resize the padding-sensitive CoMPT readout."
                )

    def _features(
        self, canonical: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        # Cache belongs to one fixed feature seed/version and canonical identity. Assembly
        # follows the query's order, including duplicates; dimensions alone are never a key.
        def key(smiles: str) -> tuple[str, int, str]:
            return FEATURE_POLICY, self.feature_seed, smiles

        unique = list(dict.fromkeys(s for s in canonical if key(s) not in self._feature_cache))
        if unique:
            chemistry, names = chemistry_matrix(unique)
            if hasattr(self, "descriptor_names_") and self.descriptor_names_ != names:
                raise ValueError("RDKit descriptor names/order differ from fitted schema")
            self.descriptor_names_ = names
            for index, smiles in enumerate(unique):
                if index % 25 == 0:
                    self._progress(f"Generating molecule features {index + 1}/{len(unique)}")
                row = generate_z3d(
                    LabelFreeIdentity("user", smiles, smiles, ()),
                    {"question": "Molecular property regression", "definition": "User toolkit"},
                    seed=molecule_seed(smiles, self.feature_seed),
                )
                status = {
                    "geometry_status": row["status"],
                    "geometry_missing_count": int(row["mask"].sum()),
                    "conformer_count": row["conformer_count"],
                }
                self._feature_cache[key(smiles)] = (chemistry[index], row["values"], status)
        return (
            np.stack([self._feature_cache[key(s)][0] for s in canonical]),
            np.stack([self._feature_cache[key(s)][1] for s in canonical]),
            [dict(self._feature_cache[key(s)][2]) for s in canonical],
        )

    @staticmethod
    def _matrix(chemistry: np.ndarray, z3d: np.ndarray, imputer: MedianImputer) -> np.ndarray:
        # Also impute rare nonfinite 2D descriptors for ExtraTrees on supported sklearn.
        return np.concatenate(
            [
                imputer.transform(np.concatenate([chemistry, z3d], axis=1)),
                (~np.isfinite(z3d)).astype(np.float32),
            ],
            axis=1,
        )

    def fit(
        self, smiles: Sequence[str], targets: Sequence[float]
    ) -> MolecularPropertyFusionRegressor:
        raw = _smiles_list(smiles)
        target = np.asarray(targets, dtype=np.float64)
        if target.ndim != 1 or len(target) != len(raw):
            raise ValueError("targets must be a one-dimensional column matching SMILES length")
        if not np.isfinite(target).all():
            raise ValueError("targets must all be finite numbers; remove missing/nonfinite labels")
        self.dependency_versions_ = _dependencies(self.mode)
        canonical = _canonical(raw)
        if len(set(canonical)) < 3:
            raise ValueError("at least three distinct canonical molecular groups are required")
        folds = grouped_folds(canonical, self.random_state)
        self._capacity(canonical)
        if any(np.std(target[fit]) == 0 for fit, _ in folds):
            raise ValueError(
                "OOF training targets have zero variation; add more labeled molecular groups"
            )
        if self.mode == "full":
            # Validate every inner split before conformer generation and tree fitting.
            from sklearn.model_selection import GroupShuffleSplit

            groups = np.asarray(canonical)
            for fold, fit in [
                *((i, f) for i, (f, _) in enumerate(folds)),
                (0, np.arange(len(raw))),
            ]:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=0.1, random_state=self.random_state + fold
                )
                inner, _ = next(splitter.split(target[fit], groups=groups[fit]))
                if not np.std(target[fit][inner]) > 0:
                    raise ValueError(
                        "CoMPT inner training targets have zero variation; add more labeled molecular groups"
                    )
        # Once fitting starts, a failed refit must not leave a usable partial model.
        self._fitted = False
        self._feature_cache.clear()
        if hasattr(self, "descriptor_names_"):
            del self.descriptor_names_
        started = time.perf_counter()
        chemistry, z3d, statuses = self._features(canonical)
        config = json.loads(files("molprop_fusion").joinpath("user_defaults.json").read_text())
        if self.validation_only:
            config["models"]["graph"].update(hidden_width=16, layers=1)
            config["graph_training"].update(max_epochs=1, patience=1, warmup_epochs=1, batch_size=8)
        from molprop_fusion.benchmark import _model

        oof = np.empty((len(raw), 5 if self.mode == "full" else 4), dtype=np.float64)
        preprocessing = []
        for fold, (fit, held) in enumerate(folds):
            self._progress(f"Fitting OOF fold {fold + 1}/3")
            imputer = MedianImputer().fit(np.concatenate([chemistry[fit], z3d[fit]], axis=1))
            preprocessing.append(
                {
                    "fit_rows": len(fit),
                    "held_rows": len(held),
                    "all_missing_columns": int(
                        (~np.isfinite(np.concatenate([chemistry[fit], z3d[fit]], axis=1)))
                        .all(axis=0)
                        .sum()
                    ),
                }
            )
            fit_x = self._matrix(chemistry[fit], z3d[fit], imputer)
            held_x = self._matrix(chemistry[held], z3d[held], imputer)
            for column, family in enumerate(FAMILIES):
                tree = _model(
                    config,
                    family,
                    "user",
                    self.random_state + fold,
                    self.n_jobs,
                    smoke=self.validation_only,
                )
                tree.fit(fit_x, target[fit])
                oof[held, column] = tree.predict(held_x)
        self.imputer_ = MedianImputer().fit(np.concatenate([chemistry, z3d], axis=1))
        train_x = self._matrix(chemistry, z3d, self.imputer_)
        self.trees_ = []
        for family in FAMILIES:
            self._progress(f"Fitting final {family}")
            tree = _model(
                config, family, "user", self.random_state, self.n_jobs, smoke=self.validation_only
            )
            tree.fit(train_x, target)
            self.trees_.append(tree)
        epochs = []
        if self.mode == "full":
            from molprop_fusion.compt_graph import fit_graph_regressor

            graph = config["models"]["graph"]
            options = {
                "hidden_width": graph["hidden_width"],
                "layers": graph["layers"],
                "heads": graph["attention_heads"],
                "dropout": graph["dropout"],
                "max_atoms": self.max_atoms,
                "atom_width": 115,
                "bond_width": 13,
            }
            training = {
                **config["graph_training"],
                "device": self.device,
                "warmup_factor": float(config["datasets"]["user"]["warmup_factor"]),
            }
            groups = np.asarray(canonical, dtype=object)
            for fold, (fit, held) in enumerate(folds):
                self._progress(f"Fitting CoMPT OOF fold {fold + 1}/3")
                result = fit_graph_regressor(
                    train_smiles=[canonical[i] for i in fit],
                    train_target=target[fit],
                    train_groups=groups[fit],
                    evaluation_smiles=[canonical[i] for i in held],
                    seed=self.random_state + fold,
                    model_options=options,
                    training_options=training,
                )
                oof[held, 4] = result.predictions
                epochs.append(result.best_epoch)
            self._progress("Fitting final CoMPT")
            result = fit_graph_regressor(
                train_smiles=canonical,
                train_target=target,
                train_groups=groups,
                evaluation_smiles=[],
                seed=self.random_state,
                model_options=options,
                training_options=training,
            )
            self.graph_ = result.predictor
            epochs.append(result.best_epoch)
        self.meta_ = {"T": _meta_state(fit_meta_model(oof[:, :4], target))}
        if self.mode == "full":
            self.meta_["T_plus_G"] = _meta_state(fit_meta_model(oof, target))
        self.training_summary_ = {
            "rows": len(raw),
            "canonical_groups": len(set(canonical)),
            "oof_columns": oof.shape[1],
            "fold_preprocessing": preprocessing,
            "geometry_status_counts": dict(Counter(s["geometry_status"] for s in statuses)),
            "tree_fits": 16,
            "graph_fits": len(epochs),
            "graph_best_epochs": epochs,
            "validation_only": self.validation_only,
            "wall_seconds": time.perf_counter() - started,
        }
        self._inference_config = (self.mode, self.feature_seed, self.max_atoms)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("model is not fitted; call fit() or load_model() first")
        if self._inference_config != (self.mode, self.feature_seed, self.max_atoms):
            raise ValueError(
                "fitted mode, feature_seed, and max_atoms are fixed; retrain a new model"
            )

    def predict(self, smiles: Sequence[str], *, return_status: bool = False) -> Any:
        """Predict T_plus_G in full mode or T in tabular mode, in input row order."""
        self._check_fitted()
        canonical = _canonical(_smiles_list(smiles))
        if not canonical:
            empty = np.empty(0, dtype=np.float64)
            return (empty, []) if return_status else empty
        self._capacity(canonical)
        unique = list(dict.fromkeys(canonical))
        chemistry, z3d, statuses = self._features(unique)
        matrix = self._matrix(chemistry, z3d, self.imputer_)
        columns = np.column_stack([tree.predict(matrix) for tree in self.trees_])
        if self.mode == "full":
            columns = np.column_stack([columns, self.graph_.predict(unique, device=self.device)])
        output = _meta_predict(self.meta_["T_plus_G" if self.mode == "full" else "T"], columns)
        if not np.isfinite(output).all():
            raise ValueError("model produced a non-finite prediction")
        positions = {s: i for i, s in enumerate(unique)}
        order = [positions[s] for s in canonical]
        output = output[order]
        ordered_statuses = [dict(statuses[i]) for i in order]
        return (output, ordered_statuses) if return_status else output

    def save(self, path: str | Path) -> None:
        """Write a new bundle directory. Load only bundles from trusted sources."""
        import joblib

        self._check_fitted()
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(
                f"bundle destination already exists: {destination}; choose a new directory"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write elsewhere first so an interrupted save does not look like a model bundle.
        import tempfile

        with tempfile.TemporaryDirectory(prefix=".fusion-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir()
            self.trees_[0].save_model(stage / "xgboost.json")
            self.trees_[1].save_model(str(stage / "catboost.cbm"))
            self.trees_[2].booster_.save_model(str(stage / "lightgbm.txt"))
            joblib.dump(self.trees_[3], stage / "extratrees.joblib")
            np.save(stage / "imputation.npy", self.imputer_.medians, allow_pickle=False)
            graph = None
            if self.mode == "full":
                import torch

                torch.save(
                    {k: v.detach().cpu() for k, v in self.graph_.model.state_dict().items()},
                    stage / "compt.pt",
                )
                graph = {
                    "model_options": self.graph_.model_options,
                    "target_mean": self.graph_.target_mean,
                    "target_scale": self.graph_.target_scale,
                    "batch_size": self.graph_.batch_size,
                }
            from molprop_fusion import __version__

            metadata = {
                "schema_version": BUNDLE_SCHEMA,
                "package_version": __version__,
                "dependency_versions": self.dependency_versions_,
                "parameters": {
                    "mode": self.mode,
                    "random_state": self.random_state,
                    "feature_seed": self.feature_seed,
                    "n_jobs": self.n_jobs,
                    "max_atoms": self.max_atoms,
                    "validation_only": self.validation_only,
                },
                "features": {
                    "policy": FEATURE_POLICY,
                    "canonicalization": "rdkit-canonical-isomeric-smiles",
                    "morgan_bits": 2048,
                    "morgan_radius": 2,
                    "morgan_chirality": False,
                    "rdkit_descriptors": self.descriptor_names_,
                    "geometry_columns": list(RAW_COLUMNS),
                    "missingness": "317-geometry-indicators",
                    "imputation": "train-median-all-missing-zero",
                },
                "fusion": self.meta_,
                "graph": graph,
                "training_summary": self.training_summary_,
            }
            (stage / "model.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            stage.rename(destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> MolecularPropertyFusionRegressor:
        """Load a trusted fitted model; defaults to CPU regardless of training hardware."""
        return load_model(path, device=device)


class _LightGBMPredictor:
    def __init__(self, booster: Any, n_jobs: int):
        self.booster = booster
        self.booster_ = booster
        self.n_jobs = n_jobs

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.booster.predict(matrix, num_threads=self.n_jobs)


def _require_bundle_files(directory: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete model bundle: missing {', '.join(missing)}; "
            "restore or copy the complete bundle directory"
        )


def load_model(path: str | Path, *, device: str = "cpu") -> MolecularPropertyFusionRegressor:
    """Load a trusted bundle. ExtraTrees uses joblib (can execute arbitrary code)."""
    import joblib

    from molprop_fusion import __version__

    directory = Path(path)
    _require_bundle_files(directory, ["model.json"])
    metadata = json.loads((directory / "model.json").read_text())
    if metadata["schema_version"] != BUNDLE_SCHEMA or metadata["package_version"] != __version__:
        raise ValueError("unsupported bundle/package version; use the training package version")
    model = MolecularPropertyFusionRegressor(**metadata["parameters"], device=device)
    model.dependency_versions_ = _dependencies(model.mode)
    saved_versions = dict(metadata["dependency_versions"])
    current_versions = dict(model.dependency_versions_)
    # CPU and CUDA wheel local-version suffixes do not change the weights format.
    for versions in (saved_versions, current_versions):
        if "torch" in versions:
            versions["torch"] = versions["torch"].split("+")[0]
    if saved_versions != current_versions:
        raise ValueError(
            "bundle dependency versions differ; recreate the exact versions listed in model.json"
        )
    from rdkit.Chem import Descriptors

    schema = metadata["features"]
    if (
        schema["policy"] != FEATURE_POLICY
        or schema["rdkit_descriptors"] != [name for name, _ in Descriptors._descList]
        or schema["geometry_columns"] != list(RAW_COLUMNS)
        or schema["canonicalization"] != "rdkit-canonical-isomeric-smiles"
    ):
        raise ValueError("bundle feature schema/names/order differ from this installation")
    expected = {
        "morgan_bits": 2048,
        "morgan_radius": 2,
        "morgan_chirality": False,
        "missingness": "317-geometry-indicators",
        "imputation": "train-median-all-missing-zero",
    }
    if any(schema.get(k) != v for k, v in expected.items()):
        raise ValueError("bundle feature configuration differs from this installation")
    required = [
        "xgboost.json",
        "catboost.cbm",
        "lightgbm.txt",
        "extratrees.joblib",
        "imputation.npy",
    ]
    if model.mode == "full":
        required.append("compt.pt")
    _require_bundle_files(directory, required)
    model.descriptor_names_ = schema["rdkit_descriptors"]
    model.imputer_ = MedianImputer()
    model.imputer_.medians = np.load(directory / "imputation.npy", allow_pickle=False)
    if model.imputer_.medians.shape != (2582,) or not np.isfinite(model.imputer_.medians).all():
        raise ValueError("invalid bundle imputation statistics")
    from catboost import CatBoostRegressor
    from lightgbm import Booster
    from xgboost import XGBRegressor

    xgb = XGBRegressor(n_jobs=model.n_jobs)
    xgb.load_model(directory / "xgboost.json")
    cat = CatBoostRegressor(thread_count=model.n_jobs)
    cat.load_model(str(directory / "catboost.cbm"))
    model.trees_ = [
        xgb,
        cat,
        _LightGBMPredictor(Booster(model_file=str(directory / "lightgbm.txt")), model.n_jobs),
        joblib.load(directory / "extratrees.joblib"),
    ]
    model.meta_ = metadata["fusion"]
    for name in ("T", "T_plus_G") if model.mode == "full" else ("T",):
        state = model.meta_[name]
        count = 4 if name == "T" else 5
        if any(
            np.asarray(state[k]).shape != (count,) or not np.isfinite(state[k]).all()
            for k in ("mean", "scale", "coefficients")
        ) or np.any(np.asarray(state["scale"]) <= 0):
            raise ValueError("invalid fusion scaler/coefficients")
    if model.mode == "full":
        import torch

        from molprop_fusion.compt_graph import CoMPTRegressor, GraphPredictor

        graph = metadata["graph"]
        if graph["model_options"]["max_atoms"] != model.max_atoms or not graph["target_scale"] > 0:
            raise ValueError("invalid graph capacity/normalization metadata")
        network = CoMPTRegressor(**graph["model_options"])
        network.load_state_dict(
            torch.load(directory / "compt.pt", map_location="cpu", weights_only=True)
        )
        network.eval()
        model.graph_ = GraphPredictor(
            network,
            graph["model_options"],
            graph["target_mean"],
            graph["target_scale"],
            graph["batch_size"],
        )
    model.training_summary_ = metadata["training_summary"]
    model._inference_config = (model.mode, model.feature_seed, model.max_atoms)
    model._fitted = True
    return model
