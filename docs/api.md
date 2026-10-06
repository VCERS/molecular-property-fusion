# Python API

```python
from molprop_fusion import MolecularPropertyFusionRegressor, load_model

model = MolecularPropertyFusionRegressor(mode="full", device="cpu", random_state=42)
model.fit(train_smiles, train_targets)
prediction, diagnostics = model.predict(new_smiles, return_status=True)
model.save("models/property")
restored = load_model("models/property", device="cpu")
prediction = restored.predict(new_smiles)
```

`train_smiles` and `new_smiles` are sequences of strings (including single-element
lists). `train_targets` is a finite one-dimensional numeric sequence of the same
length. `fit` returns the estimator. `predict` returns a float64 NumPy vector in
input order. Empty input returns shape `(0,)`. Strict errors identify the affected
row; the API never silently skips invalid molecules. With `return_status=True`,
it returns `(vector, list_of_diagnostic_dicts)`.

| Constructor argument | Default | Meaning |
| --- | --- | --- |
| `mode` | `'full'` | `'full'` (T_plus_G) or `'tabular'` (T) |
| `device` | `'auto'` | CoMPT training/inference device |
| `random_state` | `42` | Group splits and model training seed |
| `feature_seed` | `42` | Deterministic molecule geometry seed |
| `n_jobs` | `1` | Tree fitting parallelism |
| `max_atoms` | `100` | Frozen full-mode native graph padding capacity |
| `validation_only` | `False` | Deliberately tiny fits for software validation |
| `verbose` | `False` | Print feature/fitting progress to stderr |

`training_summary_` reports row/group counts, fit counts, OOF column count, geometry
statuses, graph best epochs, and elapsed time. Inspect `docs/user-guide.md` for
the default configuration source and supported dependency boundary.

`save(path)` writes all fitted inference state into a **new** directory.
`load_model(path, device='cpu')` and
`MolecularPropertyFusionRegressor.load(path, device='cpu')` restore trusted bundles.
Only the exact recorded package/dependency versions are supported. ExtraTrees
uses joblib, so loading untrusted models is unsafe. No raw training data is saved.
