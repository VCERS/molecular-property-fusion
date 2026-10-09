# Train a model for your molecular property

Use one fitted model per continuous property, with targets in a consistent unit.
Prediction quality depends on your training data, property, and chemical coverage.
The tutorial uses invented labels solely to demonstrate the workflow.

## Installation

Use Python 3.11 in a virtual environment. From this repository:

```bash
python -m pip install '.[tabular]'  # four trees and molecular features; no PyTorch
# or
python -m pip install '.[full]'     # four trees plus CoMPT; no Chemprop
```

The base installation (`pip install .`) supports configuration/aggregate verification
and package import without molecular or graph libraries. The existing `[reproduce]`
extra retains the frozen scientific versions, including the optional Chemprop
comparison. See [reproducibility.md](reproducibility.md) for paper workflows.

macOS may require `brew install libomp` for XGBoost/LightGBM. On Linux choose the
appropriate PyTorch CPU/CUDA wheels before installing `[full]` if necessary.
Python 3.11.12 and macOS ARM64 CPU were tested for this toolkit with NumPy 1.26.4,
scikit-learn 1.9.1, RDKit 2026.3.6 (217 descriptors), XGBoost 3.2.0, CatBoost 1.2.10,
LightGBM 4.7.0, joblib 1.6.0, and PyTorch 2.14.1. These software-validation versions
are separate from the frozen scientific `[reproduce]` environment. New RDKit
registries must still contain exactly 217 ordered descriptors; use the tested
RDKit version if a future registry changes.

Linux x86_64 CPU and CUDA use were also validated on NVIDIA H100 with Python
3.11.15, scikit-learn 1.9.0, RDKit 2026.3.4, PyTorch 2.13.0, and joblib 1.5.3
(the tree-library and NumPy versions listed above). This used an isolated wheel
installation, real small full-method fits, and fresh-process CPU loading of a
GPU-trained bundle with CUDA devices hidden. The same real H100-trained bundle
also loaded and predicted on a CPU-only macOS ARM64 host with matching dependency
release versions. MPS and Windows remain unvalidated.

## Your CSV

Training needs SMILES and a numeric target column; extra identifier columns are
fine. Remove missing targets, use consistent units, and include at least three
distinct canonical structures. Three is a structural minimum, not a useful
training size. Full mode also requires nonzero target variation within every
inner training split, so very small datasets may need additional groups.

```bash
molprop-fusion train --data train.csv --smiles-column smiles --target-column logS \
  --mode full --device cpu --output models/solubility
molprop-fusion predict --model models/solubility --data candidates.csv \
  --smiles-column smiles --output predictions.csv
```

`candidates.csv` needs no labels. Output preserves every original column and row
order, appending `prediction`, `prediction_status`, `prediction_error`,
`geometry_status`, `geometry_missing_count`, and `conformer_count`. Rename any
input columns that already use those output names. Duplicate molecules retain
duplicate rows. An empty query yields an empty prediction vector or a CSV header.
Training prints feature/fitting progress; `--n-jobs` controls tree parallelism.

The default invalid policy is strict: an invalid molecule stops the command before
writing output. `--invalid-policy report` instead writes blank predictions and
an actionable error for invalid SMILES or oversized graphs, while retaining
successful rows in their original positions. Model/dependency failures still stop
execution; this policy does not hide a broken installation.

## Geometry and molecules

Canonicalization uses RDKit canonical isomeric SMILES, without salt removal or
neutralization. Morgan fingerprints retain the historical no-chirality policy.
Z3D uses up to 16 ETKDGv3 conformers, with no force-field optimization or energy
weighting. The new API hashes canonical molecular identity, `feature_seed`, and
`canonical-etkdgv3-v1` to generate a reproducible conformer seed, independent of
row number and prediction batch. Equivalent canonical structures share features.
The paper commands continue to use their historical dataset/source-row seed.

`geometry_status=ok` can still have missing pharmacophore distances. Absent
pharmacophores are normal missingness, not invalid SMILES. `multiple_fragments`,
`radical_unsupported`, `no_heavy_atom`, `embed_failed`,
`nonfinite_required_descriptor`, and `exception` indicate missing geometry;
training medians and missingness indicators let valid molecules remain usable.
The API can return diagnostics with `predict(smiles, return_status=True)`.

Geometry imputation is fitted within each OOF training fold and again on the full
training set. Entirely missing columns use zero. The user API also imputes rare
nonfinite 2D descriptors with fold-local medians; the paper workflow retains its
existing preprocessing. An in-memory molecule cache is bound to the feature
policy and seed and assembles features in query order. No external dimension-only
cache is accepted by the user API.

## CoMPT capacity and devices

Full mode trains all five base predictors. Tabular mode explicitly omits CoMPT;
full mode never silently falls back. Use `device='cpu'`, `'cuda'`, or `'auto'`.
Auto chooses CUDA, then MPS, then CPU when available. Full CPU fits can be slow.
Geometry is generated on CPU in both modes.

`max_atoms=100` sets graph padding and positional capacity **before training**.
CoMPT's GRU readout depends on padded length, so loading/inference always keeps
that capacity. An oversized molecule is rejected, never truncated. Retrain with
a larger `max_atoms`, or train a separate tabular model. Changing fitted feature
parameters or capacity is not supported; construct and fit a new estimator.

## Bundles and compatibility

`model.save('models/solubility')` creates a new directory and refuses to overwrite
an existing one. `load_model('models/solubility')` defaults to CPU, regardless of
training hardware. `MolecularPropertyFusionRegressor.load(...)` is equivalent.
Keep the whole directory together.

The bundle contains native XGBoost JSON, CatBoost CBM, LightGBM text,
ExtraTrees joblib, NumPy imputation statistics, CoMPT weights loaded with
`weights_only=True`, and JSON feature schema, Ridge scaler/coefficients,
normalization, versions, seed, capacity, and a concise training summary. It does
not store raw targets, training SMILES, OOF predictions, or the feature cache.

**Load only trusted bundles:** ExtraTrees joblib can execute code when loaded.
For v0, loading requires the exact package and recorded dependency release versions
(PyTorch CPU/CUDA wheel suffixes may differ),
and matching ordered RDKit/Z3D schema. Read `model.json` to recreate the training
environment. Cross-version migration is not supported. CPU loading avoids a
GPU hardware requirement and was tested with a real H100-trained bundle. The
CUDA/CPU numerical checks use `rtol=1e-5, atol=1e-5`; exact equality across device
backends is not expected. Duplicate query rows still receive identical outputs.

## Method and cost

Morgan-2048 + RDKit-217 + Z3D-317 + geometry missingness-317 feed XGBoost,
CatBoost, LightGBM, and ExtraTrees. CoMPT uses atom/bond features and topological
distances, not the conformer coordinates. Three canonical-group-disjoint folds
produce four or five OOF prediction columns. StandardScaler followed by Ridge
(alpha 10) learns `T` from four columns and `T_plus_G` directly from five columns.
Full `predict` returns `T_plus_G`; tabular `predict` returns `T`.

Final trees fit the complete training set. The final CoMPT uses the existing
10% group-disjoint inner early-stopping procedure and selected weights; its target
normalization comes from that inner fit partition. This retains the baseline
training procedure rather than refitting the graph on its early-stopping groups.

Defaults follow the frozen ESOL configuration: 300 rounds for boosted trees,
500 ExtraTrees, and CoMPT width 256, three layers, four heads, up to 150 epochs.
These are one documented starting configuration, not optimized settings for
arbitrary properties. Each fit trains 16 tree models and, in full mode, four
CoMPT models. Graph memory grows quadratically with atom capacity. Geometry and
OOF fits dominate cost; start with a modest dataset and explicit CPU settings.
`--validation-only` / `validation_only=True` reduce trees to five and CoMPT to
width 16, one layer, one epoch. They are software checks, not scientific presets.

## Small tutorial

```bash
python examples/tutorial.py --mode tabular --output outputs/tutorial-tabular
python examples/tutorial.py --mode full --output outputs/tutorial-full
# On a CUDA host: training on GPU, loading/prediction on CPU
python examples/tutorial.py --mode full --device cuda --output outputs/tutorial-cuda
```

Both commands generate distributable synthetic CSVs, train with reduced settings,
save, load in a new process, predict unlabeled molecules, and check row outputs.
Choose a new output directory for each run. No benchmark accuracy is claimed.

Run the optional hardware checks with `pytest tests/test_cuda.py`. They skip
without CUDA. These checks include reduced full-method fits, a one-epoch native
CoMPT architecture diagnostic (width 256, three layers, capacity 100), GPU/CPU
prediction parity, batch/order/capacity checks, and loading in a new process with
no visible GPU. They are software validation rather than accuracy studies.
