# Molecular Property Fusion

This repository implements prediction-level fusion for continuous molecular
property regression. Four tree regressors use Morgan fingerprints, RDKit
descriptors, and conformer-derived geometric summaries. A separate CoMPT graph
regressor uses atom-bond graphs. Ridge regression is fitted to grouped
out-of-fold predictions from the tree models alone (`T`) or from the tree and
graph models together (`T_plus_G`).

Train a reusable model on your own labeled SMILES, then predict new molecules
from an unlabeled CSV. Models are fitted per continuous property. The complete
method (`full`) combines four trees and CoMPT; explicit `tabular` mode uses the
four trees for CPU work. Chemprop is optional and is not needed for either mode.

## Use the method

Use Python 3.11. From this checkout, install one dependency group:

```bash
python -m pip install '.[full]'
# For four-tree CPU fitting instead, install '.[tabular]' and train with --mode tabular.

molprop-fusion train --data train.csv --smiles-column smiles --target-column logS \
  --mode full --device cpu --output models/solubility
molprop-fusion predict --model models/solubility --data candidates.csv \
  --smiles-column smiles --output predictions.csv
```

Training needs a finite numeric target column. Prediction needs only SMILES and
preserves input columns and row order, including duplicates. Add
`--invalid-policy report` to keep invalid rows with blank predictions and an error.
Full CPU training can be slow; use `--mode tabular` when CoMPT is not wanted.
macOS tree libraries may require `brew install libomp`.

The Python example also requires the `[full]` installation:

```python
from molprop_fusion import MolecularPropertyFusionRegressor, load_model

model = MolecularPropertyFusionRegressor(mode="full", device="cpu", random_state=42)
model.fit(train_smiles, train_targets)
prediction = model.predict(["CCO", "c1ccccc1"])
model.save("models/property")
restored = load_model("models/property")  # CPU by default; trusted bundles only
prediction = restored.predict(["CCO"])
```

Run the small synthetic tutorial:

```bash
python examples/tutorial.py --mode tabular --output outputs/tutorial-tabular
python examples/tutorial.py --mode full --output outputs/tutorial-full
```

The tutorial uses reduced software-validation fits and invented labels; it does
not demonstrate benchmark accuracy. See the [user guide](docs/user-guide.md) for
CSV input, geometry statuses, fixed graph atom capacity, saving/loading,
installation compatibility, defaults, and compute cost, and the
[API reference](docs/api.md) for method arguments.

## Reproduce the paper

This local release candidate is **0.2.0rc1** and has not been published.
The paper routes cover ESOL, FreeSolv, Lipophilicity, AqSolDB, PPBR (human),
VDss, Half-life, Microsomal clearance, Hepatocyte clearance, BACE, and the
7,993-molecule SMILES-derived QM9 subset. Together these comprise eleven
datasets, fifteen property tasks, and ninety evaluations. The seven added
datasets contribute eleven tasks and 55 cases.

Start with [paper reproduction](docs/reproducibility.md), then follow
[expanded datasets and results](docs/expanded-reproduction.md) for source
snapshots, index-only partitions, retained aggregates, and computational costs.
The historical public version 0.1.0 at commit `4a2d86e` and its original
configurations/results remain available in public repository history. The
historical benchmark/AqSolDB configurations and aggregates in this candidate
are unchanged.

### Reproduction installation

Python 3.11 is required. A lightweight installation supports configuration and
aggregate verification:

```bash
python -m pip install -e .
molprop-fusion verify-results
```

The frozen scientific dependency set is available as an extra:

```bash
python -m pip install -e ".[reproduce]"
```

These exact versions match the recorded scientific validation environment. GPU availability,
CUDA wheels, and platform-specific packages may require selecting the
appropriate package index for the host.

## Public benchmarks

Download the three public CSV files and verify their hashes as described in
[`docs/data.md`](docs/data.md). Then run, for example:

```bash
molprop-fusion prepare \
  --dataset esol \
  --data data/delaney-processed.csv \
  --output cache/esol-features.npz

molprop-fusion benchmark \
  --dataset esol \
  --data data/delaney-processed.csv \
  --feature-cache cache/esol-features.npz \
  --output outputs/esol.json \
  --device cuda \
  --n-jobs 8
```

The `--smoke` flag limits input to 30 rows and shortens model fitting. It
checks the end-to-end code path but does not produce a scientific result.

Run the Chemprop baseline separately so test labels remain unavailable during
training and checkpoint selection:

```bash
molprop-fusion chemprop \
  --dataset esol \
  --data data/delaney-processed.csv \
  --seed 13 \
  --output outputs/esol-chemprop-seed-13.json
```

## AqSolDB transfer

The transfer entry point accepts an AqSolDB-derived CSV prepared by the user.
It requires `SMILES`, `Solubility`, and one partition column per seed named
`split_seed_1` through `split_seed_5`. Each partition column contains
`train`, `validation`, or `test`. The overlap-controlled study partitions
are not redistributed.

```bash
molprop-fusion transfer \
  --config configs/aqsoldb_transfer.yaml \
  --data /path/to/user-prepared-aqsoldb.csv \
  --output outputs/aqsoldb-transfer.json \
  --device cuda \
  --n-jobs 8
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete
workflow, expected cost, and aggregate verification.

## Provenance and licensing

[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records the exact private
source commits, original paths, source hashes, public destinations, and
transformation notes. Toolkit v0 supplements this mapping while preserving the
original source records. The export contains no private Git history.

Original repository code is released under the MIT License. The graph
implementation follows MIT-licensed CoMPT and retains its notice in
[`licenses/CoMPT-LICENSE`](licenses/CoMPT-LICENSE). Chemprop is not vendored.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
