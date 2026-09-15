# Molecular Property Fusion

This repository implements prediction-level fusion for continuous molecular
property regression. Four tree regressors use Morgan fingerprints, RDKit
descriptors, and conformer-derived geometric summaries. A separate CoMPT graph
regressor uses atom-bond graphs. Ridge regression is fitted to grouped
out-of-fold predictions from the tree models alone (`T`) or from the tree and
graph models together (`T_plus_G`).

The public workflows cover ESOL, FreeSolv, Lipophilicity, and a user-supplied
AqSolDB transfer evaluation. Datasets, row-level partitions, targets,
predictions, residuals, checkpoints, and model artifacts are not distributed.
Only compact aggregate results needed for verification are tracked.

## Install

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

These exact versions match the validated H100 environment. GPU availability,
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
transformation notes. The new repository contains no private Git history.

Original repository code is released under the MIT License. The graph
implementation follows MIT-licensed CoMPT and retains its notice in
[`licenses/CoMPT-LICENSE`](licenses/CoMPT-LICENSE). Chemprop is not vendored.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
