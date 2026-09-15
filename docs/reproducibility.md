# Reproducibility

## Environment

The authoritative validation environment uses Python 3.11.15 with:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| RDKit | 2026.3.4 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| LightGBM | 4.7.0 |
| PyTorch | 2.13.0 |
| Chemprop | 2.3.1 |

Install these versions with `python -m pip install ".[reproduce]"`. The graph
workflow was validated on an NVIDIA H100. Full runs are intentionally absent
from ordinary CPU continuous integration.

## Input checks

Download the public data files listed in [`data.md`](data.md), then check
their digests:

```bash
sha256sum data/delaney-processed.csv data/SAMPL.csv data/Lipophilicity.csv
```

The observed hashes must match `configs/public_benchmarks.yaml`.

## Feature construction

`molprop-fusion prepare` builds:

- a 2,048-bit radius-2 Morgan fingerprint without chirality;
- the frozen RDKit registry of 217 molecular descriptors;
- 317 conformer-derived geometric values plus 317 missingness indicators.

Z3D missing values are imputed using medians fitted only on the current
training fold. A column that is entirely missing in that fold receives zero.
Feature caches are generated artifacts and are ignored by Git.

## Model fitting

The benchmark runs four tree regressors, CoMPT, and two Ridge meta-models.
Three group-disjoint folds produce out-of-fold columns for Ridge. The
configuration fixes the tree parameters, graph architecture, training
schedule, seeds, maximum atom counts, and Ridge alpha.

Run the three datasets:

```bash
for dataset in esol freesolv lipophilicity; do
  molprop-fusion benchmark \
    --dataset "$dataset" \
    --data "data/$(case "$dataset" in esol) echo delaney-processed.csv;; freesolv) echo SAMPL.csv;; *) echo Lipophilicity.csv;; esac)" \
    --feature-cache "cache/$dataset-features.npz" \
    --output "outputs/$dataset.json" \
    --device cuda \
    --n-jobs 8
done
```

Case outputs contain metrics, fit counts, Ridge parameters, best epochs, and
prediction-vector hashes. They do not contain row targets, predictions,
errors, or residuals. Summarize a case file with:

```bash
molprop-fusion summarize --input outputs/esol.json --output outputs/esol-summary.json
```

## Cheap validation

Use a public file with `--smoke` to exercise CSV loading, molecular features,
all four tree families, grouped OOF fitting, CoMPT fitting, and Ridge fusion.
The command uses 30 rows and one graph epoch:

```bash
molprop-fusion benchmark \
  --dataset esol \
  --data data/delaney-processed.csv \
  --output outputs/esol-smoke.json \
  --device cuda \
  --n-jobs 1 \
  --seed 13 \
  --smoke
```

Smoke metrics are not comparable with the tracked scientific aggregates.

## Tracked aggregate verification

```bash
molprop-fusion verify-results --results-root results
```

The public benchmark aggregate authenticates source result SHA-256
`59a81c1cfc965759284ab74836b7c7c1a954d5247f1d90d15f9d3b46efee4b90`.
The AqSolDB aggregate authenticates source result SHA-256
`b028e0018891e41c59785bef0413f8ad1b16b7537462d5ef558dbf9903ec188f`.
The original source result files are not copied because they contain more
operational detail than is needed for public verification.
