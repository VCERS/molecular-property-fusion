# Reproduce the expanded paper evaluations

The 0.2.0rc1 GitHub prerelease includes seven additional datasets and eleven
property tasks, evaluated over five runs with one fixed test population per
dataset. It supplements the three-dataset benchmark (30 evaluations) and
AqSolDB (five evaluations). Transferability therefore covers eight datasets,
twelve tasks, and sixty evaluations; the combined paper covers eleven datasets,
fifteen tasks, and ninety evaluations. No new accuracy experiment was run to
prepare the reviewed candidate for this release.

## Sources and targets

Download the exact files named in [expanded_sources.json](../configs/expanded_sources.json)
from its upstream URLs and verify the listed SHA-256 hashes. The source registry
pins TDC loader revision `c310c35f27e3f506411018ac43d97b8ba23ca652`,
DeepChem 2.8.0 revision `d5b293934d427062f52e2d92c1569d53d10418f9`,
and Chemprop v2.3.1 revision `9a0b47ad19ebf440c1557787e666d442728adaad`.
The official identifiers and machine keys remain unchanged.

| Dataset | Official identifier | Target field | Native unit | Primary metric |
| --- | --- | --- | --- | --- |
| PPBR (human) | PPBR_AZ | Y, Homo sapiens rows only | percentage bound | MAE |
| VDss | VDss_Lombardo | Y | L/kg | Spearman |
| Half-life | Half_Life_Obach | Y | hour | Spearman |
| Microsomal clearance | Clearance_Microsome_AZ | Y | µL/min/mg | Spearman |
| Hepatocyte clearance | Clearance_Hepatocyte_AZ | Y | µL/min per 10⁶ cells | Spearman |
| BACE | BACE regression | pIC50 | −log10(IC50/mol/L) | RMSE |
| QM9 | QM9 | homo, lumo, gap | Hartree | MAE |
| QM9 | QM9 | mu | Debye | MAE |
| QM9 | QM9 | alpha | Bohr³ | MAE |

Every task also reports MAE and RMSE. Errors are smaller-is-better; Spearman
is larger-is-better and dimensionless. PPBR errors are percentage points.
Microsomal clearance source mL/min/g is numerically identical to µL/min/mg.
Targets retain their source scale. BACE pIC50 is already logarithmic.
The Article displays QM9 orbital/gap errors in mHartree; these files retain Hartree.

## Filtering and assignments

The `partitions/*.csv` files contain only zero-based source-row indices,
minimum retained source indices for historical geometry seeds, and five
train/validation/test assignments, in the original retained order. These are
research-generated assignments, not copies of molecular structures or targets.
Their hashes are in the registry. Omitted source rows remain omitted, so replay
retains historical-overlap and internal-parent exclusions without requiring
private identity collections. Never substitute a fresh unfiltered split.

The historical preparation excluded invalid structures, graphs exceeding 115
atoms, nonfinite fixed chemistry, and absent target fields. Geometry failures
remain masked. PPBR selects the human subset. QM9 excludes the official
3,054-ID consistency-failure list. One representative per eligible parent was
selected in SHA-256 order of `expanded-v0|parent_connectivity_key`, up to 8,000
parents. The retained QM9 population is 7,993 rows with 5,600/800/1,593
train/validation/test rows for all five tasks; it uses only SMILES-derived
features, never supplied DFT geometries or auxiliary quantum targets.

TDC tests replayed the pinned ADMET archive's structural assignment. BACE used
Bemis–Murcko scaffold groups; QM9 used Butina clusters of Morgan-2048 radius-2
fingerprints at distance 0.6, `reordering=False`, named by minimum parent key.
Whole groups ordered by SHA-256 of `seed|group` were selected when inclusion
brought their size nearer the requested fraction (20% test, 12.5% of remaining
rows validation). Test seed was 20261009; validation seeds were 1 through 5.
Parent collisions joined training/validation groups. Canonical duplicates also
remain together in three OOF folds.

Historical exact/uncharged-parent overlaps were removed only from test, using
available ESOL, FreeSolv, Lipophilicity, AqSolDB, Caco2_Wang, LD50_Zhu, and earlier
development identities. Caco2 and LD50 are overlap references, not added
benchmarks. Tests also exclude parents present in training/validation and retain
the smallest source-index representative per test parent. The recorded
historical exclusions are 187/87/70/184/184/2/7 in dataset order above.
[Preparation aggregates](../results/expanded_preparation.json) retain all counts,
filter reasons, overlap counts, and structural split sizes. Microsomal and
Hepatocyte clearance tests have only 37 and 50 molecules.

The original preparation lacked usable QM7/QM8 and some earlier-development
identities. The exported indices reproduce the exclusions actually applied;
they do not establish separation from those unresolved historical collections.
AqSolDB's separately curated memberships remain unavailable in this package.
Exact AqSolDB reruns still require the study partitions supplied by the user.

## Commands

Install the frozen `[reproduce]` environment described in [reproducibility.md](reproducibility.md).
From the checkout, with upstream snapshots in `data/expanded/`:

```bash
molprop-fusion expanded-prepare --dataset ppbr_az --source data/expanded \
  --output cache/ppbr-prepared.json
molprop-fusion expanded-run --dataset ppbr_az --endpoint Y --run 1 \
  --source data/expanded --prepared cache/ppbr-prepared.json \
  --feature-cache cache/expanded --device cuda --n-jobs 8 \
  --output outputs/ppbr-run-1.json
molprop-fusion expanded-summarize --input outputs/ppbr-run-*.json \
  --output outputs/ppbr-summary.json
molprop-fusion verify-results --results-root results
```

Prepare each of the seven machine keys in the registry; run each declared
endpoint for runs 1 through 5 to obtain the 55-case matrix. Each case runs all
four methods. Use a shared feature-cache directory to reuse geometry across
runs and the five QM9 properties. Preparation and execution verify source and
partition hashes and refuse to overwrite outputs. Installed wheels carry docs,
configs, partitions and results under `sys.prefix/share/molecular-property-fusion`;
run there, or pass absolute `--config`, `--manifest`, and `--partitions` paths.

The four-method case uses the unchanged tree, CoMPT, and Ridge profile in
[expanded_benchmarks.yaml](../configs/expanded_benchmarks.yaml). T/G/fusion use
training rows only; CoMPT makes its own grouped 10% early-stopping subset.
Chemprop uses outer validation for RMSE checkpoint selection. Its ephemeral
CSV withholds test labels and temporary predictions/checkpoints are deleted.
Only aggregate metrics, counts, and timings persist. Reduced `--validation-only`
fits are software diagnostics and cannot be combined with scientific cases.
CPU execution is supported for small diagnostics; the original matrix ran on H100.

Historical reproduction uses `source_seed(dataset, minimum_retained_source_index)`
for geometry, float32 cached Z3D values, and finite fixed chemistry filtering.
The user API's canonical-molecule seed and fold-local 2D imputation are described
in the [user guide](user-guide.md). Sharing the model profile does not make the
two input-generation paths numerically identical.

## Results and costs

[Expanded aggregates](../results/expanded_benchmarks.json) retain all 55 run
records, all primary/secondary metrics, paired differences, and cost summaries.
They report 1,265 successful fits: 880 tree, 220 CoMPT, 110 Ridge, 55 Chemprop.
Mean primary gains are 7/11 over T and 10/11 over Chemprop, or 8/12 and 11/12
with AqSolDB. Fusion is worse than T for VDss, Half-life, Microsomal clearance,
and BACE. PPBR/Hepatocyte gains are small. Chemprop is better on QM9 dipole;
favorable LUMO MAE does not imply favorable LUMO RMSE.

The complete seven-dataset four-method experiment took **27.728 hours**,
excluding AqSolDB. Shared cold feature preparation took 4.453 hours and cases
23.270 hours. T model components sum to 0.580 hours, graph/fusion increments
19.306 hours, and Chemprop enclosing workflows 0.768 hours. T+G components sum
to 19.886 hours before inputs and 24.313 hours with shared cold inputs/I/O.
These component sums are not a separately measured T+G elapsed training time.
Case orchestration accounts for a 2.616-hour difference; feature and remaining
workflow differences are retained. Do not sum nested Chemprop clocks twice.
New command summaries report measured invocation elapsed sums separately.

Maximum sampled CPU RSS was 24.75 GiB, Torch allocation 44.01 GiB, and process
GPU memory 49.00 GiB. Samples were 0.5 seconds apart; RSS can double-count shared
pages. The historical storage footprint was 2.087 GiB, not a storage peak.
Setup/acquisition durations were incomplete and are excluded. CoMPT ran
21–150 epochs (median 66); Chemprop ran 50 epochs per fit. Runtime values are
retained observations, not guarantees for another host.

## Data terms and scope of export

The export includes software, aggregate scientific observations, and original
integer-only research assignments. It contains no source datasets, molecular
identity derivatives, row-level targets/predictions/errors, or private quantum
data. Upstream datasets remain subject to their own terms; the source registry
retains the acquisition attribution and rights notes. [TDC documentation](https://tdcommons.ai/single_pred_tasks/adme/)
leaves several ADME licenses unspecified; BACE artifact-specific reuse terms
are not explicit. This release does not infer permission to redistribute
those source records from the software license. QM9's original deposit is
[Figshare 1057646](https://doi.org/10.6084/m9.figshare.1057646).
