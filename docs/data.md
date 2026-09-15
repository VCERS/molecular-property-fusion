# Data acquisition

No molecular dataset is stored in this repository. Download each public file
from its upstream host and confirm its SHA-256 digest before use.

| Dataset | Upstream file | Rows | SHA-256 |
| --- | --- | ---: | --- |
| ESOL | [delaney-processed.csv](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | 1,128 | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |
| FreeSolv | [SAMPL.csv](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv) | 642 | `ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72` |
| Lipophilicity | [Lipophilicity.csv](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv) | 4,200 | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` |

The public benchmark command reconstructs the recorded Lane K split from the
upstream row order: an 80/20 split using the case seed, followed by an equal
validation/test split of the held-out 20% using random state 42. The local
fusion models do not use the outer validation partition. Chemprop uses it only
for checkpoint selection.

## AqSolDB

AqSolDB is described by Sorkun, Khetan, and Er in
[Scientific Data](https://doi.org/10.1038/s41597-019-0151-1). The public data
deposit is available under [DOI 10.7910/DVN/OVHAW8](https://doi.org/10.7910/DVN/OVHAW8).

This repository does not redistribute AqSolDB rows or the study's
overlap-controlled memberships. Users must obtain the data under the
upstream terms and provide the required partition columns described in the
README. Consequently, the tracked AqSolDB aggregate verifies the reported
result, while exact membership-level reruns require the separately curated
inputs.

## Licensing

Dataset terms are independent of this repository's MIT software license.
Review and follow each upstream dataset's terms before downloading, copying,
or redistributing its content.
