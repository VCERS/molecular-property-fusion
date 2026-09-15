# Third-party notices

This repository is MIT-licensed for its original code. Dependencies keep their
own licenses; installing this project does not relicense them.

## CoMPT

The graph regressor in `src/molprop_fusion/compt_graph.py` is a clean PyTorch
implementation whose architecture and feature semantics follow
[jcchan23/CoMPT](https://github.com/jcchan23/CoMPT) commit
`50bb4a83fc4ac843ee92bf56421cec0ba631a90f`.

CoMPT is MIT-licensed, copyright 2021 Jianwen Chen. Its license is reproduced
in `licenses/CoMPT-LICENSE`. No CoMPT dataset, checkpoint, cache, or model
artifact is included.

## Chemprop

Chemprop is an external optional dependency and is not vendored. The reported
baseline used [chemprop/chemprop](https://github.com/chemprop/chemprop) commit
`9a0b47ad19ebf440c1557787e666d442728adaad`, tag/version `v2.3.1`, under
the MIT License. See the upstream repository for its copyright notice and
complete dependency notices.
