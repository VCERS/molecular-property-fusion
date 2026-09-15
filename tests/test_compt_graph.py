from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
compt_graph = pytest.importorskip("molprop_fusion.compt_graph")


def test_graph_features_and_forward_shape() -> None:
    node, edge, distance = compt_graph.featurize_smiles("CCO", max_atoms=8)
    assert node.shape == (8, 116)
    assert edge.shape == (8, 8, 13)
    assert distance.shape == (8, 8)
    model = compt_graph.CoMPTRegressor(hidden_width=16, heads=4, layers=1, max_atoms=8)
    node_tensor = torch.from_numpy(node[None])
    edge_tensor = torch.from_numpy(edge[None])
    distance_tensor = torch.from_numpy(distance[None])
    mask = node_tensor.abs().sum(dim=-1) != 0
    assert model(node_tensor, mask, distance_tensor, edge_tensor).shape == (1,)
