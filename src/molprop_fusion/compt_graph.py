"""Clean PyTorch implementation of the MIT-licensed CoMPT graph regressor.

Architecture and feature semantics follow jcchan23/CoMPT commit
50bb4a83fc4ac843ee92bf56421cec0ba631a90f. No MoleSG source is copied here.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _mish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.tanh(F.softplus(value))


class ScaleNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(width)))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * (self.scale / torch.norm(value, dim=-1, keepdim=True).clamp(self.epsilon))


class FeedForward(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(width, width)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropouts[0](_mish(self.linears[0](value)))


class CommunicativeAttention(nn.Module):
    def __init__(self, heads: int, width: int, dropout: float, attenuation: float = 0.1):
        super().__init__()
        if width % heads:
            raise ValueError("CoMPT width must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.attenuation_lambda = nn.Parameter(torch.tensor(attenuation))
        self.linears = nn.ModuleList([nn.Linear(width, width) for _ in range(5)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = nodes.shape
        attenuation = torch.clamp(self.attenuation_lambda, min=0.0, max=1.0)
        diffusion = attenuation * distance
        diffusion = diffusion.masked_fill(~mask.unsqueeze(1).expand(-1, length, -1), torch.inf)
        diffusion = F.softmax(-diffusion, dim=-1)
        query = (
            self.linears[0](nodes).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        )
        key = (
            self.linears[1](edges)
            .view(batch, length, length, self.heads, self.head_width)
            .permute(0, 3, 1, 2, 4)
        )
        value = (
            self.linears[2](nodes).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        )
        scale = math.sqrt(self.head_width)
        out_score = torch.einsum("bhmd,bhmnd->bhmn", query, key) / scale
        in_score = torch.einsum("bhnd,bhmnd->bhnm", query, key) / scale
        attention_mask = mask[:, None, None, :].expand(-1, self.heads, length, -1)
        out_attention = F.softmax(out_score.masked_fill(~attention_mask, -torch.inf), dim=-1)
        in_attention = F.softmax(in_score.masked_fill(~attention_mask, -torch.inf), dim=-1)
        diagonal = torch.diag_embed(torch.diagonal(out_attention, dim1=-2, dim2=-1))
        message = self.dropout((out_attention + in_attention - diagonal) * diffusion[:, None])
        node_output = torch.einsum("bhmn,bhnd->bhmd", message, value)
        edge_output = message[..., None] * key
        node_output = node_output.transpose(1, 2).contiguous().view(batch, length, width)
        edge_output = (
            edge_output.permute(0, 2, 3, 1, 4).contiguous().view(batch, length, length, width)
        )
        return _mish(self.linears[3](node_output)), _mish(self.linears[4](edge_output))


class EncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.self_attn = CommunicativeAttention(heads, width, dropout)
        self.feed_forward = FeedForward(width, dropout)
        self.norm = ScaleNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.dropout(self.norm(nodes))
        first, _ = self.self_attn(normalized, edges, distance, mask)
        first = normalized + self.dropout(self.norm(first))
        second = self.feed_forward(first)
        return normalized + first + self.dropout(self.norm(second)), edges


class Encoder(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        template = EncoderLayer(width, heads, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(template) for _ in range(layers)])
        self.norm = ScaleNorm(width)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, distance, mask)
        return self.norm(nodes)


class GruReadout(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * width, width)
        self.bias = nn.Parameter(torch.empty(width))
        self.bias.data.uniform_(-1.0 / math.sqrt(width), 1.0 / math.sqrt(width))
        self.proj = nn.Sequential(
            nn.Linear(width, width),
            nn.Mish(),
            ScaleNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numeric_mask = mask[..., None].float()
        masked = nodes * numeric_mask
        hidden = torch.max(_mish(masked + self.bias), dim=1)[0][None].repeat(2, 1, 1)
        messages, _ = self.gru(masked, hidden)
        messages = _mish(self.linear(messages))
        pooled = messages.sum(dim=1) / numeric_mask.sum(dim=1)
        return self.proj(pooled).squeeze(-1)


class CoMPTRegressor(nn.Module):
    def __init__(
        self,
        *,
        atom_width: int = 115,
        bond_width: int = 13,
        hidden_width: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        max_atoms: int = 100,
    ):
        super().__init__()
        self.node_embed = nn.Module()
        self.node_embed.lut = nn.Linear(atom_width, hidden_width)
        self.edge_embed = nn.Module()
        self.edge_embed.lut = nn.Linear(bond_width, hidden_width)
        self.pos_embed = nn.Module()
        self.pos_embed.pe = nn.Embedding(max_atoms + 1, hidden_width, padding_idx=0)
        self.encoder = Encoder(hidden_width, heads, layers, dropout)
        self.generator = GruReadout(hidden_width, dropout)
        self.hidden_width = hidden_width
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.node_embed.lut(node_features[..., :-1]) * math.sqrt(self.hidden_width)
        nodes = nodes + self.pos_embed.pe(node_features[..., -1].long())
        edges = self.edge_embed.lut(edge_features) * math.sqrt(self.hidden_width)
        return self.encoder(nodes, edges, distance, mask)

    def forward(
        self,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        distance: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(self.encode(node_features, edge_features, distance, mask), mask)


def featurize_smiles(
    smiles: str, max_atoms: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid canonical SMILES")
    atoms = molecule.GetNumAtoms()
    if not 0 < atoms <= max_atoms:
        raise ValueError(f"molecule atom count {atoms} exceeds frozen maximum {max_atoms}")
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(molecule)
    Chem.rdmolops.AssignStereochemistryFrom3D(molecule)
    AllChem.ComputeGasteigerCharges(molecule)
    nodes = np.zeros((max_atoms, 116), dtype=np.float32)
    for atom in molecule.GetAtoms():
        atomic = np.zeros(101, dtype=np.float32)
        atomic[atom.GetAtomicNum() - 1 if 1 <= atom.GetAtomicNum() <= 100 else -1] = 1
        hybrids = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
        hybrid = np.zeros(6, dtype=np.float32)
        hybrid[
            hybrids.index(atom.GetHybridization()) if atom.GetHybridization() in hybrids else -1
        ] = 1
        charges = []
        for name in ("_GasteigerCharge", "_GasteigerHCharge"):
            value = float(atom.GetDoubleProp(name))
            charges.append(value if math.isfinite(value) else 0.0)
        numeric = np.asarray(
            [
                atom.GetTotalNumHs(includeNeighbors=True) / 8,
                atom.GetDegree() / 4,
                atom.GetFormalCharge() / 8,
                atom.GetTotalValence() / 8,
                *charges,
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetIdx() + 1,
            ],
            dtype=np.float32,
        )
        nodes[atom.GetIdx()] = np.concatenate([atomic, hybrid, numeric])
    edges = np.zeros((max_atoms, max_atoms, 13), dtype=np.float32)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOZ,
    ]
    for bond in molecule.GetBonds():
        feature = np.zeros(13, dtype=np.float32)
        if bond.GetBondType() in bond_types:
            feature[bond_types.index(bond.GetBondType())] = 1
        if bond.GetStereo() in stereo_types:
            feature[4 + stereo_types.index(bond.GetStereo())] = 1
        feature[10:] = [bond.GetIsConjugated(), bond.GetIsAromatic(), bond.IsInRing()]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[begin, end] = edges[end, begin] = feature
    distance = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    distance[:atoms, :atoms] = Chem.rdmolops.GetDistanceMatrix(molecule).astype(np.float32)
    return nodes, edges, distance


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@dataclass
class GraphFitResult:
    predictions: np.ndarray
    best_epoch: int
    wall_seconds: float
    peak_accelerator_bytes: int | None


def fit_graph_regressor(
    *,
    train_smiles: Sequence[str],
    train_target: np.ndarray,
    train_groups: Sequence[str],
    evaluation_smiles: Sequence[str],
    seed: int,
    model_options: dict[str, Any],
    training_options: dict[str, Any],
) -> GraphFitResult:
    """One fit with a deterministic group-disjoint inner early-stopping split."""
    from sklearn.model_selection import GroupShuffleSplit
    from torch.utils.data import DataLoader, TensorDataset

    start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(str(training_options.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    fit_index, early_index = next(splitter.split(train_target, groups=np.asarray(train_groups)))
    if set(np.asarray(train_groups)[fit_index]) & set(np.asarray(train_groups)[early_index]):
        raise RuntimeError("inner early-stopping group overlap")
    all_smiles = [*train_smiles, *evaluation_smiles]
    graph_rows = [
        featurize_smiles(smiles, int(model_options["max_atoms"])) for smiles in all_smiles
    ]
    nodes = torch.from_numpy(np.stack([row[0] for row in graph_rows]))
    edges = torch.from_numpy(np.stack([row[1] for row in graph_rows]))
    distances = torch.from_numpy(np.stack([row[2] for row in graph_rows]))
    mean, scale = float(np.mean(train_target[fit_index])), float(np.std(train_target[fit_index]))
    if not scale > 0:
        raise ValueError("graph fit target scale is non-positive")
    standardized = torch.from_numpy(((train_target - mean) / scale).astype(np.float32))

    model = CoMPTRegressor(**model_options)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, weight_decay=float(training_options["weight_decay"])
    )
    batch_size = int(training_options["batch_size"])
    warmup_steps = max(1, (len(fit_index) // batch_size) * int(training_options["warmup_epochs"]))
    train_dataset = TensorDataset(
        nodes[fit_index], edges[fit_index], distances[fit_index], standardized[fit_index]
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False
    )
    best_loss, best_epoch, best_state, stale, step = math.inf, 0, None, 0, 0
    for epoch in range(1, int(training_options["max_epochs"]) + 1):
        model.train()
        for node, edge, distance, target in loader:
            step += 1
            rate = (
                float(training_options["warmup_factor"])
                * model.hidden_width**-0.5
                * min(step**-0.5, step * warmup_steps**-1.5)
            )
            optimizer.param_groups[0]["lr"] = rate
            node, edge, distance, target = (
                value.to(device) for value in (node, edge, distance, target)
            )
            mask = node.abs().sum(dim=-1) != 0
            loss = torch.sqrt(F.mse_loss(model(node, mask, distance, edge), target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            node, edge, distance = (
                value[early_index].to(device) for value in (nodes, edges, distances)
            )
            mask = node.abs().sum(dim=-1) != 0
            early_loss = float(
                torch.sqrt(
                    F.mse_loss(
                        model(node, mask, distance, edge), standardized[early_index].to(device)
                    )
                ).cpu()
            )
        if early_loss < best_loss:
            best_loss, best_epoch, stale = early_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(training_options["patience"]):
            break
    if best_state is None:
        raise RuntimeError("graph fit produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    evaluation = np.arange(len(train_smiles), len(all_smiles))
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(evaluation), batch_size):
            idx = evaluation[begin : begin + batch_size]
            node, edge, distance = (value[idx].to(device) for value in (nodes, edges, distances))
            mask = node.abs().sum(dim=-1) != 0
            predictions.append(model(node, mask, distance, edge).cpu().numpy())
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    output = np.concatenate(predictions).astype(np.float64) * scale + mean
    if not np.isfinite(output).all():
        raise ValueError("non-finite graph prediction")
    return GraphFitResult(output, best_epoch, time.perf_counter() - start, peak)
