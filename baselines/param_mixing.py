"""
Shared utilities: parameter mixing across graph neighbors.

Provides two doubly-stochastic mixing weight schemes:
  - Metropolis-Hastings (MH): used by Gossip FedAvg and DecDiff-VT
  - Lazy Random Walk: used by D-PSGD (Lian et al. NeurIPS 2017)

Both produce W such that W[i] is a probability distribution over {i} ∪ N(i),
enabling gossip-style decentralized averaged mixing.

Stage 2 (trainable backbone) notes
------------------------------------
In cache_features=True (stage 1), all nodes share a frozen backbone so
only head parameters need mixing — mix_head_parameters() handles this.

In cache_features=False (stage 2), each node has its own trainable backbone.
mix_model_parameters() handles this:
  - Head params: always mixed across all same-arch and cross-arch neighbors
    (head dim is the same regardless of backbone arch since num_classes=10).
  - Backbone params: mixed only among same-architecture neighbors.
    Cross-arch neighbors are excluded from the backbone average, and the
    remaining backbone weights are renormalized to sum to 1.
  - If a node has no same-arch neighbors, its backbone is left unchanged
    (self-weight = 1 for backbone).

This is the principled approach: FedAvg on backbone requires compatible
parameter spaces; head mixing is always valid since output dim is fixed.
"""

import torch
from typing import Dict, List, Optional


def metropolis_hastings_weights(neighbor_map: Dict[int, List[int]]) -> Dict[int, Dict[int, float]]:
    """
    Metropolis-Hastings mixing weights for a symmetric graph.

    W[i][j] = 1 / (1 + max(deg_i, deg_j))  for j in N(i)
    W[i][i] = 1 - sum_{j in N(i)} W[i][j]

    Yields a doubly stochastic matrix.
    """
    degrees = {i: len(nbrs) for i, nbrs in neighbor_map.items()}
    W: Dict[int, Dict[int, float]] = {}
    for i, nbrs in neighbor_map.items():
        W[i] = {}
        off_sum = 0.0
        for j in nbrs:
            w_ij = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            W[i][j] = float(w_ij)
            off_sum += w_ij
        W[i][i] = float(1.0 - off_sum)
    return W


def lazy_random_walk_weights(neighbor_map: Dict[int, List[int]]) -> Dict[int, Dict[int, float]]:
    """
    Lazy random walk weights as used in D-PSGD (Lian et al. NeurIPS 2017).

    W[i][j] = 1 / (2 * d_max)   for j in N(i)
    W[i][i] = 1 - deg_i / (2 * d_max)

    Guaranteed doubly stochastic for any undirected graph.
    """
    degrees = {i: len(nbrs) for i, nbrs in neighbor_map.items()}
    d_max = max(degrees.values()) if degrees else 1
    d_max = max(1, d_max)

    W: Dict[int, Dict[int, float]] = {}
    for i, nbrs in neighbor_map.items():
        W[i] = {}
        for j in nbrs:
            W[i][j] = 1.0 / (2.0 * d_max)
        W[i][i] = 1.0 - degrees[i] / (2.0 * d_max)
    return W


def mix_head_parameters(
    nodes: dict,
    neighbor_map: Dict[int, List[int]],
    W: Dict[int, Dict[int, float]],
) -> None:
    """
    In-place weighted averaging of each node's HEAD parameters only.
    Used in stage 1 (cache_features=True) where backbone is frozen.

    All heads are snapshotted BEFORE any update (synchronous gossip).
    """
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    for i, node in nodes.items():
        snapshots[i] = {
            name: p.data.clone()
            for name, p in node.model.head.named_parameters()
        }

    for i, node in nodes.items():
        w_map = W[i]
        new_params: Dict[str, torch.Tensor] = {}
        for name, snap in snapshots[i].items():
            new_params[name] = torch.zeros_like(snap)

        for j, w_ij in w_map.items():
            if abs(w_ij) < 1e-14 or j not in snapshots:
                continue
            for name in new_params:
                new_params[name].add_(snapshots[j][name], alpha=float(w_ij))

        for name, p in node.model.head.named_parameters():
            p.data.copy_(new_params[name])


def mix_model_parameters(
    nodes: dict,
    neighbor_map: Dict[int, List[int]],
    W: Dict[int, Dict[int, float]],
    node_arch_map: Dict[int, str],
    default_arch: str,
) -> None:
    """
    In-place weighted averaging of FULL model parameters.
    Used in stage 2 (cache_features=False) where backbone is trainable.

    Head parameters: mixed across ALL neighbors (head dim is always 10).
    Backbone parameters: mixed only among SAME-ARCHITECTURE neighbors.
        Cross-arch neighbors are excluded and remaining weights renormalized.
        If a node has no same-arch neighbors, backbone is left unchanged.

    All parameters are snapshotted BEFORE any update (synchronous gossip).
    """
    # Snapshot full named parameters for all nodes
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    for i, node in nodes.items():
        snapshots[i] = {
            name: p.data.clone()
            for name, p in node.model.named_parameters()
        }

    for i, node in nodes.items():
        w_map  = W[i]
        i_arch = node_arch_map.get(i, default_arch)

        # Separate head and backbone param names
        head_names     = {name for name, _ in node.model.head.named_parameters()}
        # Prefix head names to match full model naming
        full_head_names = {f"head.{n}" for n in head_names}

        # For backbone: compute renormalized weights over same-arch neighbors only
        backbone_w: Dict[int, float] = {}
        for j, w_ij in w_map.items():
            if abs(w_ij) < 1e-14 or j not in snapshots:
                continue
            j_arch = node_arch_map.get(j, default_arch)
            if j_arch == i_arch:
                backbone_w[j] = w_ij

        backbone_w_sum = sum(backbone_w.values())
        if backbone_w_sum < 1e-14:
            # No same-arch neighbors — leave backbone unchanged
            backbone_w = {i: 1.0}
            backbone_w_sum = 1.0

        # Accumulate mixed parameters
        new_params: Dict[str, torch.Tensor] = {}
        for name, snap in snapshots[i].items():
            new_params[name] = torch.zeros_like(snap)

        for name in new_params:
            is_head = name in full_head_names or name.startswith("head.")

            if is_head:
                # Mix head across all neighbors using original weights
                for j, w_ij in w_map.items():
                    if abs(w_ij) < 1e-14 or j not in snapshots:
                        continue
                    if name in snapshots[j]:
                        new_params[name].add_(snapshots[j][name], alpha=float(w_ij))
            else:
                # Mix backbone only across same-arch neighbors, renormalized
                for j, w_ij in backbone_w.items():
                    if j not in snapshots:
                        continue
                    norm_w = w_ij / backbone_w_sum
                    if name in snapshots[j]:
                        new_params[name].add_(snapshots[j][name], alpha=float(norm_w))

        # Write back in-place
        for name, p in node.model.named_parameters():
            p.data.copy_(new_params[name])


def mix_parameters_auto(
    nodes: dict,
    neighbor_map: Dict[int, List[int]],
    W: Dict[int, Dict[int, float]],
    cache_features: bool,
    node_arch_map: Optional[Dict[int, str]] = None,
    default_arch: str = "mobilenet_v2",
) -> None:
    """
    Dispatcher: uses mix_head_parameters in stage 1 (cache_features=True),
    mix_model_parameters in stage 2 (cache_features=False).
    """
    if cache_features:
        mix_head_parameters(nodes, neighbor_map, W)
    else:
        mix_model_parameters(
            nodes, neighbor_map, W,
            node_arch_map or {i: default_arch for i in nodes},
            default_arch,
        )