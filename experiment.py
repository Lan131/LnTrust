#!/usr/bin/env python3
"""
Connectivity tradeoff experiments for decentralized pseudo-label sharing on CIFAR-10.

Changes vs previous version:
  - Per-node LOCAL test loaders: in skewed mode, each node's test set is sampled
    with the same SkewMixtureSampler as its training data.
  - Deployable CN-best: after training, for each node i, pick best neighbor (incl. self)
    by val accuracy on i's own val data, then report that neighbor's accuracy on i's local test.
  - Training-time neighbor weighting (--neighbor_weighting train_acc).
  - --random_models: randomly assign each node mobilenet_v2 or efficientnet_b0.
    Requires --cache_features.
  - --connection_model data_similarity: parametric preferential attachment biased by
    per-node class-histogram cosine similarity. --similarity_temp controls sharpness.
  - NPZ file saving removed; metrics are logged to CSV only.
  - GradientAlignedDiscountedUCB now smoothly blends UCB1 → grad-align over `theta`
    decay steps.  alpha = min(1, _step / theta).  The UCB1 path is unchanged.
"""

import argparse
import os
import random
import sys
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T

from torchvision.models import mobilenet_v2, efficientnet_b0
try:
    from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
except Exception:
    MobileNet_V2_Weights = None
    EfficientNet_B0_Weights = None

import matplotlib.pyplot as plt


# ----------------------------
# Stats helpers
# ----------------------------
def _mean_ci95(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mu = float(sum(xs) / n)
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    s = math.sqrt(max(0.0, var))
    hw = 1.96 * (s / math.sqrt(n))
    return mu, float(hw)


# ----------------------------
# Printing / IO helpers
# ----------------------------
def _pprint(msg: str, no_flush: bool = False) -> None:
    print(msg)
    if not no_flush:
        sys.stdout.flush()


def _maybe_write_csv_header(path: str, header_line: str) -> None:
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header_line.rstrip("\n") + "\n")


def _append_csv_lines(path: str, lines: List[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def _write_hparams(path: str, args: argparse.Namespace) -> None:
    keys = sorted(vars(args).keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k in keys:
            f.write(f"{k}: {getattr(args, k)}\n")


def closed_neighborhood_best_mean(
    neighbor_map: Dict[int, List[int]],
    accs: Dict[int, float],
) -> float:
    n = len(neighbor_map)
    if n <= 0:
        return 0.0
    vals: List[float] = []
    for i in range(n):
        hood = [i] + list(neighbor_map.get(i, []))
        vals.append(max(float(accs[j]) for j in hood))
    return float(sum(vals) / max(1, len(vals)))


# ----------------------------
# Timing utilities
# ----------------------------
def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


class TimeBreakdown:
    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = device
        self._acc: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def reset(self) -> None:
        if not self.enabled:
            return
        self._acc = {}
        self._stack = []

    def add(self, key: str, dt: float) -> None:
        if not self.enabled:
            return
        self._acc[key] = float(self._acc.get(key, 0.0) + float(dt))

    def begin(self, key: str) -> None:
        if not self.enabled:
            return
        _sync_if_cuda(self.device)
        self._stack.append((str(key), time.perf_counter()))

    def end(self, key: str) -> None:
        if not self.enabled:
            return
        _sync_if_cuda(self.device)
        t1 = time.perf_counter()
        if self._stack and self._stack[-1][0] == key:
            _, t0 = self._stack.pop()
            self.add(key, t1 - t0)
            return
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == key:
                _, t0 = self._stack.pop(idx)
                self.add(key, t1 - t0)
                return

    def as_dict(self) -> Dict[str, float]:
        return dict(self._acc)


# ----------------------------
# Pretrained backbone helpers
# ----------------------------
def _get_mobilenet_v2_pretrained():
    if MobileNet_V2_Weights is not None:
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    return mobilenet_v2(pretrained=True)


def _get_efficientnet_b0_pretrained():
    if EfficientNet_B0_Weights is not None:
        return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    return efficientnet_b0(pretrained=True)


class FrozenBackboneHead(nn.Module):
    def __init__(self, arch: str, num_classes: int = 10, head_dropout_p: float = 0.5,
                 baseline_non_linearity: bool = False):
        super().__init__()
        arch = str(arch).lower()
        self.arch = arch

        if arch == "mobilenet_v2":
            base = _get_mobilenet_v2_pretrained()
            self.backbone = base.features
            for p in self.backbone.parameters():
                p.requires_grad = False
            feat_dim = int(base.last_channel)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if baseline_non_linearity:
                self.head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.Tanh(),
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(256, int(num_classes)),
                )
            else:
                self.head = nn.Sequential(
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(feat_dim, int(num_classes)),
                )
            for p in self.head.parameters():
                p.requires_grad = True

        elif arch == "efficientnet_b0":
            base = _get_efficientnet_b0_pretrained()
            self.backbone = base.features
            for p in self.backbone.parameters():
                p.requires_grad = False
            feat_dim = int(base.classifier[-1].in_features)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if baseline_non_linearity:
                self.head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(256, int(num_classes)),
                )
            else:
                self.head = nn.Sequential(
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(feat_dim, int(num_classes)),
                )
            for p in self.head.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown arch: {arch}. Choose from ['mobilenet_v2', 'efficientnet_b0'].")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.pool(f)
        return torch.flatten(f, 1)

    def forward_head(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


class HeadOnly(nn.Module):
    def __init__(self, arch: str, feat_dim: int = 1280, num_classes: int = 10,
                 head_dropout_p: float = 0.5, baseline_non_linearity: bool = False):
        super().__init__()
        arch = str(arch).lower()
        self.arch = arch

        if arch == "mobilenet_v2":
            if baseline_non_linearity:
                # Two-layer MLP head — breaks the purely linear averaging property
                # of gossip FedAvg and makes the feature-space combination nonlinear.
                # ReLU used for both archs so --baseline_non_linearity is a clean
                # experiment-level toggle: off = everyone linear, on = everyone MLP.
                self.head = nn.Sequential(
                    nn.Linear(int(feat_dim), 256),
                    nn.Tanh(),
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(256, int(num_classes)),
                )
            else:
                self.head = nn.Sequential(
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(int(feat_dim), int(num_classes)),
                )
        elif arch == "efficientnet_b0":
            if baseline_non_linearity:
                self.head = nn.Sequential(
                    nn.Linear(int(feat_dim), 256),
                    nn.ReLU(),
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(256, int(num_classes)),
                )
            else:
                self.head = nn.Sequential(
                    nn.Dropout(p=float(head_dropout_p)),
                    nn.Linear(int(feat_dim), int(num_classes)),
                )
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward_head(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ----------------------------
# Dataset wrappers
# ----------------------------
class FeatureTensorDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        assert feats.dim() == 2
        assert labels.dim() == 1
        assert feats.size(0) == labels.size(0)
        self.feats = feats.contiguous()
        self.labels = labels.contiguous()

    def __len__(self):
        return int(self.feats.size(0))

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


class UnlabeledWrapper(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x


# ----------------------------
# Training data skew helpers
# ----------------------------
def _make_favored_class_map(num_nodes: int, num_classes: int, strategy: str, seed: int) -> Dict[int, int]:
    if strategy == "round_robin":
        return {i: (i % num_classes) for i in range(num_nodes)}
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        return {i: int(rng.integers(0, num_classes)) for i in range(num_nodes)}
    raise ValueError(f"Unknown skew_strategy: {strategy}")


class SkewMixtureSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        labels: np.ndarray,
        favored_class: int,
        skew_factor: float,
        min_other_frac: float,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.favored_class = int(favored_class)
        self.skew_factor = float(skew_factor)
        self.min_other_frac = float(min_other_frac)
        self.n = int(self.labels.shape[0])
        self.fav_idx = np.flatnonzero(self.labels == self.favored_class).astype(np.int64)
        self.oth_idx = np.flatnonzero(self.labels != self.favored_class).astype(np.int64)
        self.gen = generator

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        n = self.n
        if n <= 0:
            return iter([])
        if self.fav_idx.size == 0 or self.oth_idx.size == 0:
            return iter(torch.randint(low=0, high=n, size=(n,), generator=self.gen).tolist())
        sf = max(1.0, float(self.skew_factor))
        fav_w = sf * float(self.fav_idx.size)
        oth_w = 1.0 * float(self.oth_idx.size)
        fav_frac_target = fav_w / max(1e-12, (fav_w + oth_w))
        min_other = max(0.0, min(1.0, float(self.min_other_frac)))
        fav_frac = min(fav_frac_target, 1.0 - min_other)
        n_fav = max(1, min(n - 1, int(round(n * fav_frac))))
        n_other = n - n_fav
        fav = self.fav_idx[
            torch.randint(low=0, high=self.fav_idx.size, size=(n_fav,), generator=self.gen).numpy()
        ]
        oth = self.oth_idx[
            torch.randint(low=0, high=self.oth_idx.size, size=(n_other,), generator=self.gen).numpy()
        ]
        out = np.concatenate([fav, oth], axis=0)
        perm = torch.randperm(out.shape[0], generator=self.gen).numpy()
        return iter(out[perm].tolist())


def _make_skewed_loader(
    dataset: Dataset,
    labels_np: np.ndarray,
    favored_class: int,
    skew_factor: float,
    skew_min_other_frac: float,
    seed: int,
    node_id: int,
    batch_size: int,
    num_workers: int,
    seed_offset: int = 17,
) -> DataLoader:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) * 1_000_003 + int(node_id) * 10_007 + seed_offset)
    sampler = SkewMixtureSampler(
        labels=labels_np,
        favored_class=int(favored_class),
        skew_factor=float(skew_factor),
        min_other_frac=float(skew_min_other_frac),
        generator=g,
    )
    persistent = num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )


def _make_train_loader(
    train_subset: Dataset,
    batch_size: int,
    num_workers: int,
    training_data_mode: str,
    favored_class: Optional[int],
    skew_factor: float,
    skew_min_other_frac: float,
    seed: int,
    node_id: int,
) -> DataLoader:
    persistent = (num_workers > 0)
    if training_data_mode == "iid":
        return DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
        )
    if training_data_mode == "skewed":
        if favored_class is None:
            return DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
            )
        ys: Optional[np.ndarray] = None
        if hasattr(train_subset, "labels"):
            ys = train_subset.labels.cpu().numpy()
        elif isinstance(train_subset, Subset) and hasattr(train_subset.dataset, "targets"):
            base_targets = np.asarray(train_subset.dataset.targets, dtype=np.int64)
            ys = base_targets[np.asarray(train_subset.indices, dtype=np.int64)]
        if ys is None or len(ys) == 0:
            return DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
            )
        return _make_skewed_loader(
            dataset=train_subset, labels_np=ys, favored_class=favored_class,
            skew_factor=skew_factor, skew_min_other_frac=skew_min_other_frac,
            seed=seed, node_id=node_id, batch_size=batch_size,
            num_workers=num_workers, seed_offset=17,
        )
    raise ValueError(f"Unknown training_data_mode: {training_data_mode}")


# ----------------------------
# Fast unlabeled buffers
# ----------------------------
class UnlabeledBuf:
    def __init__(self, x_cpu: torch.Tensor):
        self.x = x_cpu.contiguous()
        self.n = int(x_cpu.size(0))

    def sample_with_replacement(self, batch_size: int, generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
        if self.n <= 0:
            return None
        idx = torch.randint(low=0, high=self.n, size=(int(batch_size),), device="cpu", generator=generator)
        return self.x[idx]

    def make_perm(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.randperm(self.n, generator=generator, device="cpu")


# ----------------------------
# UCB Neighbor Bandit
# ----------------------------
class UCBNeighborBandit:
    """
    Per-node UCB1 bandit for neighbor selection.
    Reward: score(i,j) = mean_{x in U_i}  sum_c  w_i[c] * P_j(c|x)
    """

    def __init__(self, neighbor_map: Dict[int, List[int]], c: float = 1.0):
        self.c = float(c)
        self.Q: Dict[int, Dict[int, float]] = {
            i: {j: 0.5 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        self.N: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        self.T: Dict[int, int] = {i: 0 for i in neighbor_map}

    def update(self, i: int, j: int, reward: float) -> None:
        n = self.N[i][j] + 1
        self.N[i][j] = n
        self.Q[i][j] += (float(reward) - self.Q[i][j]) / n

    def set_q(self, i: int, j: int, value: float, n_pseudo: int = 1) -> None:
        """Seed Q from bootstrap scores.  Also sets N so the arm is not treated
        as unvisited (which would bypass bootstrap values entirely)."""
        self.Q[i][j] = float(value)
        self.N[i][j] = max(self.N[i][j], int(n_pseudo))

    def ucb_score(self, i: int, j: int) -> float:
        t = max(1, self.T[i])
        n = max(1, self.N[i][j])
        return self.Q[i][j] + self.c * math.sqrt(math.log(1 + t) / n)

    def select_arm(self, i: int) -> Optional[int]:
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return None
        unvisited = [j for j in nbrs if self.N[i][j] == 0]
        if unvisited:
            return unvisited[0]
        return max(nbrs, key=lambda j: self.ucb_score(i, j))

    def best_arm(self, i: int) -> Optional[int]:
        if not self.Q[i]:
            return None
        return max(self.Q[i], key=self.Q[i].get)

    def softmax_weights(self, i: int, temperature: float = 0.05) -> Dict[int, float]:
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return {}
        qs = np.array([self.Q[i][j] for j in nbrs], dtype=np.float64)
        qs = (qs - qs.max()) / max(temperature, 1e-8)
        ws = np.exp(qs)
        ws /= ws.sum()
        return {j: float(ws[k]) for k, j in enumerate(nbrs)}

    def deploy_weights(self, i: int, self_weight: float = 0.10) -> Dict[int, float]:
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return {i: 1.0}
        nw = self.softmax_weights(i, temperature=0.05)
        result = {j: w * (1.0 - self_weight) for j, w in nw.items()}
        result[i] = float(self_weight)
        return result


# ----------------------------
# Deployment UCB Bandit
# ----------------------------
class DeploymentUCBBandit:
    """Scalar UCB1 bandit for deployment-time neighbor selection.

    Separate from the training bandit. Arms are (node i, neighbor j) pairs.
    Reward = IW-weighted confidence of model j on node i's unlabeled buffer.
    Observable without labels — confidence is a reliable proxy for deployment
    quality since it tracks how well j's model generalizes to node i's distribution.

    All arms are updated simultaneously at rebootstrap time (cheap with feature
    cache), so exploration is over WHICH neighbor to commit to across rounds,
    not within a single round. UCB ensures we periodically re-evaluate
    neighbors that haven't been tried recently as their models improve.

    Score: UCB1  mu_j + c * sqrt(ln(T) / N_j)
    """

    def __init__(
        self,
        neighbor_map: Dict[int, List[int]],
        c: float = 1.0,
    ):
        self.c = float(c)
        # Running mean reward per (node, neighbor) arm
        self.mu: Dict[int, Dict[int, float]] = {
            i: {j: 0.0 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        # Pull count per arm
        self.N: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        # Total pulls per node
        self.T: Dict[int, int] = {i: 0 for i in neighbor_map}

    def update(self, i: int, j: int, reward: float) -> None:
        """Update arm (i, j) with observed reward."""
        self.N[i][j] += 1
        self.T[i]    += 1
        self.mu[i][j] += (reward - self.mu[i][j]) / self.N[i][j]

    def ucb_score(self, i: int, j: int) -> float:
        t = max(1, self.T[i])
        n = max(1, self.N[i][j])
        return self.mu[i][j] + self.c * math.sqrt(math.log(t) / n)

    def best_arm(self, i: int, hood: Optional[List[int]] = None) -> Optional[int]:
        """Pick the arm with highest UCB score. hood = [i] + neighbors."""
        candidates = hood if hood is not None else list(self.mu[i].keys())
        if not candidates:
            return None
        # Include self (index i) if not already in mu — score 0
        return max(candidates, key=lambda j: self.ucb_score(i, j) if j in self.mu[i] else 0.0)

    def seed(self, i: int, j: int, reward: float, n_pseudo: int = 3) -> None:
        """Bootstrap arm (i, j) with a pseudo-count so exploration starts reasonable."""
        self.mu[i][j] = float(reward)
        self.N[i][j]  = max(self.N[i][j], int(n_pseudo))
        self.T[i]     = max(self.T[i], int(n_pseudo) * max(1, len(self.mu[i])))


# ----------------------------
# Gradient-Aligned Discounted UCB Bandit
# ----------------------------
class GradientAlignedDiscountedUCB:
    """
    Per-node, per-class discounted UCB bandit for neighbor selection.

    Blending schedule
    -----------------
    The bandit starts as a pure UCB1 confidence bandit and transitions to a
    gradient-alignment bandit over ``theta`` decay steps.

        alpha = min(1.0, _step / theta)

    where ``_step`` is incremented each time ``decay()`` is called (once per
    pseudo epoch).  All public methods (``mu``, ``ucb_score``, ``best_arm``,
    ``softmax_weights``) use the blended estimate; the underlying accumulators
    for the two reward signals are kept separate so neither contaminates the
    other's running mean.

    Accumulators
    ------------
    S / N        -- gradient-cosine reward (range [0, 1] after mapping [-1,1]→[0,1])
    S_ucb1 / N_ucb1 -- confidence-based reward: mean_x P_j(c|x)

    mu(i,j,c) = (1 - alpha) * mu_ucb1(i,j,c)  +  alpha * mu_grad(i,j,c)

    ``set_q`` (called by the bootstrap) seeds only the UCB1 side, since
    bootstrap scores are confidence-based.  The grad-align side starts at zero
    and is populated by ``update()`` calls during training.
    """

    def __init__(
        self,
        neighbor_map: Dict[int, List[int]],
        num_classes: int = 10,
        c: float = 1.0,
        gamma: float = 0.99,
        theta: int = 150,
    ):
        self.c = float(c)
        self.gamma = float(gamma)
        self.theta = int(theta)
        self.num_classes = int(num_classes)

        # Gradient-alignment accumulators
        self.S: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        self.N: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }

        # UCB1 (confidence-based) accumulators — drives behaviour when alpha ≈ 0
        self.S_ucb1: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        self.N_ucb1: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }

        self.T: Dict[int, int] = {i: 0 for i in neighbor_map}
        # Number of decay() calls; drives the UCB1→grad-align blend.
        self._step: int = 0

    # ------------------------------------------------------------------
    # Blend schedule
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Fraction of grad-align in the blended estimate. 0 at init, 1 after theta steps."""
        return float(min(1.0, self._step / max(1, self.theta)))

    # ------------------------------------------------------------------
    # Per-signal means
    # ------------------------------------------------------------------
    def mu_grad(self, i: int, j: int, c: int) -> float:
        """Running mean of the gradient-cosine reward for arm (i,j,c)."""
        n = float(self.N[i][j][c])
        return float(self.S[i][j][c]) / n if n > 1e-9 else 0.0

    def mu_ucb1(self, i: int, j: int, c: int) -> float:
        """Running mean of the confidence-based (UCB1) reward for arm (i,j,c)."""
        n = float(self.N_ucb1[i][j][c])
        return float(self.S_ucb1[i][j][c]) / n if n > 1e-9 else 0.0

    def mu(self, i: int, j: int, c: int) -> float:
        """Blended mean: (1-alpha)*mu_ucb1 + alpha*mu_grad."""
        a = self.alpha
        return (1.0 - a) * self.mu_ucb1(i, j, c) + a * self.mu_grad(i, j, c)

    # ------------------------------------------------------------------
    # UCB score (uses blended mu; exploration term uses combined counts)
    # ------------------------------------------------------------------
    def ucb_score(self, i: int, j: int, c: int) -> float:
        t = max(1, self.T[i])
        # Use the total effective count across both reward signals so the
        # exploration bonus shrinks at the right rate regardless of alpha.
        n_grad  = float(self.N[i][j][c])
        n_ucb1  = float(self.N_ucb1[i][j][c])
        a       = self.alpha
        n_eff   = max(1e-9, (1.0 - a) * n_ucb1 + a * n_grad)
        return self.mu(i, j, c) + self.c * math.sqrt(math.log(1 + t) / n_eff)

    # ------------------------------------------------------------------
    # Discount / step counter
    # ------------------------------------------------------------------
    def decay(self) -> None:
        """Geometric discount applied to both accumulator sets; advances _step."""
        for i in self.S:
            for j in self.S[i]:
                self.S[i][j]     *= self.gamma
                self.N[i][j]     *= self.gamma
                self.S_ucb1[i][j] *= self.gamma
                self.N_ucb1[i][j] *= self.gamma
        self._step += 1

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self, i: int, j: int, c: int, reward: float) -> None:
        """Record a gradient-alignment reward for arm (i, j, c)."""
        self.S[i][j][c] += float(reward)
        self.N[i][j][c] += 1.0

    def update_ucb1(self, i: int, j: int, c: int, reward: float) -> None:
        """Record a confidence-based (UCB1) reward for arm (i, j, c)."""
        self.S_ucb1[i][j][c] += float(reward)
        self.N_ucb1[i][j][c] += 1.0

    # ------------------------------------------------------------------
    # Bootstrap (seeds the UCB1 side only — bootstrap uses confidence scores)
    # ------------------------------------------------------------------
    def set_q(self, i: int, j: int, value: float) -> None:
        """Seed the UCB1 accumulators from bootstrap confidence scores.

        The grad-align accumulators are left at zero: they have no gradient
        data yet and should not be pre-populated with confidence proxies.
        """
        if j in self.S_ucb1[i]:
            self.S_ucb1[i][j][:] = float(value)
            self.N_ucb1[i][j][:] = 1.0

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------
    def select_arm(self, i: int, c: int) -> Optional[int]:
        nbrs = list(self.S[i].keys())
        if not nbrs:
            return None
        # An arm is "unvisited" if neither accumulator has seen it for class c.
        unvisited = [
            j for j in nbrs
            if self.N[i][j][c] < 1e-9 and self.N_ucb1[i][j][c] < 1e-9
        ]
        if unvisited:
            return unvisited[0]
        return max(nbrs, key=lambda j: self.ucb_score(i, j, c))

    def best_arm(self, i: int, class_weights: Optional[np.ndarray] = None) -> Optional[int]:
        nbrs = list(self.S[i].keys())
        if not nbrs:
            return None
        C = self.num_classes
        w = class_weights if class_weights is not None else np.full(C, 1.0 / C)
        scores = {j: float(np.dot([self.mu(i, j, c) for c in range(C)], w)) for j in nbrs}
        return max(scores, key=scores.__getitem__)

    def softmax_weights(self, i: int, class_weights: Optional[np.ndarray] = None,
                        temperature: float = 0.05) -> Dict[int, float]:
        nbrs = list(self.S[i].keys())
        if not nbrs:
            return {}
        C = self.num_classes
        w = class_weights if class_weights is not None else np.full(C, 1.0 / C)
        qs = np.array(
            [float(np.dot([self.mu(i, j, c) for c in range(C)], w)) for j in nbrs],
            dtype=np.float64,
        )
        qs = (qs - qs.max()) / max(temperature, 1e-8)
        ws = np.exp(qs)
        ws /= ws.sum()
        return {j: float(ws[k]) for k, j in enumerate(nbrs)}


# ----------------------------
# Contextual Diagonal LinUCB Bandit (for Algorithm 4 / entropy_ucb)
# ----------------------------
class ContextualUCBNeighborBandit:
    """
    Per-node Diagonal Linear UCB bandit for neighbor selection.

    Approximates the full covariance matrix A with its diagonal, reducing
    storage from O(d²) to O(d) and update/score cost from O(d²) to O(d).
    With feat_dim=1280 and O(N×K) arms this makes the difference between
    ~26 GB of numpy arrays and ~26 MB.

    Context normalization
    ---------------------
    Raw MobileNetV2/EfficientNet features have L2 norm ≈ 5-18 in 1280-d
    space.  Without normalization the exploration bonus c·‖z/√a‖ starts at
    c·‖z‖ ≈ 5-18 — 5-18× larger than the [0,1] reward range — making the
    bandit effectively random regardless of c.  All contexts are L2-normalized
    to unit norm before update/score so the exploration bonus is always ≈ c
    and c=1.0 works as intended.

    Diagonal update rule (online ridge regression on normalized context):
        z̃  = z / ‖z‖₂
        a_j  ← a_j + z̃ ⊙ z̃          (accumulate squared unit features)
        b_j  ← b_j + r * z̃           (accumulate reward-weighted unit features)
        θ_j  = b_j / a_j             (ridge estimate per-feature)

    UCB score:
        µ = θ_j · z̃
        σ = √( z̃² / a_j · 1 )  = ‖z̃ / √a_j‖
        score = µ + c * σ

    Bootstrap seeding
    -----------------
    set_q seeds both the scalar Q tracker (for softmax_weights/deployment)
    AND the LinUCB (a, b) state using a unit context, so the bandit starts
    with meaningful reward estimates rather than zero.

    Scalar Q / N trackers are kept for softmax_weights / best_arm /
    deploy_weights compatibility (used at deployment time, no context needed).
    """

    def __init__(
        self,
        neighbor_map: Dict[int, List[int]],
        feat_dim: int = 1280,
        c: float = 1.0,
        lambda_: float = 1.0,
        num_classes: int = 10,
    ):
        self.c         = float(c)
        self.feat_dim  = int(feat_dim)
        self.lambda_   = float(lambda_)
        self.num_classes = int(num_classes)

        # ── Class-agnostic arms (original) ────────────────────────────
        # Diagonal LinUCB state — two vectors per arm instead of a matrix.
        self.a: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.full(feat_dim, lambda_, dtype=np.float32) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        self.b: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(feat_dim, dtype=np.float32) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }

        # Scalar trackers (interface compatibility — used by softmax_weights etc.)
        self.Q: Dict[int, Dict[int, float]] = {
            i: {j: 0.5 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        self.N: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in nbrs} for i, nbrs in neighbor_map.items()
        }
        self.T: Dict[int, int] = {i: 0 for i in neighbor_map}

        # ── Per-class arms ─────────────────────────────────────────────
        # One (a_c, b_c) pair per (node i, neighbor j, class c).
        # Context = mean feature of unlabeled examples student predicts as c.
        # Reward  = teacher j's mean confidence on those class-c examples - threshold.
        # This lets the bandit learn "for class-c images, which neighbor is best?"
        # — enabling per-class teacher specialization and potentially beating
        # the omniscient oracle (which picks one best teacher for the whole node).
        self.a_c: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {
            i: {j: {c: np.full(feat_dim, lambda_, dtype=np.float32)
                    for c in range(num_classes)}
                for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        self.b_c: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {
            i: {j: {c: np.zeros(feat_dim, dtype=np.float32)
                    for c in range(num_classes)}
                for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        # Per-class null arms — one per (node i, class c)
        self.null_a_c: Dict[int, Dict[int, np.ndarray]] = {
            i: {c: np.full(feat_dim, lambda_, dtype=np.float32)
                for c in range(num_classes)}
            for i in neighbor_map
        }
        self.null_b_c: Dict[int, Dict[int, np.ndarray]] = {
            i: {c: np.zeros(feat_dim, dtype=np.float32)
                for c in range(num_classes)}
            for i in neighbor_map
        }

        # ── Class-agnostic null arm ────────────────────────────────────
        self.null_a: Dict[int, np.ndarray] = {
            i: np.full(feat_dim, lambda_, dtype=np.float32) for i in neighbor_map
        }
        self.null_b: Dict[int, np.ndarray] = {
            i: np.zeros(feat_dim, dtype=np.float32) for i in neighbor_map
        }
        self.null_N: Dict[int, int] = {i: 0 for i in neighbor_map}

        # ── Per-class scalar running means (label-free deployment) ─────────
        # pc_mu[i][j][c] = running mean of P_j(c | x) for x predicted as class c
        # by student i.  Seeded from bootstrap IW-confidence at set_q time.
        # Updated in update_class alongside the LinUCB accumulators.
        # Used by deploy_criteria="bandit_class" to pick per-class best teachers
        # weighted by node i's class distribution — no val data required.
        self.pc_mu: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }
        self.pc_n: Dict[int, Dict[int, np.ndarray]] = {
            i: {j: np.zeros(num_classes, dtype=np.float64) for j in nbrs}
            for i, nbrs in neighbor_map.items()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(z: np.ndarray) -> np.ndarray:
        """L2-normalize context to unit norm; no-op if already zero."""
        norm = float(np.linalg.norm(z))
        return (z / norm).astype(np.float32) if norm > 1e-9 else z.astype(np.float32)

    def update(self, i: int, j: int, context: np.ndarray, reward: float) -> None:
        """Diagonal ridge update on L2-normalized context; O(d) cost."""
        z = self._normalize(context.ravel())
        self.a[i][j] += z * z
        self.b[i][j] += float(reward) * z

        # Scalar tracker
        n = self.N[i][j] + 1
        self.N[i][j] = n
        self.Q[i][j] += (float(reward) - self.Q[i][j]) / n

    def ucb_score(self, i: int, j: int, context: np.ndarray) -> float:
        """Diagonal LinUCB score on L2-normalized context; O(d) cost."""
        z     = self._normalize(context.ravel())
        theta = self.b[i][j] / self.a[i][j]          # element-wise ridge estimate
        mu    = float(np.dot(theta, z))
        var   = float(np.dot(z * z, 1.0 / self.a[i][j]))
        return mu + self.c * math.sqrt(max(0.0, var))

    def set_q(self, i: int, j: int, value: float, n_pseudo: int = 1) -> None:
        """Seed scalar Q AND LinUCB state from bootstrap confidence scores.

        Uses a uniform unit context (all directions equally likely) so the
        initial theta estimate reflects the IW-confidence bootstrap score.
        Without seeding the LinUCB, the bandit ignores the bootstrap entirely
        for teacher *selection* (ucb_score uses a/b, not Q/N).
        Also seeds per-class arms with the same value so best_arm_contextual
        starts from a meaningful baseline rather than all-zeros.
        """
        self.Q[i][j] = float(value)
        self.N[i][j] = max(self.N[i][j], int(n_pseudo))
        # Seed class-agnostic LinUCB arm
        unit_z = np.full(self.feat_dim, 1.0 / math.sqrt(self.feat_dim), dtype=np.float32)
        self.a[i][j] += unit_z * unit_z   # adds 1/feat_dim per dimension
        self.b[i][j] += float(value) * unit_z
        # Seed per-class arms and scalar means with the bootstrap IW-confidence
        # score so bandit_class deployment starts from a meaningful ranking.
        for c in range(self.num_classes):
            self.a_c[i][j][c] += unit_z * unit_z
            self.b_c[i][j][c] += float(value) * unit_z
            self.pc_mu[i][j][c] = float(value)
            self.pc_n[i][j][c]  = float(max(1, int(n_pseudo)))

    def update_null(self, i: int, context: np.ndarray, reward: float = 0.0) -> None:
        """Record one observation of the null arm.

        reward should be (student_conf - threshold) so the null arm represents
        'keep training myself' rather than a fixed zero. This ensures that a
        strong node (high student confidence) correctly outcompetes weak teachers
        that nominally pass the fixed threshold.
        """
        z = self._normalize(context.ravel())
        self.null_a[i] += z * z
        self.null_b[i] += float(reward) * z
        self.null_N[i] += 1

    def null_ucb_score(self, i: int, context: np.ndarray) -> float:
        """UCB score of the null arm — 'do nothing / keep self-training'.

        mu = running mean of (student_conf - threshold) rewards.
        Early on: large exploration bonus → bandit still tries teachers.
        Later: mu converges to the student's own quality above threshold.
        Teachers must beat this to be worth distilling from.
        """
        z     = self._normalize(context.ravel())
        theta = self.null_b[i] / self.null_a[i]
        mu    = float(np.dot(theta, z))
        var   = float(np.dot(z * z, 1.0 / self.null_a[i]))
        return mu + self.c * math.sqrt(max(0.0, var))

    # ── Per-class arm methods ──────────────────────────────────────────
    def update_class(self, i: int, j: int, c: int,
                     context: np.ndarray, reward: float) -> None:
        """Diagonal ridge update for per-class arm (i, j, c)."""
        z = self._normalize(context.ravel())
        self.a_c[i][j][c] += z * z
        self.b_c[i][j][c] += float(reward) * z
        # Scalar running mean for label-free deployment (no context needed)
        n = self.pc_n[i][j][c] + 1.0
        self.pc_n[i][j][c]  = n
        self.pc_mu[i][j][c] += (float(reward) - self.pc_mu[i][j][c]) / n

    def ucb_score_class(self, i: int, j: int, c: int,
                        context: np.ndarray) -> float:
        """UCB score for per-class arm (i, j, c)."""
        z     = self._normalize(context.ravel())
        theta = self.b_c[i][j][c] / self.a_c[i][j][c]
        mu    = float(np.dot(theta, z))
        var   = float(np.dot(z * z, 1.0 / self.a_c[i][j][c]))
        return mu + self.c * math.sqrt(max(0.0, var))

    def update_null_class(self, i: int, c: int,
                          context: np.ndarray, reward: float) -> None:
        """Update the per-class null arm for node i, class c."""
        z = self._normalize(context.ravel())
        self.null_a_c[i][c] += z * z
        self.null_b_c[i][c] += float(reward) * z

    def null_ucb_score_class(self, i: int, c: int,
                             context: np.ndarray) -> float:
        """UCB score of the per-class null arm for node i, class c."""
        z     = self._normalize(context.ravel())
        theta = self.null_b_c[i][c] / self.null_a_c[i][c]
        mu    = float(np.dot(theta, z))
        var   = float(np.dot(z * z, 1.0 / self.null_a_c[i][c]))
        return mu + self.c * math.sqrt(max(0.0, var))

    def best_arm(self, i: int) -> Optional[int]:
        if not self.Q[i]:
            return None
        return max(self.Q[i], key=self.Q[i].get)

    def best_arm_contextual(
        self,
        i: int,
        class_contexts: Dict[int, np.ndarray],
        class_weights: np.ndarray,
    ) -> Optional[int]:
        """Pick best neighbor for deployment using per-class LinUCB arms.

        score(j) = sum_c w_i[c] * mu_c(i, j, z_c)

        Uses the posterior mean (not UCB) — exploration is for training,
        deployment should exploit the learned estimates. Weights by node
        i's skew distribution so we pick the teacher that is best for
        the classes node i actually cares about.
        Falls back to scalar Q[i] if no class contexts available.
        """
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return None
        if not class_contexts:
            return max(nbrs, key=lambda j: self.Q[i].get(j, 0.0))
        scores: Dict[int, float] = {}
        for j in nbrs:
            s = 0.0; w_total = 0.0
            for c, z_ctx in class_contexts.items():
                w_c = float(class_weights[c]) if c < len(class_weights) else 0.0
                if w_c <= 0.0:
                    continue
                z     = self._normalize(z_ctx.ravel())
                theta = self.b_c[i][j][c] / self.a_c[i][j][c]
                s       += w_c * float(np.dot(theta, z))
                w_total += w_c
            scores[j] = s / max(w_total, 1e-9)
        # Fall back to bootstrapped scalar Q until per-class arms have
        # enough real updates to differentiate neighbors contextually.
        # One full pass = num_classes * num_neighbors updates.
        min_updates = self.num_classes * max(1, len(nbrs))
        if self.T.get(i, 0) < min_updates:
            return max(nbrs, key=lambda j: self.Q[i].get(j, 0.0))
        return max(nbrs, key=lambda j: scores[j])

    def softmax_weights(self, i: int, temperature: float = 0.05) -> Dict[int, float]:
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return {}
        qs  = np.array([self.Q[i][j] for j in nbrs], dtype=np.float64)
        qs  = (qs - qs.max()) / max(temperature, 1e-8)
        ws  = np.exp(qs);  ws /= ws.sum()
        return {j: float(ws[k]) for k, j in enumerate(nbrs)}

    def deploy_weights(self, i: int, self_weight: float = 0.10) -> Dict[int, float]:
        nbrs = list(self.Q[i].keys())
        if not nbrs:
            return {i: 1.0}
        nw = self.softmax_weights(i, temperature=0.05)
        result = {j: w * (1.0 - self_weight) for j, w in nw.items()}
        result[i] = float(self_weight)
        return result


def _grad_cosine_sim(
    model: nn.Module,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    z_pseudo: torch.Tensor,
    p_pseudo: torch.Tensor,
    iw: float = 1.0,
    eps: float = 1e-8,
    cache_features: bool = True,
) -> float:
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not head_params or z_val.size(0) == 0 or z_pseudo.size(0) == 0:
        return 0.0
    model.train()
    _fwd = model.forward_head if cache_features else model
    logits_val = _fwd(z_val)
    loss_val = F.cross_entropy(logits_val, y_val)
    grads_val = torch.autograd.grad(loss_val, head_params, retain_graph=False,
                                    create_graph=False, allow_unused=True)
    g_val = torch.cat([g.detach().reshape(-1) for g in grads_val if g is not None])

    y_pseudo = p_pseudo.argmax(dim=-1)  # hard labels
    logits_ps = _fwd(z_pseudo)
    loss_ps = float(iw) * F.cross_entropy(logits_ps, y_pseudo)
    grads_ps = torch.autograd.grad(loss_ps, head_params, retain_graph=False,
                                   create_graph=False, allow_unused=True)
    g_ps = torch.cat([g.detach().reshape(-1) for g in grads_ps if g is not None])

    norm = g_val.norm() * g_ps.norm()
    if norm < eps:
        return 0.0
    raw = float(torch.dot(g_val, g_ps).item() / norm.item())
    return float((raw + 1.0) / 2.0)  # map [-1,1] -> [0,1]


def _compute_head_val_gradient(
    model: nn.Module,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Flat head-parameter gradient of CE loss on a validation batch.

    Returns None if the head has no trainable parameters or the batch is empty.
    Does NOT modify model weights; uses torch.autograd.grad so .grad buffers
    are untouched.
    """
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not head_params or z_val.size(0) == 0:
        return None
    model.train()
    logits = model.forward_head(z_val) if hasattr(model, 'forward_head') and z_val.dim() <= 2 else model(z_val)
    loss = F.cross_entropy(logits, y_val)
    grads = torch.autograd.grad(
        loss, head_params, retain_graph=False, create_graph=False, allow_unused=True
    )
    parts = [g.detach().reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    g = torch.cat(parts)
    return g if g.numel() > 0 else None


def _pseudo_grad_alignment(
    model: nn.Module,
    g_val: torch.Tensor,
    z_pseudo: torch.Tensor,
    p_pseudo: torch.Tensor,
    iw: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Raw cosine similarity in [-1, 1] between a pre-computed validation gradient
    ``g_val`` and the IS-weighted pseudo-label gradient produced by ``model`` on
    ``z_pseudo``.

    Intentionally NOT remapped to [0, 1] so the τ threshold in Algorithm 1
    operates on a natural scale (τ = 0 means "any positive alignment").
    """
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not head_params or z_pseudo.size(0) == 0:
        return 0.0
    model.train()
    y_pseudo = p_pseudo.argmax(dim=-1)
    _fwd2 = model.forward_head if z_pseudo.dim() <= 2 else model
    logits = _fwd2(z_pseudo)
    loss = float(iw) * F.cross_entropy(logits, y_pseudo)
    grads = torch.autograd.grad(
        loss, head_params, retain_graph=False, create_graph=False, allow_unused=True
    )
    parts = [g.detach().reshape(-1) for g in grads if g is not None]
    if not parts:
        return 0.0
    g_ps = torch.cat(parts)
    norm = g_val.norm() * g_ps.norm()
    if norm < eps:
        return 0.0
    return float(torch.dot(g_val, g_ps).item() / norm.item())


# ----------------------------
# Connectivity graph generation
# ----------------------------
def _edges_from_neighbor_map(neighbor_map: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    edges = set()
    for i, nbrs in neighbor_map.items():
        for j in nbrs:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(edges)


def _uniform_random_graph(n: int, p: float, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(int(seed))
    neighbor_sets: Dict[int, set] = {i: set() for i in range(n)}
    e_max = (n * (n - 1)) // 2
    target_edges = int(np.floor(float(p) * e_max))
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    for i, j in all_edges[:target_edges]:
        neighbor_sets[i].add(j)
        neighbor_sets[j].add(i)
    return {i: sorted(list(neighbor_sets[i])) for i in range(n)}


def _barabasi_albert_graph_with_target_edges(n: int, target_edges: int, seed: int, ba_m: int = 0) -> Dict[int, List[int]]:
    rng = np.random.default_rng(int(seed))
    if n <= 1:
        return {0: []}
    e_max = (n * (n - 1)) // 2
    target_edges = int(max(0, min(e_max, target_edges)))
    # ba_m=0 means derive m from target_edges (original behaviour).
    # ba_m>0 overrides with the explicit preferential-attachment parameter.
    if ba_m > 0:
        m = int(max(1, min(n - 1, int(ba_m))))
    else:
        m = int(max(1, min(n - 1, round(max(1.0, target_edges / max(1.0, n))))))
    m0 = int(min(n, max(2, m + 1)))
    neighbor_sets: Dict[int, set] = {i: set() for i in range(n)}
    degrees = np.zeros(n, dtype=np.int64)
    for i in range(m0):
        for j in range(i + 1, m0):
            neighbor_sets[i].add(j); neighbor_sets[j].add(i)
            degrees[i] += 1; degrees[j] += 1
    for new_node in range(m0, n):
        deg = degrees[:new_node].astype(np.float64)
        probs = (deg / deg.sum()) if deg.sum() > 0 else np.ones(new_node) / new_node
        chosen = set()
        while len(chosen) < min(m, new_node):
            chosen.add(int(rng.choice(new_node, p=probs)))
        for v in chosen:
            neighbor_sets[new_node].add(v); neighbor_sets[v].add(new_node)
            degrees[new_node] += 1; degrees[v] += 1
    edges = _edges_from_neighbor_map({i: sorted(list(s)) for i, s in neighbor_sets.items()})
    cur = len(edges)
    if cur > target_edges:
        rng.shuffle(edges)
        for i, j in edges[:(cur - target_edges)]:
            neighbor_sets[i].discard(j); neighbor_sets[j].discard(i)
    elif cur < target_edges:
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in neighbor_sets[i]]
        rng.shuffle(missing)
        for i, j in missing[:(target_edges - cur)]:
            neighbor_sets[i].add(j); neighbor_sets[j].add(i)
    return {i: sorted(list(neighbor_sets[i])) for i in range(n)}


def _data_similarity_graph(
    n: int,
    target_edges: int,
    seed: int,
    node_class_hists: np.ndarray,
    similarity_temp: float = 1.0,
    similarity_mode: str = "softmax",
) -> Dict[int, List[int]]:
    """
    Parametric graph biased by class-distribution cosine similarity.

    similarity_mode='softmax' (default):
        Probabilistic preferential attachment — edges are sampled without
        replacement using softmax(sim / temp) as the sampling distribution.
        Lower temp concentrates weight on high-similarity pairs but is
        still stochastic: the top pair can be missed by chance.

    similarity_mode='topk':
        Deterministic k-NN phase followed by random fill.
        Each node first connects to its ceil(target_edges*2/n) most similar
        neighbours, guaranteeing the globally highest-similarity edges are
        always included.  Any remaining edge budget is filled uniformly at
        random from the unconnected pairs.  This is the maximum-advantage
        setting for our method: the UCB reward signal, pseudo-label quality,
        and CN-best pool are all maximally aligned with the graph structure.
    """
    rng = np.random.default_rng(int(seed))
    e_max = (n * (n - 1)) // 2
    target_edges = int(max(0, min(e_max, target_edges)))
    neighbor_sets: Dict[int, set] = {i: set() for i in range(n)}

    if n < 2 or target_edges == 0:
        return {i: [] for i in range(n)}

    hists = np.array(node_class_hists, dtype=np.float64)
    norms = np.linalg.norm(hists, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    hists_n = hists / norms

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sims = np.array(
        [float(np.dot(hists_n[i], hists_n[j])) for i, j in all_edges],
        dtype=np.float64,
    )

    if similarity_mode == "topk":
        # --- Deterministic phase: each node connects to its k most similar peers ---
        # k chosen so that the deterministic phase consumes roughly half the budget;
        # the rest is filled randomly.  k = ceil(target_edges * 2 / n) ensures
        # every node gets at least one guaranteed high-similarity neighbour even at
        # low density, while leaving room for random fill at high density.
        k = max(1, int(np.ceil(target_edges * 2 / n)))
        sim_matrix = hists_n @ hists_n.T  # (n, n)
        np.fill_diagonal(sim_matrix, -1.0)  # exclude self
        for i in range(n):
            top_js = np.argsort(sim_matrix[i])[::-1][:k]
            for j in top_js:
                if i != j:
                    neighbor_sets[i].add(j)
                    neighbor_sets[j].add(i)
        # --- Random fill: add remaining edges uniformly at random ---
        cur_edges = sum(len(v) for v in neighbor_sets.values()) // 2
        if cur_edges < target_edges:
            remaining_pool = [
                (i, j) for i, j in all_edges
                if j not in neighbor_sets[i]
            ]
            rng.shuffle(remaining_pool)
            for i, j in remaining_pool[:target_edges - cur_edges]:
                neighbor_sets[i].add(j)
                neighbor_sets[j].add(i)
    else:
        # --- Softmax mode (original): temperature-scaled probabilistic sampling ---
        temp = max(float(similarity_temp), 1e-8)
        log_w = sims / temp
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()
        chosen = rng.choice(
            len(all_edges),
            size=min(target_edges, len(all_edges)),
            replace=False,
            p=weights,
        )
        for idx in chosen:
            i, j = all_edges[int(idx)]
            neighbor_sets[i].add(j)
            neighbor_sets[j].add(i)

    return {i: sorted(list(neighbor_sets[i])) for i in range(n)}


def _make_neighbor_map(
    n: int,
    p: float,
    seed: int,
    connection_model: str,
    node_class_hists: Optional[np.ndarray] = None,
    similarity_temp: float = 1.0,
    similarity_mode: str = "softmax",
    ba_m: int = 0,
) -> Dict[int, List[int]]:
    e_max = (n * (n - 1)) // 2
    target_edges = int(np.floor(float(p) * e_max))
    if connection_model == "uniform":
        return _uniform_random_graph(n=n, p=p, seed=seed)
    if connection_model == "barabasi_albert":
        return _barabasi_albert_graph_with_target_edges(n=n, target_edges=target_edges, seed=seed, ba_m=int(ba_m))
    if connection_model == "data_similarity":
        if node_class_hists is None:
            raise ValueError(
                "connection_model='data_similarity' requires node_class_hists. "
                "This is passed automatically from DecentralizedPseudoLabelSystem.__init__."
            )
        return _data_similarity_graph(
            n=n, target_edges=target_edges, seed=seed,
            node_class_hists=node_class_hists,
            similarity_temp=float(similarity_temp),
            similarity_mode=str(similarity_mode),
        )
    raise ValueError(f"Unknown connection_model: {connection_model}")


def _plot_graph_colored_by_performance(
    neighbor_map, perf, hub_id, title, out_path,
) -> None:
    edges = _edges_from_neighbor_map(neighbor_map)
    n = len(neighbor_map)
    rng = np.random.default_rng(0)
    pos = {i: rng.normal(size=2) for i in range(n)}
    for _ in range(200):
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]; dist = float(np.linalg.norm(d) + 1e-6)
                pos[i] += (d/dist) * 0.005/dist; pos[j] -= (d/dist) * 0.005/dist
        for (i, j) in edges:
            d = pos[j] - pos[i]; dist = float(np.linalg.norm(d) + 1e-6)
            pos[i] += (d/dist) * 0.003*dist; pos[j] -= (d/dist) * 0.003*dist
    xs = np.array([pos[i][0] for i in range(n)], dtype=np.float32)
    ys = np.array([pos[i][1] for i in range(n)], dtype=np.float32)
    vals = np.array([float(perf.get(i, 0.0)) for i in range(n)], dtype=np.float32)
    plt.figure(figsize=(7, 6))
    for (i, j) in edges:
        plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], linewidth=0.8, alpha=0.35)
    sc = plt.scatter(xs, ys, c=vals, s=140, cmap="viridis", vmin=0.0, vmax=1.0,
                     edgecolors="k", linewidths=0.7)
    if hub_id in neighbor_map:
        plt.scatter([pos[hub_id][0]], [pos[hub_id][1]], s=260, facecolors="none",
                    edgecolors="white", linewidths=2.5)
    for i in range(n):
        plt.text(pos[i][0], pos[i][1], str(i), fontsize=9, ha="center", va="center", color="white")
    plt.colorbar(sc, label="Accuracy")
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


def _plot_graph_topology(
    neighbor_map: Dict[int, List[int]],
    node_arch_map: Optional[Dict[int, str]],
    title: str,
    out_path: str,
) -> None:
    """Save a topology visualization colored by node degree.

    Hub nodes (high degree) appear bright yellow; leaf nodes appear dark purple.
    MobileNetV2 nodes are drawn as circles (o), EfficientNet-B0 as stars (*).
    """
    edges = _edges_from_neighbor_map(neighbor_map)
    n = len(neighbor_map)
    degrees = {i: len(neighbor_map[i]) for i in range(n)}

    # Simple force-directed layout
    rng = np.random.default_rng(42)
    pos = {i: rng.normal(size=2) for i in range(n)}
    for _ in range(300):
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = float(np.linalg.norm(d) + 1e-6)
                pos[i] += (d / dist) * 0.005 / dist
                pos[j] -= (d / dist) * 0.005 / dist
        for (i, j) in edges:
            d = pos[j] - pos[i]
            dist = float(np.linalg.norm(d) + 1e-6)
            pos[i] += (d / dist) * 0.003 * dist
            pos[j] -= (d / dist) * 0.003 * dist

    degs = np.array([float(degrees[i]) for i in range(n)], dtype=np.float32)
    max_deg = float(degs.max()) if degs.max() > 0 else 1.0
    sizes = 80 + 200 * (degs / max_deg)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw edges — thickness proportional to sum of endpoint degrees
    for (i, j) in edges:
        w = 0.3 + 1.2 * (degrees[i] + degrees[j]) / (2.0 * max_deg)
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color="#aaaaaa", linewidth=w, alpha=0.4, zorder=1)

    # Split nodes by architecture for separate scatter calls (different markers)
    has_arch = node_arch_map and any(v == "efficientnet_b0" for v in node_arch_map.values())

    if has_arch:
        mnv2_idx   = [i for i in range(n) if node_arch_map.get(i, "mobilenet_v2") == "mobilenet_v2"]
        efnet_idx  = [i for i in range(n) if node_arch_map.get(i, "mobilenet_v2") == "efficientnet_b0"]

        # MobileNetV2 — circles
        if mnv2_idx:
            xs_m = np.array([pos[i][0] for i in mnv2_idx], dtype=np.float32)
            ys_m = np.array([pos[i][1] for i in mnv2_idx], dtype=np.float32)
            sc = ax.scatter(xs_m, ys_m,
                            c=[degs[i] for i in mnv2_idx],
                            s=[sizes[i] for i in mnv2_idx],
                            cmap="plasma", vmin=degs.min(), vmax=degs.max(),
                            marker='o', edgecolors="white", linewidths=0.8,
                            zorder=2, label="MobileNetV2")

        # EfficientNet-B0 — stars (larger so the star shape is visible)
        if efnet_idx:
            xs_e = np.array([pos[i][0] for i in efnet_idx], dtype=np.float32)
            ys_e = np.array([pos[i][1] for i in efnet_idx], dtype=np.float32)
            sc = ax.scatter(xs_e, ys_e,
                            c=[degs[i] for i in efnet_idx],
                            s=[sizes[i] * 1.8 for i in efnet_idx],
                            cmap="plasma", vmin=degs.min(), vmax=degs.max(),
                            marker='*', edgecolors="white", linewidths=0.5,
                            zorder=3, label="EfficientNet-B0")
    else:
        xs = np.array([pos[i][0] for i in range(n)], dtype=np.float32)
        ys = np.array([pos[i][1] for i in range(n)], dtype=np.float32)
        sc = ax.scatter(xs, ys, c=degs, s=sizes, cmap="plasma",
                        vmin=degs.min(), vmax=degs.max(),
                        marker='o', edgecolors="white", linewidths=0.8, zorder=2)

    # Degree labels on every node
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(degrees[i]),
                fontsize=6, ha="center", va="center", color="white",
                fontweight="bold", zorder=4)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Degree")

    if has_arch:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    _pprint(f"[VIZ] Topology graph saved: {out_path}")


# ----------------------------
# Node
# ----------------------------
class Node:
    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        val_loader: DataLoader,
        neighbor_ids: List[int],
        device: torch.device,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-3,
        kl_weight: float = 2.0,
        seed: int = 0,
        pseudo_disable_patience: int = 0,
        pseudo_disable_delta: float = 0.0,
        debug_time: bool = False,
        cache_features: bool = False,
    ):
        self.node_id = int(node_id)
        self.seed = int(seed)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader
        self.neighbor_ids = list(neighbor_ids)
        self.device = device
        self.cache_features = bool(cache_features)
        self.kl_weight = float(kl_weight)
        self.l2_coeff = float(weight_decay)
        self.debug_time = bool(debug_time)
        self._tb = TimeBreakdown(enabled=self.debug_time, device=self.device)
        assert hasattr(self.model, "head"), "Model must have .head"
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable_params, lr=float(lr), momentum=float(momentum), weight_decay=0.0,
        )
        self._model_params_for_l2 = trainable_params
        self.pseudo_disable_patience = int(pseudo_disable_patience)
        self.pseudo_disable_delta = float(pseudo_disable_delta)
        self.pseudo_allowed = True
        self._pseudo_started = False
        self._best_val_since_pseudo = -1.0
        self._bad_val_rounds = 0
        self._train_iter: Optional[Iterator] = None

    def _next_train_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def supervised_step(self) -> float:
        self.model.train()
        a, y = self._next_train_batch()
        y = y.to(self.device, non_blocking=True)
        if self.cache_features:
            logits = self.model.forward_head(a.to(self.device, non_blocking=True))
        else:
            logits = self.model(a.to(self.device, non_blocking=True))
        loss = F.cross_entropy(logits, y)
        if self.l2_coeff > 0.0 and self._model_params_for_l2:
            l2 = sum((p * p).sum() for p in self._model_params_for_l2)
            loss = loss + 0.5 * self.l2_coeff * l2
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        v = float(loss.detach().cpu().item())
        # accumulate for round-level loss logging
        self._sup_loss_sum = getattr(self, "_sup_loss_sum", 0.0) + v
        self._sup_loss_n   = getattr(self, "_sup_loss_n",   0)   + 1
        return v

    def pseudo_step_avg(
        self,
        z_pseudo: torch.Tensor,
        p_avg: torch.Tensor,
        kl_scale: float,
        conf_weight_tau0: float,
        agreement_beta: float = 0.0,
        node_class_weights: Optional[np.ndarray] = None,
    ) -> float:
        """KL distillation step with optional confident-agreement weighting.

        Per-example weight = w_conf * w_agree^agreement_beta, where:
          w_conf  = ((teacher_conf - tau0) / (1 - tau0)).clamp(0, 1)
                    (existing conf_weight_tau0 logic; = 1 when tau0 <= 0)
          w_agree = student's probability assigned to teacher's predicted class
                    (how much the student *already* agrees with the teacher)
          agreement_beta controls the strength of the agreement penalty:
            beta = 0  →  pure teacher-confidence weighting (original behaviour)
            beta = 1  →  weight = w_conf * w_agree  (confident AND agreed)
            beta > 1  →  harder gate: only strongly-agreed examples survive
        """
        if (self.kl_weight <= 0.0) or (not self.neighbor_ids) or (not self.pseudo_allowed):
            return 0.0
        eff_kl = float(self.kl_weight) * float(kl_scale)
        if eff_kl <= 0.0:
            return 0.0
        self.model.train()
        # In stage_2_tuning mode cache_features=False and z_pseudo contains raw
        # images — must run full backbone+head, not head-only.
        if self.cache_features:
            logits_student = self.model.forward_head(z_pseudo)
        else:
            logits_student = self.model(z_pseudo)
        log_probs_student = F.log_softmax(logits_student, dim=-1)
        C = p_avg.size(-1)
        per_ex = F.kl_div(log_probs_student, p_avg, reduction="none").sum(dim=-1) / float(C)

        # ── Confidence weight (teacher side) ────────────────────────────
        tau0 = float(conf_weight_tau0)
        if 0.0 < tau0 < 1.0:
            conf = p_avg.max(dim=-1).values                              # (B,)
            w = ((conf - tau0) / (1.0 - tau0)).clamp(0.0, 1.0)         # (B,)
        else:
            w = torch.ones(per_ex.size(0), device=per_ex.device)

        # ── Agreement weight (student × teacher) ────────────────────────
        beta = float(agreement_beta)
        if beta > 0.0:
            # teacher's argmax class per example
            teacher_cls  = p_avg.argmax(dim=-1)                         # (B,)
            probs_student = F.softmax(logits_student.detach(), dim=-1)  # (B, C)
            # student's prob of the teacher's predicted class
            w_agree = probs_student.gather(1, teacher_cls.unsqueeze(1)).squeeze(1)  # (B,)
            w = w * w_agree.pow(beta)

        loss_kl = (per_ex * w).sum() / w.sum().clamp_min(1e-6)
        loss = eff_kl * loss_kl
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        v = float(loss.detach().cpu().item())
        # accumulate for round-level loss logging (raw unscaled KL for interpretability)
        self._pseudo_loss_sum  = getattr(self, "_pseudo_loss_sum",  0.0) + float(loss_kl.detach().cpu().item())
        self._pseudo_loss_n    = getattr(self, "_pseudo_loss_n",    0)   + 1
        return v

    @torch.no_grad()
    def evaluate_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0; total = 0
        for a, y in loader:
            y = y.to(self.device, non_blocking=True)
            if self.cache_features:
                logits = self.model.forward_head(a.to(self.device, non_blocking=True))
            else:
                logits = self.model(a.to(self.device, non_blocking=True))
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        return 0.0 if total == 0 else (correct / total)

    @torch.no_grad()
    def evaluate_per_class_accuracy(self, loader: DataLoader, num_classes: int = 10) -> np.ndarray:
        self.model.eval()
        correct = np.zeros(num_classes, dtype=np.int64)
        total   = np.zeros(num_classes, dtype=np.int64)
        for a, y in loader:
            y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            a_dev = a.to(self.device, non_blocking=True)
            if self.cache_features:
                logits = self.model.forward_head(a_dev)
            else:
                logits = self.model(a_dev)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for c in range(num_classes):
                mask = (y_np == c)
                correct[c] += int((preds[mask] == c).sum())
                total[c]   += int(mask.sum())
        out = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            out[c] = float(correct[c]) / float(total[c]) if total[c] > 0 else 0.0
        return out

    def update_val_and_maybe_disable(self, val_acc: float, did_pseudo_this_round: bool) -> None:
        if self.pseudo_disable_patience <= 0:
            return
        if did_pseudo_this_round:
            self._pseudo_started = True
        if not self._pseudo_started:
            return
        if val_acc >= self._best_val_since_pseudo + self.pseudo_disable_delta:
            self._best_val_since_pseudo = float(val_acc)
            self._bad_val_rounds = 0
        else:
            self._bad_val_rounds += 1
            if self._bad_val_rounds >= self.pseudo_disable_patience:
                self.pseudo_allowed = False


# ----------------------------
# Pseudo accounting
# ----------------------------
@dataclass
class PseudoStats:
    pseudo_batches: int = 0
    pseudo_examples: int = 0

    def add_batch(self, batch_size: int) -> None:
        self.pseudo_batches += 1
        self.pseudo_examples += int(batch_size)


def _make_node_size_map(
    num_nodes, base_per_node, n_train, mode, seed, min_size, total_budget, dirichlet_alpha,
    degree_map: Optional[Dict[int, int]] = None,
) -> List[int]:
    mode = str(mode).lower()
    total_budget = min(int(total_budget) if int(total_budget) > 0 else num_nodes * base_per_node, n_train)
    if mode == "none":
        if int(total_budget) > 0 and int(total_budget) != num_nodes * base_per_node:
            base = max(int(min_size), min(int(total_budget) // num_nodes, n_train))
            remainder = int(total_budget) - base * num_nodes
            sizes = [base] * num_nodes
            for k in range(max(0, remainder)):
                sizes[k % num_nodes] += 1
            return [min(s, n_train) for s in sizes]
        return [max(min_size, min(base_per_node, n_train)) for _ in range(num_nodes)]
    if mode == "dirichlet":
        rng = np.random.default_rng(int(seed))
        # If degrees provided, bias Dirichlet concentration toward hubs so
        # higher-degree nodes receive proportionally more training data.
        if degree_map is not None and max(degree_map.values(), default=0) > 0:
            degs = np.array([float(max(degree_map.get(i, 1), 1)) for i in range(num_nodes)], dtype=np.float64)
            degs_norm = degs / max(degs.mean(), 1e-9)
            alpha_vec = np.maximum(degs_norm * float(dirichlet_alpha), 1e-6)
            # Hub floor: nodes with above-average degree get alpha >= 1.0 so
            # Dirichlet variance can't starve them. Below alpha=1.0 the variance
            # is enormous — a hub with alpha=0.018 can easily draw near-zero
            # weight even though its mean share is proportional to its degree.
            mean_deg = float(degs.mean())
            for _ni in range(num_nodes):
                if degs[_ni] >= mean_deg:
                    alpha_vec[_ni] = max(alpha_vec[_ni], 1.0)
        else:
            # Degenerate graph (p=0 or no edges) — fall back to uniform Dirichlet
            _pprint("[SIZE] dirichlet: all degrees zero, falling back to uniform allocation.")
            alpha_vec = np.full(num_nodes, max(1e-6, float(dirichlet_alpha)))
        w = rng.dirichlet(alpha=alpha_vec)
        sizes = np.maximum(np.floor(w * total_budget).astype(np.int64), min_size)
        s = int(sizes.sum())
        if s > total_budget:
            order = np.argsort(-sizes); k = 0
            over = s - total_budget
            while over > 0 and k < len(order) * 10:
                i = int(order[k % len(order)])
                if sizes[i] > min_size: sizes[i] -= 1; over -= 1
                k += 1
        elif s < total_budget:
            order = np.argsort(-w); k = 0
            under = total_budget - s
            while under > 0:
                sizes[int(order[k % len(order)])] += 1; under -= 1; k += 1
        return [int(x) for x in sizes.tolist()]
    if mode == "degree":
        # Strictly allocate budget proportional to node degree.
        # Requires degree_map; falls back to uniform if not provided.
        if degree_map is None or max(degree_map.values(), default=0) == 0:
            _pprint("[SIZE] size_skew_mode=degree but no edges in graph -- falling back to uniform.")
            return [max(min_size, min(base_per_node, n_train)) for _ in range(num_nodes)]
        degs = np.array([float(max(degree_map.get(i, 1), 1)) for i in range(num_nodes)], dtype=np.float64)
        degs_norm = degs / max(degs.sum(), 1e-9)
        raw = np.maximum(np.floor(degs_norm * total_budget).astype(np.int64), min_size)
        remainder = total_budget - int(raw.sum())
        order = np.argsort(-degs)
        for k in range(max(0, remainder)):
            raw[order[k % num_nodes]] += 1
        return [min(int(x), n_train) for x in raw.tolist()]
    raise ValueError(f"Unknown size_skew_mode: {mode}")


# ----------------------------
# Importance-weight helpers
# ----------------------------
def _skew_class_weights(
    favored_class: int,
    skew_factor: float,
    skew_min_other_frac: float,
    num_classes: int,
) -> np.ndarray:
    sf = max(1.0, float(skew_factor))
    n_other = num_classes - 1
    fav_frac = min(sf / (sf + float(n_other)), 1.0 - max(0.0, min(1.0, float(skew_min_other_frac))))
    w = np.full(num_classes, (1.0 - fav_frac) / max(1, n_other), dtype=np.float64)
    w[int(favored_class)] = fav_frac
    w /= w.sum()
    return w


def _weighted_acc(per_class_acc: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(per_class_acc, weights))


# ----------------------------
# System
# ----------------------------
class DecentralizedPseudoLabelSystem:
    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        val_fraction: float,
        test_fraction: float,
        unlabeled_fraction: float,
        per_node_sample_size: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        dropout_p: float,
        dropout_feat: float,
        kl_weight: float,
        seed: int,
        hub: int,
        num_workers_train: int,
        num_workers_eval: int,
        network_connection_p: float,
        connection_model: str,
        pseudo_conf_threshold: float,
        pseudo_entropy_threshold: float,
        pseudo_warmup_rounds: int,
        kl_ramp_rounds: int,
        pseudo_disable_patience: int,
        pseudo_disable_delta: float,
        training_data_mode: str = "iid",
        skew_factor: float = 5.0,
        skew_strategy: str = "round_robin",
        skew_seed: int = 0,
        skew_min_other_frac: float = 0.2,
        min_classes_per_node: int = 2,
        unlabeled_per_node: int = 0,
        unlabeled_pool_skew: str = "iid",
        arch: str = "mobilenet_v2",
        oracle_supervised_union: bool = False,
        oracle_bandit: bool = False,
        debug_time: bool = False,
        cache_features: bool = False,
        finetune_backbone: bool = False,
        baseline_non_linearity: bool = False,
        cache_batch_size: int = 512,
        amp_features: bool = True,
        feature_cache_path: str = "",
        size_skew_mode: str = "none",
        size_seed: int = 0,
        min_per_node_size: int = 20,
        size_total_budget: int = 0,
        size_dirichlet_alpha: float = 0.3,
        neighbor_weighting: str = "none",
        neighbor_weight_update_freq: int = 10,
        ucb_c: float = 1.0,
        bandit_type: str = "ucb1",
        grad_align_gamma: float = 0.99,
        grad_align_tau: float = 0.0,
        grad_align_iw_temp: float = 1.0,
        entropy_gate_tau: float = 0.5,
        entropy_ucb_align_tau: float = 0.5,
        pseudo_teacher_mode: str = "avg",
        similarity_temp: float = 1.0,
        similarity_mode: str = "softmax",
        ba_m: int = 0,
        random_models: bool = False,
        random_models_mnv2_frac: float = 0.8,
        random_models_hub_efnet: bool = True,
        top_k_teachers: int = 1,
        mutual_distillation: bool = False,
        pseudo_label_temp: float = 1.0,
        pseudo_examples_per_round: int = 128,
        teacher_ema: bool = False,
        teacher_ema_steps: int = 4,
        _current_stage2_round: int = 0,
        mobilenet_cache_path: str = "./Desktop/DecentralizedLearning/data/feature_cache/cifar10_mnv2_224.pt",
        efficientnet_cache_path: str = "./Desktop/DecentralizedLearning/data/feature_cache/cifar10_features_efficientnet_b0.pt",
        viz_graph: bool = False,
        _out_dir: str = ".",
        baseline_merge_val: bool = False,
        dataset: str = "cifar10",
        geo_cache: str = "",
    ):
        self._dataset = str(dataset).lower()
        self.num_classes = 100 if self._dataset == "cifar100" else 10
        self.geo_cache = str(geo_cache).strip()
        self._current_stage2_round = int(_current_stage2_round)
        self.num_nodes = int(num_nodes)
        self.feature_cache_path = str(feature_cache_path)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.unlabeled_fraction = float(unlabeled_fraction)
        self.unlabeled_per_node = int(unlabeled_per_node)  # 0 = use fraction instead
        self.per_node_sample_size = int(per_node_sample_size)
        self.lr = float(lr); self.momentum = float(momentum)
        self.weight_decay = float(weight_decay); self.dropout_p = float(dropout_p)
        self.dropout_feat = float(dropout_feat); self.kl_weight = float(kl_weight)
        self.seed = int(seed); self.hub = int(hub)
        self.num_workers_train = int(num_workers_train)
        self.num_workers_eval  = int(num_workers_eval)
        self.network_connection_p = float(network_connection_p)
        self.connection_model = str(connection_model).lower()
        self.arch = str(arch).lower()
        self.oracle_supervised_union = bool(oracle_supervised_union)
        self.oracle_bandit = bool(oracle_bandit)
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        self.pseudo_entropy_threshold = float(pseudo_entropy_threshold)
        self.pseudo_warmup_rounds = int(pseudo_warmup_rounds)
        self.kl_ramp_rounds = int(kl_ramp_rounds)
        self.pseudo_disable_patience = int(pseudo_disable_patience)
        self.pseudo_disable_delta = float(pseudo_disable_delta)
        self.training_data_mode = str(training_data_mode)
        self.skew_factor = float(skew_factor)
        self.skew_strategy = str(skew_strategy)
        self.skew_seed = int(skew_seed)
        self.skew_min_other_frac = float(skew_min_other_frac)
        self.min_classes_per_node = int(min_classes_per_node)
        self.cache_features = bool(cache_features)
        # When features are cached all data is pre-loaded into RAM as tensors.
        # DataLoader workers just add pipe overhead and file handle pressure.
        # Force to 0 to avoid "too many open files" on SLURM with 50 nodes.
        if self.cache_features:
            self.num_workers_train = 0
            self.num_workers_eval  = 0
        self.finetune_backbone = bool(finetune_backbone)
        self.baseline_non_linearity = bool(baseline_non_linearity)
        self.cache_batch_size = int(cache_batch_size)
        self.amp_features = bool(amp_features)
        self.feat_dim = 1280
        self.size_skew_mode = str(size_skew_mode); self.size_seed = int(size_seed)
        self.min_per_node_size = int(min_per_node_size)
        self.size_total_budget = int(size_total_budget)
        self.size_dirichlet_alpha = float(size_dirichlet_alpha)
        self.neighbor_weighting = str(neighbor_weighting).lower()
        self.neighbor_weight_update_freq = int(neighbor_weight_update_freq)
        self.ucb_c = float(ucb_c)
        self.bandit_type = str(bandit_type).lower()
        self.bandit_context_labeled = False  # set via run_one_setting after init
        self.baseline_merge_val = bool(baseline_merge_val)
        self._deploy_bandit: Optional[DeploymentUCBBandit] = None  # init after graph built
        self.grad_align_gamma = float(grad_align_gamma)
        self.grad_align_tau = float(grad_align_tau)
        self.grad_align_iw_temp = float(grad_align_iw_temp)
        self.entropy_gate_tau = float(entropy_gate_tau)
        self.entropy_ucb_align_tau = float(entropy_ucb_align_tau)
        self.pseudo_teacher_mode = str(pseudo_teacher_mode).lower()
        self.similarity_temp = float(similarity_temp)
        self.similarity_mode = str(similarity_mode).lower()
        self.ba_m = max(0, int(ba_m))
        self.viz_graph = bool(viz_graph)
        self._out_dir = str(_out_dir)
        self.unlabeled_pool_skew = str(unlabeled_pool_skew).lower()
        self.random_models = bool(random_models)
        self.random_models_mnv2_frac = float(np.clip(random_models_mnv2_frac, 0.0, 1.0))
        self.random_models_hub_efnet = bool(random_models_hub_efnet)
        self.top_k_teachers = max(1, int(top_k_teachers))
        self.mutual_distillation = bool(mutual_distillation)
        self.pseudo_label_temp = float(pseudo_label_temp)
        self.pseudo_examples_per_round = max(0, int(pseudo_examples_per_round))
        self.teacher_ema = bool(teacher_ema)
        self.teacher_ema_steps = max(0, int(teacher_ema_steps))
        self._ema_head_states: Dict[int, dict] = {}
        self._ema_step: int = 0
        self.mobilenet_cache_path = str(mobilenet_cache_path)
        self.efficientnet_cache_path = str(efficientnet_cache_path)

        self._neighbor_weights: Dict[int, Dict[int, float]] = {}
        self._best_neighbor: Dict[int, int] = {}
        self._bandit = None
        self._bandit_bootstrapped: bool = False
        self._node_arch_map: Dict[int, str] = {}
        self._node_test_loaders: Dict[int, DataLoader] = {}
        self._unlabeled_bufs: Optional[Dict[int, UnlabeledBuf]] = None
        self._unlabeled_loader_for_node: Dict[int, DataLoader] = {}
        self._unlabeled_feat_cpu: Dict[int, torch.Tensor] = {}
        self._unlabeled_idx_per_node: Dict[int, torch.Tensor] = {}  # raw train indices
        self._val_idx_per_node: Dict[int, torch.Tensor] = {}        # raw val train indices
        self._test_idx_per_node: Dict[int, torch.Tensor] = {}       # raw test train indices
        self.global_unlabeled_total: int = 0
        self.unlabeled_shard_sizes: List[int] = []
        self._feat_train_full: Optional[torch.Tensor] = None
        self._lab_train_full: Optional[torch.Tensor] = None
        self._feat_cache_by_arch: Dict[str, dict] = {}

        assert 0.0 <= self.network_connection_p <= 1.0
        assert self.training_data_mode in ("iid", "skewed")
        assert self.skew_factor >= 1.0
        assert self.skew_strategy in ("round_robin", "random")
        assert self.arch in ("mobilenet_v2", "efficientnet_b0")
        assert self.connection_model in ("uniform", "barabasi_albert", "data_similarity")
        assert self.neighbor_weighting in ("none", "train_acc", "ucb")
        assert self.size_skew_mode in ("none", "dirichlet", "degree")
        assert self.pseudo_teacher_mode in ("avg", "best")
        if self.random_models and not self.cache_features:
            raise ValueError("--random_models requires --cache_features")
        if self.bandit_type == "grad_align_ucb" and self.pseudo_teacher_mode != "avg":
            print("[WARNING] --pseudo_teacher_mode is ignored when --bandit_type grad_align_ucb.")

        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using Device: " + str(self.device))
        self.debug_time = bool(debug_time)
        self._tb = TimeBreakdown(enabled=self.debug_time, device=self.device)
        self._last_round_timing: Dict[str, float] = {}
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        input_size = 224
        cifar_mean = (0.485, 0.456, 0.406)
        cifar_std  = (0.229, 0.224, 0.225)
        transform_train = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=cifar_mean, std=cifar_std),
        ])
        transform_test = T.Compose([
            T.Resize(input_size + 32),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=cifar_mean, std=cifar_std),
        ])
        _ds_cls = torchvision.datasets.CIFAR100 if self._dataset == "cifar100" else torchvision.datasets.CIFAR10
        full_train_aug  = _ds_cls(root="./data", train=True,  download=True, transform=transform_train)
        full_train_eval = _ds_cls(root="./data", train=True,  download=True, transform=transform_test)
        full_test       = _ds_cls(root="./data", train=False, download=True, transform=transform_test)
        self._full_test = full_test
        self._full_test_labels_np = np.array(full_test.targets, dtype=np.int64)

        n_train = len(full_train_aug)
        all_indices = list(range(n_train))

        if self.random_models:
            _arch_rng = np.random.default_rng(int(self.seed) + 99_991)
            _mnv2_frac = self.random_models_mnv2_frac
            if self.random_models_hub_efnet:
                # Degree-based assignment deferred — graph not built yet.
                # Temporarily assign all to mobilenet_v2; will be reassigned
                # after neighbor_map is available (see below).
                for _ni in range(self.num_nodes):
                    self._node_arch_map[_ni] = "mobilenet_v2"
            else:
                # Random assignment.
                for _ni in range(self.num_nodes):
                    self._node_arch_map[_ni] = (
                        "mobilenet_v2" if _arch_rng.random() < _mnv2_frac else "efficientnet_b0"
                    )
                _n_mnv2 = sum(1 for a in self._node_arch_map.values() if a == "mobilenet_v2")
                _pprint(f"[RANDOM_MODELS] mobilenet_v2={_n_mnv2}  efficientnet_b0={self.num_nodes - _n_mnv2}  "
                        f"(target frac={_mnv2_frac:.2f})")
        else:
            for _ni in range(self.num_nodes):
                self._node_arch_map[_ni] = self.arch

        self.favored_class_map = _make_favored_class_map(
            num_nodes=self.num_nodes, num_classes=self.num_classes,
            strategy=self.skew_strategy, seed=self.skew_seed,
        )

        self._node_skew_weights: Dict[int, np.ndarray] = {}
        for _ni in range(self.num_nodes):
            if self.training_data_mode == "skewed":
                self._node_skew_weights[_ni] = _skew_class_weights(
                    self.favored_class_map[_ni], self.skew_factor,
                    self.skew_min_other_frac, self.num_classes,
                )
            else:
                self._node_skew_weights[_ni] = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float64)

        # ── Preliminary graph for degree-aware size allocation ─────────
        # For size_skew_mode in ('dirichlet', 'degree') we need node degrees
        # before splitting data. Build the graph now with a placeholder class
        # histogram (uniform); it will be rebuilt below with the real histograms
        # once the data split is known.  For 'barabasi_albert' and 'uniform'
        # this preliminary graph IS the final graph (class hists are unused).
        _prelim_graph_seed = int(self.seed * 1_000_003 + int(round(self.network_connection_p * 1e6)))
        _prelim_hists = np.ones((self.num_nodes, self.num_classes), dtype=np.float64) / float(self.num_classes)
        _prelim_neighbor_map = _make_neighbor_map(
            n=self.num_nodes, p=self.network_connection_p,
            seed=_prelim_graph_seed, connection_model=self.connection_model,
            node_class_hists=_prelim_hists,
            similarity_temp=self.similarity_temp,
            similarity_mode=self.similarity_mode,
            ba_m=self.ba_m,
        )
        _prelim_degrees: Dict[int, int] = {i: len(nbrs) for i, nbrs in _prelim_neighbor_map.items()}

        targets_np = np.asarray(full_train_aug.targets, dtype=np.int64)
        Kmin = int(self.min_classes_per_node)
        node_indices: List[List[int]] = []
        _all_indices_arr = np.asarray(all_indices, dtype=np.int64)

        # Pre-compute per-node split sizes
        _tf = float(self.test_fraction)
        _vf = float(self.val_fraction)

        total_budget = (
            self.size_total_budget if self.size_total_budget > 0
            else int(self.num_nodes * self.per_node_sample_size)
        )
        node_sizes = _make_node_size_map(
            num_nodes=self.num_nodes, base_per_node=self.per_node_sample_size,
            n_train=n_train, mode=self.size_skew_mode, seed=self.size_seed,
            min_size=self.min_per_node_size, total_budget=total_budget,
            dirichlet_alpha=self.size_dirichlet_alpha,
            degree_map=_prelim_degrees,
        )
        if self.size_skew_mode in ("dirichlet", "degree"):
            _deg_sorted = sorted(_prelim_degrees.items(), key=lambda x: -x[1])[:5]
            _size_sorted = sorted(enumerate(node_sizes), key=lambda x: -x[1])[:5]
            _pprint(f"[SIZE] mode={self.size_skew_mode}  top-5 hubs by degree: "
                    + "  ".join(f"n{i}(deg={d},sz={node_sizes[i]})" for i, d in _deg_sorted))
            _pprint(f"[SIZE] top-5 nodes by size: "
                    + "  ".join(f"n{i}(sz={s})" for i, s in _size_sorted))

        self.node_sizes = node_sizes

        train_subset_idxs: Dict[int, List[int]] = {}
        val_subset_idxs:   Dict[int, List[int]] = {}
        test_subset_idxs:  Dict[int, List[int]] = {}

        for i in range(self.num_nodes):
            sz = int(min(node_sizes[i], n_train))
            if sz <= 0:
                node_indices.append([])
                train_subset_idxs[i] = []; val_subset_idxs[i] = []; test_subset_idxs[i] = []
                continue

            if self.training_data_mode == "skewed":
                # Draw ONE pool of sz examples with skewed probabilities,
                # then PARTITION into train / val / test with no overlap.
                # Within each class, examples are allocated proportionally
                # so all three splits share the same class distribution.
                _class_probs = self._node_skew_weights[i]   # (C,)
                _idx_probs   = _class_probs[targets_np[_all_indices_arr]]
                _idx_probs   = _idx_probs / _idx_probs.sum()
                _rng         = np.random.default_rng(self.seed * 10_000 + i)
                avail = len(_all_indices_arr)

                # Single draw — all sz examples come from here, no second draw.
                all_node = _rng.choice(
                    _all_indices_arr, size=min(sz, avail),
                    replace=False, p=_idx_probs,
                )
                _rng.shuffle(all_node)
                all_labels = targets_np[all_node]

                # Stratified partition: within each class, assign examples to
                # test, then val, then train — guarantees no overlap and
                # preserves the skewed distribution across all splits.
                tr_idx_list: List[int] = []
                va_idx_list: List[int] = []
                te_idx_list: List[int] = []
                for _c in range(self.num_classes):
                    _c_positions = np.where(all_labels == _c)[0]
                    _rng.shuffle(_c_positions)
                    nc = len(_c_positions)
                    if nc == 0:
                        continue
                    # Allocate test first (fixed fraction), then val, rest to train.
                    nc_te = max(0, int(round(nc * _tf)))
                    nc_va = max(0, int(round(nc * _vf)))
                    nc_tr = nc - nc_te - nc_va
                    # If rounding consumed too many, steal from val first
                    if nc_tr < 0:
                        nc_va = max(0, nc_va + nc_tr)
                        nc_tr = nc - nc_te - nc_va
                    if nc_tr < 0:
                        nc_te = max(0, nc_te + nc_tr)
                        nc_tr = nc - nc_te - nc_va
                    te_idx_list.extend(all_node[_c_positions[:nc_te]].tolist())
                    va_idx_list.extend(all_node[_c_positions[nc_te:nc_te + nc_va]].tolist())
                    tr_idx_list.extend(all_node[_c_positions[nc_te + nc_va:]].tolist())

                tr_idx = tr_idx_list
                va_idx = va_idx_list
                te_idx = te_idx_list

                # Safety net: tiny nodes (sz≤5) with high val_fraction can
                # end up with 0 train examples after per-class rounding.
                # Steal from val first, then test, to guarantee ≥1 train example.
                if len(tr_idx) == 0 and (len(va_idx) + len(te_idx)) > 0:
                    if va_idx:
                        tr_idx.append(va_idx.pop())
                    elif te_idx:
                        tr_idx.append(te_idx.pop())

                node_indices.append(tr_idx)
                train_subset_idxs[i] = tr_idx
                val_subset_idxs[i]   = va_idx
                test_subset_idxs[i]  = te_idx
            else:
                if Kmin <= 1:
                    chosen = random.sample(all_indices, k=sz)
                else:
                    ok = False; last_cand = None
                    for _ in range(200):
                        cand = random.sample(all_indices, k=sz); last_cand = cand
                        if np.unique(targets_np[np.asarray(cand, dtype=np.int64)]).size >= Kmin:
                            chosen = cand; ok = True; break
                    if not ok:
                        chosen = last_cand or random.sample(all_indices, k=sz)
                node_indices.append(chosen)
                _rng2 = np.random.default_rng(self.seed * 10_000 + i + 1)
                _rng2.shuffle(chosen := np.asarray(chosen))
                n_te = max(1, int(sz * _tf)); n_va = max(1, int(sz * _vf)); n_tr = sz - n_te - n_va
                if n_tr < 1: n_tr = 1
                train_subset_idxs[i] = chosen[:n_tr].tolist()
                val_subset_idxs[i]   = chosen[n_tr:n_tr+n_va].tolist()
                test_subset_idxs[i]  = chosen[n_tr+n_va:].tolist()

        _num_classes = self.num_classes
        _node_class_hists = np.zeros((self.num_nodes, _num_classes), dtype=np.float64)
        for _ni in range(self.num_nodes):
            _idxs = node_indices[_ni]
            if _idxs:
                _labs = targets_np[np.asarray(_idxs, dtype=np.int64)]
                for _c in range(_num_classes):
                    _node_class_hists[_ni, _c] = float((_labs == _c).sum())
        self._node_class_hists = _node_class_hists

        union_labeled = set(idx for idxs in node_indices for idx in idxs)
        complement = [ix for ix in all_indices if ix not in union_labeled]
        if self.unlabeled_per_node > 0:
            self.global_unlabeled_total = self.unlabeled_per_node * self.num_nodes
        else:
            self.global_unlabeled_total = max(0, int(sum(node_sizes) * self.unlabeled_fraction))
        if self.global_unlabeled_total == 0:
            global_unlabeled = []
        elif len(complement) >= self.global_unlabeled_total:
            global_unlabeled = random.sample(complement, k=self.global_unlabeled_total)
        elif len(complement) > 0:
            global_unlabeled = complement[:]
            global_unlabeled.extend(
                random.choices(all_indices, k=self.global_unlabeled_total - len(global_unlabeled))
            )
        else:
            global_unlabeled = random.choices(all_indices, k=self.global_unlabeled_total)

        _complement_arr = np.asarray(global_unlabeled if global_unlabeled else all_indices, dtype=np.int64)
        _comp_targets = targets_np[_complement_arr]
        n_per_node = max(1, self.global_unlabeled_total // self.num_nodes)

        # Pre-bucket complement indices by class for stratified sampling.
        _comp_by_class: Dict[int, np.ndarray] = {
            c: _complement_arr[_comp_targets == c] for c in range(self.num_classes)
        }

        unlabeled_shard_map: Dict[int, List[int]] = {}
        for _ni in range(self.num_nodes):
            _rng_u = np.random.default_rng(self.seed * 100_000 + _ni + 999)
            if self.unlabeled_pool_skew == "skewed" and self.training_data_mode == "skewed" and hasattr(self, "_node_skew_weights"):
                # Stratified: draw exactly round(n_per_node * w_c) examples from
                # each class bucket so the empirical distribution matches _node_skew_weights
                # exactly (up to rounding), regardless of shard size.
                _cls_probs = self._node_skew_weights[_ni]
                _chosen_parts: List[np.ndarray] = []
                _remaining = n_per_node
                _nc = self.num_classes
                for _c in range(_nc):
                    # Last class gets the remainder to avoid off-by-one totals.
                    _n_c = int(round(_cls_probs[_c] * n_per_node)) if _c < _nc - 1 else _remaining
                    _n_c = max(0, min(_n_c, _remaining))
                    _bucket = _comp_by_class[_c]
                    if len(_bucket) == 0 or _n_c == 0:
                        _remaining -= _n_c
                        continue
                    _chosen_parts.append(
                        _rng_u.choice(_bucket, size=min(_n_c, len(_bucket)),
                                      replace=(_n_c > len(_bucket)))
                    )
                    _remaining -= _n_c
                _chosen = np.concatenate(_chosen_parts) if _chosen_parts else np.array([], dtype=np.int64)
            else:
                # IID: uniform sampling (default)
                _chosen = _rng_u.choice(
                    _complement_arr,
                    size=min(n_per_node, len(_complement_arr)),
                    replace=(n_per_node > len(_complement_arr)),
                )
            unlabeled_shard_map[_ni] = _chosen.tolist()
        self.unlabeled_shard_sizes = [len(unlabeled_shard_map[i]) for i in range(self.num_nodes)]

        # Per-node unlabeled class histogram (used by dist check below).
        _num_classes_u = self.num_classes
        self._unlabeled_class_dists: Dict[int, np.ndarray] = {}
        for _ni in range(self.num_nodes):
            _idxs_u = np.asarray(unlabeled_shard_map.get(_ni, []), dtype=np.int64)
            if _idxs_u.size == 0:
                self._unlabeled_class_dists[_ni] = np.full(_num_classes_u, 1.0 / _num_classes_u)
            else:
                _labs_u = targets_np[_idxs_u]
                _counts_u = np.bincount(_labs_u, minlength=_num_classes_u).astype(np.float64)
                _total_u  = _counts_u.sum()
                self._unlabeled_class_dists[_ni] = _counts_u / _total_u if _total_u > 0 else np.full(_num_classes_u, 1.0 / _num_classes_u)

        graph_seed = int(self.seed * 1_000_003 + int(round(self.network_connection_p * 1e6)))
        self.neighbor_map = _make_neighbor_map(
            n=self.num_nodes,
            p=self.network_connection_p,
            seed=graph_seed,
            connection_model=self.connection_model,
            node_class_hists=_node_class_hists,
            similarity_temp=self.similarity_temp,
            similarity_mode=self.similarity_mode,
            ba_m=self.ba_m,
        )

        if self.connection_model == "data_similarity":
            _n_edges = sum(len(v) for v in self.neighbor_map.values()) // 2
            _pprint(
                f"[GRAPH] data_similarity  temp={self.similarity_temp:.3f}  "
                f"mode={self.similarity_mode}  "
                f"p={self.network_connection_p:.3f}  edges={_n_edges}"
            )

        # ── Topology summary (always printed) ─────────────────────────
        _deg_all = sorted([len(nbrs) for nbrs in self.neighbor_map.values()], reverse=True)
        _n_edges_total = sum(_deg_all) // 2
        _deg_arr = np.array(_deg_all, dtype=np.float64)
        _isolated = int((_deg_arr == 0).sum())
        _ba_m_actual = (
            self.ba_m if self.ba_m > 0
            else int(max(1, min(self.num_nodes - 1, round(max(1.0, _n_edges_total / max(1.0, self.num_nodes))))))
            if self.connection_model == "barabasi_albert" else 0
        )
        _ba_str = f"  ba_m={_ba_m_actual}" if self.connection_model == "barabasi_albert" else ""
        _pprint(
            f"[TOPOLOGY] conn={self.connection_model}  p={self.network_connection_p:.3f}  "
            f"edges={_n_edges_total}{_ba_str}  "
            f"deg: min={int(_deg_arr.min())} mean={_deg_arr.mean():.1f} "
            f"median={float(np.median(_deg_arr)):.1f} max={int(_deg_arr.max())}  "
            f"isolated={_isolated}"
        )
        # Print top-10 hubs and their degrees
        _top_hubs = [(i, len(self.neighbor_map[i])) for i in range(self.num_nodes)]
        _top_hubs.sort(key=lambda x: -x[1])
        _hub_str = "  ".join(f"n{i}({d})" for i, d in _top_hubs[:10])
        _pprint(f"[TOPOLOGY] top-10 hubs: {_hub_str}")
        # Degree histogram in brackets
        _bins = [0, 1, 2, 5, 10, 20, 50, 999]
        _hist_parts = []
        for lo, hi in zip(_bins[:-1], _bins[1:]):
            count = int(((lo <= _deg_arr) & (_deg_arr < hi)).sum())
            if count > 0:
                _hist_parts.append(f"[{lo},{hi}): {count}")
        _pprint(f"[TOPOLOGY] degree histogram: {', '.join(_hist_parts)}")

        # ── Topology visualization ──────────────────────────────────────
        if getattr(self, "viz_graph", False):
            _viz_path = os.path.join(
                str(getattr(self, "_out_dir", ".")),
                f"topology_seed{self.seed}_p{self.network_connection_p:.3f}"
                f"_{self.connection_model}.png"
            )
            _viz_title = (
                f"Topology: {self.connection_model}  p={self.network_connection_p:.3f}  "
                f"nodes={self.num_nodes}  edges={_n_edges_total}  seed={self.seed}"
                + (f"  ba_m={self.ba_m}" if self.connection_model == "barabasi_albert" and self.ba_m > 0 else "")
            )
            _plot_graph_topology(
                neighbor_map=self.neighbor_map,
                node_arch_map=self._node_arch_map if self.random_models else None,
                title=_viz_title,
                out_path=_viz_path,
            )

        # ── Degree-based arch assignment (hub_efnet mode) ──────────────
        # Assign EfficientNet-B0 to the top-(1-mnv2_frac) highest-degree
        # nodes so heterogeneous models are concentrated at hubs.  This
        # makes parameter-averaging baselines structurally incompatible at
        # the most-connected nodes, directly testing our output-only claim.
        if self.random_models and self.random_models_hub_efnet:
            _degrees = {i: len(nbrs) for i, nbrs in self.neighbor_map.items()}
            _n_efnet = max(0, min(
                self.num_nodes,
                int(round(self.num_nodes * (1.0 - self.random_models_mnv2_frac)))
            ))
            # Ties broken by node index for reproducibility.
            _sorted_by_degree = sorted(_degrees.keys(), key=lambda i: (-_degrees[i], i))
            _efnet_set = set(_sorted_by_degree[:_n_efnet])
            for _ni in range(self.num_nodes):
                self._node_arch_map[_ni] = (
                    "efficientnet_b0" if _ni in _efnet_set else "mobilenet_v2"
                )
            _n_mnv2 = self.num_nodes - _n_efnet
            _deg_efnet = [_degrees[i] for i in _efnet_set]
            _deg_mnv2  = [_degrees[i] for i in range(self.num_nodes) if i not in _efnet_set]
            _pprint(
                f"[RANDOM_MODELS] hub_efnet=True  "
                f"mobilenet_v2={_n_mnv2} (avg_deg={np.mean(_deg_mnv2):.1f})  "
                f"efficientnet_b0={_n_efnet} (avg_deg={np.mean(_deg_efnet) if _deg_efnet else 0:.1f})  "
                f"(target mnv2_frac={self.random_models_mnv2_frac:.2f})"
            )
            # Re-run viz after hub assignment so arch colors are correct
            if getattr(self, "viz_graph", False):
                _plot_graph_topology(
                    neighbor_map=self.neighbor_map,
                    node_arch_map=self._node_arch_map,
                    title=_viz_title,
                    out_path=_viz_path,
                )

        if self.neighbor_weighting == "ucb":
            if self.bandit_type == "grad_align_ucb":
                self._bandit = GradientAlignedDiscountedUCB(
                    self.neighbor_map, num_classes=self.num_classes,
                    c=self.ucb_c, gamma=self.grad_align_gamma,
                )
            elif self.bandit_type == "ucb1":
                self._bandit = UCBNeighborBandit(self.neighbor_map, c=self.ucb_c)
            elif self.bandit_type == "entropy_ucb":
                # Algorithm 4: contextual LinUCB; context = mean feature of
                # the current unlabeled mini-batch; reward = logit-space
                # gradient alignment, gated by H(w_i).
                self._bandit = ContextualUCBNeighborBandit(
                    self.neighbor_map, feat_dim=self.feat_dim, c=self.ucb_c,
                    num_classes=self.num_classes,
                )
            # grad_align (greedy Algorithm 1) needs no bandit state

            # Deployment bandit — separate from training bandit.
            # Scalar UCB1 over neighbors; reward = IW-conf on unlabeled buffer.
            # Bootstrapped and updated at rebootstrap_freq. Used by deploy_criteria=bandit.
            self._deploy_bandit = DeploymentUCBBandit(self.neighbor_map, c=self.ucb_c)

        # train/val/test splits already built per-node in the sampling loop above.

        oracle_union_train_idx: List[int] = []
        if self.oracle_supervised_union:
            u = set()
            for i in range(self.num_nodes):
                u.update(train_subset_idxs[i])
            oracle_union_train_idx = sorted(u)

        self._feature_extractor: Optional[FrozenBackboneHead] = None
        if self.cache_features and not self.random_models:
            self._feature_extractor = FrozenBackboneHead(
                arch=self.arch, num_classes=self.num_classes, head_dropout_p=self.dropout_p,
            ).to(self.device)
            self._feature_extractor.eval()

        def _extract_features_for_subset(ds, idxs, wants_labels, extractor):
            idxs = list(idxs)
            if len(idxs) == 0:
                return torch.empty((0, self.feat_dim)), torch.empty((0,), dtype=torch.long)
            uniq = sorted(set(idxs))
            loader = DataLoader(
                Subset(ds, uniq), batch_size=self.cache_batch_size,
                shuffle=False, num_workers=0, pin_memory=False,
            )
            feats_list, labs_list = [], []
            with torch.no_grad():
                for batch in loader:
                    x, y = batch[0].to(self.device, non_blocking=True), batch[1]
                    if self.amp_features and self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            z = extractor.forward_features(x)
                    else:
                        z = extractor.forward_features(x)
                    feats_list.append(z.detach().cpu().float())
                    if wants_labels:
                        labs_list.append(y.detach().cpu().long())
            feats_uniq = torch.cat(feats_list, dim=0) if feats_list else torch.empty((0, self.feat_dim))
            labs_uniq  = (
                torch.cat(labs_list, dim=0) if (wants_labels and labs_list)
                else torch.empty((0,), dtype=torch.long)
            )
            if len(uniq) != len(idxs):
                pos   = {ix: j for j, ix in enumerate(uniq)}
                order = [pos[ix] for ix in idxs]
                return feats_uniq[order], (labs_uniq[order] if wants_labels else torch.empty((0,), dtype=torch.long))
            return feats_uniq, labs_uniq

        if self.cache_features:

            # 2026-04-22: tolerate caches without a pre-split test set.
            # Background: CIFAR caches (produced by precompute_features.py)
            # always contain feats_train/labs_train AND feats_test/labs_test
            # because they pre-split the dataset. EuroSAT geo caches
            # (produced by precompute_eurosat_features.py) contain only the
            # train pool — per-node test splits are carved out later by
            # _apply_geo_cache from the unified pool. The old code did
            #   obj.get("feats_test", obj.get("feat_test")).contiguous()
            # which raised AttributeError: 'NoneType' object has no
            # attribute 'contiguous' on the EuroSAT cache. Now the loader
            # returns the test keys only when present so downstream code
            # can check for their absence and skip the global test_loader.
            def _load_one_cache(path: str, arch_name: str) -> dict:
                path = os.path.abspath(path.strip())
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"[FEATCACHE] Cache not found for {arch_name}: {path}\n"
                        f"Run precompute_features.py --arch {arch_name} --out_path {path}"
                    )
                _pprint(f"[FEATCACHE] Loading {arch_name} from: {path}")
                obj = torch.load(path, map_location="cpu")
                ft = obj.get("feats_train", obj.get("feat_train"))
                lt = obj.get("labs_train",  obj.get("lab_train"))
                if ft is None or lt is None:
                    raise RuntimeError(
                        f"[FEATCACHE] Cache at {path} is missing feats_train/labs_train. "
                        f"This is not a usable feature cache."
                    )
                out = {
                    "feats_train": ft.contiguous().float(),
                    "labs_train":  lt.contiguous().long(),
                }
                fe = obj.get("feats_test", obj.get("feat_test"))
                le = obj.get("labs_test",  obj.get("lab_test"))
                if fe is not None and le is not None:
                    out["feats_test"] = fe.contiguous().float()
                    out["labs_test"]  = le.contiguous().long()
                return out

            if self.random_models:
                self._feat_cache_by_arch["mobilenet_v2"]    = _load_one_cache(
                    self.mobilenet_cache_path, "mobilenet_v2")
                self._feat_cache_by_arch["efficientnet_b0"] = _load_one_cache(
                    self.efficientnet_cache_path, "efficientnet_b0")
                self._feat_train_full = self._feat_cache_by_arch["mobilenet_v2"]["feats_train"]
                self._lab_train_full  = self._feat_cache_by_arch["mobilenet_v2"]["labs_train"]
            else:
                cache_path = self.feature_cache_path.strip()
                if cache_path:
                    cache_path = os.path.abspath(cache_path)
                    if os.path.isdir(cache_path) or cache_path.endswith(os.sep):
                        os.makedirs(cache_path, exist_ok=True)
                        cache_path = os.path.join(
                            cache_path, f"cifar10_{self.arch}_{input_size}_eval.pt"
                        )
                else:
                    cache_path = ""

                if cache_path and os.path.isfile(cache_path):
                    obj = _load_one_cache(cache_path, self.arch)
                else:
                    _pprint("[FEATCACHE] Building feature cache (train_eval + test).")
                    assert self._feature_extractor is not None
                    ft, lt = _extract_features_for_subset(
                        full_train_eval, range(len(full_train_eval)), True, self._feature_extractor)
                    fe, le = _extract_features_for_subset(
                        full_test, range(len(full_test)), True, self._feature_extractor)
                    obj = {
                        "feats_train": ft.contiguous(), "labs_train": lt.contiguous(),
                        "feats_test":  fe.contiguous(), "labs_test":  le.contiguous(),
                    }
                    if cache_path:
                        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                        torch.save(
                            {"version": 2, "arch": self.arch, "feat_dim": self.feat_dim, **obj},
                            cache_path,
                        )
                        _pprint(f"[FEATCACHE] Saved to: {cache_path}")

                self._feat_cache_by_arch[self.arch] = obj
                self._feat_train_full = obj["feats_train"]
                self._lab_train_full  = obj["labs_train"]

            _default_arch     = "mobilenet_v2" if self.random_models else self.arch
            _default_test_obj = self._feat_cache_by_arch[_default_arch]

            # ── Auto-infer num_classes from cache labels ───────────────
            # Guards against the common mistake of pointing to a CIFAR-100
            # cache without passing --dataset cifar100, which leaves
            # self.num_classes=10 and causes IndexError on label 10+.
            _inferred_nc = int(self._lab_train_full.max().item()) + 1
            if _inferred_nc != self.num_classes:
                _pprint(
                    f"[FEATCACHE] WARNING: cache has {_inferred_nc} classes but "
                    f"--dataset implies {self.num_classes}. "
                    f"Updating num_classes to {_inferred_nc}. "
                    f"Pass --dataset cifar100 to silence this warning."
                )
                self.num_classes = _inferred_nc

            # 2026-04-22: build the global test_loader only if the cache
            # actually has a test split. CIFAR caches always do → this
            # branch behaves identically to the old code. EuroSAT geo
            # caches don't (one unified pool, per-node splits carved out
            # later by _apply_geo_cache) → we set test_loader to None,
            # and evaluate_all_nodes_on_test already checks
            # _node_test_loaders first so the fallback is never taken.
            if "feats_test" in _default_test_obj and "labs_test" in _default_test_obj:
                self._feats_test_cached = _default_test_obj["feats_test"]
                self._labs_test_cached  = _default_test_obj["labs_test"]
                test_ds = FeatureTensorDataset(self._feats_test_cached, self._labs_test_cached)
                self._test_ds_cached = test_ds
                self.test_loader = DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers_eval, pin_memory=True,
                    persistent_workers=(self.num_workers_eval > 0),
                )
            else:
                self._feats_test_cached = None
                self._labs_test_cached  = None
                self._test_ds_cached    = None
                self.test_loader        = None
                _pprint(
                    "[FEATCACHE] Cache has no pre-split test set — global "
                    "test_loader disabled. Per-node test loaders must be "
                    "populated by _apply_geo_cache (or equivalent)."
                )

        # 2026-04-22: short-circuit the synthetic CIFAR-based partition
        # when --geo_cache is set. The partition loop above filled
        # train_subset_idxs / val_subset_idxs / test_subset_idxs /
        # unlabeled_shard_map with indices drawn from range(50000)
        # (CIFAR-10's train pool). Those indices are invalid for the
        # EuroSAT feature cache (27000 rows), so slicing at line ~2643
        # raises IndexError whenever a synthetic index exceeds 27000
        # (happens with moderate-to-large per-node sizes + --random_models
        # + dirichlet sizing). Clear the dicts so every node gets empty
        # loaders here; _apply_geo_cache at the end of __init__ rebuilds
        # them with EuroSAT-valid indices. The `_slice` helper below
        # already handles empty idx lists (returns empty tensors).
        if self.geo_cache:
            _pprint(
                "[GEO] --geo_cache set: clearing synthetic CIFAR partition "
                "before per-node loader construction. Loaders will be rebuilt "
                "by _apply_geo_cache from geographic node_assign."
            )
            train_subset_idxs  = {i: [] for i in range(self.num_nodes)}
            val_subset_idxs    = {i: [] for i in range(self.num_nodes)}
            test_subset_idxs   = {i: [] for i in range(self.num_nodes)}
            unlabeled_shard_map = {i: [] for i in range(self.num_nodes)}
            oracle_union_train_idx = []

        self.nodes: Dict[int, Node] = {}
        for i in range(self.num_nodes):
            train_idx     = oracle_union_train_idx if self.oracle_supervised_union else train_subset_idxs[i]
            val_idx       = val_subset_idxs[i]
            unlabeled_idx = unlabeled_shard_map.get(i, [])
            favored       = self.favored_class_map[i] if self.training_data_mode == "skewed" else None
            _node_arch    = self._node_arch_map[i]

            if self.cache_features:
                _arch_cache = self._feat_cache_by_arch[_node_arch]
                _ft_full    = _arch_cache["feats_train"]
                _lb_full    = _arch_cache["labs_train"]

                def _slice(idxs, _ft=_ft_full, _lb=_lb_full):
                    if not idxs:
                        return torch.empty((0, self.feat_dim)), torch.empty((0,), dtype=torch.long)
                    rows = torch.as_tensor(idxs, dtype=torch.long)
                    return _ft[rows], _lb[rows]

                ft, lt = _slice(train_idx)
                fv, lv = _slice(val_idx)
                fu, _  = _slice(unlabeled_idx)
                # Store raw indices so cross-arch teachers can look up the same
                # examples in their own feature cache (different backbone → different
                # feature space; passing i's features to j's head is garbage).
                self._unlabeled_idx_per_node[i] = torch.as_tensor(
                    unlabeled_idx, dtype=torch.long
                ) if unlabeled_idx else torch.empty((0,), dtype=torch.long)
                self._val_idx_per_node[i] = torch.as_tensor(
                    val_idx, dtype=torch.long
                ) if val_idx else torch.empty((0,), dtype=torch.long)
                self._test_idx_per_node[i] = torch.as_tensor(
                    test_subset_idxs[i], dtype=torch.long
                ) if test_subset_idxs.get(i) else torch.empty((0,), dtype=torch.long)

                # Data is already class-skewed by construction (node_indices
                # sampled with skewed probabilities), so plain loaders suffice.
                # train/val/test all share the same local skewed distribution.
                test_idx = test_subset_idxs[i]
                fe, le   = _slice(test_idx)

                train_ds = FeatureTensorDataset(ft, lt)
                val_ds   = FeatureTensorDataset(fv, lv)
                test_ds  = FeatureTensorDataset(fe, le)

                # ── Baseline val-merge ─────────────────────────────────
                # For baselines, merge the val set into training so they
                # use all labeled data. The split is identical to ours so
                # stage 1 pretrained weights are exactly the same; baselines
                # just also train on the val portion we reserve for selection.
                if getattr(self, "baseline_merge_val", False) and fv.size(0) > 0:
                    ft_merged = torch.cat([ft, fv], dim=0)
                    lt_merged = torch.cat([lt, lv], dim=0)
                    train_ds  = FeatureTensorDataset(ft_merged, lt_merged)

                _nw_eval = self.num_workers_eval
                _nw_half = max(1, _nw_eval // 2)
                # 2026-04-22: shuffle=True on an empty dataset raises
                # ValueError because RandomSampler requires num_samples>0.
                # When --geo_cache is set, we deliberately create empty
                # placeholder loaders that _apply_geo_cache will replace;
                # use shuffle=(train_ds has rows) to avoid the crash here.
                # CIFAR behavior unchanged (train_ds always non-empty).
                _train_shuffle = (len(train_ds) > 0)
                train_loader = DataLoader(
                    train_ds, batch_size=self.batch_size, shuffle=_train_shuffle,
                    num_workers=self.num_workers_train, pin_memory=True,
                    persistent_workers=(self.num_workers_train > 0),
                )
                train_eval_loader = DataLoader(
                    train_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=_nw_eval, pin_memory=True,
                    persistent_workers=(_nw_eval > 0),
                )
                val_loader = DataLoader(
                    val_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=_nw_half, pin_memory=True,
                    persistent_workers=(_nw_half > 0),
                )
                self._unlabeled_feat_cpu[i] = fu
                # Store raw indices for stage_2_tuning mode
                if not hasattr(self, "_stored_train_idx"):
                    self._stored_train_idx: Dict[int, List[int]] = {}
                    self._stored_val_idx:   Dict[int, List[int]] = {}
                    self._stored_test_idx:  Dict[int, List[int]] = {}
                self._stored_train_idx[i] = train_idx
                self._stored_val_idx[i]   = val_idx
                self._stored_test_idx[i]  = test_subset_idxs[i]
                self._node_test_loaders[i] = DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=_nw_eval, pin_memory=True,
                    persistent_workers=(_nw_eval > 0),
                )
            else:
                train_loader = _make_train_loader(
                    Subset(full_train_aug, train_idx), self.batch_size,
                    self.num_workers_train, self.training_data_mode, favored,
                    self.skew_factor, self.skew_min_other_frac, self.seed, i,
                )
                train_eval_loader = DataLoader(
                    Subset(full_train_eval, train_idx), batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers_eval, pin_memory=True,
                    persistent_workers=(self.num_workers_eval > 0),
                )
                # Data is already class-skewed by construction; plain loaders suffice.
                _nw_half = max(1, self.num_workers_eval // 2)
                test_idx     = test_subset_idxs[i]
                _val_subset  = Subset(full_train_eval, val_idx)
                _test_subset = Subset(full_train_eval, test_idx)
                val_loader = DataLoader(
                    _val_subset, batch_size=self.batch_size, shuffle=False,
                    num_workers=_nw_half, pin_memory=True,
                    persistent_workers=(_nw_half > 0),
                )
                self._node_test_loaders[i] = DataLoader(
                    _test_subset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers_eval, pin_memory=True,
                    persistent_workers=(self.num_workers_eval > 0),
                )
                self._unlabeled_loader_for_node[i] = DataLoader(
                    UnlabeledWrapper(Subset(full_train_eval, unlabeled_idx)),
                    batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers_eval, pin_memory=True,
                    persistent_workers=(self.num_workers_eval > 0),
                )

            model = (
                HeadOnly(arch=_node_arch, feat_dim=self.feat_dim, num_classes=self.num_classes,
                         head_dropout_p=self.dropout_p, baseline_non_linearity=self.baseline_non_linearity)
                if self.cache_features else
                FrozenBackboneHead(arch=_node_arch, num_classes=self.num_classes, head_dropout_p=self.dropout_p, baseline_non_linearity=self.baseline_non_linearity)
            )
            if self.finetune_backbone and not self.cache_features:
                for p in model.backbone.parameters():
                    p.requires_grad = True

            self.nodes[i] = Node(
                node_id=i, model=model, train_loader=train_loader,
                train_eval_loader=train_eval_loader, val_loader=val_loader,
                neighbor_ids=self.neighbor_map[i], device=self.device,
                lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                kl_weight=self.kl_weight, seed=self.seed,
                pseudo_disable_patience=self.pseudo_disable_patience,
                pseudo_disable_delta=self.pseudo_disable_delta,
                debug_time=self.debug_time, cache_features=self.cache_features,
            )

        # Global test_loader kept as fallback for IID mode; in skewed mode
        # evaluation always uses per-node _node_test_loaders.
        if not self.cache_features:
            self.test_loader = DataLoader(
                full_test, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers_eval, pin_memory=True,
                persistent_workers=(self.num_workers_eval > 0),
            )

        for i, node_i in self.nodes.items():
            nbrs = node_i.neighbor_ids
            if nbrs:
                w = 1.0 / len(nbrs)
                self._neighbor_weights[i] = {j: w for j in nbrs}
            else:
                self._neighbor_weights[i] = {}

        # ── Distribution sanity check ──────────────────────────────────
        # Compute per-node class distribution for val and test splits and
        # report average KL(val || test) to confirm they match.
        # 2026-04-22: skip this block entirely when --geo_cache is set.
        # In that case __init__ hasn't populated real loaders yet (the
        # synthetic partition was cleared by the short-circuit above);
        # _apply_geo_cache runs its own [DIST CHECK post-geo] against the
        # real loaders at the end of __init__. Running the check here on
        # the empty placeholders produces misleading "KL=0.0000 (OK)"
        # output that looks like all distributions match.
        if self.geo_cache:
            _pprint("[DIST CHECK] skipped (--geo_cache set; will run "
                    "post-override as [DIST CHECK post-geo]).")
        else:
            def _split_class_dist(loader: DataLoader) -> np.ndarray:
                counts = np.zeros(self.num_classes, dtype=np.float64)
                for _, labels in loader:
                    for lbl in (labels.numpy() if hasattr(labels, "numpy") else labels):
                        counts[int(lbl)] += 1
                total = counts.sum()
                return counts / total if total > 0 else counts

            def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
                # Only compute KL over classes present in BOTH distributions.
                # Classes with zero mass in either split inflate KL to infinity
                # even when the distributions match on present classes.
                mask = (p > 0) & (q > 0)
                if mask.sum() == 0:
                    return 0.0
                p_ = p[mask] + eps; q_ = q[mask] + eps
                p_ /= p_.sum(); q_ /= q_.sum()
                return float(np.sum(p_ * np.log(p_ / q_)))

            kl_vals = []
            for i, node_i in self.nodes.items():
                test_loader_i = self._node_test_loaders.get(i)
                if test_loader_i is None:
                    continue
                val_dist  = _split_class_dist(node_i.val_loader)
                test_dist = _split_class_dist(test_loader_i)
                kl_vals.append(_kl(val_dist, test_dist))

            kl_train_val = []
            for i, node_i in self.nodes.items():
                train_dist = _split_class_dist(node_i.train_eval_loader)
                val_dist   = _split_class_dist(node_i.val_loader)
                kl_train_val.append(_kl(train_dist, val_dist))

            if kl_vals:
                avg_kl = float(np.mean(kl_vals))
                max_kl = float(np.max(kl_vals))
                print(f"[DIST CHECK] val-vs-test  KL: avg={avg_kl:.4f}  max={max_kl:.4f}  "
                      f"({'OK' if avg_kl < 0.10 else 'HIGH — val/test distributions differ!'})",
                      flush=True)
            if kl_train_val:
                avg_kl2 = float(np.mean(kl_train_val))
                max_kl2 = float(np.max(kl_train_val))
                print(f"[DIST CHECK] train-vs-val  KL: avg={avg_kl2:.4f}  max={max_kl2:.4f}  "
                      f"({'OK (rare classes go to train by design)' if avg_kl2 < 5.0 else 'HIGH — train/val distributions differ!'})",
                      flush=True)

            kl_train_test = []
            for i, node_i in self.nodes.items():
                test_loader_i = self._node_test_loaders.get(i)
                if test_loader_i is None:
                    continue
                train_dist = _split_class_dist(node_i.train_eval_loader)
                test_dist  = _split_class_dist(test_loader_i)
                kl_train_test.append(_kl(train_dist, test_dist))
            if kl_train_test:
                avg_kl3 = float(np.mean(kl_train_test))
                max_kl3 = float(np.max(kl_train_test))
                print(f"[DIST CHECK] train-vs-test KL: avg={avg_kl3:.4f}  max={max_kl3:.4f}  "
                      f"({'OK' if avg_kl3 < 0.10 else 'HIGH — train/test distributions differ!'})",
                      flush=True)

            # ── Unlabeled-buffer distribution check ────────────────────────
            # Compare KL(expected || empirical_unlabeled) where expected is the
            # node's exact target distribution (_node_skew_weights). This is the
            # ground-truth reference — no loader-sampling noise.
            #
            # unlabeled_pool_skew=skewed → KL should be ~0 (shard drawn from same
            #   distribution as training; pseudo-label signal is on-distribution)
            # unlabeled_pool_skew=iid   → KL will be HIGH (~1.7 nats at skew=100);
            #   IS-weighting is supposed to correct this but the signal is weaker.
            #
            # FAIL threshold for skewed mode: avg KL > 0.05 means the sampling
            # code is broken and shards do NOT match the intended distribution.
            if hasattr(self, "_unlabeled_class_dists") and hasattr(self, "_node_skew_weights"):
                kl_unlab_vs_expected = []
                for i in range(self.num_nodes):
                    unlab_dist   = self._unlabeled_class_dists.get(i)
                    expected_dist = self._node_skew_weights.get(i)
                    if unlab_dist is None or expected_dist is None:
                        continue
                    # KL(expected || empirical): how far is the actual shard from
                    # the target distribution the node's training data was drawn from.
                    kl_unlab_vs_expected.append(_kl(expected_dist, unlab_dist))

                if kl_unlab_vs_expected:
                    avg_kl4  = float(np.mean(kl_unlab_vs_expected))
                    max_kl4  = float(np.max(kl_unlab_vs_expected))
                    worst_ni = int(np.argmax(kl_unlab_vs_expected))
                    worst_fav = self.favored_class_map.get(worst_ni, "?")

                    if self.unlabeled_pool_skew == "skewed" and self.training_data_mode == "skewed":
                        if avg_kl4 < 0.05:
                            _verdict = "PASS — shard matches node skew distribution"
                        else:
                            _verdict = "FAIL — shard does NOT match node skew! Check sampling code"
                    else:
                        # IID pool: high KL is expected and correct
                        _verdict = f"expected (iid pool vs skewed train)"

                    print(
                        f"[DIST CHECK] unlabeled KL(expected‖empirical): "
                        f"avg={avg_kl4:.4f}  max={max_kl4:.4f}  "
                        f"worst=node{worst_ni}(fav={worst_fav})  [{_verdict}]",
                        flush=True,
                    )

                    # Favored-class coverage: for skewed mode the unlabeled shard
                    # should have ~fav_frac of the favored class. Print per-node
                    # favored-class fraction so a misconfigured shard is obvious.
                    if self.unlabeled_pool_skew == "skewed" and self.training_data_mode == "skewed":
                        fav_fracs_got  = []
                        fav_fracs_want = []
                        for i in range(self.num_nodes):
                            unlab_dist    = self._unlabeled_class_dists.get(i)
                            expected_dist = self._node_skew_weights.get(i)
                            fav           = self.favored_class_map.get(i, 0)
                            if unlab_dist is None or expected_dist is None:
                                continue
                            fav_fracs_got.append(unlab_dist[fav])
                            fav_fracs_want.append(expected_dist[fav])
                        if fav_fracs_got:
                            print(
                                f"[DIST CHECK]   favored-class coverage in unlabeled shard: "
                                f"avg={np.mean(fav_fracs_got):.3f}  "
                                f"min={np.min(fav_fracs_got):.3f}  "
                                f"(target ≈ {np.mean(fav_fracs_want):.3f})",
                                flush=True,
                            )

        # ── Geographic override (EuroSAT geo_cache) ───────────────────
        if self.geo_cache:
            self._apply_geo_cache()

    # ------------------------------------------------------------------
    # Geographic cache override
    # ------------------------------------------------------------------

    def _apply_geo_cache(self):
        """Override system internals with geographic data from a precomputed cache.

        Replaces the synthetic data partition, graph topology, architecture
        assignments, and class distributions with real geographic data from
        precompute_eurosat_features.py.
        """
        blob = torch.load(self.geo_cache, map_location="cpu")
        node_assign = blob["node_assign"].numpy()       # [N_total]
        adj_list = blob["adj_list"]                      # {i: [j, ...]}
        arch_map = blob.get("arch_map", {})              # {i: str}
        labs = blob["labs_train"].numpy()                 # [N_total]
        meta = blob.get("meta", {})
        C = meta.get("num_classes", self.num_classes)
        num_nodes = meta.get("num_nodes", len(adj_list))
        class_names = blob.get("class_names", [])

        # Update num_classes if cache disagrees
        if C != self.num_classes:
            _pprint(f"[GEO] Updating num_classes from {self.num_classes} to {C}")
            self.num_classes = C

        _pprint(f"[GEO] Loading geographic cache: {self.geo_cache}")
        _pprint(f"[GEO] {len(labs)} images, {num_nodes} nodes, {C} classes")

        # ── 1. Override graph topology ────────────────────────────────
        for i, node in self.nodes.items():
            nbrs = adj_list.get(i, adj_list.get(str(i), []))
            if isinstance(nbrs, list):
                node.neighbor_ids = set(int(j) for j in nbrs)
            else:
                node.neighbor_ids = set()
        # Also update neighbor_map
        self.neighbor_map = {
            i: sorted(int(j) for j in adj_list.get(i, adj_list.get(str(i), [])))
            for i in range(num_nodes)
        }

        degrees = [len(self.nodes[i].neighbor_ids) for i in range(num_nodes)]
        n_edges = sum(degrees) // 2
        _pprint(f"[GEO] Graph: {n_edges} edges, degree min={min(degrees)} "
                f"mean={np.mean(degrees):.1f} max={max(degrees)}")

        # ── 2. Override architecture map ──────────────────────────────
        # 2026-04-22: only apply the arch_map override when random_models
        # is enabled. The EuroSAT precompute script populates arch_map
        # assuming a heterogeneous setup (top 10% degree hubs →
        # efficientnet_b0, rest → mobilenet_v2) that mirrors
        # --random_models. When random_models is off, experiment.py's
        # __init__ only loaded a single feature cache (self.arch); if we
        # applied arch_map anyway, lookups for the "other" arch would
        # miss and downstream loader construction would silently skip
        # those nodes — which is what produced the
        #   [GEO] WARNING: No feature cache for node X arch=efficientnet_b0
        # messages. For homogeneous runs, every node stays on self.arch
        # (the arch_map in the cache is ignored, not an error).
        if arch_map and self.random_models:
            for i_key, arch in arch_map.items():
                i = int(i_key)
                if i in self._node_arch_map:
                    self._node_arch_map[i] = arch
            n_mn = sum(1 for v in self._node_arch_map.values()
                       if v == "mobilenet_v2")
            n_ef = sum(1 for v in self._node_arch_map.values()
                       if v == "efficientnet_b0")
            _pprint(f"[GEO] Architectures (from geo cache, random_models=True): "
                    f"MobileNetV2={n_mn} EfficientNet-B0={n_ef}")
        elif arch_map:
            _pprint(f"[GEO] arch_map present in cache but --random_models is "
                    f"off; keeping all nodes on self.arch={self.arch} "
                    f"(arch_map ignored).")

        # ── 3. Partition each node's images into train/val/test ───────
        _vf = float(self.val_fraction)
        _tf = float(self.test_fraction)
        # 2026-04-22: honor --per_node_sample_size as a ceiling. Previously
        # every node used ALL of its geographic region's images (e.g. node 8
        # with 10958 regional images trained on ~7400 of them regardless of
        # --per_node_sample_size=1000), producing Stage 1 test accuracy
        # ~20pt higher than CIFAR baselines simply because large-region
        # nodes had 7x more training data than the CLI implied. Capping to
        # per_node_sample_size puts the geo setup on equal footing with
        # the synthetic baselines.
        _cap = int(getattr(self, "per_node_sample_size", 0) or 0)
        if _cap > 0:
            _pprint(f"[GEO] Capping per-node sample size at "
                    f"per_node_sample_size={_cap} (uncapped region sizes "
                    f"were used in previous builds).")
        # 2026-04-22: track actual post-cap sizes so the [GEO] Node
        # sizes summary below reflects reality (previously printed the
        # uncapped regional counts, which was misleading).
        _actual_sizes: Dict[int, int] = {}
        _actual_labels: Dict[int, np.ndarray] = {}
        for i in range(num_nodes):
            node_indices = np.where(node_assign == i)[0]
            rng_i = np.random.default_rng(self.seed * 10_000 + i)
            rng_i.shuffle(node_indices)
            if _cap > 0 and len(node_indices) > _cap:
                node_indices = node_indices[:_cap]

            n = len(node_indices)
            _actual_sizes[i] = n
            _actual_labels[i] = labs[node_indices]
            n_test = max(1, int(n * _tf))
            n_val = max(1, int(n * _vf))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - n_train) // 2)
                n_test = n - n_train - n_val

            test_idx = node_indices[:n_test]
            val_idx = node_indices[n_test:n_test + n_val]
            train_idx = node_indices[n_test + n_val:]

            # Store index lists
            self._stored_train_idx[i] = train_idx.tolist()
            self._stored_val_idx[i] = val_idx.tolist()
            self._stored_test_idx[i] = test_idx.tolist()

            # Store as tensors
            train_t = torch.as_tensor(train_idx, dtype=torch.long)
            val_t = torch.as_tensor(val_idx, dtype=torch.long)
            test_t = torch.as_tensor(test_idx, dtype=torch.long)
            self._val_idx_per_node[i] = val_t
            self._test_idx_per_node[i] = test_t

            # Clear unlabeled pool for this node
            self._unlabeled_idx_per_node[i] = torch.tensor([], dtype=torch.long)

            # Get feature cache for this node's architecture
            arch = self._node_arch_map.get(i, self.arch)
            cache = self._feat_cache_by_arch.get(arch, {})
            ft = cache.get("feats_train")
            lb = cache.get("labs_train")
            if ft is None or lb is None:
                # 2026-04-22: upgraded from silent-skip to hard error.
                # With the random_models gate above, reaching this branch
                # means either (a) --random_models is on but a required
                # per-arch cache was never loaded, or (b) a loaded cache
                # is missing feats_train/labs_train. Either way, leaving
                # the node with null loaders would just defer the crash
                # to the training loop. Fail fast with guidance instead.
                available = sorted(a for a, c in self._feat_cache_by_arch.items()
                                   if c.get("feats_train") is not None)
                raise RuntimeError(
                    f"[GEO] No feature cache for node {i} arch={arch!r}. "
                    f"Available arches in self._feat_cache_by_arch: {available}. "
                    f"If you passed --random_models, make sure BOTH "
                    f"--mobilenet_cache_path and --efficientnet_cache_path "
                    f"point at real files containing feats_train/labs_train. "
                    f"If you are NOT using heterogeneous models, drop "
                    f"--random_models (arch_map from the cache will then be "
                    f"ignored and every node will use --arch={self.arch!r})."
                )

            node = self.nodes[i]

            # Build train loaders
            train_ds = FeatureTensorDataset(ft[train_t], lb[train_t])
            node.train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=0, pin_memory=True)
            node.train_eval_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True)
            node._train_iter = None

            # Build val loader
            val_ds = FeatureTensorDataset(ft[val_t], lb[val_t])
            node.val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True)

            # Build test loader
            test_ds = FeatureTensorDataset(ft[test_t], lb[test_t])
            self._node_test_loaders[i] = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True)

            # ── 4. Empirical class distribution (no synthetic skew) ───
            node_labels = labs[node_assign == i]
            counts = np.bincount(node_labels, minlength=C).astype(np.float64)
            counts += 1e-8
            self._node_skew_weights[i] = counts / counts.sum()

            # Favored class = most common
            self.favored_class_map[i] = int(np.argmax(counts))

        # Rebuild unlabeled buffers (empty — no shared pool for geo)
        self._unlabeled_bufs = None
        self._unlabeled_feat_cpu = {i: torch.empty((0, self.feat_dim))
                                    for i in range(num_nodes)}

        # ── 5. Rebuild neighbor weights ───────────────────────────────
        for i, node in self.nodes.items():
            nbrs = list(node.neighbor_ids)
            if nbrs:
                w = 1.0 / len(nbrs)
                self._neighbor_weights[i] = {j: w for j in nbrs}
            else:
                self._neighbor_weights[i] = {}

        # ── 6. Print node statistics ──────────────────────────────────
        # Use post-cap sizes/labels rather than the raw regional counts.
        sizes = [_actual_sizes[i] for i in range(num_nodes)]
        _pprint(f"[GEO] Node sizes (post-cap): min={min(sizes)} "
                f"mean={np.mean(sizes):.0f} max={max(sizes)}")

        top5 = sorted(range(num_nodes), key=lambda x: -sizes[x])[:5]
        for i in top5:
            node_labels = _actual_labels[i]
            counts = np.bincount(node_labels, minlength=C)
            top3c = np.argsort(-counts)[:3]
            if class_names:
                desc = ", ".join(f"{class_names[c]}={counts[c]}" for c in top3c)
            else:
                desc = ", ".join(f"c{c}={counts[c]}" for c in top3c)
            _pprint(f"  node {i:2d} (n={sizes[i]:4d}, deg={degrees[i]:2d}): {desc}")

        _pprint(f"[GEO] Override complete: {num_nodes} nodes, "
                f"{sum(sizes)} images partitioned")

        # 2026-04-22: re-run the per-node distribution check against the
        # real (post-override) loaders. The original [DIST CHECK] block
        # in __init__ fires before _apply_geo_cache, so for geo runs it
        # was reporting KL=0.0000 on empty placeholder loaders — a lie.
        def _post_geo_class_dist(loader):
            counts = np.zeros(self.num_classes, dtype=np.float64)
            for _, labels in loader:
                for lbl in (labels.numpy() if hasattr(labels, "numpy") else labels):
                    counts[int(lbl)] += 1
            total = counts.sum()
            return counts / total if total > 0 else counts

        def _post_geo_kl(p, q, eps=1e-9):
            mask = (p > 0) & (q > 0)
            if mask.sum() == 0:
                return 0.0
            p_ = p[mask] + eps; q_ = q[mask] + eps
            p_ /= p_.sum(); q_ /= q_.sum()
            return float(np.sum(p_ * np.log(p_ / q_)))

        _vv = []  # val vs test
        _tv = []  # train vs val
        _tt = []  # train vs test
        for i, node_i in self.nodes.items():
            tl = self._node_test_loaders.get(i)
            if tl is None:
                continue
            tr = _post_geo_class_dist(node_i.train_eval_loader)
            vl = _post_geo_class_dist(node_i.val_loader)
            te = _post_geo_class_dist(tl)
            _vv.append(_post_geo_kl(vl, te))
            _tv.append(_post_geo_kl(tr, vl))
            _tt.append(_post_geo_kl(tr, te))

        if _vv:
            _pprint(f"[DIST CHECK post-geo] val-vs-test  KL: "
                    f"avg={np.mean(_vv):.4f}  max={np.max(_vv):.4f}")
        if _tv:
            _pprint(f"[DIST CHECK post-geo] train-vs-val  KL: "
                    f"avg={np.mean(_tv):.4f}  max={np.max(_tv):.4f}")
        if _tt:
            _pprint(f"[DIST CHECK post-geo] train-vs-test KL: "
                    f"avg={np.mean(_tt):.4f}  max={np.max(_tt):.4f}")

    # ------------------------------------------------------------------
    # Neighbor weight helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_j_on_i_loader(
        self, j: int, i: int, loader: DataLoader,
        idx_store: Optional[Dict[int, torch.Tensor]] = None,
    ) -> float:
        """Evaluate model j's accuracy on a loader built from node i's feature space.

        When j and i share the same arch, this is identical to
        nodes[j].evaluate_accuracy(loader).  When they differ, we look up
        the same dataset positions in j's own feature cache using idx_store.

        idx_store should be _val_idx_per_node when loader is a val loader,
        or _test_idx_per_node when loader is a test loader.
        If None, tries _val_idx_per_node then _test_idx_per_node.
        """
        _j_arch = self._node_arch_map.get(j, self.arch)
        _i_arch = self._node_arch_map.get(i, self.arch)

        # Fast path: same arch — loader features are valid for j's head.
        # Also fast path when not in cache mode — just run the full model.
        if not self.cache_features or not (self.random_models and _j_arch != _i_arch):
            return self.nodes[j].evaluate_accuracy(loader)

        if _j_arch not in self._feat_cache_by_arch:
            return self.nodes[j].evaluate_accuracy(loader)  # fallback

        _j_feats = self._feat_cache_by_arch[_j_arch]["feats_train"]
        model_j  = self.nodes[j].model
        model_j.eval()

        # Determine which index store to use
        if idx_store is not None:
            _stores = [idx_store]
        else:
            _stores = [self._val_idx_per_node, self._test_idx_per_node]

        correct = 0; total = 0; ex_pos = 0
        for a, y in loader:
            B = a.size(0)
            y = y.to(self.device, non_blocking=True)
            _raw_idx = None
            for _store in _stores:
                _all_idx = _store.get(i)
                if _all_idx is not None and _all_idx.numel() > 0:
                    _chunk = _all_idx[ex_pos: ex_pos + B]
                    if _chunk.numel() == B:
                        _raw_idx = _chunk
                        break
            if _raw_idx is None:
                # Fallback: same arch or index lookup failed — use loader features directly.
                # This is correct for same-arch; for cross-arch it means the index store
                # didn't cover this batch (shouldn't happen with correct idx_store).
                logits = model_j.forward_head(a.to(self.device, non_blocking=True))
            else:
                z_j    = _j_feats[_raw_idx].to(self.device, non_blocking=True)
                logits = model_j.forward_head(z_j)
            correct   += (logits.argmax(dim=-1) == y).sum().item()
            total     += B
            ex_pos    += B  # track by examples, not by batch index * batch_size
        return 0.0 if total == 0 else correct / total

    def update_neighbor_weights_from_train(self) -> None:
        with torch.no_grad():
            for i, node_i in self.nodes.items():
                nbrs = node_i.neighbor_ids
                if not nbrs:
                    continue
                raw: Dict[int, float] = {}
                for j in nbrs:
                    raw[j] = self.nodes[j].evaluate_accuracy(node_i.train_eval_loader)
                accs = np.array([raw[j] for j in nbrs], dtype=np.float64)
                accs -= accs.max()
                exp_accs = np.exp(accs)
                norm_weights = exp_accs / exp_accs.sum()
                self._neighbor_weights[i] = {j: float(norm_weights[k]) for k, j in enumerate(nbrs)}

    def get_last_round_timing(self) -> Dict[str, float]:
        return dict(self._last_round_timing)

    def _feats_for_teacher(
        self, j: int, batch_idx: Optional[torch.Tensor], z_fallback: torch.Tensor
    ) -> torch.Tensor:
        """Return features for teacher j on a given batch.

        If teacher j shares the same arch as the student (or random_models is
        off), return z_fallback directly (no copy).  Otherwise look up the same
        images in j's own feature cache using the raw dataset indices batch_idx.
        Falls back to z_fallback if indices are unavailable.
        """
        if not self.cache_features:
            # In full-tuning mode, z_fallback is raw images (or student features).
            # For cross-arch teachers we can't use student features — return fallback
            # and let the caller handle encoding through teacher's backbone.
            return z_fallback
        if not (self.random_models and len(self._feat_cache_by_arch) > 1):
            return z_fallback
        if batch_idx is None or batch_idx.numel() == 0:
            return z_fallback
        _j_arch = self._node_arch_map.get(j, self.arch)
        if _j_arch not in self._feat_cache_by_arch:
            return z_fallback
        _j_cache = self._feat_cache_by_arch[_j_arch]["feats_train"]
        return _j_cache[batch_idx].to(self.device, non_blocking=True)

    # ------------------------------------------------------------------
    # IW-unlabeled scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _iw_conf_score_one(self, i: int, j: int, buf_x: torch.Tensor,
                           buf_idx: Optional[torch.Tensor] = None,
                           weight_override: Optional[np.ndarray] = None) -> float:
        w_arr = weight_override if weight_override is not None else self._node_skew_weights[i]
        w_i = torch.tensor(w_arr, dtype=torch.float32, device=self.device)
        teacher = self.nodes[j]
        teacher.model.eval()
        total = 0.0
        count = 0
        for s in range(0, buf_x.size(0), self.batch_size):
            xb      = buf_x[s:s + self.batch_size]
            idx_b   = buf_idx[s:s + self.batch_size] if buf_idx is not None else None
            z_j     = self._feats_for_teacher(j, idx_b, xb.to(self.device, non_blocking=True))
            probs   = F.softmax(teacher.model(z_j) if not self.cache_features else teacher.model.forward_head(z_j), dim=-1)
            total  += float((probs * w_i.unsqueeze(0)).sum(dim=-1).sum().item())
            count  += z_j.size(0)
        return total / max(1, count)

    @torch.no_grad()
    def _iw_val_score_one(self, i: int, j: int) -> float:
        """IW-weighted probability neighbor j assigns to the correct label on
        node i's val set.

            score = (1/|val|) * sum_{(z,y) in val_i} w_i[y] * P_j(y | z)

        This is the clean bandit signal for val_bandit deployment:
        - Uses node i's TRUE labels (not pseudo-labels)
        - Weights by node i's class distribution so favored classes matter more
        - A collapsed model that always predicts class 0 scores w_i[0] on class-0
          examples but 0 on all others — correctly penalized
        - Node i's own model scores high because it was trained on this distribution
        - Strictly better than IW-confidence (which rewards confident collapse)
        """
        node_i   = self.nodes[i]
        node_j   = self.nodes[j]
        w_i      = self._node_skew_weights[i]  # (C,) numpy
        _j_arch  = self._node_arch_map.get(j, self.arch)
        _i_arch  = self._node_arch_map.get(i, self.arch)
        _has_multi = self.random_models and len(self._feat_cache_by_arch) > 1

        val_idx  = self._val_idx_per_node.get(i)  # raw dataset indices for val
        node_j.model.eval()
        total = 0.0
        count = 0
        for xb, yb in node_i.val_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            n  = xb.size(0)
            # Get the right features for teacher j (handles cross-arch)
            if (self.cache_features and _has_multi and _j_arch != _i_arch
                    and val_idx is not None
                    and _j_arch in self._feat_cache_by_arch):
                # val_loader iterates in order — compute batch offset
                _j_feats = self._feat_cache_by_arch[_j_arch]["feats_train"]
                # use the same index slice as the val loader
                z_j = _j_feats[val_idx[count:count + n]].to(self.device, non_blocking=True)
            else:
                z_j = xb
            probs  = F.softmax(node_j.model(z_j) if not self.cache_features else node_j.model.forward_head(z_j), dim=-1)  # (n, C)
            # P_j(y | x) for the true label y
            p_correct = probs[torch.arange(n, device=self.device), yb]   # (n,)
            # IW weight for each example's true class
            w_tensor  = torch.tensor(w_i, dtype=torch.float32, device=self.device)
            iw        = w_tensor[yb]                                       # (n,)
            total    += float((iw * p_correct).sum().item())
            count    += n
        return total / max(1, count)

    @torch.no_grad()
    def _iw_conf_score_per_class(self, i: int, j: int, buf_x: torch.Tensor) -> np.ndarray:
        C = self.num_classes
        teacher = self.nodes[j]
        teacher.model.eval()
        total = np.zeros(C, dtype=np.float64)
        count = 0
        for s in range(0, buf_x.size(0), self.batch_size):
            xb = buf_x[s:s + self.batch_size].to(self.device, non_blocking=True)
            probs = F.softmax(teacher.model(xb) if not self.cache_features else teacher.model.forward_head(xb), dim=-1)
            total += probs.sum(dim=0).cpu().numpy().astype(np.float64)
            count += xb.size(0)
        return total / max(1, count)

    @torch.no_grad()
    def switch_to_full_tuning(self) -> None:
        """Switch from frozen-backbone cache mode to full backbone finetuning.

        Called at the start of Stage 2 when --stage_2_tuning is set.
        Each node's HeadOnly model is replaced with a FrozenBackboneHead
        whose backbone is then unfrozen, head weights are transferred,
        and the optimizer is rebuilt to include backbone parameters.
        After this, cache_features is set False so all forward passes
        run through the full model on raw images.
        """
        if not self.cache_features:
            return  # already in full-tuning mode
        _pprint("[STAGE2_TUNE] Switching to full backbone finetuning...")

        # Load raw CIFAR-10 for image-space training
        import torchvision
        import torchvision.transforms as T
        cifar_mean = (0.485, 0.456, 0.406)
        cifar_std  = (0.229, 0.224, 0.225)
        transform_train = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.Resize(96),   # 96 instead of 224: 5x faster, still meaningful finetuning
            T.ToTensor(),
            T.Normalize(mean=cifar_mean, std=cifar_std),
        ])
        transform_eval = T.Compose([
            T.Resize(96),
            T.ToTensor(),
            T.Normalize(mean=cifar_mean, std=cifar_std),
        ])
        _ds_cls_s2 = torchvision.datasets.CIFAR100 if self._dataset == "cifar100" else torchvision.datasets.CIFAR10
        full_train_aug  = _ds_cls_s2(
            root="./data", train=True, download=True, transform=transform_train,
        )
        full_train_eval = _ds_cls_s2(
            root="./data", train=True, download=True, transform=transform_eval,
        )

        from torch.utils.data import DataLoader, Subset
        nw = 0  # avoid "too many open files" on SLURM

        for i, node in self.nodes.items():
            _arch = self._node_arch_map.get(i, self.arch)

            # Build new full model with same architecture
            new_model = FrozenBackboneHead(
                arch=_arch, num_classes=self.num_classes,
                head_dropout_p=self.dropout_p,
                baseline_non_linearity=self.baseline_non_linearity,
            ).to(self.device)

            # Transfer head weights from HeadOnly
            new_model.head.load_state_dict(node.model.head.state_dict())

            # Unfreeze backbone
            for p in new_model.backbone.parameters():
                p.requires_grad = True

            # Rebuild loaders from raw images using stored index sets
            train_idx = self._stored_train_idx.get(i, [])
            val_idx   = self._stored_val_idx.get(i, [])
            test_idx  = self._stored_test_idx.get(i, [])

            train_loader = DataLoader(
                Subset(full_train_aug, train_idx),
                batch_size=self.batch_size, shuffle=True,
                num_workers=nw, pin_memory=True,
            )
            train_eval_loader = DataLoader(
                Subset(full_train_eval, train_idx),
                batch_size=self.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True,
            )
            val_loader = DataLoader(
                Subset(full_train_eval, val_idx),
                batch_size=self.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True,
            ) if val_idx else node.val_loader
            test_loader = DataLoader(
                Subset(full_train_eval, test_idx),
                batch_size=self.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True,
            )

            # Replace node model and loaders
            node.model           = new_model
            node.train_loader    = train_loader
            node.train_eval_loader = train_eval_loader
            node.val_loader      = val_loader
            self._node_test_loaders[i] = test_loader

            # Rebuild optimizer with all params
            node.optimizer = torch.optim.SGD(
                [p for p in new_model.parameters() if p.requires_grad],
                lr=self.lr * 0.1,  # lower LR for finetuning
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        # Switch system and all nodes to image mode
        self.cache_features = False
        for node in self.nodes.values():
            node.cache_features = False
            node._train_iter = None  # force iterator reset on next batch

        # Rebuild unlabeled loaders from raw images
        import torchvision.transforms as _T2
        _ul_transform = _T2.Compose([
            # Store at 32×32 to keep buffer memory manageable.
            # Images are resized to 224 on-the-fly during the backbone forward pass.
            _T2.ToTensor(),
            _T2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        _ul_cifar = _ds_cls_s2(
            root="./data", train=True, download=True, transform=_ul_transform,
        )
        for i in self.nodes:
            ul_idx = self._unlabeled_idx_per_node.get(i)
            if ul_idx is not None and ul_idx.numel() > 0:
                self._unlabeled_loader_for_node[i] = DataLoader(
                    UnlabeledWrapper(Subset(_ul_cifar, ul_idx.tolist())),
                    batch_size=self.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True,
                )
        self._unlabeled_bufs = None  # force rebuild

        _pprint("[STAGE2_TUNE] Done — all nodes now finetune full backbone.")

    def update_weights_from_unlabeled_conf(
        self, node_subset: Optional[List[int]] = None,
    ) -> None:
        """Bootstrap: score (node, neighbor) pairs with IW-unlabeled confidence.

        Parameters
        ----------
        node_subset : optional list of node ids to rebootstrap. If None,
                      all nodes are rebootstrapped (original behaviour).
        """
        bufs = self._build_unlabeled_bufs_if_needed()
        degree_prior = float(getattr(self, "degree_prior", 0.0))
        # Precompute degree info for degree-informed prior
        if degree_prior > 0.0:
            _all_degrees = {j: len(self.neighbor_map.get(j, []))
                           for j in self.nodes}
            _max_log_deg = math.log(1.0 + max(_all_degrees.values(), default=1))

        iter_nodes = node_subset if node_subset is not None else list(self.nodes.keys())
        for i in iter_nodes:
            nbrs = self.nodes[i].neighbor_ids
            if not nbrs:
                continue
            buf = bufs.get(i)
            if buf is None or buf.n == 0:
                continue
            buf_idx = self._unlabeled_idx_per_node.get(i)  # raw dataset indices
            scores = {j: self._iw_conf_score_one(i, j, buf.x, buf_idx) for j in nbrs}
            scores[i] = self._iw_conf_score_one(i, i, buf.x, buf_idx)

            # ── Degree-informed prior ──────────────────────────────────
            # Bias toward high-degree neighbors: they have seen more diverse
            # data and are more likely to be effective teachers (especially
            # hub nodes in scale-free topologies).  The bonus is additive on
            # the IW-confidence score and log-scaled so it saturates for
            # very high degree, preventing hubs from dominating permanently.
            if degree_prior > 0.0:
                for j in nbrs:
                    scores[j] += degree_prior * (
                        math.log(1.0 + _all_degrees[j]) / max(_max_log_deg, 1e-9)
                    )

            self._self_iw_scores: Dict[int, float] = getattr(self, "_self_iw_scores", {})
            self._self_iw_scores[i] = scores[i]
            self._best_neighbor[i] = max((j for j in scores if j != i), key=scores.get)
            nbr_scores = {j: scores[j] for j in nbrs}
            raw = np.array([nbr_scores[j] for j in nbrs], dtype=np.float64)
            shifted = (raw - raw.max()) / 0.05
            ws = np.exp(shifted)
            ws /= ws.sum()
            self._neighbor_weights[i] = {j: float(ws[k]) for k, j in enumerate(nbrs)}
            if self._bandit is not None:
                for j, sc in nbr_scores.items():
                    self._bandit.set_q(i, j, sc)
        self._bandit_bootstrapped = True

    @torch.no_grad()
    def update_bandit_staggered(self) -> None:
        """Staggered UCB: evaluate ONE arm per node per call. O(N) cost."""
        if self._bandit is None or self.bandit_type != "ucb1":
            return
        bufs = self._build_unlabeled_bufs_if_needed()
        for i in self.nodes.keys():
            j = self._bandit.select_arm(i)
            if j is None:
                continue
            buf = bufs.get(i)
            if buf is None or buf.n == 0:
                continue
            buf_idx = self._unlabeled_idx_per_node.get(i)
            reward = self._iw_conf_score_one(i, j, buf.x, buf_idx)
            self._bandit.update(i, j, reward)
            self._best_neighbor[i] = self._bandit.best_arm(i)
            self._neighbor_weights[i] = self._bandit.softmax_weights(i)

    # ------------------------------------------------------------------
    # Pseudo-label teachers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _best_neighbor_pseudo_from_features(
        self, student_id: int, z: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node = self.nodes[student_id]
        nbrs = node.neighbor_ids
        if not nbrs:
            return z[:0], torch.empty((0, self.num_classes), device=self.device)
        j = self._best_neighbor.get(student_id)
        if j is None or j not in set(nbrs):
            return self._avg_neighbor_pseudo_from_features(student_id, z, batch_idx)
        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)
        teacher = self.nodes[j]
        teacher.model.eval()
        z_j   = self._feats_for_teacher(j, batch_idx, z)
        probs = F.softmax(teacher.model(z_j) if not self.cache_features else teacher.model.forward_head(z_j), dim=-1)
        keep = torch.ones(probs.size(0), device=self.device, dtype=torch.bool)
        if ct > 0.0:
            keep = keep & (probs.max(dim=-1).values >= ct)
        if et >= 0.0:
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            keep = keep & (ent <= et)
        if not keep.any():
            return z[:0], torch.empty((0, self.num_classes), device=self.device)
        return z[keep], probs[keep].detach()

    @torch.no_grad()
    def _best_neighbor_pseudo_from_images(
        self, student_id: int, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node = self.nodes[student_id]
        nbrs = node.neighbor_ids
        if not nbrs:
            return x[:0], torch.empty((0, self.num_classes), device=self.device)
        j = self._best_neighbor.get(student_id)
        if j is None or j not in set(nbrs):
            return self._avg_neighbor_pseudo_from_images(student_id, x)
        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)
        teacher = self.nodes[j]
        teacher.model.eval()
        probs = F.softmax(teacher.model(x), dim=-1)
        keep = torch.ones(probs.size(0), device=self.device, dtype=torch.bool)
        if ct > 0.0:
            keep = keep & (probs.max(dim=-1).values >= ct)
        if et >= 0.0:
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            keep = keep & (ent <= et)
        if not keep.any():
            return x[:0], torch.empty((0, self.num_classes), device=self.device)
        return x[keep], probs[keep].detach()

    def _kl_scale_for_round(self, round_idx: int) -> float:
        r = int(round_idx)
        if r < self.pseudo_warmup_rounds:
            return 0.0
        if self.kl_ramp_rounds <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (r - self.pseudo_warmup_rounds + 1) / float(self.kl_ramp_rounds))))

    @torch.no_grad()
    def _build_unlabeled_bufs_if_needed(self) -> Dict[int, "UnlabeledBuf"]:
        if self._unlabeled_bufs is not None:
            return self._unlabeled_bufs
        out: Dict[int, UnlabeledBuf] = {}
        if self.cache_features:
            for i in self.nodes.keys():
                feats = self._unlabeled_feat_cpu.get(i, torch.empty((0, self.feat_dim)))
                out[i] = UnlabeledBuf(feats)
        else:
            for i in self.nodes.keys():
                loader = self._unlabeled_loader_for_node.get(i)
                if loader is not None:
                    # In non-cache mode, pre-loading 2000 images per node is expensive.
                    # Instead load a smaller sample (up to batch_size*10) on demand.
                    _max_samples = min(self.batch_size * 10, 500)
                    xs = []
                    count = 0
                    for xb in loader:
                        xs.append(xb.detach().cpu())
                        count += xb.size(0)
                        if count >= _max_samples:
                            break
                    x_all = (
                        torch.cat(xs, dim=0)[:_max_samples].contiguous()
                        if xs else torch.empty((0, 3, 32, 32))
                    )
                else:
                    x_all = torch.empty((0, 3, 32, 32))
                out[i] = UnlabeledBuf(x_all)
        self._unlabeled_bufs = out
        return out

    @torch.no_grad()
    def _avg_neighbor_pseudo_from_features(
        self, student_id: int, z: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node = self.nodes[student_id]
        nbrs = node.neighbor_ids
        if not nbrs:
            return z[:0], torch.empty((0, self.num_classes), device=self.device)
        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)
        sum_p = torch.zeros((z.size(0), self.num_classes), device=self.device)
        cnt   = torch.zeros((z.size(0),),    device=self.device)
        weights = self._neighbor_weights.get(student_id, {})
        for j in nbrs:
            w_j     = float(weights.get(j, 1.0 / len(nbrs)))
            teacher = self.nodes[j]
            teacher.model.eval()
            z_j     = self._feats_for_teacher(j, batch_idx, z)
            probs   = F.softmax(teacher.model(z_j) if not self.cache_features else teacher.model.forward_head(z_j), dim=-1)
            keep = torch.ones(probs.size(0), device=self.device, dtype=torch.bool)
            if ct > 0.0:
                keep = keep & (probs.max(dim=-1).values >= ct)
            if et >= 0.0:
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                keep = keep & (ent <= et)
            if keep.any():
                kp    = keep.float() * w_j
                sum_p = sum_p + probs * kp.unsqueeze(-1)
                cnt   = cnt   + kp
        valid = cnt > 0.0
        if not valid.any():
            return z[:0], torch.empty((0, self.num_classes), device=self.device)
        return z[valid], sum_p[valid] / cnt[valid].unsqueeze(-1)

    @torch.no_grad()
    def _avg_neighbor_pseudo_from_images(
        self, student_id: int, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node = self.nodes[student_id]
        nbrs = node.neighbor_ids
        if not nbrs:
            return x[:0], torch.empty((0, self.num_classes), device=self.device)
        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)
        sum_p = torch.zeros((x.size(0), self.num_classes), device=self.device)
        cnt   = torch.zeros((x.size(0),),    device=self.device)
        weights = self._neighbor_weights.get(student_id, {})
        for j in nbrs:
            w_j = float(weights.get(j, 1.0 / len(nbrs)))
            teacher = self.nodes[j]
            teacher.model.eval()
            probs = F.softmax(teacher.model(x), dim=-1)
            keep = torch.ones(probs.size(0), device=self.device, dtype=torch.bool)
            if ct > 0.0:
                keep = keep & (probs.max(dim=-1).values >= ct)
            if et >= 0.0:
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                keep = keep & (ent <= et)
            if keep.any():
                kp    = keep.float() * w_j
                sum_p = sum_p + probs * kp.unsqueeze(-1)
                cnt   = cnt   + kp
        valid = cnt > 0.0
        if not valid.any():
            return x[:0], torch.empty((0, self.num_classes), device=self.device)
        return x[valid], sum_p[valid] / cnt[valid].unsqueeze(-1)

    # ------------------------------------------------------------------
    # Stale pseudo-label cache  (stage_2_tuning fast path)
    # ------------------------------------------------------------------
    def _compute_stale_pseudo_labels(self, no_flush: bool = False) -> None:
        """Run every teacher over every node's full unlabeled buffer ONCE and
        cache the resulting soft labels.  Also caches val backbone features for
        each (student_i, teacher_j) pair so that val EMA updates can use head-only
        forward passes instead of full backbone passes every round.

        Stores
        ------
        _stale_teacher_probs : Dict[(node_i, teacher_j), Tensor[buf_n, 10]] CPU
        _stale_avg_labels    : Dict[node_i, (x_cpu, avg_soft_y_cpu, valid_cpu)]
        _stale_val_feats     : Dict[(node_i, teacher_j), Tensor[n_val, feat_dim]] CPU
            Backbone features of teacher_j on node_i's val data.  Used by the
            val EMA update to avoid re-running the backbone every round.
        """
        bufs = self._build_unlabeled_bufs_if_needed()
        self._stale_teacher_probs: Dict[Tuple[int, int], torch.Tensor] = {}
        self._stale_avg_labels:    Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._stale_val_feats:     Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)
        n_pairs = 0

        # ── Stage-2 val feature precompute ───────────────────────────────────
        # For each teacher j, compute j's backbone features over every student
        # i's val data (only when j is in i's closed neighborhood).  Cost: same
        # backbone passes as the pseudo-label loop below, but amortised over
        # stale_refresh_freq rounds instead of paid every round for val EMA.
        _do_val_cache = (
            not self.cache_features
            and any(hasattr(n.model, "forward_features") for n in self.nodes.values())
        )
        if _do_val_cache:
            with torch.no_grad():
                for i, node_i in self.nodes.items():
                    hood = [i] + list(node_i.neighbor_ids)
                    # Collect all val images for node i (constant across rounds)
                    val_imgs: List[torch.Tensor] = []
                    val_labs: List[torch.Tensor] = []
                    for xb, yb in node_i.val_loader:
                        val_imgs.append(xb); val_labs.append(yb)
                    if not val_imgs:
                        continue
                    val_x_cpu = torch.cat(val_imgs, 0)   # [n_val, 3, H, W]
                    val_y_cpu = torch.cat(val_labs, 0)   # [n_val]
                    val_x_dev = val_x_cpu.to(self.device, non_blocking=True)

                    for j in hood:
                        teacher = self.nodes[j]
                        if not hasattr(teacher.model, "forward_features"):
                            continue
                        teacher.model.eval()
                        feat_chunks: List[torch.Tensor] = []
                        for s in range(0, val_x_dev.size(0), self.batch_size):
                            feat_chunks.append(
                                teacher.model.forward_features(val_x_dev[s: s + self.batch_size])
                            )
                        feats_cpu = torch.cat(feat_chunks, 0).cpu()   # [n_val, D]
                        self._stale_val_feats[(i, j)] = (feats_cpu, val_y_cpu)

        for i, node in self.nodes.items():
            buf = bufs.get(i)
            if buf is None or buf.n <= 0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            n      = buf.n
            nbrs   = node.neighbor_ids
            wts    = self._neighbor_weights.get(i, {})
            x_all  = buf.x.to(self.device, non_blocking=True)   # [n, 3, H, W]
            sum_p  = torch.zeros(n, self.num_classes, device=self.device)
            cnt    = torch.zeros(n,     device=self.device)

            for j in nbrs:
                w_j     = float(wts.get(j, 1.0 / len(nbrs)))
                teacher = self.nodes[j]
                teacher.model.eval()

                chunks: List[torch.Tensor] = []
                with torch.no_grad():
                    for s in range(0, n, self.batch_size):
                        xb = x_all[s: s + self.batch_size]
                        if not self.cache_features and hasattr(teacher.model, "forward_features"):
                            z    = teacher.model.forward_features(xb)
                            prbs = F.softmax(teacher.model.forward_head(z), dim=-1)
                        elif self.cache_features:
                            prbs = F.softmax(teacher.model.forward_head(xb), dim=-1)
                        else:
                            prbs = F.softmax(teacher.model(xb), dim=-1)
                        chunks.append(prbs)

                probs_all = torch.cat(chunks, 0)
                self._stale_teacher_probs[(i, j)] = probs_all.cpu()
                n_pairs += 1

                keep = torch.ones(n, dtype=torch.bool, device=self.device)
                if ct > 0.0:
                    keep &= probs_all.max(-1).values >= ct
                if et >= 0.0:
                    ent   = -(probs_all * probs_all.clamp_min(1e-12).log()).sum(-1)
                    keep &= ent <= et
                kp     = keep.float() * w_j
                sum_p += probs_all * kp.unsqueeze(-1)
                cnt   += kp

            valid = cnt > 0
            if not valid.any():
                continue
            avg_y               = torch.zeros(n, self.num_classes, device=self.device)
            avg_y[valid]        = sum_p[valid] / cnt[valid].unsqueeze(-1)
            self._stale_avg_labels[i] = (buf.x.cpu(), avg_y.cpu(), valid.cpu())

        mem_mb = sum(v.numel() * 4 for v in self._stale_teacher_probs.values()) // (1024 * 1024)
        val_pairs = len(self._stale_val_feats)
        _pprint(
            f"[STALE_LABELS] Cached {n_pairs} (node,teacher) pairs for "
            f"{len(self._stale_avg_labels)} nodes.  "
            f"Val feat cache: {val_pairs} pairs.  "
            f"Device={self.device}  prob_cache={mem_mb} MB",
            no_flush=no_flush,
        )

    def _pseudo_epoch_from_stale_labels(
        self,
        steps_per_node: int,
        kl_scale:       float,
        conf_weight_tau0: float,
        agreement_beta:   float = 0.0,
    ) -> Dict[int, "PseudoStats"]:
        """Pseudo-label distillation using cached (stale) teacher outputs.

        Teacher inference cost: ZERO per step (labels pre-cached).
        Student cost: 1 backbone forward+backward per step (unavoidable —
        the student's own weights must be updated via the KL loss).

        For entropy_ucb bandit: per-teacher probs are looked up from cache
        and bandit rewards are updated from those stale confidence scores.
        The bandit still learns which teacher is most useful; it just uses
        slightly stale confidence estimates between cache refreshes.
        """
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        if kl_scale <= 0.0 or not getattr(self, "_stale_avg_labels", None):
            return stats

        _round   = int(getattr(self, "_current_stage2_round", 0))
        is_bandit = isinstance(self._bandit, ContextualUCBNeighborBandit)
        bufs     = self._build_unlabeled_bufs_if_needed()

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            stale = self._stale_avg_labels.get(i)
            if stale is None:
                continue
            x_cpu, soft_y_cpu, valid_cpu = stale
            valid_idx = valid_cpu.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + _round * 1_000_003)
            w_i = self._node_skew_weights[i]

            # ── Bandit reward update from stale per-teacher probs ──────────
            # Runs once per node (not per step) — cheap since no backbone call.
            if is_bandit:
                bandit: ContextualUCBNeighborBandit = self._bandit  # type: ignore
                for j in node.neighbor_ids:
                    probs_cpu = self._stale_teacher_probs.get((i, j))
                    if probs_cpu is None:
                        continue
                    probs_dev   = probs_cpu.to(self.device, non_blocking=True)
                    pred_cls    = probs_dev.argmax(-1)
                    for c in range(bandit.num_classes):
                        mask = (pred_cls == c)
                        if mask.sum() < 4:
                            continue
                        r_c   = float(probs_dev[mask, c].mean()) - float(conf_weight_tau0)
                        ctx_c = np.zeros(bandit.feat_dim, dtype=np.float64)
                        bandit.update_class(i, j, c, ctx_c, r_c)

            # ── Student distillation steps ─────────────────────────────────
            for _ in range(int(steps_per_node)):
                perm  = torch.randint(0, valid_idx.numel(), (self.batch_size,), generator=g)
                idx   = valid_idx[perm]
                x_b   = x_cpu[idx].to(self.device, non_blocking=True)
                p_b   = soft_y_cpu[idx].to(self.device, non_blocking=True)
                node.pseudo_step_avg(
                    x_b, p_b,
                    kl_scale=kl_scale,
                    conf_weight_tau0=conf_weight_tau0,
                    agreement_beta=agreement_beta,
                    node_class_weights=w_i,
                )
                stats[i].add_batch(x_b.size(0))

        return stats

    def _merge_val_into_train_loaders(self) -> None:
        """Rebuild each node's train loader to include its val data.

        Called AFTER stage 1 completes for baselines and p=0 (independent
        learning), where the val set serves no purpose (no neighbors to
        evaluate, no bandit to update).  Stage 1 is identical for all methods
        because this is called post-pretrain; only stage 2 supervised steps
        benefit from the extra data.

        Works in cache_features mode by slicing the full feature cache using
        the stored val indices.  No-ops gracefully if val indices are missing.
        """
        _nw = self.num_workers_train
        for i, node in self.nodes.items():
            _arch      = self._node_arch_map.get(i, self.arch)
            _cache     = self._feat_cache_by_arch.get(_arch, {})
            _ft_full   = _cache.get("feats_train")
            _lb_full   = _cache.get("labs_train")
            _val_idx   = self._val_idx_per_node.get(i)

            if _ft_full is None or _val_idx is None or _val_idx.numel() == 0:
                continue

            # Current train dataset features/labels
            cur_ds = node.train_eval_loader.dataset
            if not isinstance(cur_ds, FeatureTensorDataset):
                continue

            # Fetch val features from the full cache
            fv = _ft_full[_val_idx]
            lv = _lb_full[_val_idx]

            ft_merged = torch.cat([cur_ds.feats, fv], dim=0)
            lt_merged = torch.cat([cur_ds.labels, lv], dim=0)
            merged_ds = FeatureTensorDataset(ft_merged, lt_merged)

            node.train_loader = DataLoader(
                merged_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=_nw, pin_memory=True,
                persistent_workers=(_nw > 0),
            )
            node.train_eval_loader = DataLoader(
                merged_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=_nw, pin_memory=True,
                persistent_workers=(_nw > 0),
            )
            node._train_iter = None   # force iterator reset

        _pprint(
            f"[VAL_MERGE] Merged val into train for {len(self.nodes)} nodes "
            f"(stage 2 supervised steps now use full labeled set)."
        )

    def _carve_val_from_train_for_stage2(self) -> None:
        """Randomly carve val_fraction of each node's training data into a fresh val set.

        Called between Stage 1 and Stage 2 for our method (baseline='none', p>0).

        Stage 1 trained on ALL data (train+val merged via baseline_merge_val=True)
        so pretrain performance has exact parity with baselines.  Now we need a
        val set for Stage 2 deployment scoring (val_ema, bandit selection).

        By randomly selecting val examples from the Stage 1 training pool:
        - Stage 1 parity: identical training data to baselines ✓
        - Fresh val: no subset was held out, so no memorization bias ✓
        - Stage 2 cost: slightly less training data — compensated by pseudo-labels ✓
        """
        vf = float(self.val_fraction)
        if vf <= 0.0:
            return

        _nw_train = self.num_workers_train
        _nw_eval  = self.num_workers_eval
        _nw_half  = max(0, _nw_eval // 2)

        n_carved = 0
        for i, node in self.nodes.items():
            _arch  = self._node_arch_map.get(i, self.arch)
            _cache = self._feat_cache_by_arch.get(_arch, {})
            _ft_full = _cache.get("feats_train")
            _lb_full = _cache.get("labs_train")

            if _ft_full is None:
                continue

            # Get the full set of raw dataset indices this node trained on in Stage 1.
            # With baseline_merge_val=True, this is the original train + val merged.
            train_raw = list(self._stored_train_idx.get(i, []))
            val_raw   = list(self._stored_val_idx.get(i, []))
            all_raw   = train_raw + val_raw

            if len(all_raw) < 2:
                continue

            n_total = len(all_raw)
            n_val   = max(1, int(round(n_total * vf)))
            n_train = n_total - n_val
            if n_train < 1:
                n_train = 1
                n_val   = n_total - 1

            # Random split — deterministic per (seed, node) but independent of
            # the original train/val partition so it's truly "fresh".
            rng = np.random.default_rng(self.seed * 10_000 + i + 777_777)
            perm = rng.permutation(n_total)

            all_raw_arr     = np.array(all_raw, dtype=np.int64)
            new_train_idx   = all_raw_arr[perm[:n_train]].tolist()
            new_val_idx     = all_raw_arr[perm[n_train:]].tolist()

            # Slice features from the full cache
            new_train_rows = torch.as_tensor(new_train_idx, dtype=torch.long)
            new_val_rows   = torch.as_tensor(new_val_idx, dtype=torch.long)

            train_ds = FeatureTensorDataset(_ft_full[new_train_rows], _lb_full[new_train_rows])
            val_ds   = FeatureTensorDataset(_ft_full[new_val_rows],   _lb_full[new_val_rows])

            # Rebuild loaders
            node.train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=_nw_train, pin_memory=True,
                persistent_workers=(_nw_train > 0),
            )
            node.train_eval_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=_nw_eval, pin_memory=True,
                persistent_workers=(_nw_eval > 0),
            )
            node.val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=_nw_half, pin_memory=True,
                persistent_workers=(_nw_half > 0),
            )
            node._train_iter = None  # force iterator reset

            # Update raw index stores for cross-arch eval and stage_2_tuning
            self._val_idx_per_node[i]  = new_val_rows
            self._stored_train_idx[i]  = new_train_idx
            self._stored_val_idx[i]    = new_val_idx
            n_carved += 1

        _pprint(
            f"[VAL_CARVE] Carved val from training for {n_carved} nodes "
            f"(val_fraction={vf:.2f}) — Stage 2 val is fresh, not memorized from Stage 1."
        )

    def supervised_steps_synchronous(
        self, sup_steps_total: int, sup_steps_per_node: int, round_idx: int,
    ) -> None:
        if sup_steps_total > 0:
            node_ids = list(self.nodes.keys())
            rng = random.Random(self.seed * 1_000_003 + 1337 + int(round_idx))
            for _ in range(int(sup_steps_total)):
                self.nodes[node_ids[rng.randrange(len(node_ids))]].supervised_step()
        else:
            for i in self.nodes.keys():
                for _ in range(int(sup_steps_per_node)):
                    self.nodes[i].supervised_step()

    # ------------------------------------------------------------------
    # Pseudo epochs
    # ------------------------------------------------------------------
    def _pseudo_epoch_fixed_steps(
        self, steps_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        if steps_per_node <= 0 or kl_scale <= 0.0:
            return stats

        # Greedy gradient alignment (Algorithm 1) — no bandit state, no bootstrap needed.
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "grad_align"):
            return self._pseudo_epoch_greedy_grad_align_steps(steps_per_node, kl_scale, conf_weight_tau0, agreement_beta)

        # Discounted UCB + gradient alignment (Algorithm 2) — enters after bootstrap.
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "grad_align_ucb"
                and self._bandit_bootstrapped):
            return self._pseudo_epoch_grad_align_ucb_steps(steps_per_node, kl_scale, conf_weight_tau0, agreement_beta)

        # Entropy-Gated UCB with Logit-Space Gradient Alignment (Algorithm 4).
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "entropy_ucb"):
            return self._pseudo_epoch_entropy_gated_ucb_steps(steps_per_node, kl_scale, conf_weight_tau0, agreement_beta)

        # Omniscient oracle: always picks the best neighbor by ground-truth test accuracy.
        if self.bandit_type == "omniscient":
            return self._pseudo_epoch_omniscient_steps(steps_per_node, kl_scale, conf_weight_tau0, agreement_beta)

        _round = int(getattr(self, "_current_stage2_round", 0))
        bufs = self._build_unlabeled_bufs_if_needed()
        gens = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + _round * 1_000_003)
            gens[i] = g
        use_best = (self.pseudo_teacher_mode == "best")
        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue
            for _ in range(int(steps_per_node)):
                xb = buf.sample_with_replacement(self.batch_size, generator=gens[i])
                if xb is None:
                    continue
                if self.cache_features:
                    z = xb.to(self.device, non_blocking=True)
                    z_kept, p_avg = (
                        self._best_neighbor_pseudo_from_features(i, z)
                        if use_best else
                        self._avg_neighbor_pseudo_from_features(i, z)
                    )
                    if z_kept.size(0) == 0:
                        continue
                    stats[i].add_batch(z_kept.size(0))
                    node.pseudo_step_avg(z_kept, p_avg, kl_scale=kl_scale,
                                         conf_weight_tau0=conf_weight_tau0, agreement_beta=agreement_beta)
                else:
                    x = xb.to(self.device, non_blocking=True)
                    x_kept, p_avg = (
                        self._best_neighbor_pseudo_from_images(i, x)
                        if use_best else
                        self._avg_neighbor_pseudo_from_images(i, x)
                    )
                    if x_kept.size(0) == 0:
                        continue
                    stats[i].add_batch(x_kept.size(0))
                    node.pseudo_step_avg(x_kept, p_avg, kl_scale=kl_scale,
                                         conf_weight_tau0=conf_weight_tau0, agreement_beta=agreement_beta)
        return stats

    def _pseudo_epoch_omniscient_steps(
        self, steps_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        """Oracle ceiling: at every step, pick the best neighbor by ground-truth test accuracy.

        This is NOT a real algorithm — it requires access to held-out test labels
        which are unavailable in practice. Its purpose is to bound how much better
        any bandit-based selection strategy could theoretically perform.

        Teacher selection: for each node i, evaluate every neighbor j on node i's
        local test set (cross-arch aware), pick the highest-accuracy j, distill
        from j's pseudo-labels on node i's unlabeled buffer.

        The oracle best-neighbor is recomputed every `steps_per_node` steps (once
        per pseudo epoch), not per-batch, to keep cost manageable.
        """
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        if steps_per_node <= 0 or kl_scale <= 0.0:
            return stats

        bufs = self._build_unlabeled_bufs_if_needed()
        _round = int(getattr(self, "_current_stage2_round", 0))
        gens: Dict[int, torch.Generator] = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + _round * 1_000_003)
            gens[i] = g

        ct = float(self.pseudo_conf_threshold)
        et = float(self.pseudo_entropy_threshold)

        # Pre-compute oracle best teacher for each node using ground-truth test acc.
        oracle_teacher: Dict[int, int] = {}
        with torch.no_grad():
            for i, node in self.nodes.items():
                nbrs = node.neighbor_ids
                if not nbrs:
                    continue
                test_loader_i = self._node_test_loaders.get(i)
                if test_loader_i is None:
                    # Fallback: use val accuracy
                    best_j = max(nbrs, key=lambda j: self.evaluate_j_on_i_loader(
                        j, i, node.val_loader, self._val_idx_per_node))
                else:
                    best_j = max(nbrs, key=lambda j: self.evaluate_j_on_i_loader(
                        j, i, test_loader_i, self._test_idx_per_node))
                oracle_teacher[i] = best_j
                self._best_neighbor[i] = best_j  # for diagnostics

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue

            j_star = oracle_teacher.get(i)
            if j_star is None:
                continue

            _i_unlab_idx = self._unlabeled_idx_per_node.get(i)
            _has_multi_arch = self.random_models and len(self._feat_cache_by_arch) > 1

            for _ in range(int(steps_per_node)):
                pos_idx = torch.randint(0, buf.n, (self.batch_size,),
                                        generator=gens[i], device="cpu")
                xb = buf.x[pos_idx]
                z_train = xb.to(self.device, non_blocking=True)

                batch_train_idx = (
                    _i_unlab_idx[pos_idx]
                    if (_has_multi_arch and _i_unlab_idx is not None and _i_unlab_idx.numel() > 0)
                    else None
                )

                teacher = self.nodes[j_star]
                teacher.model.eval()
                with torch.no_grad():
                    z_j   = self._feats_for_teacher(j_star, batch_train_idx, z_train)
                    probs = F.softmax(teacher.model(z_j) if not self.cache_features else teacher.model.forward_head(z_j), dim=-1)

                keep = torch.ones(probs.size(0), device=self.device, dtype=torch.bool)
                if ct > 0.0:
                    keep = keep & (probs.max(dim=-1).values >= ct)
                if et >= 0.0:
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    keep = keep & (ent <= et)
                if not keep.any():
                    continue

                z_kept = z_train[keep]
                p_kept = probs[keep].detach()
                node.pseudo_step_avg(z_kept, p_kept, kl_scale=kl_scale,
                                     conf_weight_tau0=conf_weight_tau0,
                                     agreement_beta=agreement_beta,
                                     node_class_weights=self._node_skew_weights[i])
                stats[i].add_batch(z_kept.size(0))

        return stats

    def _pseudo_epoch_grad_align_ucb_steps(
        self, steps_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        assert isinstance(self._bandit, GradientAlignedDiscountedUCB)
        bandit: GradientAlignedDiscountedUCB = self._bandit
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        bufs  = self._build_unlabeled_bufs_if_needed()
        _round = int(getattr(self, "_current_stage2_round", 0))
        gens  = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + _round * 1_000_003)
            gens[i] = g

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue
            w_i = self._node_skew_weights[i]

            # Group val set by true label.
            val_by_class: Dict[int, torch.Tensor] = {}
            for z_val_batch, y_val_batch in node.val_loader:
                z_val_batch = z_val_batch.to(self.device, non_blocking=True)
                y_val_batch = y_val_batch.to(self.device, non_blocking=True)
                for c in y_val_batch.unique().tolist():
                    c = int(c)
                    mask = (y_val_batch == c)
                    chunk = z_val_batch[mask]
                    val_by_class[c] = (
                        chunk if c not in val_by_class
                        else torch.cat([val_by_class[c], chunk], dim=0)
                    )

            # Get model predictions on full unlabeled buffer.
            node.model.eval()
            all_preds = []
            z_full = buf.x.to(self.device, non_blocking=True)
            with torch.no_grad():
                for s in range(0, z_full.size(0), self.batch_size):
                    logits = node.model(z_full[s:s + self.batch_size]) if not self.cache_features else node.model.forward_head(z_full[s:s + self.batch_size])
                    all_preds.append(logits.argmax(dim=-1))
            all_preds = torch.cat(all_preds, dim=0)

            buf_by_class: Dict[int, torch.Tensor] = {}
            for c in all_preds.unique().tolist():
                c = int(c)
                buf_by_class[c] = z_full[all_preds == c]

            # -------------------------------------------------------
            # Reward updates: both grad-align and UCB1 signals.
            # The bandit's mu() blends them via alpha = _step / theta.
            # -------------------------------------------------------
            for c, z_buf_c in buf_by_class.items():
                if z_buf_c.size(0) < 8:
                    continue
                z_val_c = val_by_class.get(c)
                if z_val_c is None or z_val_c.size(0) < 8:
                    continue
                j_star = bandit.select_arm(i, c)
                if j_star is None:
                    continue
                teacher = self.nodes[j_star]
                teacher.model.eval()
                with torch.no_grad():
                    p_j_c = F.softmax(teacher.model(z_buf_c) if not self.cache_features else teacher.model.forward_head(z_buf_c), dim=-1).detach()

                # --- Grad-align reward ---
                y_val_c = torch.full((z_val_c.size(0),), c, dtype=torch.long, device=self.device)
                iw_c    = float(w_i[c] ** self.grad_align_iw_temp)
                n_val   = z_val_c.size(0)
                grad_reward = _grad_cosine_sim(
                    node.model, z_val_c, y_val_c, z_buf_c, p_j_c, iw=iw_c,
                    cache_features=self.cache_features,
                )
                grad_reward = grad_reward * min(1.0, n_val / 32.0)
                node.optimizer.zero_grad(set_to_none=True)
                bandit.update(i, j_star, c, grad_reward)

                # --- UCB1 (confidence) reward — always recorded so the UCB1
                #     side stays warm even as alpha grows toward 1. ---
                ucb1_reward = float(p_j_c[:, c].mean())
                bandit.update_ucb1(i, j_star, c, ucb1_reward)

            # -------------------------------------------------------
            # Distillation steps: per-class best teacher via blended mu.
            # -------------------------------------------------------
            for _ in range(int(steps_per_node)):
                xb = buf.sample_with_replacement(self.batch_size, generator=gens[i])
                if xb is None:
                    continue
                z = xb.to(self.device, non_blocking=True)
                node.model.eval()
                with torch.no_grad():
                    pred_classes = (node.model(z) if not self.cache_features else node.model.forward_head(z)).argmax(dim=-1)

                z_consensus = torch.zeros_like(z)
                p_consensus = torch.zeros((z.size(0), self.num_classes), device=self.device)
                used = torch.zeros(z.size(0), dtype=torch.bool, device=self.device)

                ct = float(self.pseudo_conf_threshold)
                et = float(self.pseudo_entropy_threshold)
                t_bandit = max(1, bandit.T[i])

                for c in pred_classes.unique().tolist():
                    c = int(c)
                    mask = (pred_classes == c)
                    z_c  = z[mask]
                    if z_c.size(0) == 0:
                        continue
                    nbrs = node.neighbor_ids
                    # Select top-k arms by blended UCB score for class c.
                    k = min(self.top_k_teachers, len(nbrs))
                    top_k_c = sorted(
                        nbrs,
                        key=lambda j: bandit.ucb_score(i, j, c),
                        reverse=True,
                    )[:k]
                    # Average pseudo-labels uniformly across top-k teachers.
                    with torch.no_grad():
                        p_j_c = sum(
                            F.softmax(self.nodes[j].model(self._feats_for_teacher(j, None, z_c)) if not self.cache_features else self.nodes[j].model.forward_head(self._feats_for_teacher(j, None, z_c)), dim=-1)
                            for j in top_k_c
                        ) / len(top_k_c)
                    p_j_c = p_j_c.detach()

                    keep_c = torch.ones(p_j_c.size(0), dtype=torch.bool, device=self.device)
                    if ct > 0.0:
                        keep_c = keep_c & (p_j_c.max(dim=-1).values >= ct)
                    if et >= 0.0:
                        ent = -(p_j_c * p_j_c.clamp_min(1e-12).log()).sum(dim=-1)
                        keep_c = keep_c & (ent <= et)
                    if not keep_c.any():
                        continue

                    orig_indices = mask.nonzero(as_tuple=True)[0]
                    kept_indices = orig_indices[keep_c]
                    z_consensus[kept_indices] = z_c[keep_c]
                    p_consensus[kept_indices] = p_j_c[keep_c]
                    used[kept_indices] = True

                if not used.any():
                    continue

                z_final = z_consensus[used]
                p_final = p_consensus[used]
                y_hard  = p_final.argmax(dim=-1)
                node.model.train()
                logits = node.model(z_final) if not self.cache_features else node.model.forward_head(z_final)
                loss   = float(node.kl_weight) * kl_scale * F.cross_entropy(logits, y_hard)
                node.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                node.optimizer.step()
                stats[i].add_batch(z_final.size(0))

                # Refresh best-neighbor cache and softmax weights.
                j_best = bandit.best_arm(i, class_weights=w_i)
                if j_best is not None:
                    self._best_neighbor[i] = j_best
                self._neighbor_weights[i] = bandit.softmax_weights(i, class_weights=w_i)

        return stats

    def _pseudo_epoch_greedy_grad_align_steps(
        self, steps_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        """Algorithm 1: Decentralized Learning via Greedy Gradient Alignment.

        Every distillation step, all neighbors are evaluated (full information).
        For each predicted class c the neighbor whose IS-weighted pseudo-label
        gradient has the highest cosine alignment with the local validation
        gradient is selected greedily.  The τ safeguard (self.grad_align_tau)
        rejects the entire class partition when no neighbor achieves positive
        (or above-threshold) alignment, preventing negative transfer.

        There is no bandit state; this method can run from the very first
        pseudo-label round without a bootstrap phase.
        """
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        if steps_per_node <= 0 or kl_scale <= 0.0:
            return stats

        bufs = self._build_unlabeled_bufs_if_needed()
        _round = int(getattr(self, "_current_stage2_round", 0))
        gens: Dict[int, torch.Generator] = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + _round * 1_000_003)
            gens[i] = g

        ct  = float(self.pseudo_conf_threshold)
        et  = float(self.pseudo_entropy_threshold)
        tau = float(self.grad_align_tau)

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue
            w_i = self._node_skew_weights[i]

            # Pre-group validation features by true label so we only iterate
            # the val loader once per node per epoch.
            val_by_class: Dict[int, torch.Tensor] = {}
            for z_val_batch, y_val_batch in node.val_loader:
                z_val_batch = z_val_batch.to(self.device, non_blocking=True)
                y_val_batch = y_val_batch.to(self.device, non_blocking=True)
                for c in y_val_batch.unique().tolist():
                    c = int(c)
                    mask = (y_val_batch == c)
                    chunk = z_val_batch[mask]
                    val_by_class[c] = (
                        chunk if c not in val_by_class
                        else torch.cat([val_by_class[c], chunk], dim=0)
                    )

            for _ in range(int(steps_per_node)):
                xb = buf.sample_with_replacement(self.batch_size, generator=gens[i])
                if xb is None:
                    continue
                z = xb.to(self.device, non_blocking=True)

                # Route each example to a predicted class.
                node.model.eval()
                with torch.no_grad():
                    pred_classes = (node.model(z) if not self.cache_features else node.model.forward_head(z)).argmax(dim=-1)

                z_consensus = torch.zeros_like(z)
                p_consensus = torch.zeros((z.size(0), self.num_classes), device=self.device)
                used        = torch.zeros(z.size(0), dtype=torch.bool, device=self.device)

                for c in pred_classes.unique().tolist():
                    c    = int(c)
                    mask = (pred_classes == c)
                    z_c  = z[mask]
                    if z_c.size(0) == 0:
                        continue

                    z_val_c = val_by_class.get(c)
                    if z_val_c is None or z_val_c.size(0) < 4:
                        continue  # not enough val data for a reliable gradient

                    y_val_c = torch.full(
                        (z_val_c.size(0),), c, dtype=torch.long, device=self.device
                    )

                    # Step 2: Compute the "true north" validation gradient once.
                    g_val_c = _compute_head_val_gradient(node.model, z_val_c, y_val_c)
                    if g_val_c is None:
                        continue

                    # Step 3: Full-information evaluation — score every neighbor.
                    scored: List[Tuple[float, int, torch.Tensor]] = []

                    for j in node.neighbor_ids:
                        teacher = self.nodes[j]
                        teacher.model.eval()
                        with torch.no_grad():
                            p_j_c = F.softmax(
                                teacher.model(z_c) if not self.cache_features else teacher.model.forward_head(z_c), dim=-1
                            ).detach()
                        score = _pseudo_grad_alignment(
                            node.model, g_val_c, z_c, p_j_c,
                            iw=float(w_i[c] ** self.grad_align_iw_temp)
                        )
                        if score > tau:
                            scored.append((score, j, p_j_c))

                    # Step 4: τ safeguard — skip class if no neighbor is helpful.
                    if not scored:
                        continue

                    # Take top-k by alignment score; average their pseudo-labels.
                    scored.sort(key=lambda t: t[0], reverse=True)
                    top_k = scored[:min(self.top_k_teachers, len(scored))]
                    best_probs = sum(p for _, _, p in top_k) / len(top_k)
                    self._best_neighbor[i] = top_k[0][1]  # highest-aligned for diagnostics

                    # Optional confidence / entropy gating on the chosen teacher.
                    keep_c = torch.ones(best_probs.size(0), dtype=torch.bool, device=self.device)
                    if ct > 0.0:
                        keep_c = keep_c & (best_probs.max(dim=-1).values >= ct)
                    if et >= 0.0:
                        ent    = -(best_probs * best_probs.clamp_min(1e-12).log()).sum(dim=-1)
                        keep_c = keep_c & (ent <= et)
                    if not keep_c.any():
                        continue

                    orig_indices = mask.nonzero(as_tuple=True)[0]
                    kept_indices = orig_indices[keep_c]
                    z_consensus[kept_indices] = z_c[keep_c]
                    p_consensus[kept_indices] = best_probs[keep_c]
                    used[kept_indices]         = True

                if not used.any():
                    continue

                # Step 5: Single optimization step on the stitched consensus batch.
                z_final = z_consensus[used]
                p_final = p_consensus[used]
                y_hard  = p_final.argmax(dim=-1)
                node.model.train()
                logits = node.model(z_final) if not self.cache_features else node.model.forward_head(z_final)
                loss   = float(node.kl_weight) * kl_scale * F.cross_entropy(logits, y_hard)
                node.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                node.optimizer.step()
                stats[i].add_batch(z_final.size(0))

        return stats

    def _pseudo_epoch_entropy_gated_ucb_steps(
        self, steps_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        """Algorithm 4: Entropy-Gated Decentralized UCB with Logit-Space Gradient Alignment.

        Per-step flow
        -------------
        1. Entropy gate: precomputed H(w_i) (fixed scalar from node's class
           distribution) selects the reference regime:
             H(w_i) > τH  →  balanced:  g_ref = p − y  (val loader, true labels)
             H(w_i) ≤ τH  →  skewed:    g_ref = p − w_i (unlabeled buf, IS weights)
        2. For ALL neighbors j evaluate alignment reward r_j and update
           ContextualUCBNeighborBandit with the batch mean-feature as context.
        3. Filter candidates: j with r_j > entropy_ucb_align_tau.
           Skip step entirely if no candidate passes (protect the model).
        4. Select top-k teachers from candidates by contextual UCB score.
        5. Average their pseudo-labels uniformly and train on the result.

        Degeneracy properties
        ---------------------
        entropy_ucb_align_tau → −∞  : no filtering → pure contextual UCB1
        entropy_gate_tau = 0        : always balanced regime (true-label g_ref)
        top_k_teachers = 1          : single best arm (original behaviour)
        """
        assert isinstance(self._bandit, ContextualUCBNeighborBandit), (
            "Algorithm 4 requires bandit_type='entropy_ucb' which uses ContextualUCBNeighborBandit"
        )
        bandit: ContextualUCBNeighborBandit = self._bandit
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        bufs  = self._build_unlabeled_bufs_if_needed()
        gens: Dict[int, torch.Generator] = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i)
            gens[i] = g

        ct        = float(self.pseudo_conf_threshold)
        align_tau = float(self.entropy_ucb_align_tau)
        et        = float(self.pseudo_entropy_threshold)

        # mutual_queue[j] = list of (i, reward, z_train_i) meaning:
        # node j should learn from node i's predictions on j's unlabeled buf,
        # using reward as a quality proxy (higher = more aligned teacher).
        # Populated during the main loop; consumed in the mutual-distillation pass.
        mutual_queue: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in self.nodes.keys()
        }

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue

            # ── Distillation gate ─────────────────────────────────────────
            # Skip collection + communication entirely if self is already the
            # strongest teacher in the neighborhood.  Uses IW-confidence scores
            # from the last bootstrap/rebootstrap (updated every rebootstrap_freq
            # rounds) vs the bandit's best-neighbor running mean.
            # Margin = deploy_self_margin so both decisions use the same threshold.
            _self_iw_map = getattr(self, "_self_iw_scores", {})
            if i in _self_iw_map and node.neighbor_ids:
                _self_q   = float(_self_iw_map[i])
                _best_nbr = max(bandit.Q[i].get(j, 0.0) for j in node.neighbor_ids)
                _gate_margin = float(getattr(self, "deploy_self_margin", 0.05))
                if _self_q > _best_nbr + _gate_margin:
                    continue  # Self is clearly strongest — no distillation needed

            w_i        = self._node_skew_weights[i]          # (C,) numpy float64
            w_i_tensor = torch.tensor(w_i, dtype=torch.float32, device=self.device)

            # ── Entropy gate: fixed scalar per node ──────────────────────
            H_wi = float(
                -np.sum(w_i * np.log(np.clip(w_i, 1e-12, None))) / math.log(10)
            )
            use_balanced = (H_wi > float(self.entropy_gate_tau)) and len(node.val_loader.dataset) > 0
            val_iter     = iter(node.val_loader) if use_balanced else None
            _val_step_i  = 0  # tracks which batch we're on in the val loader

            # ── Collection phase ──────────────────────────────────────────
            # Run steps_per_node collection batches. For each batch that
            # passes the alignment gate, score every surviving example by
            # max teacher confidence and accumulate into a pool.
            # After all collection steps, sort the pool globally by score
            # and train on the top pseudo_examples_per_round examples.
            # This gives a hard budget and ensures the highest-quality
            # examples (most confident teachers) are always selected,
            # regardless of which batch or step they came from.
            pool_z:     List[torch.Tensor] = []   # features (in student's arch space)
            pool_p:     List[torch.Tensor] = []   # combined pseudo-label dists
            pool_score: List[torch.Tensor] = []   # per-example quality score
            _best_nbr_this_node: Optional[int] = None

            # Raw unlabeled indices for node i — used to look up the same images
            # in each teacher's own feature cache when architectures differ.
            _i_unlab_idx = self._unlabeled_idx_per_node.get(i)  # (N_unlab,) or None
            _i_arch      = self._node_arch_map.get(i, self.arch)
            _has_multi_arch = self.random_models and len(self._feat_cache_by_arch) > 1

            for step_idx in range(int(steps_per_node)):
                # Sample a random batch of positions into i's unlabeled buffer
                pos_idx = torch.randint(
                    low=0, high=buf.n, size=(self.batch_size,),
                    generator=gens[i], device="cpu"
                ) if buf.n > 0 else None
                if pos_idx is None:
                    continue

                xb = buf.x[pos_idx]
                z_train = xb.to(self.device, non_blocking=True)

                # Resolve the raw training-set indices for this batch so each
                # teacher can look up features in its own backbone's cache.
                if _has_multi_arch and _i_unlab_idx is not None and _i_unlab_idx.numel() > 0:
                    batch_train_idx = _i_unlab_idx[pos_idx]  # actual dataset row numbers
                else:
                    batch_train_idx = None

                node.model.eval()
                with torch.no_grad():
                    # In stage_2_tuning mode cache_features=False, z_train is raw images.
                    # Must use full forward (backbone + head) not forward_head alone.
                    if self.cache_features:
                        p_train = F.softmax(node.model.forward_head(z_train), dim=-1)
                    else:
                        p_train = F.softmax(node.model(z_train), dim=-1)
                    # Bandit always expects feat_dim-dimensional context vectors.
                    # Extract backbone features when buffer contains raw images.
                    if z_train.shape[-1] != self.feat_dim:
                        z_feat_train = node.model.forward_features(z_train)
                    else:
                        z_feat_train = z_train

                # Build reference gradient
                if use_balanced:
                    try:
                        z_ref, y_ref = next(val_iter)
                    except StopIteration:
                        val_iter    = iter(node.val_loader)
                        _val_step_i = 0
                        try:
                            z_ref, y_ref = next(val_iter)
                        except StopIteration:
                            # val_loader genuinely empty — fall back to skewed regime
                            use_balanced = False
                            val_iter     = None
                            continue
                    z_ref = z_ref.to(self.device, non_blocking=True)
                    y_ref = y_ref.to(self.device, non_blocking=True)
                    # Recover raw dataset indices for this val batch so cross-arch
                    # teachers can look up the same images in their own cache.
                    # val_loader is shuffle=False so batch _val_step_i always
                    # contains val_idx[_val_step_i*B : (_val_step_i+1)*B].
                    _val_idx_i = self._val_idx_per_node.get(i)
                    if _val_idx_i is not None and _val_idx_i.numel() > 0:
                        _bs      = z_ref.size(0)
                        _start   = _val_step_i * self.batch_size
                        _val_batch_idx = _val_idx_i[_start: _start + _bs]
                    else:
                        _val_batch_idx = None
                    _val_step_i += 1
                    with torch.no_grad():
                        if self.cache_features:
                            p_ref = F.softmax(node.model.forward_head(z_ref), dim=-1)
                        else:
                            p_ref = F.softmax(node.model(z_ref), dim=-1)
                    y_onehot = F.one_hot(y_ref, num_classes=self.num_classes).float()
                    g_ref   = p_ref - y_onehot
                    z_align = z_ref
                    p_align = p_ref
                else:
                    _val_batch_idx = None
                    g_ref   = p_train - w_i_tensor.unsqueeze(0)
                    z_align = z_train
                    p_align = p_train

                n_ref        = g_ref.norm(dim=-1).clamp_min(1e-8)
                # Use feat_dim-dimensional representation for bandit context.
                # In cache_features mode, z_align is already 1280-d features.
                # In stage_2_tuning mode (cache_features=False), z_align is raw
                # images — use zero context so the bandit degrades to scalar UCB.
                if self.cache_features:
                    z_feat_align = z_feat_train if (z_align is z_train) else z_align
                    z_context_np = z_feat_align.detach().mean(dim=0).cpu().numpy().astype(np.float64)
                else:
                    z_context_np = np.zeros(bandit.feat_dim, dtype=np.float64)

                # Student predicted class per example — used for per-class arm routing.
                pred_classes_train = p_train.argmax(dim=-1)  # (B,)

                # Evaluate all neighbors, update bandit (class-agnostic AND per-class)
                nbrs      = node.neighbor_ids
                rewards:   Dict[int, float]        = {}
                p_by_nbr: Dict[int, torch.Tensor] = {}

                for j in nbrs:
                    nbr     = self.nodes[j]
                    _j_arch = self._node_arch_map.get(j, self.arch)
                    nbr.model.eval()
                    with torch.no_grad():
                        _T = max(float(self.pseudo_label_temp), 1e-4)

                        # ── Cross-arch feature lookup ──────────────────────
                        # If teacher j has a different backbone from student i,
                        # j's head was trained on j's feature distribution and
                        # will produce garbage on i's features.  Look up the
                        # same batch images in j's own feature cache instead.
                        if (_has_multi_arch
                                and _j_arch != _i_arch
                                and batch_train_idx is not None
                                and _j_arch in self._feat_cache_by_arch):
                            _j_feats_full = self._feat_cache_by_arch[_j_arch]["feats_train"]
                            z_train_j = _j_feats_full[batch_train_idx].to(
                                self.device, non_blocking=True
                            )
                        else:
                            z_train_j = z_train  # same arch — use student's features

                        # In stage_2_tuning mode cache_features=False and
                        # z_train_j contains raw images (not 1280-d features).
                        # forward_head expects features — use full model instead.
                        if self.cache_features:
                            _logits_train = nbr.model.forward_head(z_train_j)
                        else:
                            _logits_train = nbr.model(z_train_j)
                        p_j_train = (F.softmax(_logits_train, dim=-1) if _T == 1.0
                                     else F.softmax(_logits_train / _T, dim=-1))
                        # Alignment reward: teacher must run on its own features.
                        # In balanced mode z_align = student val features (wrong arch).
                        # In skewed mode z_align = z_train = z_train_j already.
                        if use_balanced:
                            z_align_j = self._feats_for_teacher(j, _val_batch_idx, z_align)
                        else:
                            z_align_j = z_train_j
                        if self.cache_features:
                            p_j_align = F.softmax(nbr.model.forward_head(z_align_j), dim=-1)
                        else:
                            p_j_align = F.softmax(nbr.model(z_align_j), dim=-1)

                    mean_conf = float(p_j_train.max(dim=-1).values.mean().item())
                    if ct > 0.0 and mean_conf < ct:
                        r = 0.0
                    else:
                        g_j   = p_align - p_j_align
                        n_j   = g_j.norm(dim=-1).clamp_min(1e-8)
                        dots  = (g_ref * g_j).sum(dim=-1)
                        cos_s = (dots / (n_ref * n_j)).mean().item()
                        r     = float(0.5 * (1.0 + cos_s))

                    rewards[j]   = r
                    p_by_nbr[j]  = p_j_train
                    bandit.update(i, j, z_context_np, r)

                    # ── Per-class arm updates ──────────────────────────────
                    # Route examples by student prediction; update per-class
                    # arm (i, j, c) with teacher j's mean confidence on those
                    # examples. Lets the bandit learn "for class-c inputs,
                    # which neighbor is best?" — the class-conditioning that
                    # enables specialised per-class knowledge transfer.
                    for _c in pred_classes_train.unique().tolist():
                        _c = int(_c)
                        _mask_c = (pred_classes_train == _c)
                        if _mask_c.sum() < 4:
                            continue
                        _r_c = float(p_j_train[_mask_c, _c].mean().item())
                        _z_c = z_feat_train[_mask_c].detach().mean(dim=0).cpu().numpy().astype(np.float64)
                        bandit.update_class(i, j, _c, _z_c, _r_c)

                bandit.T[i] += 1

                # Candidate filter + teacher selection
                if getattr(self, "oracle_bandit", False):
                    # ── Oracle bandit: omniscient selection ────────────────
                    # Pick top-k neighbors by ground-truth test accuracy on
                    # node i's own test set. No alignment gate, no exploration.
                    # Establishes the theoretical ceiling for bandit selection.
                    _test_loader_i = self._node_test_loaders.get(i)
                    if _test_loader_i is None or not nbrs:
                        continue
                    oracle_scores = {
                        j: self.evaluate_j_on_i_loader(
                            j, i, _test_loader_i, self._test_idx_per_node
                        )
                        for j in nbrs
                    }
                    k = min(self.top_k_teachers, len(nbrs))
                    top_k = sorted(nbrs, key=lambda j: oracle_scores[j], reverse=True)[:k]
                    candidates = top_k  # all pass — oracle has no gate
                    _best_nbr_this_node = top_k[0]
                else:
                    # Alignment gate: τ_align = 0.5 ensures cos > 0, guaranteeing
                    # positive weight-space inner product (Corollary 4.3).
                    candidates = [j for j in nbrs if rewards[j] > align_tau]
                    if not candidates:
                        continue

                    # ── Class-conditioned teacher selection ────────────────
                    # Route unlabeled examples by student predicted class c,
                    # then select the top-k teachers for class c using the
                    # per-class LinUCB arm score. This answers "for class-c
                    # inputs specifically, which neighbor is most informative?"
                    # rather than picking one teacher for the whole mixed batch.
                    z_consensus = torch.zeros_like(z_train)
                    p_consensus = torch.zeros((z_train.size(0), self.num_classes), device=self.device)
                    used_mask   = torch.zeros(z_train.size(0), dtype=torch.bool, device=self.device)

                    for _c in pred_classes_train.unique().tolist():
                        _c = int(_c)
                        _mask_c = (pred_classes_train == _c)
                        _z_c    = z_train[_mask_c]
                        if _z_c.size(0) == 0:
                            continue

                        # Per-class context: mean feature of class-c examples
                        # (zero vector in image mode — bandit degrades to scalar UCB)
                        if self.cache_features:
                            _z_ctx_c = _z_c.detach().mean(dim=0).cpu().numpy().astype(np.float64)
                        else:
                            _z_ctx_c = np.zeros(bandit.feat_dim, dtype=np.float64)

                        # Select top-k teachers for class c by per-class UCB score
                        _k_c    = min(self.top_k_teachers, len(candidates))
                        _top_k_c = sorted(
                            candidates,
                            key=lambda j: bandit.ucb_score_class(i, j, _c, _z_ctx_c),
                            reverse=True,
                        )[:_k_c]

                        if _best_nbr_this_node is None:
                            _best_nbr_this_node = _top_k_c[0]

                        # Combine pseudo-labels from top-k teachers for class c
                        if len(_top_k_c) == 1:
                            _p_c = p_by_nbr[_top_k_c[0]][_mask_c]
                        else:
                            _stack_c = torch.stack([p_by_nbr[j][_mask_c] for j in _top_k_c], dim=0)  # (k, n_c, C)
                            _conf_c  = _stack_c[:, :, _c]                                             # (k, n_c)
                            _iw_c    = w_i_tensor[_c]                                                 # scalar
                            _raw_w_c = _conf_c * _iw_c                                               # (k, n_c)
                            _norm_c  = _raw_w_c.sum(dim=0, keepdim=True).clamp_min(1e-8)
                            _p_c     = ((_raw_w_c / _norm_c).unsqueeze(-1) * _stack_c).sum(dim=0)    # (n_c, C)

                        # Per-class confidence / entropy gate
                        _keep_c = torch.ones(_p_c.size(0), dtype=torch.bool, device=self.device)
                        if ct > 0.0:
                            _keep_c = _keep_c & (_p_c.max(dim=-1).values >= ct)
                        if et >= 0.0:
                            _ent_c  = -(_p_c * _p_c.clamp_min(1e-12).log()).sum(dim=-1)
                            _keep_c = _keep_c & (_ent_c <= et)
                        if not _keep_c.any():
                            continue

                        _orig_idx = _mask_c.nonzero(as_tuple=True)[0]
                        _kept_idx = _orig_idx[_keep_c]
                        z_consensus[_kept_idx] = _z_c[_keep_c]
                        p_consensus[_kept_idx] = _p_c[_keep_c]
                        used_mask[_kept_idx]   = True

                        # Queue per-class best teacher for mutual distillation
                        if self.mutual_distillation:
                            _best_j_c = _top_k_c[0]
                            if self.nodes[_best_j_c].pseudo_allowed and self.nodes[_best_j_c].kl_weight > 0.0:
                                mutual_queue[_best_j_c].append((i, rewards.get(_best_j_c, 0.0)))

                    if not used_mask.any():
                        continue

                    # Accumulate into pool — score = max teacher confidence × IW
                    z_kept   = z_consensus[used_mask].detach().cpu()
                    p_kept   = p_consensus[used_mask].detach().cpu()
                    conf     = p_kept.max(dim=-1).values
                    pred_cls = p_kept.argmax(dim=-1)
                    w_i_cpu  = torch.tensor(w_i, dtype=torch.float32)
                    iw_sel   = w_i_cpu[pred_cls]
                    scores   = conf * iw_sel
                    pool_z.append(z_kept)
                    pool_p.append(p_kept)
                    pool_score.append(scores)

            # ── Selection phase: global top-N across all collected examples ─
            if pool_z:
                all_z     = torch.cat(pool_z,     dim=0)   # (M, feat_dim)
                all_p     = torch.cat(pool_p,     dim=0)   # (M, C)
                all_score = torch.cat(pool_score, dim=0)   # (M,)

                cap = int(self.pseudo_examples_per_round)
                if cap > 0 and all_z.size(0) > cap:
                    # Keep globally top-cap examples by teacher confidence
                    topN_idx = torch.topk(all_score, k=cap, largest=True, sorted=False).indices
                    all_z = all_z[topN_idx]
                    all_p = all_p[topN_idx]

                # ── Training phase: run optimizer in mini-batches ──────────
                perm = torch.randperm(all_z.size(0))
                for b_start in range(0, all_z.size(0), self.batch_size):
                    b_idx = perm[b_start:b_start + self.batch_size]
                    z_tr  = all_z[b_idx].to(self.device, non_blocking=True)
                    p_tr  = all_p[b_idx].to(self.device, non_blocking=True)
                    node.pseudo_step_avg(
                        z_tr, p_tr,
                        kl_scale=kl_scale,
                        conf_weight_tau0=conf_weight_tau0,
                        agreement_beta=agreement_beta,
                    )
                    stats[i].add_batch(z_tr.size(0))

                if _best_nbr_this_node is not None:
                    self._best_neighbor[i] = _best_nbr_this_node

            # Refresh neighbor-weight cache from bandit scalar Q-values.
            self._neighbor_weights[i] = bandit.softmax_weights(i)

        # ── Mutual distillation pass ──────────────────────────────────────
        # For each node j that was selected as a teacher for some node i,
        # also let i act as a teacher for j.  Node j distills from i's
        # predictions on j's OWN unlabeled buffer.
        #
        # Knowledge flows in both directions along each active edge, recovering
        # the tight coupling of DML-style mutual distillation while keeping the
        # alignment gate as a quality filter in both directions.
        #
        # Communication cost: ~2× pseudo-label traffic; no parameter sharing.
        # Gate: only runs when --mutual_distillation is set.
        if self.mutual_distillation:
            for j, donors in mutual_queue.items():
                if not donors:
                    continue
                node_j = self.nodes[j]
                if node_j.kl_weight <= 0.0 or not node_j.pseudo_allowed:
                    continue
                buf_j = bufs.get(j)
                if buf_j is None or buf_j.n <= 0:
                    continue

                w_j   = self._node_skew_weights[j]
                w_j_t = torch.tensor(w_j, dtype=torch.float32, device=self.device)

                # Sample a batch from j's buffer with index tracking for cross-arch lookup.
                _j_buf_n   = buf_j.n
                _j_pos_idx = torch.randint(0, _j_buf_n, (self.batch_size,),
                                           generator=gens[j], device="cpu")
                xb_j = buf_j.x[_j_pos_idx]
                z_j  = xb_j.to(self.device, non_blocking=True)

                # Raw dataset indices for j's unlabeled batch (for cross-arch donors).
                _j_unlab_idx = self._unlabeled_idx_per_node.get(j)
                _j_batch_idx = (
                    _j_unlab_idx[_j_pos_idx]
                    if (_j_unlab_idx is not None and _j_unlab_idx.numel() > 0)
                    else None
                )

                node_j.model.eval()
                with torch.no_grad():
                    if self.cache_features:
                        p_j_self = F.softmax(node_j.model.forward_head(z_j), dim=-1)
                    else:
                        p_j_self = F.softmax(node_j.model(z_j), dim=-1)
                    pred_c_j = p_j_self.argmax(dim=-1)   # (B,)

                # Collect donor pseudo-labels with per-class confidence weighting.
                donor_probs: List[torch.Tensor] = []
                donor_weights_raw: List[torch.Tensor] = []
                for (i_donor, _r) in donors:
                    node_i = self.nodes[i_donor]
                    node_i.model.eval()
                    with torch.no_grad():
                        # Donor i must run on its own feature space.
                        z_i_on_j    = self._feats_for_teacher(i_donor, _j_batch_idx, z_j)
                        if self.cache_features:
                            p_i_on_zj = F.softmax(node_i.model.forward_head(z_i_on_j), dim=-1)
                        else:
                            p_i_on_zj = F.softmax(node_i.model(z_i_on_j), dim=-1)
                    conf_ij = p_i_on_zj[
                        torch.arange(z_j.size(0), device=self.device), pred_c_j
                    ]                                    # (B,)
                    raw_w = conf_ij * w_j_t[pred_c_j]   # (B,)
                    donor_probs.append(p_i_on_zj)
                    donor_weights_raw.append(raw_w)

                if not donor_probs:
                    continue

                if len(donor_probs) == 1:
                    p_mutual = donor_probs[0]
                else:
                    stack_m = torch.stack(donor_probs, dim=0)        # (D, B, C)
                    raw_wm  = torch.stack(donor_weights_raw, dim=0)  # (D, B)
                    wm      = raw_wm / raw_wm.sum(dim=0, keepdim=True).clamp_min(1e-8)
                    p_mutual = (wm.unsqueeze(-1) * stack_m).sum(dim=0)  # (B, C)

                keep_m = torch.ones(p_mutual.size(0), dtype=torch.bool, device=self.device)
                if ct > 0.0:
                    keep_m = keep_m & (p_mutual.max(dim=-1).values >= ct)
                if et >= 0.0:
                    ent_m  = -(p_mutual * p_mutual.clamp_min(1e-12).log()).sum(dim=-1)
                    keep_m = keep_m & (ent_m <= et)
                if not keep_m.any():
                    continue

                node_j.pseudo_step_avg(
                    z_j[keep_m], p_mutual[keep_m],
                    kl_scale=kl_scale,
                    conf_weight_tau0=conf_weight_tau0,
                    agreement_beta=agreement_beta,
                )
                stats[j].add_batch(int(keep_m.sum().item()))

        return stats

    def _pseudo_epoch_fixed_examples(
        self, examples_per_node: int, kl_scale: float, conf_weight_tau0: float, agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        stats = {i: PseudoStats() for i in self.nodes.keys()}
        if examples_per_node <= 0 or kl_scale <= 0.0:
            return stats

        # Greedy gradient alignment (Algorithm 1) — no bandit state, no bootstrap needed.
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "grad_align"):
            bufs     = self._build_unlabeled_bufs_if_needed()
            max_buf  = max((bufs[i].n for i in self.nodes if bufs[i].n > 0),
                           default=self.batch_size)
            buf_steps  = max(1, max_buf // max(1, self.batch_size))
            req_steps  = max(1, int(examples_per_node) // max(1, self.batch_size))
            steps      = min(req_steps, buf_steps)
            return self._pseudo_epoch_greedy_grad_align_steps(steps, kl_scale, conf_weight_tau0, agreement_beta)

        # Discounted UCB + gradient alignment (Algorithm 2) — enters after bootstrap.
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "grad_align_ucb"
                and self._bandit_bootstrapped):
            self._bandit.decay()
            bufs     = self._build_unlabeled_bufs_if_needed()
            max_buf  = max((bufs[i].n for i in self.nodes if bufs[i].n > 0),
                           default=self.batch_size)
            buf_steps  = max(1, max_buf // max(1, self.batch_size))
            req_steps  = max(1, int(examples_per_node) // max(1, self.batch_size))
            steps      = min(req_steps, buf_steps)
            return self._pseudo_epoch_grad_align_ucb_steps(steps, kl_scale, conf_weight_tau0, agreement_beta)

        # Entropy-Gated UCB with Logit-Space Gradient Alignment (Algorithm 4).
        if (self.neighbor_weighting == "ucb"
                and self.bandit_type == "entropy_ucb"):
            bufs     = self._build_unlabeled_bufs_if_needed()
            max_buf  = max((bufs[i].n for i in self.nodes if bufs[i].n > 0),
                           default=self.batch_size)
            buf_steps  = max(1, max_buf // max(1, self.batch_size))
            req_steps  = max(1, int(examples_per_node) // max(1, self.batch_size))
            steps      = min(req_steps, buf_steps)
            return self._pseudo_epoch_entropy_gated_ucb_steps(steps, kl_scale, conf_weight_tau0, agreement_beta)

        # Omniscient oracle: always picks the best neighbor by ground-truth test accuracy.
        if self.bandit_type == "omniscient":
            bufs     = self._build_unlabeled_bufs_if_needed()
            max_buf  = max((bufs[i].n for i in self.nodes if bufs[i].n > 0), default=self.batch_size)
            steps    = max(1, min(int(examples_per_node) // max(1, self.batch_size), max_buf // max(1, self.batch_size)))
            return self._pseudo_epoch_omniscient_steps(steps, kl_scale, conf_weight_tau0, agreement_beta)
        # 2026-04-22: Vanilla UCB1 fall-through path. Previously this
        # block referenced `bufs` without ever assigning it, producing
        # `UnboundLocalError: cannot access local variable 'bufs'` at
        # the first stage-2 round with examples_per_node > 0. All the
        # early-return branches above (grad_align, grad_align_ucb,
        # entropy_ucb, omniscient) assign `bufs` before returning, so
        # this path needs the same call. The bug was latent because
        # prior runs typically used a non-vanilla bandit_type that hit
        # an early-return branch; bandit_type='ucb1' (the default) is
        # what exposes it.
        bufs = self._build_unlabeled_bufs_if_needed()
        gens = {}
        for i in self.nodes.keys():
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i)
            gens[i] = g
        perm = {i: bufs[i].make_perm(generator=gens[i]) for i in self.nodes.keys()}
        off  = {i: 0 for i in self.nodes.keys()}
        use_best = (self.pseudo_teacher_mode == "best")

        for i, node in self.nodes.items():
            if node.kl_weight <= 0.0 or not node.neighbor_ids or not node.pseudo_allowed:
                continue
            buf = bufs[i]
            if buf.n <= 0:
                continue
            remaining = int(examples_per_node)
            while remaining > 0:
                start = off[i]
                if start >= buf.n:
                    break
                take = min(self.batch_size, remaining, buf.n - start)
                xb   = buf.x[perm[i][start:start + take]]
                off[i] = start + take
                if self.cache_features:
                    z = xb.to(self.device, non_blocking=True)
                    z_kept, p_avg = (
                        self._best_neighbor_pseudo_from_features(i, z)
                        if use_best else
                        self._avg_neighbor_pseudo_from_features(i, z)
                    )
                    if z_kept.size(0) > 0:
                        stats[i].add_batch(z_kept.size(0))
                        node.pseudo_step_avg(z_kept, p_avg, kl_scale=kl_scale,
                                             conf_weight_tau0=conf_weight_tau0, agreement_beta=agreement_beta)
                else:
                    x = xb.to(self.device, non_blocking=True)
                    x_kept, p_avg = (
                        self._best_neighbor_pseudo_from_images(i, x)
                        if use_best else
                        self._avg_neighbor_pseudo_from_images(i, x)
                    )
                    if x_kept.size(0) > 0:
                        with torch.no_grad():
                            z_kept = node.model.forward_features(x_kept)
                        stats[i].add_batch(z_kept.size(0))
                        node.pseudo_step_avg(z_kept, p_avg, kl_scale=kl_scale,
                                             conf_weight_tau0=conf_weight_tau0, agreement_beta=agreement_beta)
                remaining -= int(take)
        return stats

    # ------------------------------------------------------------------
    # Teacher EMA — local Mean-Teacher-style self-distillation
    # ------------------------------------------------------------------
    def _ema_update_heads(self, ema_decay: float) -> None:
        """Update EMA full model state dict from current student weights (warm-up ramped).

        In stage 1 (cache_features=True) the backbone is frozen so only head
        weights change — but we track the full state_dict anyway so the EMA is
        ready for stage 2 without a key-mismatch crash.
        In stage 2 (cache_features=False) the backbone is trainable; the EMA
        must track backbone + head so the teacher reflects encoder evolution.
        On the first call after a stage switch the key sets may differ — we
        reinitialise the EMA from the current student rather than crashing.
        """
        self._ema_step += 1
        decay = min(float(ema_decay),
                    (1.0 + self._ema_step) / (10.0 + self._ema_step))
        for i, node in self.nodes.items():
            student_sd = {k: v.detach().cpu().float()
                          for k, v in node.model.state_dict().items()}
            if i not in self._ema_head_states:
                self._ema_head_states[i] = {k: v.clone() for k, v in student_sd.items()}
            elif set(self._ema_head_states[i].keys()) != set(student_sd.keys()):
                # Key mismatch after stage switch — reinitialise from student
                self._ema_head_states[i] = {k: v.clone() for k, v in student_sd.items()}
            else:
                ema_sd = self._ema_head_states[i]
                with torch.no_grad():
                    for k in ema_sd:
                        pe, ps = ema_sd[k], student_sd[k]
                        if pe.dtype.is_floating_point:
                            pe.mul_(decay).add_(ps, alpha=1.0 - decay)
                        else:
                            pe.copy_(ps)

    def _ema_self_distil(self, round_idx: int, kl_scale: float) -> None:
        """Self-distillation from EMA teacher on local unlabeled buffer.

        Works in both stage 1 (cache_features=True, features) and stage 2
        (cache_features=False, raw images). The EMA now tracks the full model
        so the teacher's backbone is also smoothed in stage 2.
        """
        if not self._ema_head_states:
            return
        bufs = self._build_unlabeled_bufs_if_needed()
        ct = float(self.pseudo_conf_threshold)
        for i, node in self.nodes.items():
            buf = bufs.get(i)
            if buf is None or buf.n == 0:
                continue
            if not node.pseudo_allowed or node.kl_weight <= 0.0:
                continue
            ema_sd_dev   = {k: v.to(self.device) for k, v in self._ema_head_states[i].items()}
            student_snap = {k: v.clone() for k, v in node.model.state_dict().items()}
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed * 10_000 + i + round_idx * 997)
            for _ in range(self.teacher_ema_steps):
                xb = buf.sample_with_replacement(self.batch_size, generator=g)
                if xb is None:
                    continue
                z = xb.to(self.device, non_blocking=True)
                # Teacher inference with EMA weights
                node.model.load_state_dict(ema_sd_dev, strict=False)
                node.model.eval()
                with torch.no_grad():
                    probs = F.softmax(
                        node.model.forward_head(z) if self.cache_features else node.model(z),
                        dim=-1,
                    )
                if ct > 0.0:
                    keep = probs.max(dim=-1).values >= ct
                    if not keep.any():
                        node.model.load_state_dict(student_snap, strict=False)
                        continue
                    z, probs = z[keep], probs[keep]
                # Student gradient step with restored weights
                node.model.load_state_dict(student_snap, strict=False)
                node.model.train()
                eff_kl = float(node.kl_weight) * float(kl_scale)
                log_ps = F.log_softmax(
                    node.model.forward_head(z) if self.cache_features else node.model(z),
                    dim=-1,
                )
                loss = eff_kl * F.kl_div(log_ps, probs.detach(), reduction="batchmean")
                node.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                node.optimizer.step()
                student_snap = {k: v.clone() for k, v in node.model.state_dict().items()}
            node.model.load_state_dict(student_snap, strict=False)
            node.model.train()

    # ------------------------------------------------------------------
    # run_round
    # ------------------------------------------------------------------
    def run_round(
        self,
        round_idx: int,
        pseudo_epochs: int,
        pseudo_mode: str,
        steps_per_node: int,
        examples_per_node: int,
        cap_examples_per_node: int,
        conf_weight_tau0: float,
        sup_steps_total: int,
        sup_steps_per_node: int,
        agreement_beta: float = 0.0,
    ) -> Dict[int, PseudoStats]:
        if self.debug_time:
            self._tb.reset()
            self._tb.begin("round.total")

        # Reset per-round loss accumulators on every node.
        for _node in self.nodes.values():
            _node._sup_loss_sum   = 0.0; _node._sup_loss_n   = 0
            _node._pseudo_loss_sum = 0.0; _node._pseudo_loss_n = 0

        self.supervised_steps_synchronous(
            sup_steps_total=sup_steps_total,
            sup_steps_per_node=sup_steps_per_node,
            round_idx=int(round_idx),
        )

        # ── Teacher EMA: update + self-distil on local unlabeled buffer ──
        if self.teacher_ema:
            self._ema_update_heads(ema_decay=getattr(self, "_teacher_ema_alpha", 0.1))
            kl_scale_ema = self._kl_scale_for_round(round_idx)
            if kl_scale_ema > 0.0 and self.teacher_ema_steps > 0:
                self._ema_self_distil(round_idx=int(round_idx), kl_scale=kl_scale_ema)

        if (self.neighbor_weight_update_freq > 0
                and (int(round_idx) % self.neighbor_weight_update_freq == 0)):
            if self.neighbor_weighting == "ucb":
                self.update_bandit_staggered()
            elif self.neighbor_weighting == "train_acc":
                self.update_neighbor_weights_from_train()

        kl_scale = self._kl_scale_for_round(round_idx)
        pseudo_stats_total = {i: PseudoStats() for i in self.nodes.keys()}
        if pseudo_epochs <= 0 or kl_scale <= 0.0:
            if self.debug_time:
                self._tb.end("round.total")
                self._last_round_timing = self._tb.as_dict()
            return pseudo_stats_total

        # ── Adaptive KL scale: maintain sup/kl gradient ratio ─────────────
        # As supervised loss falls during training, kl_weight × kl_scale stays
        # fixed while sup_loss halves → distillation contributes a shrinking
        # fraction of the total gradient. We track a slow EMA of the mean
        # supervised loss across nodes and rescale kl_scale so the ratio:
        #   (kl_weight × kl_scale × kl_loss) / sup_loss_ema
        # stays at its value when pseudo-labels first became active (round
        # pseudo_warmup_rounds). anchor_sup_loss is set on the first active
        # round; thereafter kl_scale is boosted by anchor/current whenever
        # ── Supervised-loss EMA (tracking only, no kl_scale modification) ──
        # The adaptive KL scaling (anchor/current ratio) was found to cause
        # test accuracy degradation: as sup_loss naturally decreases during
        # training, the KL weight ramps up (1.0→2.3×), amplifying bad
        # pseudo-labels from weak neighbors and poisoning the model.
        # We track the EMA for logging but do NOT modify kl_scale.
        _sup_vals = [n._sup_loss_sum / max(1, n._sup_loss_n)
                     for n in self.nodes.values() if getattr(n, "_sup_loss_n", 0) > 0]
        if _sup_vals:
            _cur_sup = float(sum(_sup_vals) / len(_sup_vals))
            _ema_decay = 0.05
            _prev_ema  = getattr(self, "_sup_loss_ema", None)
            _sup_ema   = _cur_sup if _prev_ema is None else (
                (1.0 - _ema_decay) * _prev_ema + _ema_decay * _cur_sup
            )
            self._sup_loss_ema = _sup_ema
            if not hasattr(self, "_anchor_sup_loss"):
                self._anchor_sup_loss = float(_sup_ema)
        self._current_stage2_round = getattr(self, "_current_stage2_round", -1) + 1
        for _ in range(int(pseudo_epochs)):
            if pseudo_mode == "steps":
                st = self._pseudo_epoch_fixed_steps(steps_per_node, kl_scale, conf_weight_tau0, agreement_beta)
            elif pseudo_mode == "examples":
                st = self._pseudo_epoch_fixed_examples(examples_per_node, kl_scale, conf_weight_tau0, agreement_beta)
            elif pseudo_mode == "all":
                st = self._pseudo_epoch_fixed_examples(cap_examples_per_node, kl_scale, conf_weight_tau0, agreement_beta)
            else:
                raise ValueError(f"Unknown pseudo_mode: {pseudo_mode}")
            for i in self.nodes.keys():
                pseudo_stats_total[i].pseudo_batches  += st[i].pseudo_batches
                pseudo_stats_total[i].pseudo_examples += st[i].pseudo_examples

        if self.debug_time:
            self._tb.end("round.total")
            self._last_round_timing = self._tb.as_dict()
        return pseudo_stats_total

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_all_nodes_on_val(self) -> Dict[int, float]:
        return {i: node.evaluate_accuracy(node.val_loader) for i, node in self.nodes.items()}

    def evaluate_all_nodes_on_test(self) -> Dict[int, float]:
        """Overall accuracy per node on its local (skewed) test set."""
        if self._node_test_loaders:
            return {
                i: node.evaluate_accuracy(self._node_test_loaders[i])
                for i, node in self.nodes.items()
            }
        return {
            i: node.evaluate_accuracy(self.test_loader)
            for i, node in self.nodes.items()
        }

    def evaluate_all_nodes_per_class_on_test(self) -> Dict[int, np.ndarray]:
        # Always use per-node local test loaders (same skewed distribution as
        # train/val). Falls back to global test_loader only in IID mode where
        # all distributions are identical.
        if self._node_test_loaders:
            return {
                i: node.evaluate_per_class_accuracy(self._node_test_loaders[i], num_classes=self.num_classes)
                for i, node in self.nodes.items()
            }
        return {
            i: node.evaluate_per_class_accuracy(self.test_loader, num_classes=self.num_classes)
            for i, node in self.nodes.items()
        }

    def evaluate_all_nodes_per_class_on_train(self) -> Dict[int, np.ndarray]:
        return {
            i: node.evaluate_per_class_accuracy(node.train_eval_loader, num_classes=self.num_classes)
            for i, node in self.nodes.items()
        }

    def weighted_test_acc(
        self, per_class_test: Dict[int, np.ndarray], node_i: int, node_j: int,
    ) -> float:
        return _weighted_acc(per_class_test[node_j], self._node_skew_weights[node_i])


# ----------------------------
# Deployable CN-best
# ----------------------------
def compute_metrics_from_per_class(
    system: "DecentralizedPseudoLabelSystem",
    per_class_test: Dict[int, np.ndarray],
    per_class_train: Dict[int, np.ndarray],
    deploy_criteria: str = "val_query",
    test_acc: Optional[Dict[int, float]] = None,
    train_acc: Optional[Dict[int, float]] = None,
    debug_val_query: bool = False,
) -> Tuple[float, float, float]:
    """Compute test, cn_local, and train metrics.

    Uses overall accuracy (test_acc / train_acc) when provided — correct for
    per-node skewed test sets. Falls back to mean per-class accuracy when not
    provided (IID / legacy mode).
    """
    node_ids = sorted(system.nodes.keys())
    test_vals, cn_local_vals, train_vals, self_vals = [], [], [], []
    bandit        = system._bandit
    bandit_ready  = (bandit is not None and getattr(system, "_bandit_bootstrapped", False))
    deploy_criteria = str(deploy_criteria).lower()
    assert deploy_criteria in ("val_query", "val_ema", "val_ucb", "dist_overlap", "bandit", "bandit_soft", "val_bandit", "bandit_class", "self"), \
        f"Unknown deploy_criteria: {deploy_criteria}"

    # Increment call counter so cn_scores cache knows when to refresh
    system._cn_call_counter = getattr(system, "_cn_call_counter", 0) + 1

    def _node_test_score(j: int) -> float:
        """Overall accuracy for node j on its own local test set."""
        return test_acc[j] if test_acc is not None else float(per_class_test[j].mean())

    def _node_train_score(i: int) -> float:
        return train_acc[i] if train_acc is not None else float(per_class_train[i].mean())

    with torch.no_grad():
        for i in node_ids:
            node_i   = system.nodes[i]
            hood     = [i] + list(node_i.neighbor_ids)
            w_i      = system._node_skew_weights[i]
            hood_set = set(hood)

            node_i_test_loader = system._node_test_loaders.get(i)

            # cn_local: oracle best neighbor evaluated on NODE I's test set.
            # Expensive in stage_2_tuning (full backbone pass per neighbor pair).
            # Cache for 3 eval blocks so the metric stays accurate but cost is
            # amortised — oracle ranking barely changes round-to-round.
            _cn_cache     = getattr(system, "_cn_scores_cache", {})
            _cn_age       = getattr(system, "_cn_scores_age", {})
            _cn_call_idx  = getattr(system, "_cn_call_counter", 0)
            _cn_refresh   = 3   # recompute every this many eval blocks
            if (i not in _cn_cache
                    or (_cn_call_idx - _cn_age.get(i, -999)) >= _cn_refresh):
                if node_i_test_loader is not None:
                    _fresh = {j: system.evaluate_j_on_i_loader(
                                    j, i, node_i_test_loader, system._test_idx_per_node)
                              for j in hood}
                else:
                    _fresh = {j: _node_test_score(j) for j in hood}
                _cn_cache[i] = _fresh
                _cn_age[i]   = _cn_call_idx
                system._cn_scores_cache = _cn_cache
                system._cn_scores_age   = _cn_age
            cn_scores = _cn_cache[i]
            j_oracle = max(hood, key=lambda j: cn_scores[j])
            cn_local_vals.append(cn_scores[j_oracle])
            # Store oracle info for the per-node print block and debug file
            if not hasattr(system, "_debug_cn_oracle"):
                system._debug_cn_oracle = {}
            system._debug_cn_oracle[i] = (j_oracle, cn_scores[j_oracle])

            # "self" mode: deploy node i's own model only (baselines and pretrain).
            # CN-local is already recorded above — only test/self/train use self here.
            if deploy_criteria == "self":
                test_vals.append(_node_test_score(i))
                self_vals.append(_node_test_score(i))
                train_vals.append(_node_train_score(i))
                continue

            use_bandit_deploy = (
                bandit_ready
                and node_i.neighbor_ids
                and deploy_criteria in ("bandit", "bandit_soft")
            )

            if use_bandit_deploy:
                if deploy_criteria == "bandit_soft":
                    dw = bandit.deploy_weights(i, self_weight=0.10)
                    dw = {j: v for j, v in dw.items() if j in hood_set}
                    if dw:
                        tw = sum(dw.values())
                        pc = sum((dw[j] / tw) * per_class_test[j] for j in dw)
                        test_vals.append(float(pc.mean()))
                    else:
                        test_vals.append(_node_test_score(i))
                else:  # bandit
                    # Use the dedicated deployment score tracker — IW-confidence
                    # of each neighbor on node i's unlabeled buffer, updated at
                    # bootstrap/rebootstrap. No exploration bonus at deploy time.
                    deploy_scores = getattr(system, "_deploy_scores", {}).get(i, {})
                    if deploy_scores:
                        j_star = max(
                            (j for j in hood if j in deploy_scores),
                            key=lambda j: deploy_scores.get(j, 0.0),
                            default=i,
                        )
                    else:
                        j_star = bandit.best_arm(i) if bandit is not None else i
                    if j_star is None or j_star not in hood_set:
                        j_star = i
                    # Gate: if node i's bootstrapped IW-confidence exceeds
                    # the scalar Q estimate for the chosen neighbor, deploy self.
                    # This prevents cold contextual arms from overriding a
                    # strong node's own model early in training.
                    self_iw_scores = getattr(system, "_self_iw_scores", {})
                    self_score = self_iw_scores.get(i, None)
                    if self_score is not None and j_star != i and hasattr(bandit, "Q"):
                        best_nbr_q = bandit.Q[i].get(j_star, 0.0)
                        if self_score > best_nbr_q:
                            j_star = i
                    if node_i_test_loader is not None:
                        test_vals.append(system.evaluate_j_on_i_loader(j_star, i, node_i_test_loader, system._test_idx_per_node))
                    else:
                        test_vals.append(_node_test_score(j_star))
            else:
                if deploy_criteria == "bandit_class" and bandit_ready and isinstance(bandit, ContextualUCBNeighborBandit) and node_i.neighbor_ids:
                    # Class-conditioned deployment using per-class bandit arm means.
                    # No val data required — uses pc_mu[i][j][c]: running mean of
                    # P_j(c | x) for unlabeled examples predicted as class c.
                    # Deploy score for neighbor j = sum_c w_i[c] * pc_mu[i][j][c].
                    # Self score = IW-confidence from last bootstrap/rebootstrap.
                    # NOTE: do NOT append to test_vals here — fall through to the
                    # common test_vals.append block below (like val_ema does).
                    _w_i_bc   = system._node_skew_weights[i]
                    _self_q   = getattr(system, "_self_iw_scores", {}).get(i, 0.0)
                    bc_scores = {i: _self_q}
                    for j in node_i.neighbor_ids:
                        if j in bandit.pc_mu.get(i, {}):
                            bc_scores[j] = float(np.dot(bandit.pc_mu[i][j], _w_i_bc))
                        else:
                            bc_scores[j] = bandit.Q[i].get(j, 0.0)
                    j_deploy = max(hood, key=lambda j: bc_scores.get(j, 0.0))
                    _margin  = float(getattr(system, "deploy_self_margin", 0.05))
                    if j_deploy != i and bc_scores.get(j_deploy, 0.0) < bc_scores.get(i, 0.0) + _margin:
                        j_deploy = i
                    val_scores = bc_scores  # used by debug_val_query block below
                elif deploy_criteria in ("val_ema", "val_bandit"):
                    ema_cache = getattr(system, "_val_ema_scores", {})
                    if ema_cache and i in ema_cache:
                        val_scores = {j: ema_cache[i].get(j, 0.0) for j in hood}
                    else:
                        val_scores = {
                            j: system.evaluate_j_on_i_loader(j, i, node_i.val_loader, system._val_idx_per_node)
                            for j in hood
                        }
                    if deploy_criteria == "val_bandit":
                        _blend_alpha = float(getattr(system, "deploy_blend_alpha", 0.7))
                        _deploy_sc   = getattr(system, "_deploy_scores", {}).get(i, {})
                        if _deploy_sc:
                            _max_val  = max(val_scores.values()) or 1.0
                            _max_conf = max(_deploy_sc.values()) or 1.0
                            combined = {
                                j: _blend_alpha * (val_scores.get(j, 0.0) / _max_val)
                                   + (1.0 - _blend_alpha) * (_deploy_sc.get(j, 0.0) / _max_conf)
                                for j in hood
                            }
                        else:
                            combined = val_scores
                        _scores_for_margin = combined
                        j_deploy = max(hood, key=lambda j: combined[j])
                    else:
                        _scores_for_margin = val_scores
                        j_deploy = max(hood, key=lambda j: val_scores[j])
                    # Self-protection margin: neighbor must beat self by at least
                    # deploy_self_margin to avoid small-val-set noise causing
                    # strong nodes to deploy to marginally higher-scoring but
                    # weaker neighbors.
                    _margin = float(getattr(system, "deploy_self_margin", 0.05))
                    if j_deploy != i and _margin > 0.0:
                        if _scores_for_margin.get(j_deploy, 0.0) < _scores_for_margin.get(i, 0.0) + _margin:
                            j_deploy = i
                else:
                    val_scores = {
                        j: system.evaluate_j_on_i_loader(j, i, node_i.val_loader, system._val_idx_per_node)
                        for j in hood
                    }
                    j_deploy = max(hood, key=lambda j: val_scores[j])
                if node_i_test_loader is not None:
                    deploy_test = system.evaluate_j_on_i_loader(j_deploy, i, node_i_test_loader, system._test_idx_per_node)
                    test_vals.append(deploy_test)
                else:
                    deploy_test = _node_test_score(j_deploy)
                    test_vals.append(deploy_test)

                # Debug: for first 3 nodes, print val vs test scores for all
                # neighbors to expose val/test ranking mismatches.
                if debug_val_query and i < 3 and node_i_test_loader is not None:
                    print(f"  [VAL_DBG] node={i}  oracle={j_oracle}(test={cn_scores[j_oracle]:.3f})  "
                          f"deploy={j_deploy}(val={val_scores[j_deploy]:.3f} test={deploy_test:.3f})  "
                          f"match={'YES' if j_deploy==j_oracle else 'NO'}", flush=True)
                    # Print all neighbor scores sorted by test (oracle order) vs val order
                    by_test = sorted(hood, key=lambda j: cn_scores[j], reverse=True)[:5]
                    by_val  = sorted(hood, key=lambda j: val_scores[j], reverse=True)[:5]
                    print(f"    top-5 by test: " +
                          "  ".join(f"n{j}(t={cn_scores[j]:.3f},v={val_scores[j]:.3f})" for j in by_test),
                          flush=True)
                    print(f"    top-5 by val:  " +
                          "  ".join(f"n{j}(v={val_scores[j]:.3f},t={cn_scores[j]:.3f})" for j in by_val),
                          flush=True)

            self_vals.append(_node_test_score(i))
            train_vals.append(_node_train_score(i))

    n = max(1, len(node_ids))
    return (float(sum(test_vals) / n), float(sum(cn_local_vals) / n),
            float(sum(train_vals) / n), float(sum(self_vals) / n))


def print_distribution_report(
    system: DecentralizedPseudoLabelSystem,
    num_nodes_to_check: int = 5,
    no_flush: bool = False,
) -> None:
    node_ids    = sorted(system.nodes.keys())[:int(num_nodes_to_check)]
    num_classes = system.num_classes
    _pprint(
        "[DISTRIB] Importance weights (= train and test distribution, identical by construction)",
        no_flush=no_flush,
    )
    _pprint(
        f"  {'node':>4}  {'favored':>7}  " + "  ".join(f" c{c}" for c in range(num_classes)),
        no_flush=no_flush,
    )
    for i in node_ids:
        favored = system.favored_class_map.get(i, -1)
        w       = system._node_skew_weights[i]
        parts   = [f"{'*' if c == favored else ' '}{w[c]:.3f}" for c in range(num_classes)]
        _pprint(f"  {i:>4}  {favored:>7}  " + "  ".join(parts), no_flush=no_flush)


# ----------------------------
# Run debug file writer
# ----------------------------

def write_run_debug_file(
    run_id: str,
    argv: list,
    system: "DecentralizedPseudoLabelSystem",
    seed: int,
    p: float,
    test_hist: list,
    cn_local_hist: list,
    self_hist: list,
    deploy_criteria: str,
    out_dir: str,
    final_sup_losses: dict,    # {node_i: float}
    final_pseudo_losses: dict, # {node_i: float}
    baseline: str = "none",
) -> None:
    """Write {out_dir}/{run_id}_seed{seed}_p{p10}.txt with full diagnostics.

    Sections
    --------
    1. Full reproducing command  (python3 experiment.py <all args>)
    2. Run-level summary
    3. Per-node table:
         node | fav | deg | self | cn_node | cn_acc | deploy_node | deploy_acc |
         dep_val | sup_loss | kl_loss | class_dist
         bandit top-5 arms on next line
    4. Deploy mismatches (deployed to a worse node than oracle by >2 pp)
    5. Missed gains (self < best neighbor by >5 pp)
    """
    import os
    import torch
    import numpy as np

    p10   = int(round(float(p) * 10))
    fname = f"{run_id}_seed{seed}_p{p10}.txt"
    fpath = os.path.join(out_dir, fname)

    node_ids    = sorted(system.nodes.keys())
    num_classes = system.num_classes

    # ── Reconstruct the full command ─────────────────────────────────────────
    # argv[0] is the script path; ensure it looks like experiment.py
    script = argv[0] if (argv and not argv[0].startswith("-")) else "experiment.py"
    parts   = [f"python3 {script}"] + [str(a) for a in argv[1:]]
    cmd_str = " \\\n  ".join(parts)

    # ── Per-node self accuracy (own model on own local test set) ─────────────
    @torch.no_grad()
    def _acc(j: int, i: int) -> float:
        loader = system._node_test_loaders.get(i)
        if loader is None:
            return 0.0
        if j == i:
            return float(system.nodes[i].evaluate_accuracy(loader))
        return float(system.evaluate_j_on_i_loader(
            j, i, loader, system._test_idx_per_node))

    self_accs = {i: _acc(i, i) for i in node_ids}

    # ── Oracle (CN-local) and deploy target per node ─────────────────────────
    deploy_self_margin = float(getattr(system, "deploy_self_margin", 0.05))
    ema_cache          = getattr(system, "_val_ema_scores", {})

    per_node_info: dict = {}
    for i in node_ids:
        node_i = system.nodes[i]
        hood   = [i] + list(node_i.neighbor_ids)

        # Oracle: best neighbor by test accuracy on node i's local test set
        cn_accs  = {j: _acc(j, i) for j in hood}
        j_oracle = max(hood, key=lambda j: cn_accs[j])

        # Val scores for deployment
        if deploy_criteria in ("val_ema", "val_bandit") and ema_cache.get(i):
            val_sc = {j: ema_cache[i].get(j, 0.0) for j in hood}
        else:
            with torch.no_grad():
                val_sc = {
                    j: float(system.evaluate_j_on_i_loader(
                        j, i, node_i.val_loader, system._val_idx_per_node))
                    for j in hood
                }

        j_deploy = max(hood, key=lambda j: val_sc[j])
        if j_deploy != i and deploy_self_margin > 0.0:
            if val_sc.get(j_deploy, 0.0) < val_sc.get(i, 0.0) + deploy_self_margin:
                j_deploy = i

        per_node_info[i] = dict(
            fav      = system.favored_class_map.get(i, -1),
            skew_w   = system._node_skew_weights.get(i, np.zeros(num_classes)),
            self_acc = self_accs[i],
            j_oracle = j_oracle,
            cn_acc   = cn_accs[j_oracle],
            j_deploy = j_deploy,
            dep_acc  = _acc(j_deploy, i),
            dep_val  = val_sc.get(j_deploy, 0.0),
            neighbors= list(node_i.neighbor_ids),
            degree   = len(node_i.neighbor_ids),
            sup_loss = final_sup_losses.get(i, float("nan")),
            kl_loss  = final_pseudo_losses.get(i, float("nan")),
        )

    # ── Bandit pull counts ───────────────────────────────────────────────────
    bandit = getattr(system, "_bandit", None)

    def _bandit_arms(i: int, top_k: int = 5) -> list:
        """Return [(neighbor_j, pulls, mean_reward)] sorted by pulls desc."""
        if bandit is None:
            return []
        N_i = getattr(bandit, "N", {}).get(i, {})
        Q_d = getattr(bandit, "Q", None) or getattr(bandit, "mu", None) or {}
        Q_i = Q_d.get(i, {}) if isinstance(Q_d, dict) else {}
        arms = [
            (j, int(N_i.get(j, 0)), float(Q_i.get(j, 0.0)))
            for j in per_node_info[i]["neighbors"]
        ]
        arms.sort(key=lambda x: x[1], reverse=True)
        return arms[:top_k]

    # ── Build output ─────────────────────────────────────────────────────────
    W     = 100
    lines = []

    def _sec(title: str) -> None:
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    # Sections 1 (command) and 2 (live log) were already written at run start.
    # Append: final run summary, per-node final state, and analysis sections.
    _sec(f"FINAL SUMMARY  —  {run_id}  seed={seed}  p={p:.3f}")
    lines.append(f"  rounds_ran   : {len(test_hist)}")
    lines.append(f"  deploy_crit  : {deploy_criteria}")
    if test_hist:
        lines.append(f"  final test   : {test_hist[-1]:.4f}  (best: {max(test_hist):.4f})")
    if cn_local_hist:
        lines.append(f"  final cn_loc : {cn_local_hist[-1]:.4f}  (best: {max(cn_local_hist):.4f})")
    if self_hist:
        lines.append(f"  final self   : {self_hist[-1]:.4f}  (best: {max(self_hist):.4f})")

    # Final per-node state (with class dist and bandit arms)
    _sec("FINAL PER-NODE STATE")
    lines.append(
        f"\n  {'node':>4} {'fav':>3} {'deg':>3} | "
        f"{'self':>6} {'cn_nd':>6} {'cn_acc':>6} "
        f"{'dep_nd':>6} {'dep_acc':>7} {'dep_val':>7} | "
        f"{'sup_L':>6} {'kl_L':>6} | class distribution"
    )
    lines.append("  " + "-" * (W - 2))

    for i in node_ids:
        d        = per_node_info[i]
        fav      = d["fav"]
        j_oracle = d["j_oracle"]
        j_deploy = d["j_deploy"]
        cn_str   = "self"          if j_oracle == i else f"n{j_oracle:02d}"
        dep_str  = "self"          if j_deploy == i else f"n{j_deploy:02d}"

        # Class distribution — full for ≤20 classes, top-5 for cifar100
        w = d["skew_w"]
        if num_classes <= 20:
            dist_str = "  ".join(
                f"{'*' if c == fav else ' '}{w[c]:.3f}"
                for c in range(num_classes)
            )
        else:
            top5     = sorted(range(num_classes), key=lambda c: w[c], reverse=True)[:5]
            dist_str = "  ".join(
                f"c{c}({'*' if c == fav else ''}{w[c]:.3f})" for c in top5
            )

        sup_s = f"{d['sup_loss']:.4f}"  if not np.isnan(d['sup_loss'])  else "   n/a"
        kl_s  = f"{d['kl_loss']:.4f}"  if not np.isnan(d['kl_loss'])   else "   n/a"

        lines.append(
            f"  n{i:02d}  c{fav:02d}  {d['degree']:>2}d | "
            f"{d['self_acc']:6.3f} {cn_str:>6} {d['cn_acc']:6.3f} "
            f"{dep_str:>6} {d['dep_acc']:7.3f} {d['dep_val']:7.3f} | "
            f"{sup_s:>6} {kl_s:>6} | {dist_str}"
        )

        # Bandit arm pulls — indented below each node row
        arms = _bandit_arms(i)
        if arms:
            arm_str = "  ".join(
                f"n{j:02d}(pulls={n},mu={mu:.3f})" for j, n, mu in arms
            )
            lines.append(f"           bandit: {arm_str}")

    # 4. Deploy mismatches — only meaningful when our method is active
    _is_collaborative = (baseline == "none" and abs(float(p)) > 1e-9)
    if _is_collaborative:
        _sec("4. DEPLOY MISMATCHES  (deployed to worse node than oracle by >2 pp)")
        mismatches = [
            i for i in node_ids
            if per_node_info[i]["j_deploy"] != per_node_info[i]["j_oracle"]
            and per_node_info[i]["dep_acc"]  < per_node_info[i]["cn_acc"] - 0.02
        ]
        if mismatches:
            lines.append(
                f"  {'node':>4}  {'oracle':>7} {'orac_acc':>8}  "
                f"{'deploy':>7} {'dep_acc':>8}  {'gap':>6}"
            )
            for i in mismatches:
                d   = per_node_info[i]
                gap = d["cn_acc"] - d["dep_acc"]
                lines.append(
                    f"  n{i:02d}  "
                    f"n{d['j_oracle']:02d}     {d['cn_acc']:8.3f}  "
                    f"{'self' if d['j_deploy']==i else 'n'+str(d['j_deploy']):>7} "
                    f"{d['dep_acc']:8.3f}  {gap:6.3f}"
                )
        else:
            lines.append("  (none — all nodes deployed optimally or near-optimally)")

        # 5. Missed gains
        _sec("5. MISSED GAINS  (self < best_neighbor by >5 pp)")
        found_missed = False
        for i in node_ids:
            d   = per_node_info[i]
            gap = d["cn_acc"] - d["self_acc"]
            if gap > 0.05:
                lines.append(
                    f"  n{i:02d}  self={d['self_acc']:.3f}  "
                    f"oracle=n{d['j_oracle']:02d}({d['cn_acc']:.3f})  "
                    f"gap={gap:.3f}  fav=c{d['fav']:02d}  deg={d['degree']}"
                )
                found_missed = True
        if not found_missed:
            lines.append("  (none — all nodes are close to their oracle ceiling)")

    lines.append("")
    lines.append(f"[END OF REPORT]  {run_id}")

    os.makedirs(out_dir, exist_ok=True)
    # Append sections 4-5 to the live log already written during training
    with open(fpath, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[DEBUG FILE] Final sections appended → {fpath}", flush=True)


# ----------------------------
# Pretrain checkpoint helpers
# ----------------------------

def _ckpt_path_for_seed(base_path: str, seed: int) -> str:
    """Return a per-seed checkpoint path.

    If *base_path* already ends in ``.pt`` or ``.pth`` the seed is injected
    before the extension, e.g. ``/tmp/pretrain.pt`` → ``/tmp/pretrain_seed0.pt``.
    Otherwise ``_seed{seed}.pt`` is appended directly.
    """
    base_path = str(base_path).strip()
    if not base_path:
        return ""
    root, ext = os.path.splitext(base_path)
    if ext.lower() in (".pt", ".pth"):
        return f"{root}_seed{seed}{ext}"
    return f"{base_path}_seed{seed}.pt"


def _save_pretrain_checkpoint(
    path: str,
    system: "DecentralizedPseudoLabelSystem",
    pre_rounds: int,
    pre_best: float,
    last_test: float,
    last_cn_local: float,
    last_train: float,
    test_hist: List[float],
    cn_local_hist: List[float],
    train_hist: List[float],
    eval_mask_hist: List[bool],
) -> None:
    """Persist all per-node model + optimiser states after stage-1 pretraining.

    The checkpoint stores everything needed to skip stage 1 on future runs:
      - model and optimiser state_dicts for every node
      - pre_rounds / pre_best so stage-2 patience is initialised correctly
      - the stage-1 metric history so learning curves are complete
      - a metadata dict (seed, num_nodes, arch) for sanity-checking on load
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    node_states = {}
    for i, node in system.nodes.items():
        node_states[i] = {
            "model":     node.model.state_dict(),
            "optimizer": node.optimizer.state_dict(),
            "pseudo_allowed": node.pseudo_allowed,
        }
    ckpt = {
        "version":       1,
        "seed":          system.seed,
        "num_nodes":     system.num_nodes,
        "arch":          system.arch,
        "node_arch_map": system._node_arch_map,
        "pre_rounds":    int(pre_rounds),
        "pre_best":      float(pre_best),
        "last_test":     float(last_test),
        "last_cn_local": float(last_cn_local),
        "last_train":    float(last_train),
        "test_hist":     list(test_hist),
        "cn_local_hist": list(cn_local_hist),
        "train_hist":    list(train_hist),
        "eval_mask_hist": list(eval_mask_hist),
        "node_states":   node_states,
    }
    torch.save(ckpt, path)
    print(f"[CKPT] Saved pretrain checkpoint → {path}", flush=True)


def _load_pretrain_checkpoint(
    path: str,
    system: "DecentralizedPseudoLabelSystem",
    no_flush: bool = False,
) -> Optional[dict]:
    """Load node weights + stage-1 history from a previously saved checkpoint.

    Returns the checkpoint dict on success, or None if the file is missing or
    the metadata doesn't match (in which case a warning is printed and stage 1
    will run normally).
    """
    if not path or not os.path.isfile(path):
        _pprint(f"[CKPT] No checkpoint found at {path!r} — running stage 1.", no_flush=no_flush)
        return None

    ckpt = torch.load(path, map_location="cpu")

    # Sanity checks
    mismatches = []
    if ckpt.get("seed") != system.seed:
        mismatches.append(f"seed: ckpt={ckpt.get('seed')} vs current={system.seed}")
    if ckpt.get("num_nodes") != system.num_nodes:
        mismatches.append(f"num_nodes: ckpt={ckpt.get('num_nodes')} vs current={system.num_nodes}")
    if mismatches:
        _pprint(
            f"[CKPT] WARNING: Checkpoint metadata mismatch — running stage 1 instead.\n"
            + "\n".join(f"  {m}" for m in mismatches),
            no_flush=no_flush,
        )
        return None

    # Restore per-node states
    node_states = ckpt["node_states"]
    for i, node in system.nodes.items():
        if i not in node_states:
            _pprint(f"[CKPT] WARNING: node {i} missing from checkpoint — skipping.", no_flush=no_flush)
            continue
        ns = node_states[i]
        node.model.load_state_dict(ns["model"])
        node.optimizer.load_state_dict(ns["optimizer"])
        node.pseudo_allowed = bool(ns.get("pseudo_allowed", True))

    _pprint(
        f"[CKPT] Loaded pretrain checkpoint from {path!r}  "
        f"(pre_rounds={ckpt['pre_rounds']}  pre_best={ckpt['pre_best']:.4f})",
        no_flush=no_flush,
    )
    return ckpt


# ----------------------------
# Experiment runner
# ----------------------------
def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _p10(p: float) -> int:
    return int(round(float(p) * 10))


@dataclass
class RunSummary:
    exp: str; seed: int; p: float; p10: int
    budget_steps: int; budget_examples: int; max_rounds: int; rounds_ran: int
    unlabeled_fraction: float; unlabeled_per_node: int; per_node_sample_size: int; kl_weight: float
    test_final: float; cn_local_final: float; train_final: float; self_final: float
    pseudo_batches_per_round_mean: float; pseudo_examples_per_round_mean: float
    is_oracle: bool = False


@dataclass
class CurveRecord:
    exp: str; seed: int; p: float; p10: int
    budget_steps: int; budget_examples: int
    rounds: np.ndarray
    test: np.ndarray; cn_local: np.ndarray; train: np.ndarray
    eval_mask: np.ndarray; pretrain_rounds_ran: int; is_oracle: bool = False


def run_one_setting(
    exp: str,
    seed: int,
    p: float,
    budget_steps: int,
    budget_examples: int,
    out_dir: str,
    num_nodes: int,
    batch_size: int,
    val_fraction: float,
    test_fraction: float,
    unlabeled_fraction: float,
    per_node_sample_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    dropout_p: float,
    dropout_feat: float,
    kl_weight: float,
    pseudo_epochs: int,
    max_rounds: int,
    patience: int,
    tol: float,
    num_workers_train: int,
    num_workers_eval: int,
    pseudo_cap_examples_per_node: int,
    pseudo_conf_threshold: float,
    pseudo_entropy_threshold: float,
    pseudo_warmup_rounds: int,
    kl_ramp_rounds: int,
    pseudo_disable_patience: int,
    pseudo_disable_delta: float,
    conf_weight_tau0: float,
    training_data_mode: str,
    skew_factor: float,
    skew_strategy: str,
    skew_seed: int,
    skew_min_other_frac: float,
    min_classes_per_node: int,
    pretrain_min_rounds: int,
    pretrain_patience: int,
    pretrain_tol: float,
    pretrain_max_rounds: int,
    unlabeled_per_node: int = 0,
    unlabeled_pool_skew: str = "iid",
    eval_freq: int = 10,
    arch: str = "mobilenet_v2",
    oracle_supervised: bool = False,
    connection_model: str = "uniform",
    plot_graph: bool = False,
    print_every: int = 1,
    no_flush: bool = False,
    debug_time: bool = False,
    sup_steps_per_node: int = 2,
    sup_steps_total: int = 0,
    cache_features: bool = False,
    finetune_backbone: bool = False,
    stage_2_tuning: bool = False,
    baseline_non_linearity: bool = False,
    cache_batch_size: int = 512,
    amp_features: bool = True,
    feature_cache_path: str = "",
    size_skew_mode: str = "none",
    size_seed: int = 0,
    min_per_node_size: int = 20,
    size_total_budget: int = 0,
    size_dirichlet_alpha: float = 0.3,
    baseline: str = "none",
    baseline_our_deployment: bool = False,
    baseline_merge_val: bool = False,
    baseline_rho: float = 0.05,
    fedmd_public_size: int = 2000,
    neighbor_weighting: str = "none",
    neighbor_weight_update_freq: int = 10,
    ucb_c: float = 1.0,
    bandit_type: str = "ucb1",
    grad_align_gamma: float = 0.99,
    grad_align_tau: float = 0.0,
    grad_align_iw_temp: float = 1.0,
    entropy_gate_tau: float = 0.5,
    entropy_ucb_align_tau: float = 0.5,
    agreement_beta: float = 0.0,
    deploy_criteria: str = "val_query",
    deploy_ema_freq: int = 5,
    deploy_ema_alpha: float = 0.1,
    pseudo_teacher_mode: str = "avg",
    verify_distribution: bool = False,
    similarity_temp: float = 1.0,
    similarity_mode: str = "softmax",
    ba_m: int = 0,
    random_models: bool = False,
    random_models_mnv2_frac: float = 0.8,
    random_models_hub_efnet: bool = True,
    top_k_teachers: int = 1,
    mutual_distillation: bool = False,
    pseudo_label_temp: float = 1.0,
    pseudo_examples_per_round: int = 128,
    teacher_ema: bool = False,
    teacher_ema_steps: int = 4,
    mobilenet_cache_path: str = "",
    efficientnet_cache_path: str = "",
    debug_val_query: bool = False,
    pretrain_checkpoint_save: str = "",
    pretrain_checkpoint_load: str = "",
    stage2_checkpoint_save: str = "",
    rebootstrap_freq: int = 10,
    viz_graph: bool = False,
    bandit_context_labeled: bool = False,
    deploy_blend_alpha: float = 0.7,
    deploy_self_margin: float = 0.05,
    run_id: str = "",
    degree_prior: float = 0.0,
    topo_rebootstrap: bool = False,
    dataset: str = "cifar10",
    stage2_sup_alpha: float = 1.0,
    geo_cache: str = "",
) -> Tuple[RunSummary, CurveRecord]:
    _set_all_seeds(seed)
    eff_pseudo_epochs = 0 if (oracle_supervised or baseline != "none") else int(pseudo_epochs)

    # ── Baseline data protocol ─────────────────────────────────────────
    # All methods use the same val/test split so stage 1 pretrained weights
    # are identical. For baselines (which don't use val for selection/bandit),
    # we merge the val set INTO the training loader so they effectively train
    # on all labeled data — without changing the split itself.
    # Use --baseline_our_deployment to skip this merge (ablation: baselines
    # train on only the train split, same as our method's training set).

    system = DecentralizedPseudoLabelSystem(
        num_nodes=num_nodes, batch_size=batch_size, val_fraction=float(val_fraction),
        test_fraction=test_fraction, unlabeled_fraction=unlabeled_fraction, unlabeled_per_node=int(unlabeled_per_node), unlabeled_pool_skew=str(unlabeled_pool_skew), per_node_sample_size=per_node_sample_size,
        lr=lr, momentum=momentum, weight_decay=weight_decay,
        dropout_p=dropout_p, dropout_feat=dropout_feat, kl_weight=kl_weight,
        seed=seed, hub=0, num_workers_train=num_workers_train, num_workers_eval=num_workers_eval,
        network_connection_p=p, connection_model=connection_model,
        pseudo_conf_threshold=pseudo_conf_threshold,
        pseudo_entropy_threshold=pseudo_entropy_threshold,
        pseudo_warmup_rounds=pseudo_warmup_rounds, kl_ramp_rounds=kl_ramp_rounds,
        pseudo_disable_patience=pseudo_disable_patience,
        pseudo_disable_delta=pseudo_disable_delta,
        training_data_mode=training_data_mode, skew_factor=skew_factor,
        skew_strategy=skew_strategy, skew_seed=skew_seed,
        skew_min_other_frac=skew_min_other_frac,
        min_classes_per_node=min_classes_per_node,
        arch=arch, oracle_supervised_union=bool(oracle_supervised),
        debug_time=bool(debug_time),
        cache_features=bool(cache_features), cache_batch_size=int(cache_batch_size),
        baseline_non_linearity=bool(baseline_non_linearity),
        amp_features=bool(amp_features), feature_cache_path=str(feature_cache_path),
        size_skew_mode=size_skew_mode, size_seed=size_seed,
        min_per_node_size=min_per_node_size, size_total_budget=size_total_budget,
        size_dirichlet_alpha=size_dirichlet_alpha,
        neighbor_weighting="none" if baseline != "none" else neighbor_weighting,
        neighbor_weight_update_freq=neighbor_weight_update_freq,
        ucb_c=float(ucb_c), bandit_type=str(bandit_type),
        grad_align_gamma=float(grad_align_gamma),
        grad_align_tau=float(grad_align_tau),
        grad_align_iw_temp=float(grad_align_iw_temp),
        entropy_gate_tau=float(entropy_gate_tau),
        entropy_ucb_align_tau=float(entropy_ucb_align_tau),
        pseudo_teacher_mode="avg" if baseline != "none" else str(pseudo_teacher_mode),
        similarity_temp=float(similarity_temp),
        similarity_mode=str(similarity_mode),
        ba_m=int(ba_m),
        random_models=bool(random_models),
        random_models_mnv2_frac=float(random_models_mnv2_frac),
        random_models_hub_efnet=bool(random_models_hub_efnet),
        top_k_teachers=int(top_k_teachers),
        mutual_distillation=bool(mutual_distillation),
        pseudo_label_temp=float(pseudo_label_temp),
        pseudo_examples_per_round=int(pseudo_examples_per_round),
        teacher_ema=bool(teacher_ema),
        teacher_ema_steps=int(teacher_ema_steps),
        mobilenet_cache_path=str(mobilenet_cache_path),
        efficientnet_cache_path=str(efficientnet_cache_path),
        viz_graph=bool(viz_graph),
        _out_dir=str(out_dir),
        baseline_merge_val=True,  # Stage 1 parity: ALL methods train on full labeled data
        dataset=str(dataset),
        geo_cache=str(geo_cache),
    )
    # Share deploy_ema_alpha as the EMA decay for teacher_ema so users
    # only need one alpha flag for both val_ema and teacher_ema.
    system._teacher_ema_alpha = float(deploy_ema_alpha)
    system.bandit_context_labeled = bool(bandit_context_labeled)
    system.deploy_blend_alpha     = float(deploy_blend_alpha)
    system.deploy_self_margin     = float(deploy_self_margin)
    system.deploy_criteria_name   = str(deploy_criteria).lower()
    system.val_fraction           = float(val_fraction)
    system.degree_prior           = float(degree_prior)

    baseline_obj = None
    if baseline != "none":
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from baselines import make_baseline as _make_baseline
        baseline_obj = _make_baseline(
            baseline, system,
            baseline_rho=float(baseline_rho),
            fedmd_public_size=int(fedmd_public_size),
        )
        if baseline_obj is not None:
            _pprint(f"[BASELINE] {baseline} initialized", no_flush=no_flush)

    if verify_distribution:
        print_distribution_report(system, num_nodes_to_check=5, no_flush=no_flush)

    node_ids = sorted(system.nodes.keys())
    hub_id   = 0

    test_hist: List[float]      = []
    cn_local_hist: List[float]  = []
    train_hist: List[float]     = []
    self_hist: List[float]      = []
    eval_mask_hist: List[bool]  = []
    last_test = 0.0; last_cn_local = 0.0; last_train = 0.0; last_self = 0.0
    total_pseudo_batches  = 0
    total_pseudo_examples = 0

    tag = (
        f"steps{budget_steps}"       if exp == "steps"    else
        f"ex{budget_examples}"       if exp == "examples" else
        f"cap{pseudo_cap_examples_per_node}"
    )
    if oracle_supervised: tag = f"{tag}_oracle"
    if cache_features:    tag = f"{tag}_featcache"
    if random_models:     tag = f"{tag}_randmodels"
    if connection_model != "uniform":
        tag = f"{tag}_{connection_model}"
        if connection_model == "data_similarity":
            tag = f"{tag}_t{similarity_temp:.2f}"

    # ── Debug file: open now so we can append every round update ─────────────
    _dbg_p10  = int(round(float(p) * 10))
    _dbg_path = os.path.join(out_dir, f"{run_id}_seed{seed}_p{_dbg_p10}.txt")
    _is_collaborative = (baseline == "none" and abs(float(p)) > 1e-9)
    _dbg_script = sys.argv[0] if (sys.argv and not sys.argv[0].startswith("-")) else "experiment.py"
    _dbg_cmd    = " \\\n  ".join([f"python3 {_dbg_script}"] + [str(a) for a in sys.argv[1:]])
    with open(_dbg_path, "w", encoding="utf-8") as _dbgf:
        _dbgf.write(
            f"{'='*100}\n"
            f"  RUN DEBUG LOG  —  {run_id}  seed={seed}  p={p:.3f}  baseline={baseline}\n"
            f"{'='*100}\n\n"
            f"## COMMAND\n{_dbg_cmd}\n\n"
            f"## LIVE ROUND LOG\n"
            f"  {'round':>6}  {'stage':>5}  {'test':>7}  {'cn_loc':>7}  {'self':>7}  {'train':>7}\n"
            f"  {'-'*60}\n"
        )

    _is_independent = (baseline == "none" and abs(float(p)) < 1e-9)
    _pprint(
        f"[START] exp={exp} seed={seed} p={p:.3f} arch={arch} "
        f"conn={connection_model}"
        + (f" temp={similarity_temp:.3f}" if connection_model == "data_similarity" else "")
        + f" random_models={random_models} baseline={'independent_learning' if _is_independent else baseline} "
        f"{'(ORACLE) ' if oracle_supervised else ''}"
        f"train_mode={training_data_mode} skew={skew_factor} "
        f"neighbor_weighting={neighbor_weighting} "
        f"max_rounds={max_rounds} pretrain_max={pretrain_max_rounds} "
        f"device={system.device}",
        no_flush=no_flush,
    )
    _pprint(
        f"[HPARAMS] "
        f"bandit_type={system.bandit_type} "
        f"ucb_c={system.ucb_c:.3f} "
        f"pseudo_conf_threshold={system.pseudo_conf_threshold:.3f} "
        f"pseudo_entropy_threshold={system.pseudo_entropy_threshold:.3f} "
        f"teacher_mode={system.pseudo_teacher_mode} "
        f"kl_weight={system.kl_weight:.3f} "
        f"entropy_gate_tau={system.entropy_gate_tau:.3f} "
        f"entropy_ucb_align_tau={system.entropy_ucb_align_tau:.3f} "
        f"grad_align_tau={system.grad_align_tau:.3f} "
        f"grad_align_gamma={system.grad_align_gamma:.3f} "
        f"warmup={system.pseudo_warmup_rounds} "
        f"kl_ramp={system.kl_ramp_rounds} "
        f"unlabeled_frac={system.unlabeled_fraction:.3f} "
        f"skew_strategy={system.skew_strategy} "
        f"skew_min_other_frac={system.skew_min_other_frac:.3f} "
        f"deploy={deploy_criteria}",
        no_flush=no_flush,
    )

    pseudo_mode          = exp
    eff_budget_examples  = int(budget_examples)
    if exp == "examples" and pseudo_cap_examples_per_node > 0:
        eff_budget_examples = min(eff_budget_examples, int(pseudo_cap_examples_per_node))

    def _should_eval(r: int) -> bool:
        ef = int(eval_freq)
        return ef <= 1 or r == 1 or r % ef == 0

    t_stage1 = 0.0; t_stage2 = 0.0; t_eval_tt = 0.0; t_eval_val = 0.0

    # ============================
    # Stage 1: supervised pretrain
    # ============================
    pre_best      = 0.0; pre_no_improve = 0; pre_rounds = 0
    pretrain_cap  = (
        int(max_rounds) if int(pretrain_max_rounds) <= 0
        else min(int(pretrain_max_rounds), int(max_rounds))
    )

    # ------------------------------------------------------------------
    # Checkpoint load: skip stage 1 entirely when a valid checkpoint exists.
    # Pass the exact path to the checkpoint file.
    # ------------------------------------------------------------------
    _ckpt_loaded = False
    if pretrain_checkpoint_load:
        _ckpt_file = pretrain_checkpoint_load
        _ckpt = _load_pretrain_checkpoint(_ckpt_file, system, no_flush=no_flush)
        if _ckpt is not None:
            pre_rounds       = int(_ckpt["pre_rounds"])
            pre_best         = float(_ckpt["pre_best"])
            last_test        = float(_ckpt.get("last_test",     0.0))
            last_cn_local    = float(_ckpt.get("last_cn_local", 0.0))
            last_train       = float(_ckpt.get("last_train",    0.0))
            test_hist        = list(_ckpt.get("test_hist",      []))
            cn_local_hist    = list(_ckpt.get("cn_local_hist",  []))
            train_hist       = list(_ckpt.get("train_hist",     []))
            eval_mask_hist   = list(_ckpt.get("eval_mask_hist", []))
            _ckpt_loaded     = True
            _pprint(
                f"[CKPT] Skipping stage 1 — restored {pre_rounds} pretrain rounds "
                f"(test={last_test:.4f}  train={last_train:.4f})",
                no_flush=no_flush,
            )

    if not _ckpt_loaded:
        _pprint(
            f"[STAGE 1] supervised pretrain: "
            f"min={pretrain_min_rounds} patience={pretrain_patience} cap={pretrain_cap}",
            no_flush=no_flush,
        )

        # Print round-0 metrics before any training.
        _ta0       = system.evaluate_all_nodes_on_test()
        _tra0      = {i: node.evaluate_accuracy(node.train_eval_loader)
                      for i, node in system.nodes.items()}
        # per_class only needed for bandit_soft deploy mode; "self" mode doesn't use it
        _pc_test0  = system.evaluate_all_nodes_per_class_on_test() if deploy_criteria == "bandit_soft" else {}
        _t0, _cn0, _tr0, _s0 = compute_metrics_from_per_class(
            system, _pc_test0, {}, deploy_criteria="self",
            test_acc=_ta0, train_acc=_tra0)
        _pprint(
            f"[{run_id}]   [PRE 000/{pretrain_cap}] test={_t0:.4f}  cn_local={_cn0:.4f}  self={_s0:.4f}  train={_tr0:.4f}",
            no_flush=no_flush,
        )

        for r in range(pretrain_cap):
            t0 = time.perf_counter()
            system.run_round(
                round_idx=r, pseudo_epochs=0, pseudo_mode=pseudo_mode,
                steps_per_node=budget_steps, examples_per_node=eff_budget_examples,
                cap_examples_per_node=int(pseudo_cap_examples_per_node),
                conf_weight_tau0=float(conf_weight_tau0),
                sup_steps_total=int(sup_steps_total),
                sup_steps_per_node=int(sup_steps_per_node),
                agreement_beta=float(agreement_beta),
            )
            # Baselines do NOT get collaboration during pretrain — stage 1 is
            # supervised-only for all methods to ensure a fair comparison.
            if system.device.type == "cuda":
                torch.cuda.synchronize()
            t_stage1  += time.perf_counter() - t0
            pre_rounds += 1
            do_eval    = _should_eval(pre_rounds) or (pre_rounds == pretrain_cap)

            if do_eval:
                te0 = time.perf_counter()
                # per_class_train never needed: train_acc is always provided to compute_metrics
                # per_class_test only needed for bandit_soft; stage-1 always uses "self" mode
                test_acc_overall  = system.evaluate_all_nodes_on_test()
                train_acc_overall = {i: node.evaluate_accuracy(node.train_eval_loader)
                                     for i, node in system.nodes.items()}
                last_test, last_cn_local, last_train, last_self = compute_metrics_from_per_class(
                    system, {}, {}, deploy_criteria="self",
                    test_acc=test_acc_overall, train_acc=train_acc_overall)
                if system.device.type == "cuda":
                    torch.cuda.synchronize()
                t_eval_tt += time.perf_counter() - te0
                if last_test > pre_best + pretrain_tol:
                    pre_best = last_test; pre_no_improve = 0
                else:
                    pre_no_improve += 1

            for hist, val in [
                (test_hist, last_test), (cn_local_hist, last_cn_local), (train_hist, last_train),
                (self_hist, last_self),
            ]:
                hist.append(val)
            eval_mask_hist.append(bool(do_eval))

            if print_every > 0 and (r == 0 or (r + 1) % print_every == 0):
                _pprint(
                    f"[{run_id}]   [PRE {r+1:03d}/{pretrain_cap}] "
                    f"test={last_test:.4f}  cn_local={last_cn_local:.4f}  self={last_self:.4f}  train={last_train:.4f}  "
                    f"no_improve={pre_no_improve}/{pretrain_patience}",
                    no_flush=no_flush,
                )
                with open(_dbg_path, "a", encoding="utf-8") as _dbgf:
                    _dbgf.write(
                        f"  {r+1:>6}  {'PRE':>5}  {last_test:7.4f}  {last_cn_local:7.4f}  "
                        f"{last_self:7.4f}  {last_train:7.4f}\n"
                    )
            if pre_rounds >= pretrain_min_rounds and pre_no_improve >= pretrain_patience:
                _pprint(f"  [PRETRAIN STOP] rounds={pre_rounds}", no_flush=no_flush)
                break

        # ------------------------------------------------------------------
        # Checkpoint save: persist stage-1 weights immediately after pretrain
        # so subsequent hparam-search trials can skip this phase.
        # ------------------------------------------------------------------
        if pretrain_checkpoint_save:
            _save_file = pretrain_checkpoint_save
            _save_pretrain_checkpoint(
                path=_save_file, system=system,
                pre_rounds=pre_rounds, pre_best=pre_best,
                last_test=last_test, last_cn_local=last_cn_local,
                last_train=last_train,
                test_hist=test_hist, cn_local_hist=cn_local_hist,
                train_hist=train_hist, eval_mask_hist=eval_mask_hist,
            )

    # ============================
    # Carve val from training for Stage 2 (our method only)
    # ============================
    # Stage 1 trained on ALL data (train+val merged) for parity with baselines.
    # Now randomly carve out val_fraction of each node's training data as a
    # fresh val set for Stage 2 deployment scoring.  Baselines and p=0 don't
    # need val (no neighbor selection), so they keep training on everything.
    _is_our_method = (baseline == "none" and abs(float(p)) > 1e-9)
    if _is_our_method:
        system._carve_val_from_train_for_stage2()

    # ============================
    # UCB bootstrap (after pretrain)
    # ============================
    if system.neighbor_weighting == "ucb" and system._bandit is not None:
        _pprint("[BANDIT] Bootstrapping IW-unlabeled confidence scores...", no_flush=no_flush)
        system.update_weights_from_unlabeled_conf()
        _pprint("[BANDIT] Bootstrap complete.", no_flush=no_flush)

    # ============================
    # Stage 2: pseudo-label phase
    # ============================
    remaining = max_rounds - pre_rounds

    # ── Stage 2 backbone unfreeze ──────────────────────────────────────
    # When --stage_2_tuning is set, switch from frozen-backbone cache mode
    # to full backbone finetuning. Stage 1 is identical across all methods.
    # Stage 2 unfreezes the backbone so DESA's KD can train the encoder.
    if stage_2_tuning and system.cache_features:
        system.switch_to_full_tuning()
        system._unlabeled_bufs = None  # force rebuild from raw images
        _pprint("[STAGE2_TUNE] Pre-building unlabeled buffers from raw images...", no_flush=no_flush)
        _ = system._build_unlabeled_bufs_if_needed()
        _pprint("[STAGE2_TUNE] Unlabeled buffers ready.", no_flush=no_flush)
        # Pre-cache val backbone features for every (student_i, teacher_j) pair so that
        # val EMA updates throughout Stage 2 can use head-only forward passes (~1000x
        # cheaper than re-running the full backbone every round at deploy_ema_freq=1).
        # This one-time cost is amortised over all Stage 2 rounds.
        if baseline == "none" and deploy_criteria in ("val_ema", "val_bandit"):
            _pprint("[STAGE2_TUNE] Pre-caching val backbone features for EMA scoring...", no_flush=no_flush)
            system._stale_val_feats: dict = {}
            with torch.no_grad():
                for _i, _node_i in system.nodes.items():
                    _hood_i = [_i] + list(_node_i.neighbor_ids)
                    _val_imgs, _val_labs = [], []
                    for _xb, _yb in _node_i.val_loader:
                        _val_imgs.append(_xb); _val_labs.append(_yb)
                    if not _val_imgs:
                        continue
                    _val_x_cpu = torch.cat(_val_imgs, 0)
                    _val_y_cpu = torch.cat(_val_labs, 0)
                    _val_x_dev = _val_x_cpu.to(system.device, non_blocking=True)
                    for _j in _hood_i:
                        _teacher = system.nodes[_j]
                        if not hasattr(_teacher.model, "forward_features"):
                            continue
                        _teacher.model.eval()
                        _chunks = []
                        for _s in range(0, _val_x_dev.size(0), system.batch_size):
                            _chunks.append(
                                _teacher.model.forward_features(_val_x_dev[_s:_s + system.batch_size])
                            )
                        _feats_cpu = torch.cat(_chunks, 0).cpu()
                        system._stale_val_feats[(_i, _j)] = (_feats_cpu, _val_y_cpu)
            _pprint(f"[STAGE2_TUNE] Val feature cache ready: {len(system._stale_val_feats)} pairs.", no_flush=no_flush)

    _pprint(
        f"[STAGE 2] {'(ORACLE)' if oracle_supervised else 'pseudo'}: "
        f"remaining={remaining}  teacher_mode={system.pseudo_teacher_mode}",
        no_flush=no_flush,
    )
    stage2_best = pre_best; stage2_no_improve = 0

    # EMA val-score cache: {node_i: {node_j: float}}.
    # Updated every deploy_ema_freq rounds; used when deploy_criteria == "val_ema".
    system._val_ema_scores: Dict[int, Dict[int, float]] = {}  # type: ignore[attr-defined]
    _ema_alpha = float(deploy_ema_alpha)
    _ema_freq  = max(1, int(deploy_ema_freq))

    _rebootstrap_freq = int(rebootstrap_freq) if rebootstrap_freq and rebootstrap_freq > 0 else 0

    # ── Topology-aware rebootstrap setup ──────────────────────────────
    # When topo_rebootstrap is enabled, low-degree (leaf) nodes get one
    # extra rebootstrap at the midpoint between full rebootstraps.
    # With freq=50: full at 50, leaf-only at 25, full at 100, leaf at 75.
    _topo_reboot = bool(topo_rebootstrap)
    _leaf_nodes: List[int] = []
    if _topo_reboot and _rebootstrap_freq > 0:
        _degrees = {i: len(system.neighbor_map[i]) for i in system.nodes}
        _median_deg = max(1.0, float(np.median(list(_degrees.values()))))
        _leaf_nodes = sorted([i for i, d in _degrees.items() if d <= _median_deg])
        _pprint(
            f"[TOPO_REBOOT] enabled: {len(_leaf_nodes)} leaf nodes "
            f"(deg ≤ {_median_deg:.0f}) get midpoint rebootstrap "
            f"(at t={_rebootstrap_freq // 2}, {_rebootstrap_freq + _rebootstrap_freq // 2}, ...), "
            f"full rebootstrap every {_rebootstrap_freq} rounds",
            no_flush=no_flush,
        )

    for t in range(remaining):

        # Periodic re-bootstrap: refresh IW-confidence scores so the
        # bandit tracks model improvements during Stage 2.
        if (
            _rebootstrap_freq > 0
            and t > 0
            and system.neighbor_weighting == "ucb"
            and system._bandit is not None
        ):
            if t % _rebootstrap_freq == 0:
                # Full rebootstrap: all nodes
                _pprint(f"[BANDIT] Re-bootstrapping all nodes at stage-2 round {t}...", no_flush=no_flush)
                system.update_weights_from_unlabeled_conf()
                # Refresh stale val-feature cache so EMA scoring stays accurate
                # as teacher backbones evolve during Stage 2 full-tuning.
                _stale_val_cache = getattr(system, "_stale_val_feats", {})
                if _stale_val_cache and not system.cache_features:
                    with torch.no_grad():
                        for (_ci, _cj), (_old_f, _old_y) in list(_stale_val_cache.items()):
                            _teacher_c = system.nodes[_cj]
                            if not hasattr(_teacher_c.model, "forward_features"):
                                continue
                            _teacher_c.model.eval()
                            _val_x_dev = system.nodes[_ci].val_loader.dataset
                            # Re-use cached val images via node's val_loader
                            _chunks_c = []
                            for _xb_c, _ in system.nodes[_ci].val_loader:
                                _xb_c = _xb_c.to(system.device, non_blocking=True)
                                _chunks_c.append(_teacher_c.model.forward_features(_xb_c))
                            if _chunks_c:
                                _stale_val_cache[(_ci, _cj)] = (
                                    torch.cat(_chunks_c, 0).cpu(), _old_y
                                )
            elif (
                _topo_reboot
                and _leaf_nodes
                and t % _rebootstrap_freq == _rebootstrap_freq // 2
            ):
                # Leaf-only rebootstrap at the midpoint between full rebootstraps
                system.update_weights_from_unlabeled_conf(node_subset=_leaf_nodes)

        t0 = time.perf_counter()
        pseudo_stats = system.run_round(
            round_idx=t, pseudo_epochs=eff_pseudo_epochs, pseudo_mode=pseudo_mode,
            steps_per_node=budget_steps, examples_per_node=eff_budget_examples,
            cap_examples_per_node=int(pseudo_cap_examples_per_node),
            conf_weight_tau0=float(conf_weight_tau0),
            sup_steps_total=int(sup_steps_total),
            sup_steps_per_node=int(sup_steps_per_node),
            agreement_beta=float(agreement_beta),
        )
        if baseline_obj is not None:
            baseline_obj.post_round_hook(t)
        if system.device.type == "cuda":
            torch.cuda.synchronize()
        t_stage2 += time.perf_counter() - t0

        for i in node_ids:
            total_pseudo_batches  += pseudo_stats[i].pseudo_batches
            total_pseudo_examples += pseudo_stats[i].pseudo_examples

        if not oracle_supervised:
            if system._bandit is None:
                tv0 = time.perf_counter()
                val_accs = system.evaluate_all_nodes_on_val()
                if system.device.type == "cuda":
                    torch.cuda.synchronize()
                t_eval_val += time.perf_counter() - tv0
                for i in node_ids:
                    system.nodes[i].update_val_and_maybe_disable(
                        val_accs[i],
                        did_pseudo_this_round=(pseudo_stats[i].pseudo_examples > 0),
                    )
            else:
                for i in node_ids:
                    bandit = system._bandit
                    if isinstance(bandit, GradientAlignedDiscountedUCB):
                        j_best = bandit.best_arm(i, class_weights=system._node_skew_weights[i])
                        raw = (
                            float(np.dot(
                                [bandit.mu(i, j_best, c) for c in range(bandit.num_classes)],
                                system._node_skew_weights[i],
                            ))
                            if j_best is not None else 0.0
                        )
                        proxy = (raw + 1.0) / 2.0
                    else:
                        j_best = bandit.best_arm(i)
                        proxy  = bandit.Q[i].get(j_best, 0.5) if j_best is not None else 0.5
                    system.nodes[i].update_val_and_maybe_disable(
                        proxy,
                        did_pseudo_this_round=(pseudo_stats[i].pseudo_examples > 0),
                    )

        global_round = pre_rounds + t + 1
        do_eval      = (
            _should_eval(global_round)
            or (t == remaining - 1)
            or (stage2_no_improve >= patience)
        )

        # ── EMA val-score update (val_ema deploy mode) ──────────────────
        # Skip for baselines — they don't do deployment selection, so
        # evaluating every neighbor on the val loader is wasted compute
        # and would incorrectly apply our deployment protocol to them.
        if baseline == "none" and deploy_criteria in ("val_ema", "val_bandit") and (t % _ema_freq == 0 or t == 0):
            with torch.no_grad():
                _stale_val = getattr(system, "_stale_val_feats", {})
                for i in node_ids:
                    node_i = system.nodes[i]
                    hood   = [i] + list(node_i.neighbor_ids)
                    new_scores: Dict[int, float] = {}
                    for j in hood:
                        key = (i, j)
                        if key in _stale_val:
                            # Fast path: teacher backbone features pre-cached —
                            # only run teacher's head (linear layer, ~1000x cheaper).
                            feats_cpu, labs_cpu = _stale_val[key]
                            feats_dev = feats_cpu.to(system.device, non_blocking=True)
                            labs_dev  = labs_cpu.to(system.device, non_blocking=True)
                            logits = system.nodes[j].model.forward_head(feats_dev)
                            correct = (logits.argmax(-1) == labs_dev).sum().item()
                            new_scores[j] = correct / max(1, labs_dev.size(0))
                        else:
                            new_scores[j] = float(system.evaluate_j_on_i_loader(
                                j, i, node_i.val_loader, system._val_idx_per_node))
                    if i not in system._val_ema_scores:
                        system._val_ema_scores[i] = dict(new_scores)
                    else:
                        old = system._val_ema_scores[i]
                        system._val_ema_scores[i] = {
                            j: (1.0 - _ema_alpha) * old.get(j, s) + _ema_alpha * s
                            for j, s in new_scores.items()
                        }

        _eff_deploy = "self" if baseline != "none" else deploy_criteria

        if do_eval:
            te0 = time.perf_counter()
            # per_class_test is only needed for bandit_soft weighting; all other
            # deploy modes use test_acc (overall) which is already computed below.
            # per_class_train is never needed: train_acc is always provided.
            per_class_test = (
                system.evaluate_all_nodes_per_class_on_test()
                if _eff_deploy == "bandit_soft" else {}
            )
            test_acc_overall  = system.evaluate_all_nodes_on_test()
            train_acc_overall = {i: node.evaluate_accuracy(node.train_eval_loader)
                                 for i, node in system.nodes.items()}
            _dbg = debug_val_query and deploy_criteria == "val_query"
            last_test, last_cn_local, last_train, last_self = compute_metrics_from_per_class(
                system, per_class_test, {}, deploy_criteria=_eff_deploy,
                test_acc=test_acc_overall, train_acc=train_acc_overall,
                debug_val_query=_dbg)
            if system.device.type == "cuda":
                torch.cuda.synchronize()
            t_eval_tt += time.perf_counter() - te0
            if last_test > stage2_best + tol:
                stage2_best = last_test; stage2_no_improve = 0
            else:
                stage2_no_improve += 1

        for hist, val in [
            (test_hist, last_test), (cn_local_hist, last_cn_local), (train_hist, last_train),
            (self_hist, last_self),
        ]:
            hist.append(val)
        eval_mask_hist.append(bool(do_eval))

        if print_every > 0 and (
            t == 0 or global_round % print_every == 0 or stage2_no_improve >= patience
        ):
            mean_ex  = total_pseudo_examples / max(1, t + 1)
            disabled = sum(1 for i in node_ids if not system.nodes[i].pseudo_allowed)
            # Average sup and pseudo losses across all nodes this round.
            _sup_losses   = [n._sup_loss_sum / max(1, n._sup_loss_n)
                             for n in system.nodes.values() if n._sup_loss_n > 0]
            _pseudo_losses = [n._pseudo_loss_sum / max(1, n._pseudo_loss_n)
                              for n in system.nodes.values() if n._pseudo_loss_n > 0]
            _sup_str    = f"{sum(_sup_losses)/len(_sup_losses):.3f}"   if _sup_losses   else "n/a"
            _pseudo_str = f"{sum(_pseudo_losses)/len(_pseudo_losses):.3f}" if _pseudo_losses else "n/a"
            _kl_scale_eff = getattr(system, "_sup_loss_ema", None)
            _anchor       = getattr(system, "_anchor_sup_loss", None)
            _kl_adapt_str = (
                f"  kl_adapt×{min(4.0, _anchor/_kl_scale_eff):.2f}"
                if (_kl_scale_eff and _anchor and _kl_scale_eff < _anchor)
                else ""
            )
            _pprint(
                f"[{run_id}]   [{'ORC' if oracle_supervised else 'PSE'} r {global_round:03d}/{max_rounds}] "
                f"test={last_test:.4f}  cn_local={last_cn_local:.4f}  self={last_self:.4f}  train={last_train:.4f}  "
                f"ex/round={mean_ex:.1f}  disabled={disabled}/{len(node_ids)}  "
                f"no_improve={stage2_no_improve}/{patience}  "
                f"sup_loss={_sup_str}  kl_loss={_pseudo_str}{_kl_adapt_str}"
                + (f"  bandit_alpha={system._bandit.alpha:.3f}" if isinstance(system._bandit, GradientAlignedDiscountedUCB) else "")
                + (f"  tau={system.grad_align_tau:.3f}" if system.bandit_type == "grad_align" else ""),
                no_flush=no_flush,
            )

            # ── Per-node breakdown: self accuracy, best neighbor, best neighbor's accuracy ──
            if do_eval:
                _test_acc_all  = test_acc_overall  # already computed above — avoids duplicate full-eval pass
                _val_ema_cache = getattr(system, "_val_ema_scores", {})
                _deploy_scores = getattr(system, "_deploy_scores", {})
                _cn_oracle     = getattr(system, "_debug_cn_oracle", {})
                _per_node_lines = []
                for _ni in node_ids:
                    _self_acc  = float(_test_acc_all[_ni])
                    _fav       = system.favored_class_map.get(_ni, -1)
                    _deg       = len(system.neighbor_map.get(_ni, []))
                    _arch      = "eff" if system._node_arch_map.get(_ni, "") == "efficientnet_b0" else "mob"
                    _hood      = [_ni] + list(system.nodes[_ni].neighbor_ids)

                    # Per-node pseudo stats
                    _n_pseudo = pseudo_stats[_ni].pseudo_examples if _ni in pseudo_stats else 0
                    _node_obj = system.nodes[_ni]
                    _node_sup = (
                        _node_obj._sup_loss_sum / max(1, _node_obj._sup_loss_n)
                        if getattr(_node_obj, "_sup_loss_n", 0) > 0 else 0.0
                    )
                    _node_kl  = (
                        _node_obj._pseudo_loss_sum / max(1, _node_obj._pseudo_loss_n)
                        if getattr(_node_obj, "_pseudo_loss_n", 0) > 0 else 0.0
                    )

                    # Oracle best neighbor from cn_local computation (populated by compute_metrics)
                    _oracle_j, _oracle_acc = _cn_oracle.get(_ni, (_ni, _self_acc))
                    _oracle_str = "self" if _oracle_j == _ni else f"n{_oracle_j:02d}"

                    # Use the actual deployment criterion to determine _deployed
                    _margin = float(getattr(system, "deploy_self_margin", 0.05))
                    if _eff_deploy == "bandit_class" and bandit is not None and hasattr(bandit, "pc_mu"):
                        _w_ni = system._node_skew_weights[_ni]
                        _self_q_ni = getattr(system, "_self_iw_scores", {}).get(_ni, 0.0)
                        _bc_sc = {_ni: _self_q_ni}
                        for _j in system.nodes[_ni].neighbor_ids:
                            if _j in bandit.pc_mu.get(_ni, {}):
                                _bc_sc[_j] = float(np.dot(bandit.pc_mu[_ni][_j], _w_ni))
                            else:
                                _bc_sc[_j] = bandit.Q[_ni].get(_j, 0.0)
                        _deployed = max(_hood, key=lambda j: _bc_sc.get(j, 0.0))
                        if _deployed != _ni and _bc_sc.get(_deployed, 0.0) < _bc_sc.get(_ni, 0.0) + _margin:
                            _deployed = _ni
                    elif _eff_deploy in ("val_ema", "val_bandit") and _val_ema_cache.get(_ni):
                        _val_sc   = _val_ema_cache[_ni]
                        _deployed = max(_hood, key=lambda j: _val_sc.get(j, 0.0))
                        if _deployed != _ni and _val_sc.get(_deployed, 0.0) < _val_sc.get(_ni, 0.0) + _margin:
                            _deployed = _ni
                    elif _eff_deploy == "self" or not system.nodes[_ni].neighbor_ids:
                        _deployed = _ni
                    elif _deploy_scores.get(_ni):
                        _dsc = _deploy_scores[_ni]
                        _deployed = max(_hood, key=lambda j: _dsc.get(j, 0.0), default=_ni)
                    else:
                        _deployed = system._best_neighbor.get(_ni, _ni) or _ni
                    # Show deployed model's accuracy on THIS node's test set
                    _tl = system._node_test_loaders.get(_ni)
                    if _tl is not None and _deployed != _ni:
                        _deploy_acc = system.evaluate_j_on_i_loader(_deployed, _ni, _tl, system._test_idx_per_node)
                    else:
                        _deploy_acc = float(_test_acc_all[_deployed])
                    _deploy_str = "self" if _deployed == _ni else f"n{_deployed:02d}"
                    _per_node_lines.append(
                        f"n{_ni:02d} {_arch} d{_deg:02d} c{_fav} | "
                        f"self={_self_acc:.3f}  deploy→{_deploy_str}={_deploy_acc:.3f}  "
                        f"oracle→{_oracle_str}={_oracle_acc:.3f}  "
                        f"ex={_n_pseudo}  sup={_node_sup:.4f}  kl={_node_kl:.4f}"
                    )
                for _row in range(0, len(_per_node_lines), 2):
                    _pprint("  " + "    ".join(_per_node_lines[_row:_row + 2]), no_flush=no_flush)
                # Append per-node detail to debug file
                with open(_dbg_path, "a", encoding="utf-8") as _dbgf:
                    _dbgf.write(f"\n  --- PSE r{global_round:03d}/{max_rounds}  "
                                f"test={last_test:.4f}  cn_loc={last_cn_local:.4f}  "
                                f"self={last_self:.4f}  ex/rnd={mean_ex:.0f}  "
                                f"sup={_sup_str}  kl={_pseudo_str} ---\n")
                    _dbgf.write(f"  {'node':>4} {'fav':>3} {'deg':>3} | "
                                f"{'self':>6} {'deploy':>6} {'dep_acc':>7} {'oracle':>6} {'ora_acc':>7} | "
                                f"{'ex':>5} {'sup':>7} {'kl':>7}\n")
                    _dbgf.write(f"  {'-'*85}\n")
                    for _pnl in _per_node_lines:
                        _dbgf.write(f"  {_pnl}\n")
            else:
                # print_every fired but no full eval — write compact round line only
                with open(_dbg_path, "a", encoding="utf-8") as _dbgf:
                    _dbgf.write(
                        f"  {global_round:>6}  {'PSE':>5}  {last_test:7.4f}  {last_cn_local:7.4f}  "
                        f"{last_self:7.4f}  {last_train:7.4f}\n"
                    )
        if stage2_no_improve >= patience:
            _pprint(f"  [EARLY STOP]", no_flush=no_flush)
            break

    rounds_ran  = len(test_hist)
    rounds_arr  = np.arange(1, rounds_ran + 1, dtype=np.int32)
    p10         = _p10(p)

    summary = RunSummary(
        exp=exp, seed=seed, p=p, p10=p10,
        budget_steps=budget_steps,
        budget_examples=eff_budget_examples if exp == "examples" else budget_examples,
        max_rounds=max_rounds, rounds_ran=rounds_ran,
        unlabeled_fraction=unlabeled_fraction,
        unlabeled_per_node=unlabeled_per_node,
        per_node_sample_size=per_node_sample_size,
        kl_weight=kl_weight,
        test_final=float(test_hist[-1])      if rounds_ran   else 0.0,
        cn_local_final=float(cn_local_hist[-1]) if cn_local_hist else 0.0,
        train_final=float(train_hist[-1])    if rounds_ran   else 0.0,
        self_final=float(self_hist[-1])      if self_hist    else 0.0,
        pseudo_batches_per_round_mean=float(total_pseudo_batches  / max(1, rounds_ran)),
        pseudo_examples_per_round_mean=float(total_pseudo_examples / max(1, rounds_ran)),
        is_oracle=bool(oracle_supervised),
    )

    curve = CurveRecord(
        exp=exp, seed=seed, p=float(p), p10=p10,
        budget_steps=int(budget_steps),
        budget_examples=int(summary.budget_examples),
        rounds=rounds_arr,
        test=np.array(test_hist,      dtype=np.float32),
        cn_local=np.array(cn_local_hist, dtype=np.float32),
        train=np.array(train_hist,    dtype=np.float32),
        eval_mask=np.array(eval_mask_hist, dtype=np.bool_),
        pretrain_rounds_ran=int(pre_rounds),
        is_oracle=bool(oracle_supervised),
    )

    if plot_graph:
        pc_test        = system.evaluate_all_nodes_per_class_on_test()
        final_weighted = {i: system.weighted_test_acc(pc_test, i, i) for i in system.nodes}
        ts_str         = time.strftime("%Y%m%d_%H%M%S")
        gpath = os.path.join(out_dir, f"graph_{exp}_seed{seed}_p{p10}_{tag}_{ts_str}.png")
        _plot_graph_colored_by_performance(
            system.neighbor_map, final_weighted, hub_id,
            f"exp={exp} seed={seed} p={p:.2f} conn={connection_model} | color=IW test acc",
            gpath,
        )
        _pprint(f"[GRAPH] Saved: {gpath}", no_flush=no_flush)

    _pprint(
        f"[END] exp={exp} seed={seed} p={p:.3f} conn={connection_model} "
        f"rounds={rounds_ran} "
        f"test={summary.test_final:.4f}  cn_local={summary.cn_local_final:.4f}  "
        f"train={summary.train_final:.4f}",
        no_flush=no_flush,
    )

    # ---- Communication cost summary ----
    _num_classes = system.num_classes
    _feat_dim    = system.feat_dim
    # Head param count: depends on whether head is 1-layer or 2-layer
    _head_params = sum(p.numel() for node in system.nodes.values()
                       for p in node.model.head.parameters())
    _head_params_per_node = _head_params // max(1, len(system.nodes))
    # Unlabeled batch size used for pseudo-label exchange
    _unlab_per_node = max(1, system.global_unlabeled_total // max(1, system.num_nodes))
    _our_cost   = _unlab_per_node * _num_classes          # soft labels on unlabeled data
    _mutual_mult = 2 if (system.bandit_type == "entropy_ucb" and system.neighbor_weighting == "ucb") else 1
    _our_cost_effective = _our_cost * _mutual_mult
    _desa_cost  = 10 * _num_classes * _feat_dim            # ipc=10 anchors per class
    _gossip_cost = _head_params_per_node                   # full head parameters
    _dml_cost   = 0                                        # uses local labeled data (private)
    _pprint(
        f"[COMM] floats/node/round — "
        f"ours={_our_cost_effective:,}"
        + (f" (2×{_our_cost:,} mutual)" if _mutual_mult == 2 else "")
        + f"  desa={_desa_cost:,} ({_desa_cost//_our_cost_effective if _our_cost_effective>0 else '?'}x)  "
        f"gossip={_gossip_cost:,} ({_gossip_cost//_our_cost_effective if _our_cost_effective>0 else '?'}x)  "
        f"dml=private_data",
        no_flush=no_flush,
    )
    if debug_time:
        _pprint(
            f"[TIME] stage1={t_stage1:.1f}s stage2={t_stage2:.1f}s "
            f"eval_tt={t_eval_tt:.1f}s eval_val={t_eval_val:.1f}s",
            no_flush=no_flush,
        )

    # ------------------------------------------------------------------
    # Stage-2 checkpoint: save final model weights after pseudo-label phase.
    # Useful for qualitative analysis of which examples were selected.
    # ------------------------------------------------------------------
    if stage2_checkpoint_save:
        _save_file = _ckpt_path_for_seed(stage2_checkpoint_save, seed)
        _save_pretrain_checkpoint(
            path=_save_file, system=system,
            pre_rounds=pre_rounds + len(test_hist) - pre_rounds,
            pre_best=stage2_best,
            last_test=last_test, last_cn_local=last_cn_local,
            last_train=last_train,
            test_hist=test_hist, cn_local_hist=cn_local_hist,
            train_hist=train_hist, eval_mask_hist=eval_mask_hist,
        )

    # ------------------------------------------------------------------
    # Debug report: write per-node diagnostic file named after run_id.
    # ------------------------------------------------------------------
    _final_sup    = {i: (n._sup_loss_sum    / max(1, n._sup_loss_n))
                     for i, n in system.nodes.items() if getattr(n, "_sup_loss_n", 0) > 0}
    _final_pseudo = {i: (n._pseudo_loss_sum / max(1, n._pseudo_loss_n))
                     for i, n in system.nodes.items() if getattr(n, "_pseudo_loss_n", 0) > 0}
    write_run_debug_file(
        run_id=run_id,
        argv=sys.argv,
        system=system,
        seed=seed,
        p=p,
        test_hist=test_hist,
        cn_local_hist=cn_local_hist,
        self_hist=self_hist,
        deploy_criteria=deploy_criteria,
        out_dir=out_dir,
        final_sup_losses=_final_sup,
        final_pseudo_losses=_final_pseudo,
        baseline=baseline,
    )

    return summary, curve


def _shared_run_kwargs(args, amp_features: bool) -> dict:
    return dict(
        out_dir=os.path.abspath(args.out_dir),
        num_nodes=args.num_nodes, batch_size=args.batch_size,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction, unlabeled_fraction=args.unlabeled_fraction, unlabeled_per_node=args.unlabeled_per_node, unlabeled_pool_skew=args.unlabeled_pool_skew,
        per_node_sample_size=args.per_node_sample_size,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        dropout_p=args.dropout_p, dropout_feat=args.dropout_feat,
        kl_weight=args.kl_weight, pseudo_epochs=args.pseudo_epochs,
        max_rounds=args.max_rounds, patience=args.patience, tol=args.tol,
        num_workers_train=args.num_workers_train, num_workers_eval=args.num_workers_eval,
        pseudo_cap_examples_per_node=args.pseudo_cap_examples_per_node,
        pseudo_conf_threshold=args.pseudo_conf_threshold,
        pseudo_entropy_threshold=args.pseudo_entropy_threshold,
        pseudo_warmup_rounds=args.pseudo_warmup_rounds,
        kl_ramp_rounds=args.kl_ramp_rounds,
        pseudo_disable_patience=args.pseudo_disable_patience,
        pseudo_disable_delta=args.pseudo_disable_delta,
        conf_weight_tau0=args.conf_weight_tau0,
        agreement_beta=args.agreement_beta,
        training_data_mode=args.training_data_mode,
        skew_factor=args.skew_factor, skew_strategy=args.skew_strategy,
        skew_seed=args.skew_seed, skew_min_other_frac=args.skew_min_other_frac,
        min_classes_per_node=args.min_classes_per_node,
        pretrain_min_rounds=args.pretrain_min_rounds,
        pretrain_patience=args.pretrain_patience, pretrain_tol=args.pretrain_tol,
        pretrain_max_rounds=args.pretrain_max_rounds,
        eval_freq=args.eval_freq, arch=args.arch,
        connection_model=args.connection_model,
        plot_graph=args.plot_graph, print_every=args.print_every,
        no_flush=args.no_flush, debug_time=args.debug_time,
        sup_steps_per_node=args.sup_steps_per_node,
        sup_steps_total=args.sup_steps_total,
        cache_features=args.cache_features,
        finetune_backbone=args.finetune_backbone, stage_2_tuning=args.stage_2_tuning,
        baseline_non_linearity=args.baseline_non_linearity, cache_batch_size=args.cache_batch_size,
        amp_features=amp_features, feature_cache_path=args.feature_cache_path,
        size_skew_mode=args.size_skew_mode, size_seed=args.size_seed,
        min_per_node_size=args.min_per_node_size,
        size_total_budget=args.size_total_budget,
        size_dirichlet_alpha=args.size_dirichlet_alpha,
        baseline=args.baseline, baseline_our_deployment=args.baseline_our_deployment,
        baseline_merge_val=args.baseline_merge_val,
        baseline_rho=args.baseline_rho,
        fedmd_public_size=args.fedmd_public_size,
        neighbor_weighting=args.neighbor_weighting,
        neighbor_weight_update_freq=args.neighbor_weight_update_freq,
        ucb_c=args.ucb_c, bandit_type=args.bandit_type,
        grad_align_gamma=args.grad_align_gamma,
        grad_align_tau=args.grad_align_tau,
        grad_align_iw_temp=args.grad_align_iw_temp,
        entropy_gate_tau=args.entropy_gate_tau,
        entropy_ucb_align_tau=args.entropy_ucb_align_tau,
        deploy_criteria=args.deploy_criteria,
        deploy_ema_freq=args.deploy_ema_freq,
        deploy_ema_alpha=args.deploy_ema_alpha,
        pseudo_teacher_mode=args.pseudo_teacher_mode,
        verify_distribution=args.verify_distribution,
        similarity_temp=args.similarity_temp,
        similarity_mode=args.similarity_mode,
        ba_m=args.ba_m,
        random_models=args.random_models,
        random_models_mnv2_frac=args.random_models_mnv2_frac,
        random_models_hub_efnet=args.random_models_hub_efnet,
        top_k_teachers=args.top_k_teachers,
        mutual_distillation=args.mutual_distillation,
        pseudo_label_temp=args.pseudo_label_temp,
        pseudo_examples_per_round=args.pseudo_examples_per_round,
        teacher_ema=args.teacher_ema,
        teacher_ema_steps=args.teacher_ema_steps,
        mobilenet_cache_path=args.mobilenet_cache_path,
        efficientnet_cache_path=args.efficientnet_cache_path,
        debug_val_query=args.debug_val_query,
        pretrain_checkpoint_save=args.pretrain_checkpoint_save,
        pretrain_checkpoint_load=args.pretrain_checkpoint_load,
        stage2_checkpoint_save=args.stage2_checkpoint_save,
        rebootstrap_freq=args.rebootstrap_freq,
        viz_graph=args.viz_graph,
        bandit_context_labeled=args.bandit_context_labeled,
        deploy_blend_alpha=args.deploy_blend_alpha,
        deploy_self_margin=args.deploy_self_margin,
        degree_prior=args.degree_prior,
        topo_rebootstrap=args.topo_rebootstrap,
        dataset=args.dataset,
        stage2_sup_alpha=args.stage2_sup_alpha,
        geo_cache=getattr(args, "geo_cache", ""),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--feature_cache_path", type=str, default="")
    parser.add_argument("--exp_control", type=str, default="none",
                        choices=["none", "budget", "data"])
    parser.add_argument("--p_list", type=float, nargs="+", required=True)
    parser.add_argument("--seed_list", type=int, nargs="+", default=[0])
    parser.add_argument("--budgets_steps", type=int, nargs="+", default=[10])
    parser.add_argument("--budgets_examples", type=int, nargs="+", default=[5000])
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--no_flush", action="store_true")
    parser.add_argument("--debug_time", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--arch", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "efficientnet_b0"])
    parser.add_argument(
        "--connection_model", type=str, default="uniform",
        choices=["uniform", "barabasi_albert", "data_similarity"],
    )
    parser.add_argument(
        "--similarity_temp", type=float, default=1.0,
        help=(
            "Temperature for --connection_model data_similarity. "
            "Controls sharpness of similarity-based preferential attachment. "
            "temp < 1 : strongly prefer similar nodes (echo chambers); "
            "temp = 1 : balanced default; "
            "temp >> 1 : approaches uniform random graph."
        ),
    )
    parser.add_argument(
        "--similarity_mode", type=str, default="softmax",
        choices=["softmax", "topk"],
        help=(
            "Edge selection strategy for --connection_model data_similarity. "
            "'softmax' (default): probabilistic sampling weighted by softmax(sim/temp) — "
            "still stochastic so the top pair can be missed by chance. "
            "'topk': deterministic k-NN phase (k = ceil(2*target_edges/n) most similar "
            "neighbours guaranteed per node) followed by uniform random fill of remaining "
            "edge budget. Maximum advantage for our method: UCB rewards, pseudo-label "
            "quality, and CN-best selection pool are all maximally similarity-aligned."
        ),
    )
    parser.add_argument(
        "--ba_m", type=int, default=0,
        help=(
            "Barabási-Albert preferential attachment parameter m: number of edges each "
            "new node attaches to when joining the network. Only used when "
            "--connection_model barabasi_albert is set. "
            "m=0 (default): derive m automatically from --p_list (original behaviour, "
            "m ≈ target_edges / n, which at high p gives near-uniform degree). "
            "m=1: pure tree — extreme hub-and-spoke, very sparse. "
            "m=2: classic scale-free BA with genuine power-law degree distribution. "
            "m=3-5: moderately hub-dominated, good balance for experiments. "
            "When ba_m > 0, --p_list still controls the random-fill edge budget "
            "added on top of the BA-generated edges, allowing you to tune density "
            "independently of the hub structure."
        ),
    )
    parser.add_argument("--plot_graph", action="store_true")
    parser.add_argument(
        "--viz_graph", action="store_true", default=False,
        help=(
            "Save a topology visualization image immediately after the graph is built "
            "(before any training). Saved to --out_dir as "
            "topology_seed{N}_p{P}_{conn_model}.png. "
            "Nodes colored by degree (bright = high degree hub), sized by degree. "
            "If --random_models is set, EfficientNet-B0 nodes get a red border. "
            "Useful for verifying BA hub-and-spoke structure before committing to a long run."
        ),
    )
    parser.add_argument("--add_supervised_oracle_p0", action="store_true")
    parser.add_argument("--sup_steps_per_node", type=int, default=2)
    parser.add_argument("--sup_steps_total", type=int, default=0)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--baseline_non_linearity", action="store_true", default=False,
                        help="Replace the single linear classification head with a 2-layer MLP "
                             "(Linear->ReLU->Dropout->Linear) for ALL nodes regardless of backbone. "
                             "Applies uniformly to both MobileNetV2 and EfficientNet-B0 nodes so "
                             "--baseline_non_linearity is a clean experiment-level toggle: "
                             "off = everyone gets a linear head, on = everyone gets the MLP. "
                             "In --random_models mode this ensures head structure does not vary "
                             "by architecture, removing a confound from mixed-arch experiments.")
    parser.add_argument("--finetune_backbone", action="store_true", default=False,
                        help="Unfreeze backbone for full fine-tuning. Incompatible with --cache_features.")
    parser.add_argument("--stage_2_tuning", action="store_true", default=False,
                        help=(
                            "Stage 1 uses frozen backbone + feature cache (fast). "
                            "At Stage 2 start, unfreeze backbone and switch to raw image training. "
                            "Enables faithful DESA comparison — trainable encoder during collaboration."
                        ))
    parser.add_argument("--cache_batch_size", type=int, default=512)
    parser.add_argument("--no_amp_features", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.15,
                        help="Fraction of each node's local data held out as a per-node local test set (same skewed distribution as train/val).")
    parser.add_argument("--unlabeled_fraction", type=float, default=0.4)
    parser.add_argument("--unlabeled_per_node", type=int, default=0,
                        help="Direct unlabeled buffer size per node. Overrides --unlabeled_fraction when > 0.")
    parser.add_argument("--unlabeled_pool_skew", type=str, default="iid",
                        choices=["iid", "skewed"],
                        help="iid (default): uniform draws. skewed: match node class distribution.")
    parser.add_argument("--per_node_sample_size", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--dropout_feat", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_warmup_rounds", type=int, default=13)
    parser.add_argument("--kl_ramp_rounds", type=int, default=25)
    parser.add_argument("--pseudo_conf_threshold", type=float, default=0.0)
    parser.add_argument("--pseudo_entropy_threshold", type=float, default=-1.0)
    parser.add_argument("--conf_weight_tau0", type=float, default=0.7)
    parser.add_argument("--agreement_beta", type=float, default=0.0,
                        help="Exponent on student-agreement weight in pseudo_step_avg. "
                             "0=pure teacher-confidence (default); 1=weight by conf*agree; "
                             ">1=harder gate requiring strong confident agreement.")
    parser.add_argument("--pseudo_cap_examples_per_node", type=int, default=2_000_000_000)
    parser.add_argument("--pseudo_disable_patience", type=int, default=100)
    parser.add_argument("--pseudo_disable_delta", type=float, default=0.0)
    parser.add_argument("--training_data_mode", type=str, default="iid",
                        choices=["iid", "skewed"])
    parser.add_argument("--skew_factor", type=float, default=5.0)
    parser.add_argument("--skew_strategy", type=str, default="round_robin",
                        choices=["round_robin", "random"])
    parser.add_argument("--skew_seed", type=int, default=0)
    parser.add_argument("--skew_min_other_frac", type=float, default=0.2)
    parser.add_argument("--min_classes_per_node", type=int, default=2)
    parser.add_argument("--pseudo_epochs", type=int, default=1)
    parser.add_argument("--max_rounds", type=int, default=120)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--pretrain_min_rounds", type=int, default=100)
    parser.add_argument("--pretrain_patience", type=int, default=100)
    parser.add_argument("--pretrain_tol", type=float, default=1e-4)
    parser.add_argument("--pretrain_max_rounds", type=int, default=100)
    parser.add_argument("--num_workers_train", type=int, default=2)
    parser.add_argument("--num_workers_eval", type=int, default=2)
    parser.add_argument("--size_skew_mode", type=str, default="none",
                        choices=["none", "dirichlet", "degree"])
    parser.add_argument("--size_seed", type=int, default=0)
    parser.add_argument("--min_per_node_size", type=int, default=20)
    parser.add_argument("--size_total_budget", type=int, default=0)
    parser.add_argument("--size_dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument(
        "--baseline_our_deployment", action="store_true", default=False,
        help=(
            "Apply our val/test split protocol to baselines (val_fraction, test_fraction "
            "as specified). Default (off): baselines train on all labeled data "
            "(val_fraction=0), keeping only the test split. This is the fair comparison — "
            "baselines don't use the val set for anything so reserving 70%% of data for it "
            "unfairly starves them. Use this flag as an ablation."
        ),
    )
    parser.add_argument(
        "--baseline_merge_val", action="store_true", default=False,
        help=(
            "Also merge val into train for our method (baseline=none, p>0). "
            "By default our method keeps val held-out for val_ema deployment scoring. "
            "Baselines and p=0 always merge regardless of this flag."
        ),
    )
    parser.add_argument("--baseline_rho", type=float, default=0.05)
    parser.add_argument("--fedmd_public_size", type=int, default=2000)
    parser.add_argument("--neighbor_weighting", type=str, default="none",
                        choices=["none", "train_acc", "ucb"])
    parser.add_argument("--neighbor_weight_update_freq", type=int, default=10)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument(
        "--rebootstrap_freq", type=int, default=10,
        help=(
            "How often (in Stage-2 rounds) to re-run the full UCB bootstrap sweep, "
            "refreshing IW-confidence scores for all (node, neighbor) pairs. "
            "This corrects staleness as models improve during Stage 2. "
            "Set to 0 to disable (single bootstrap at Stage-2 start only). "
            "Only active when --neighbor_weighting ucb is set."
        ),
    )
    parser.add_argument(
        "--top_k_teachers", type=int, default=1,
        help=(
            "Number of top-ranked teachers to distill from at each pseudo-label step. "
            "k=1 (default): use only the single best-scoring neighbor (original behaviour). "
            "k>1: select the k highest-scoring neighbors (by bandit UCB score for "
            "grad_align_ucb/entropy_ucb, or by gradient alignment score for grad_align), "
            "then average their soft pseudo-labels uniformly before the KL distillation step. "
            "Effective k is clamped to the number of available candidates that pass the "
            "alignment gate (τ safeguard), so setting k larger than the neighborhood size "
            "is safe — it simply uses all passing neighbors."
        ),
    )
    parser.add_argument(
        "--pseudo_examples_per_round", type=int, default=128,
        help=(
            "Maximum number of pseudo-labeled examples to train on per node per round "
            "(entropy_ucb only). All collection batches are scored by max teacher "
            "confidence and the globally top-N examples are selected for training. "
            "Default 128 = one batch per node, giving a 1:1 ratio with the single "
            "supervised step. Increase to allow more pseudo-label signal per round. "
            "Set to 0 to disable the cap (use all collected examples)."
        ),
    )
    parser.add_argument(
        "--mutual_distillation", action="store_true", default=False,
        help=(
            "Enable bidirectional pseudo-label distillation (entropy_ucb only). "
            "After node i distills from its top-k neighbors j, each j also distills "
            "from i's predictions on j's own unlabeled buffer. Knowledge flows in both "
            "directions along each active edge, recovering the tight coupling of "
            "DML-style mutual learning. Uses the same per-class confidence weighting "
            "and alignment gate as the forward pass. "
            "Communication cost: ~2× pseudo-label traffic (soft labels in both directions). "
            "No additional parameter sharing — fully output-only."
        ),
    )
    parser.add_argument(
        "--pseudo_label_temp", type=float, default=1.0,
        help=(
            "Temperature applied to teacher logits when generating pseudo-labels "
            "(entropy_ucb only). T < 1 sharpens (more decisive), T > 1 softens, "
            "T = 1.0 (default) is standard softmax. The alignment reward is always "
            "computed at T=1 so the bandit's cosine similarity is not distorted."
        ),
    )
    parser.add_argument(
        "--teacher_ema", action="store_true", default=False,
        help=(
            "Enable local Mean-Teacher-style EMA self-distillation. After each "
            "supervised step, each node's head EMA is updated (decay = "
            "--deploy_ema_alpha, shared with val_ema), then --teacher_ema_steps "
            "self-distillation steps distill from the EMA teacher into the student "
            "using the local unlabeled buffer. Zero cross-node communication. "
            "Freely combinable with --bandit_type entropy_ucb. Unlike "
            "--baseline mean_teacher this does NOT disable neighbor collaboration."
        ),
    )
    parser.add_argument(
        "--teacher_ema_steps", type=int, default=4,
        help=(
            "Number of EMA self-distillation steps per round when --teacher_ema "
            "is set. Default 4 matches the MeanTeacherBaseline for fair comparison."
        ),
    )
    parser.add_argument("--bandit_type", type=str, default="ucb1",
                        choices=["ucb1", "grad_align", "grad_align_ucb", "entropy_ucb", "omniscient"])
    parser.add_argument(
        "--bandit_context_labeled", action="store_true", default=False,
        help=(
            "Use labeled val error signal as LinUCB context instead of the unlabeled "
            "batch mean (entropy_ucb only). Context = mean feature of val examples the "
            "student currently gets wrong — captures current weaknesses so the bandit "
            "adapts which teacher is most useful as training progresses. "
            "Default (off): mean feature of the current unlabeled batch."
        ),
    )
    parser.add_argument("--grad_align_gamma", type=float, default=0.99,
                        help="Discount factor for grad_align_ucb (Algorithm 2). Ignored by grad_align.")
    parser.add_argument("--entropy_gate_tau", type=float, default=0.5,
                        help=(
                            "Entropy gate threshold τH for Algorithm 4 (entropy_ucb). "
                            "The normalized Shannon entropy H(w_i) of the node's local class "
                            "distribution is compared against this threshold to select the "
                            "gradient computation regime: "
                            "H(w_i) > τH → balanced regime (true labels as reference gradient); "
                            "H(w_i) ≤ τH → skewed regime (importance weights w_i as synthetic "
                            "reference gradient). "
                            "τH = 0.5 is a symmetric default; τH → 0 always uses true gradients; "
                            "τH → 1 always uses synthetic gradients."
                        ))
    parser.add_argument("--entropy_ucb_align_tau", type=float, default=0.0,
                        help=(
                            "Optimization gate threshold τ for Algorithm 4 (entropy_ucb). "
                            "A distillation step is only taken when the logit-space alignment "
                            "reward r > τ, preventing negative-transfer updates. "
                            "Reward is now raw cosine similarity in [-1, 1]. "
                            "τ = 0.0 (default) accepts only positively aligned teachers "
                            "(cosine > 0), which is the natural threshold. "
                            "τ < 0 accepts more updates including weakly anti-aligned teachers. "
                            "τ > 0 is more conservative — only strongly aligned teachers pass."
                        ))
    parser.add_argument("--grad_align_tau", type=float, default=0.0,
                        help=(
                            "Alignment threshold τ for grad_align (Algorithm 1). "
                            "A neighbor is only used for class c if its gradient cosine "
                            "similarity exceeds τ. τ=0 accepts any positive alignment; "
                            "τ<0 accepts all neighbors regardless of sign."
                        ))
    parser.add_argument("--grad_align_iw_temp", type=float, default=0,
                        help=(
                            "Temperature for IS weights in grad-align distillation loss. "
                            "1.0 = raw skew weights (default, preserves current behaviour); "
                            "0.5 = sqrt-softened (compresses favoured/minority ratio); "
                            "0.0 = uniform weighting (equivalent to UCB1 selection-only). "
                            "Applies to both --bandit_type grad_align and grad_align_ucb."
                        ))
    parser.add_argument("--deploy_criteria", type=str, default="val_query",
                        choices=["val_query", "val_ema", "val_ucb", "dist_overlap", "bandit", "bandit_soft", "val_bandit", "bandit_class"])
    parser.add_argument("--deploy_ema_freq", type=int, default=5,
                        help="How often (in rounds) to refresh val-score EMA for val_ema deploy mode.")
    parser.add_argument("--deploy_ema_alpha", type=float, default=0.1,
                        help="EMA decay: new_ema = (1-alpha)*old + alpha*new.  Lower = slower.")
    parser.add_argument("--deploy_blend_alpha", type=float, default=0.7,
                        help="val_bandit blend: alpha*val_ema + (1-alpha)*iw_conf. 1.0=pure val_ema, 0.0=pure bandit.")
    parser.add_argument("--deploy_self_margin", type=float, default=0.05,
                        help="Neighbor must beat self by this margin on val to trigger deployment away from self.")
    parser.add_argument("--pseudo_teacher_mode", type=str, default="avg",
                        choices=["avg", "best"])
    parser.add_argument("--verify_distribution", action="store_true")
    parser.add_argument("--debug_val_query", action="store_true", default=False,
                        help="Print per-node val vs test ranking debug info for first few stage-2 evals")
    parser.add_argument(
        "--pretrain_checkpoint_save", type=str, default="",
        metavar="PATH",
        help=(
            "If set, save per-node model + optimiser weights to PATH after stage 1 "
            "completes (one file per seed: PATH_seed{N}.pt).  Subsequent runs with "
            "--pretrain_checkpoint_load pointing to the same PATH will skip stage 1 "
            "entirely, which is the main speedup for hyperparameter search."
        ),
    )
    parser.add_argument(
        "--pretrain_checkpoint_load", type=str, default="",
        metavar="PATH",
        help=(
            "If set and the per-seed checkpoint file exists (PATH_seed{N}.pt), "
            "load pre-trained weights and skip stage 1.  The stage-1 metric history "
            "is also restored so learning curves are complete.  If the file is "
            "missing or the metadata doesn't match (seed / num_nodes), stage 1 runs "
            "normally with a warning.  Intended for hyperparameter search: run one "
            "warmup trial with --pretrain_checkpoint_save, then pass the same path "
            "via --pretrain_checkpoint_load in all search trials."
        ),
    )
    parser.add_argument(
        "--stage2_checkpoint_save", type=str, default="",
        metavar="PATH",
        help=(
            "If set, save final post-stage-2 model weights to PATH_seed{N}.pt "
            "after the pseudo-label phase completes.  Same format as "
            "--pretrain_checkpoint_save.  Useful for qualitative analysis of "
            "which unlabeled examples are selected by the agreement gate."
        ),
    )
    parser.add_argument(
        "--random_models", action="store_true",
        help=(
            "Randomly assign each node mobilenet_v2 or efficientnet_b0. "
            "Requires --cache_features. Both caches must be precomputed."
        ),
    )
    parser.add_argument(
        "--random_models_mnv2_frac", type=float, default=0.8,
        help=(
            "Fraction of nodes assigned mobilenet_v2 when --random_models is set. "
            "The remaining nodes get efficientnet_b0. "
            "Default 0.8 (80%% MobileNetV2 / 20%% EfficientNet-B0), which creates a "
            "hub-and-spoke-style heterogeneity where most nodes share the same "
            "feature space but a minority use a different backbone. "
            "Set to 0.5 for a 50/50 split (original behaviour). "
            "Set to 1.0 or 0.0 for a homogeneous network."
        ),
    )
    parser.add_argument(
        "--random_models_hub_efnet", action="store_true", default=True,
        help=(
            "When --random_models is set, assign EfficientNet-B0 to the highest-degree "
            "nodes (hubs). The top (1 - --random_models_mnv2_frac) fraction of nodes "
            "by degree receive EfficientNet-B0. Default: True. "
            "Use --random_models_random_arch to assign architectures randomly instead."
        ),
    )
    parser.add_argument(
        "--random_models_random_arch", action="store_false", dest="random_models_hub_efnet",
        help="Assign architectures randomly instead of hub-based (overrides --random_models_hub_efnet).",
    )
    parser.add_argument(
        "--mobilenet_cache_path", type=str,
        default="./DecentralizedLearning/data/feature_cache/cifar10_mnv2_224.pt",
        help="Path to precomputed MobileNetV2 feature cache (.pt). Used with --random_models.",
    )
    parser.add_argument(
        "--efficientnet_cache_path", type=str,
        default="./DecentralizedLearning/data/feature_cache/cifar10_features_efficientnet_b0.pt",
        help="Path to precomputed EfficientNet-B0 feature cache (.pt). Used with --random_models.",
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100","eurosat"],
                        help="Dataset to use. cifar100 has 100 classes — much harder "
                             "with the same data budget, making collaboration essential.")
    parser.add_argument("--geo_cache", type=str, default="",
                        help="Path to precomputed EuroSAT geographic cache (.pt). "
                             "Overrides graph, data partition, and architecture assignments "
                             "with real geographic data from precompute_eurosat_features.py.")
    parser.add_argument(
        "--degree_prior", type=float, default=0.0,
        help=(
            "Strength of degree-informed UCB prior.  At bootstrap time, each "
            "neighbor's IW-confidence score is augmented by "
            "degree_prior * log(1 + deg(j)) / log(1 + max_deg).  "
            "This biases the bandit toward high-degree hub neighbors, which "
            "have seen more diverse training data and are more likely to be "
            "effective teachers — especially in scale-free (BA) topologies "
            "where hubs aggregate knowledge from many specialists.  "
            "0.0 (default): no degree bias (original behaviour).  "
            "0.1-0.3: recommended range for BA topologies."
        ),
    )
    parser.add_argument(
        "--topo_rebootstrap", action="store_true", default=False,
        help=(
            "Topology-aware rebootstrapping.  When enabled, low-degree (leaf) "
            "nodes rebootstrap their bandit scores at 2× the base frequency "
            "(--rebootstrap_freq // 2), while hub nodes rebootstrap at the "
            "base frequency.  On BA topologies, leaf nodes have degree 2 and "
            "a very small teacher pool — frequent recalibration lets them "
            "quickly detect when their few neighbors improve during Stage 2.  "
            "The leaf/hub boundary is the median degree of the graph."
        ),
    )
    parser.add_argument(
        "--stage2_sup_alpha", type=float, default=1.0,
        help=(
            "Scaling factor for supervised CE loss in Stage 2.  "
            "1.0 (default): full supervised + distillation (original behaviour).  "
            "0.0: distillation only in Stage 2 (pure knowledge transfer).  "
            "0.1-0.5: reduced supervision so distillation signal is not overwhelmed."
        ),
    )

    args     = parser.parse_args()
    out_dir  = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Generate a short unique run ID for tracking this invocation.
    import uuid, datetime
    _run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6].upper()
    _pprint(f"[RUN_ID] {_run_id}")

    # Print the full reconstructed CLI call so logs are self-contained.
    _pprint(
        "[CMD] python3 " + " \\\n  ".join(sys.argv[1:] if len(sys.argv) > 1 else [])
    )

    exp_list     = {"none": ["all"], "budget": ["steps"], "data": ["examples"]}[args.exp_control]
    amp_features = not args.no_amp_features
    if args.baseline == "none":
        print("With Bandit Type: "+str(args.bandit_type))

    _pprint(
        f"[CONFIG] out_dir={out_dir} exp_list={exp_list} arch={args.arch} "
        f"conn={args.connection_model}"
        + (f" similarity_temp={args.similarity_temp}" if args.connection_model == "data_similarity" else "")
        + f" random_models={args.random_models} "
        f"p_list={args.p_list} seeds={args.seed_list} baseline={args.baseline} "
        f"train_mode={args.training_data_mode} skew={args.skew_factor} "
        f"neighbor_weighting={args.neighbor_weighting} "
        f"freq={args.neighbor_weight_update_freq} "
        f"max_rounds={args.max_rounds} pretrain_max={args.pretrain_max_rounds} "
        f"cache_features={args.cache_features}",
        no_flush=args.no_flush,
    )

    hparams_path = os.path.join(out_dir, "connectivity_experiments_hparams.txt")
    _write_hparams(hparams_path, args)

    summaries: List[RunSummary]  = []
    curves:    List[CurveRecord] = []
    shared = _shared_run_kwargs(args, amp_features)

    def _run(exp_, seed_, p_, bsteps_, bex_, oracle_=False):
        s, c = run_one_setting(
            exp=exp_, seed=seed_, p=p_,
            budget_steps=int(bsteps_), budget_examples=int(bex_),
            oracle_supervised=bool(oracle_),
            run_id=_run_id,
            **shared,
        )
        summaries.append(s)
        curves.append(c)

    for exp in exp_list:
        for seed in args.seed_list:
            for p in args.p_list:
                if exp == "steps":
                    for B in args.budgets_steps:
                        _run(exp, seed, p, B, 0)
                        if args.add_supervised_oracle_p0 and abs(float(p)) < 1e-12:
                            _run(exp, seed, p, B, 0, oracle_=True)
                elif exp == "examples":
                    for M in args.budgets_examples:
                        _run(exp, seed, p, 0, M)
                        if args.add_supervised_oracle_p0 and abs(float(p)) < 1e-12:
                            _run(exp, seed, p, 0, M, oracle_=True)
                elif exp == "all":
                    _run(exp, seed, p, 0, 0)
                    if args.add_supervised_oracle_p0 and abs(float(p)) < 1e-12:
                        _run(exp, seed, p, 0, 0, oracle_=True)

    # Write CSV log.
    log_path   = os.path.join(out_dir, "connectivity_experiments_log.txt")
    csv_header = (
        "exp,seed,p,p10,num_nodes,connection_model,similarity_temp,"
        "budget_steps,budget_examples,rounds_ran,"
        "test_final,cn_local_final,train_final,pseudo_examples_per_round_mean"
    )
    _maybe_write_csv_header(log_path, csv_header)
    csv_lines = [
        f"{s.exp},{s.seed},{s.p:.3f},{s.p10},{args.num_nodes},"
        f"{args.connection_model},{args.similarity_temp:.4f},"
        f"{s.budget_steps},{s.budget_examples},{s.rounds_ran},"
        f"{s.test_final:.4f},{s.cn_local_final:.4f},{s.train_final:.4f},"
        f"{s.pseudo_examples_per_round_mean:.1f}"
        for s in summaries
    ]
    _append_csv_lines(log_path, csv_lines)

    # Aggregate CI across seeds.
    agg: Dict[Tuple, List[RunSummary]] = defaultdict(list)
    for s in summaries:
        agg[(s.exp, s.p, s.budget_steps, s.budget_examples, s.is_oracle)].append(s)

    _pprint("\n[CI] Final metrics across seeds:", no_flush=args.no_flush)
    _pprint(
        f"  {'exp':7s}  {'budget':10s}  {'p':>4}  {'type':>6}  {'n':>2}  "
        f"{'test (primary)':>18}  {'self':>18}  {'cn_local (oracle)':>22}",
        no_flush=args.no_flush,
    )
    for key in sorted(agg.keys(), key=lambda k: (k[0], int(k[4]), k[1], k[2], k[3])):
        lst               = agg[key]
        exp, p, bs, be, is_oracle = key
        btag  = f"steps{bs}" if exp == "steps" else f"ex{be}" if exp == "examples" else "allcap"
        tag2  = "oracle" if is_oracle else "normal"
        mu_t, hw_t = _mean_ci95([x.test_final     for x in lst])
        mu_s, hw_s = _mean_ci95([x.self_final     for x in lst])
        mu_c, hw_c = _mean_ci95([x.cn_local_final for x in lst])
        _pprint(
            f"  {exp:7s}  {btag:10s}  {p:>4.2f}  {tag2:>6}  {len(lst):>2}  "
            f"test={mu_t:.4f}±{hw_t:.4f}  self={mu_s:.4f}±{hw_s:.4f}  cn_local={mu_c:.4f}±{hw_c:.4f}",
            no_flush=args.no_flush,
        )

    _pprint(f"\nWrote CSV to: {log_path}", no_flush=args.no_flush)

    # Plots.
    def _btag(c):
        return (
            f"steps{c.budget_steps}" if c.exp == "steps" else
            f"ex{c.budget_examples}" if c.exp == "examples" else
            "allcap"
        )

    groups: Dict[Tuple, List[CurveRecord]] = defaultdict(list)
    for c in curves:
        groups[(c.exp, c.seed, _btag(c))].append(c)
    ts_str = time.strftime("%Y%m%d_%H%M%S")

    for (exp, seed, btag), group in groups.items():
        group_sorted = sorted(group, key=lambda x: (int(x.is_oracle), x.p))
        for metric_name, getter in [
            ("test",     lambda c: c.test),
            ("cn_local", lambda c: c.cn_local),
            ("train",    lambda c: c.train),
        ]:
            plt.figure()
            for c in group_sorted:
                m = c.eval_mask
                x = c.rounds[m]
                y = getter(c)[m]
                if c.is_oracle:
                    line, = plt.plot(x, y, linestyle="--", label="oracle")
                else:
                    line, = plt.plot(x, y, label=f"p={c.p:.2f}")
                plt.axvline(
                    x=float(c.pretrain_rounds_ran) + 0.5,
                    linestyle="--", linewidth=1.0, color=line.get_color(), alpha=0.25,
                )
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            title_suffix = " [PRIMARY]" if metric_name == "test" else ""
            plt.title(
                f"{metric_name}{title_suffix} | {args.num_nodes} nodes | "
                f"conn={args.connection_model} | seed={seed}"
            )
            plt.grid(True); plt.legend(); plt.tight_layout()
            path = os.path.join(
                out_dir, f"curve_{metric_name}_{exp}_seed{seed}_{btag}_{ts_str}.png"
            )
            plt.savefig(path, dpi=200); plt.close()
            _pprint(f"Saved: {path}", no_flush=args.no_flush)


if __name__ == "__main__":
    main()