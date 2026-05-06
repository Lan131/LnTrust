#!/usr/bin/env python3
"""
Learned Neighbor Trust for Decentralized Pseudo-Label Learning.

Each node learns a trust function w(i,j) that estimates how useful
neighbor j's predictions are for node i.  The trust function is a
small MLP that takes features describing the (i,j) relationship:

    Features:
      - Node i's class distribution           (C dims)
      - Neighbor j's class distribution        (C dims)
      - Distribution overlap min(w_i, w_j)     (C dims)
      - j's per-class accuracy on i's val      (C dims)
      - j's overall val accuracy on i's val    (1 dim)
      - Degree of i, degree of j               (2 dims)

The trust model is optimised to minimise ensemble CE on held-out
validation data.  At pseudo-label time, for each unlabeled example:

    y* = argmax  sum_j  w(i,j) * p_j(x)
    L_pseudo = CE(f_i(x), y*)

Hard labels, plain CE.  No temperature, no KL, no entropy gates.
The trust model is the entire contribution.

Usage:
    python3 learned_trust.py \
        --seed_list 0 1 2 --pretrain_max_rounds 50 --max_rounds 200 \
        --mobilenet_cache_path /path/to/mnv2.pt \
        --efficientnet_cache_path /path/to/efnet.pt

-----------------------------------------------------------------------
Revision history:
  2026-04-24: Exposed --conf_threshold as a CLI hyperparameter to enable
              the tau ablation (Table 8 in the paper). Previously the
              pseudo-label confidence filter used a hardcoded rule
              (max(0.2, 2/C) if C>10 else 1/C+0.1) inside
              trust_pseudo_round with no way to override it. The new
              flag threads through run_experiment() into that function;
              when unset the default rule is preserved byte-for-byte so
              pre-existing runs reproduce. Set to 0.0 to disable the
              filter entirely.
  2026-04-30: Replaced oracle _node_skew_weights[i] with empirical w_i
              computed from val_labels[i] everywhere w_i is used as a
              trust feature or weighting factor (build_trust_features,
              build_coarse_trust_features, trust gate, importance
              weighting in distillation). A node does not know the data-
              generating skew parameters; it only has its labeled samples.
              Val labels are protocol-correct (the node holds them for
              probing), prevent training-data overfit, and are the right
              source of distributional information for all trust decisions.
              Fallback to _node_skew_weights retained when val_labels is
              unavailable (first call before probing). _draw_fresh_val
              keeps _node_skew_weights for sampling weights (data
              generation, not model-facing). EuroSAT geo cache is unaffected
              since it already sets _node_skew_weights from real counts.
-----------------------------------------------------------------------
"""

import argparse
import copy
import os
import sys
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as exp

# Module-level flag: drop entropy(w_i) and deg_i from coarse trust features
# These are constant across arms for a given node, so they cancel in softmax.
_DROP_SELF_ENT_DEG = True


# =====================================================================
#  w_i helper — empirical from val labels, not oracle skew params
# =====================================================================

def _get_w_i(system, i, val_labels=None):
    """Return node i's empirical class distribution.

    2026-04-30: Estimated from the node's full labeled allocation
    (train + val indices combined) rather than the oracle skew parameters.
    Using the combined set gives the largest possible sample for the
    estimate, reducing variance in the noisy-w_i regime without
    introducing any protocol violation — the node holds all these labels.
    Minor circularity: val labels feed both w_i estimation and the trust
    model objective, but w_i is a pure counting statistic so there is no
    overfitting path.

    val_labels accepted but unused; kept for call-site compatibility.
    Fallback to _node_skew_weights when no indices are available.
    """
    C = system.num_classes
    arch = system._node_arch_map.get(i, system.arch)
    cache = system._feat_cache_by_arch.get(arch, {})
    lb_full = cache.get("labs_train")
    train_idx = system._stored_train_idx.get(i, [])
    val_idx   = system._stored_val_idx.get(i, [])
    all_idx   = list(train_idx) + list(val_idx)
    if lb_full is not None and len(all_idx) > 0:
        idx_t = torch.as_tensor(all_idx, dtype=torch.long)
        y = lb_full[idx_t].numpy()
        counts = np.bincount(y, minlength=C).astype(np.float64)
        counts += 1e-8
        return counts / counts.sum()
    return system._node_skew_weights[i]


# =====================================================================
#  Trust Feature Extraction
# =====================================================================

@torch.no_grad()
def compute_probe_responses(system, include_self=True):
    """For every (i,j) pair where j is neighbor of i (or i itself),
    compute j's per-class accuracy on i's val set.

    Returns:
        probe: {i: {j: np.array of shape (C,)}}  per-class accuracy
        val_preds: {i: {j: Tensor (N_val, C)}}    raw softmax preds
        val_labels: {i: Tensor (N_val,)}           ground truth
    """
    device = system.device
    C = system.num_classes
    probe = {}
    all_preds = {}
    all_labels = {}

    for i, node in system.nodes.items():
        val_idx = system._val_idx_per_node.get(i)
        if val_idx is None or val_idx.numel() == 0:
            continue

        any_cache = next(iter(system._feat_cache_by_arch.values()))
        y_val = any_cache["labs_train"][val_idx].to(device)
        all_labels[i] = y_val

        nbrs = list(node.neighbor_ids) if node.neighbor_ids else []
        candidates = ([i] if include_self else []) + nbrs

        probe[i] = {}
        all_preds[i] = {}

        for j in candidates:
            j_arch = system._node_arch_map.get(j, system.arch)
            j_cache = system._feat_cache_by_arch.get(j_arch, {})
            ft_full = j_cache.get("feats_train")
            if ft_full is None:
                continue

            z_val = ft_full[val_idx].to(device)
            system.nodes[j].model.eval()

            preds_j = []
            for s in range(0, z_val.size(0), system.batch_size):
                zb = z_val[s:s + system.batch_size]
                logits = system.nodes[j].model.forward_head(zb)
                preds_j.append(F.softmax(logits, dim=-1))
            preds_j = torch.cat(preds_j, dim=0)
            all_preds[i][j] = preds_j

            # Per-class accuracy
            pred_classes = preds_j.argmax(dim=1)
            pc_acc = np.zeros(C, dtype=np.float64)
            y_np = y_val.cpu().numpy()
            pred_np = pred_classes.cpu().numpy()
            for c in range(C):
                mask = (y_np == c)
                if mask.sum() > 0:
                    pc_acc[c] = float((pred_np[mask] == c).mean())
            probe[i][j] = pc_acc

    return probe, all_preds, all_labels


@torch.no_grad()
def estimate_neighbor_distributions(system):
    """Estimate each node's class distribution from its predictions
    on IID unlabeled data.  No private data access needed — just
    count argmax predictions on a shared pool.

    Returns {j: np.array (C,)} estimated class distribution.
    """
    device = system.device
    C = system.num_classes
    bufs = system._build_unlabeled_bufs_if_needed()
    estimated = {}

    for j, node in system.nodes.items():
        # Use any available unlabeled buffer (IID pool)
        # Pick a neighbor's buffer or own buffer — IID so doesn't matter
        buf = bufs.get(j)
        if buf is None or buf.n <= 0:
            # Try any buffer
            for b in bufs.values():
                if b.n > 0:
                    buf = b
                    break
        if buf is None or buf.n <= 0:
            estimated[j] = np.full(C, 1.0 / C, dtype=np.float64)
            continue

        n_sample = min(500, buf.n)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(system.seed * 77_777 + j)
        perm = buf.make_perm(generator=gen)[:n_sample]
        z = buf.x[perm].to(device)

        node.model.eval()
        all_preds = []
        for s in range(0, z.size(0), system.batch_size):
            zb = z[s:s + system.batch_size]
            if system.cache_features:
                logits = node.model.forward_head(zb)
            else:
                logits = node.model(zb)
            all_preds.append(logits.argmax(dim=1).cpu())

        pred_classes = torch.cat(all_preds).numpy()
        counts = np.bincount(pred_classes, minlength=C).astype(np.float64)
        counts += 1e-8  # smoothing
        estimated[j] = counts / counts.sum()

    return estimated


def build_trust_features(system, probe, estimated_dists, i, j,
                         coarse=False, drop_self_ent_deg=None,
                         val_labels=None):
    """Build feature vector for the (i, j) trust prediction."""
    if coarse:
        return build_coarse_trust_features(system, probe, estimated_dists, i, j,
                                           drop_self_ent_deg=drop_self_ent_deg,
                                           val_labels=val_labels)
    C = system.num_classes
    w_i = _get_w_i(system, i, val_labels)          # (C,) — empirical from val
    w_j = estimated_dists.get(j, np.full(C, 1.0/C))  # (C,) — estimated
    overlap = np.minimum(w_i, w_j)                # (C,)
    pc_acc = probe.get(i, {}).get(j, np.zeros(C))  # (C,)
    overall_acc = float(pc_acc.mean())
    deg_i = len(system.nodes[i].neighbor_ids) if system.nodes[i].neighbor_ids else 0
    deg_j = len(system.nodes[j].neighbor_ids) if system.nodes[j].neighbor_ids else 0

    feat = np.concatenate([
        w_i,                    # C
        w_j,                    # C
        overlap,                # C
        pc_acc,                 # C
        [overall_acc],          # 1
        [deg_i / 50.0],         # 1  (normalised)
        [deg_j / 50.0],         # 1
    ])
    return feat.astype(np.float32)


def build_coarse_trust_features(system, probe, estimated_dists, i, j,
                                drop_self_ent_deg=None, val_labels=None):
    """Aggregate trust features for large C (e.g. CIFAR-100).

    Instead of 4C per-class features, uses 8 (or 6) scalar summaries:
      1. Entropy of w_i (how specialized is node i)  [dropped if drop_self_ent_deg]
      2. Entropy of w_j (how specialized is neighbor j)
      3. Total overlap: sum(min(w_i, w_j))
      4. Overall probe accuracy: mean(pc_acc)
      5. Weighted probe accuracy: sum(w_i * pc_acc)  (acc on classes i cares about)
      6. KL(w_i || w_j)  (distribution mismatch)
      7. deg_i / n  [dropped if drop_self_ent_deg]
      8. deg_j / n

    Returns np.array of shape (6,) or (8,)
    """
    if drop_self_ent_deg is None:
        drop_self_ent_deg = _DROP_SELF_ENT_DEG
    C = system.num_classes
    w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical from val
    w_j = estimated_dists.get(j, np.full(C, 1.0/C))
    pc_acc = probe.get(i, {}).get(j, np.zeros(C))
    deg_i = len(system.nodes[i].neighbor_ids) if system.nodes[i].neighbor_ids else 0
    deg_j = len(system.nodes[j].neighbor_ids) if system.nodes[j].neighbor_ids else 0

    # Entropy (normalized to [0,1])
    def _ent(p):
        p = np.clip(p, 1e-10, 1.0)
        return -float(np.sum(p * np.log(p))) / max(1e-8, np.log(C))

    if drop_self_ent_deg:
        feat = np.array([
            _ent(w_j),                                  # how specialized is j
            float(np.minimum(w_i, w_j).sum()),          # total overlap
            float(pc_acc.mean()),                        # overall probe acc
            float((w_i * pc_acc).sum()),                 # weighted probe acc
            float(np.sum(w_i * np.log(w_i / np.clip(w_j, 1e-10, 1.0)))),  # KL
            deg_j / 50.0,
        ], dtype=np.float32)
    else:
        feat = np.array([
            _ent(w_i),                                  # how specialized is i
            _ent(w_j),                                  # how specialized is j
            float(np.minimum(w_i, w_j).sum()),          # total overlap
            float(pc_acc.mean()),                        # overall probe acc
            float((w_i * pc_acc).sum()),                 # weighted probe acc
            float(np.sum(w_i * np.log(w_i / np.clip(w_j, 1e-10, 1.0)))),  # KL
            deg_i / 50.0,
            deg_j / 50.0,
        ], dtype=np.float32)
    return feat


# =====================================================================
#  Trust Model
# =====================================================================

class TrustModel(nn.Module):
    """Small MLP: features(i,j) → scalar trust weight."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: (B, D) → (B,) raw logits (no softmax — applied externally)"""
        return self.net(x).squeeze(-1)



# =====================================================================
#  Hedge (Multiplicative Weights) Baseline
# =====================================================================

class HedgeWeights:
    def __init__(self, neighbor_map, eta=0.5, include_self=True):
        self.eta = eta
        self.include_self = include_self
        self.log_w = {}
        for i, nbrs in neighbor_map.items():
            arms = ([i] if include_self else []) + list(nbrs)
            self.log_w[i] = {j: 0.0 for j in arms}

    def update_on_val(self, node_id, probe):
        if node_id not in self.log_w:
            return
        pc_acc = probe.get(node_id, {})
        for j in self.log_w[node_id]:
            if j in pc_acc:
                loss_j = 1.0 - float(pc_acc[j].mean())
                self.log_w[node_id][j] -= self.eta * loss_j

    def get_weights(self, node_id, include_self=None):
        lw = self.log_w.get(node_id, {})
        if not lw:
            return {}
        if include_self is not None:
            lw = {j: v for j, v in lw.items()
                  if include_self or j != node_id}
        if not lw:
            return {}
        max_lw = max(lw.values())
        exp_w = {j: np.exp(v - max_lw) for j, v in lw.items()}
        total = sum(exp_w.values())
        return {j: v / total for j, v in exp_w.items()}


class UniformWeights:
    def __init__(self, neighbor_map, include_self=True):
        self.include_self = include_self
        self.neighbor_map = neighbor_map

    def get_weights(self, node_id, include_self=None):
        inc = include_self if include_self is not None else self.include_self
        nbrs = list(self.neighbor_map.get(node_id, []))
        arms = ([node_id] if inc else []) + nbrs
        if not arms:
            return {}
        w = 1.0 / len(arms)
        return {j: w for j in arms}


class ValOptWeights:
    """Validation-optimized weights via gradient descent on val CE.

    Recomputes per-node simplex weights by running Adam on val predictions
    each time update_on_val is called.  Used for both training-time
    distillation and deployment-time ensemble (Algorithm 3 everywhere).
    """
    def __init__(self, neighbor_map, val_preds, val_labels, device,
                 lr=0.05, steps=300, tau=0.01, include_self=True):
        self.neighbor_map = neighbor_map
        self.val_preds = val_preds
        self.val_labels = val_labels
        self.device = device
        self.lr = lr
        self.steps = steps
        self.tau = tau
        self.include_self = include_self
        self._cached = {}  # {node_id: {j: weight}}

    def update_on_val(self, node_id, probe=None):
        """Recompute val-optimized weights for node_id."""
        nbrs = list(self.neighbor_map.get(node_id, []))
        hood = ([node_id] if self.include_self else []) + nbrs
        preds_i = {}
        for j in hood:
            if j in self.val_preds.get(node_id, {}):
                preds_i[j] = self.val_preds[node_id][j].to(self.device)
        y = self.val_labels.get(node_id)
        if y is None or len(y) == 0 or not preds_i:
            self._cached[node_id] = {}
            return
        y = y.to(self.device)
        # Import here to avoid circular ref at class definition time
        self._cached[node_id] = val_optimized_deploy_weights(
            preds_i, y, hood, self.device,
            lr=self.lr, steps=self.steps, tau=self.tau)

    def update_all(self):
        for i in self.neighbor_map:
            self.update_on_val(i)

    def get_weights(self, node_id, include_self=None):
        w = self._cached.get(node_id, {})
        if not w:
            return {}
        if include_self is not None and not include_self:
            w = {j: v for j, v in w.items() if j != node_id}
            total = sum(w.values())
            if total > 1e-8:
                w = {j: v / total for j, v in w.items()}
        return w


def train_trust_model(
    system, trust_model, probe, estimated_dists, val_preds, val_labels,
    device, lr=0.01, steps=200, include_self=True, coarse_trust=False,
    iw_trust=False,
):
    """Train the trust model to minimise ensemble CE on val data.

    For each node i, the ensemble prediction is:
        p_ens(x) = sum_j  softmax(trust(feat_ij))_j  *  p_j(x)
    Loss = CE(p_ens, y_val) averaged across all nodes.

    If iw_trust=True, uses distribution-weighted CE:
        Loss = sum_c w_i^c * CE_c(p_ens, y_val)
    This aligns the trust training objective with the deployment gate,
    which compares distribution-weighted accuracy.
    """
    optimizer = torch.optim.Adam(trust_model.parameters(), lr=lr)

    # Pre-compute feature tensors for all (i,j) pairs
    pair_data = []  # list of (i, arms, feat_tensor, preds_stack, y_val)
    for i in system.nodes:
        if i not in val_preds or i not in val_labels:
            continue
        nbrs = list(system.nodes[i].neighbor_ids) if system.nodes[i].neighbor_ids else []
        arms = ([i] if include_self else []) + nbrs
        arms = [j for j in arms if j in val_preds.get(i, {})]
        if len(arms) < 1:
            continue

        feats = []
        preds = []
        for j in arms:
            f = build_trust_features(system, probe, estimated_dists, i, j,
                                     coarse=coarse_trust, val_labels=val_labels)
            feats.append(f)
            preds.append(val_preds[i][j])

        feat_t = torch.tensor(np.stack(feats), dtype=torch.float32,
                              device=device)  # (A, D)
        preds_t = torch.stack(preds, dim=0)  # (A, N_val, C)
        y_t = val_labels[i]  # (N_val,)

        # Class weights for distribution-weighted CE
        cw_t = None
        if iw_trust:
            w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            cw_t = torch.tensor(w_i, dtype=torch.float32, device=device)

        pair_data.append((i, arms, feat_t, preds_t, y_t, cw_t))

    if not pair_data:
        return 0.0

    best_loss = float('inf')
    best_state = None

    for step in range(steps):
        total_loss = 0.0
        n_nodes = 0

        for i, arms, feat_t, preds_t, y_t, cw_t in pair_data:
            logits = trust_model(feat_t)
            w = F.softmax(logits, dim=0)

            p_ens = torch.einsum('a,anc->nc', w, preds_t)
            log_p = (p_ens + 1e-8).log()
            loss = F.nll_loss(log_p, y_t, weight=cw_t)

            total_loss += loss
            n_nodes += 1

        avg_loss = total_loss / max(1, n_nodes)
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        if avg_loss.item() < best_loss:
            best_loss = avg_loss.item()
            best_state = copy.deepcopy(trust_model.state_dict())

    # Restore best
    if best_state is not None:
        trust_model.load_state_dict(best_state)

    return best_loss


def train_trust_models_per_node(
    system, trust_models, probe, estimated_dists, val_preds, val_labels,
    device, lr=0.01, steps=200, include_self=True, coarse_trust=False,
    iw_trust=False,
):
    """Train one trust model per node on that node's val data only.

    trust_models: {node_id: TrustModel}
    Each node i's model is trained to minimise ensemble CE on i's val:
        min_θi  CE( Σ_j softmax(trust_i(feat_ij)) · p_j(x_val_i), y_val_i )
    If iw_trust=True, uses distribution-weighted CE.
    Fully decentralized — no data shared between nodes.
    """
    C = system.num_classes
    total_loss = 0.0
    n_trained = 0

    for i, model_i in trust_models.items():
        if i not in val_preds or i not in val_labels:
            continue
        nbrs = list(system.nodes[i].neighbor_ids) if system.nodes[i].neighbor_ids else []
        arms = ([i] if include_self else []) + nbrs
        arms = [j for j in arms if j in val_preds.get(i, {})]
        if len(arms) < 1:
            continue

        feats = []
        preds = []
        for j in arms:
            f = build_trust_features(system, probe, estimated_dists, i, j,
                                     coarse=coarse_trust, val_labels=val_labels)
            feats.append(f)
            preds.append(val_preds[i][j])

        feat_t = torch.tensor(np.stack(feats), dtype=torch.float32,
                              device=device)
        preds_t = torch.stack(preds, dim=0)
        y_t = val_labels[i]

        # Class weights for distribution-weighted CE
        cw_t = None
        if iw_trust:
            w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            cw_t = torch.tensor(w_i, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(model_i.parameters(), lr=lr)
        best_loss_i = float('inf')
        best_state_i = None

        for step in range(steps):
            logits = model_i(feat_t)
            w = F.softmax(logits, dim=0)
            p_ens = torch.einsum('a,anc->nc', w, preds_t)
            log_p = (p_ens + 1e-8).log()
            loss = F.nll_loss(log_p, y_t, weight=cw_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss_i:
                best_loss_i = loss.item()
                best_state_i = copy.deepcopy(model_i.state_dict())

        if best_state_i is not None:
            model_i.load_state_dict(best_state_i)

        total_loss += best_loss_i
        n_trained += 1

    return total_loss / max(1, n_trained)


# =====================================================================
#  Get learned weights for a specific node
# =====================================================================

@torch.no_grad()
def get_trust_weights(system, trust_model, probe, estimated_dists,
                      node_id, device, include_self=True,
                      hedge_boost=None, coarse_trust=False,
                      val_labels=None):
    """Return {j: weight} using the trust model for node i."""
    if isinstance(trust_model, (HedgeWeights, UniformWeights, ValOptWeights)):
        return trust_model.get_weights(node_id, include_self=include_self)

    if isinstance(trust_model, dict):
        model = trust_model.get(node_id)
        if model is None:
            return {}
    else:
        model = trust_model

    node = system.nodes[node_id]
    nbrs = list(node.neighbor_ids) if node.neighbor_ids else []
    arms = ([node_id] if include_self else []) + nbrs

    if not arms:
        return {}

    feats = []
    valid_arms = []
    for j in arms:
        f = build_trust_features(system, probe, estimated_dists, node_id, j,
                                 coarse=coarse_trust, val_labels=val_labels)
        feats.append(f)
        valid_arms.append(j)

    if not valid_arms:
        return {}

    feat_t = torch.tensor(np.stack(feats), dtype=torch.float32,
                          device=device)
    logits = model(feat_t)

    if hedge_boost is not None:
        lw = hedge_boost.log_w.get(node_id, {})
        boost = torch.tensor(
            [lw.get(j, 0.0) for j in valid_arms],
            dtype=torch.float32, device=device)
        logits = logits + boost
    w = F.softmax(logits, dim=0)
    return dict(zip(valid_arms, w.cpu().tolist()))


@torch.no_grad()
def trust_feature_importance(system, trust_model, probe, estimated_dists,
                             val_preds, val_labels, device,
                             coarse_trust=True, n_repeats=10):
    """Permutation importance for the 8 coarse trust features.

    For each feature dimension, shuffles it across all (i,j) pairs
    and measures the increase in val CE loss.  Higher = more important.

    Returns list of (feature_name, mean_importance, std_importance).
    """
    feat_names_full = [
        "entropy(w_i)",      # 0: how specialized is node i
        "entropy(w_j)",      # 1: how specialized is neighbor j
        "overlap",           # 2: sum(min(w_i, w_j))
        "probe_acc",         # 3: mean per-class probe accuracy
        "wtd_probe_acc",     # 4: distribution-weighted probe accuracy
        "KL(w_i||w_j)",      # 5: distribution mismatch
        "deg_i",             # 6: normalized degree of i
        "deg_j",             # 7: normalized degree of j
    ]
    feat_names_drop = [
        "entropy(w_j)",      # 0: how specialized is neighbor j
        "overlap",           # 1: sum(min(w_i, w_j))
        "probe_acc",         # 2: mean per-class probe accuracy
        "wtd_probe_acc",     # 3: distribution-weighted probe accuracy
        "KL(w_i||w_j)",      # 4: distribution mismatch
        "deg_j",             # 5: normalized degree of j
    ]
    feat_names = feat_names_drop if _DROP_SELF_ENT_DEG else feat_names_full

    if not isinstance(trust_model, dict):
        _log = getattr(system, '_pprint', print)
        _log("[FEAT_IMP] Skipping — only implemented for per-node trust models")
        return []

    # Collect features and val data per node
    node_data = {}
    for i in system.nodes:
        model_i = trust_model.get(i)
        if model_i is None:
            continue
        if i not in val_preds or i not in val_labels:
            continue
        node = system.nodes[i]
        nbrs = list(node.neighbor_ids) if node.neighbor_ids else []
        arms = [i] + nbrs
        arms = [j for j in arms if j in val_preds.get(i, {})]
        if len(arms) < 2:
            continue

        feats = []
        for j in arms:
            f = build_trust_features(system, probe, estimated_dists, i, j,
                                     coarse=coarse_trust, val_labels=val_labels)
            feats.append(f)
        feat_t = torch.tensor(np.stack(feats), dtype=torch.float32,
                              device=device)

        P = torch.stack([val_preds[i][j].to(device) for j in arms], dim=0)
        y = val_labels[i].to(device)

        node_data[i] = {
            "model": model_i, "feats": feat_t, "arms": arms,
            "P": P, "y": y,
        }

    if not node_data:
        return []

    def _compute_loss(nd, feat_override=None):
        """CE loss for node using given features."""
        f = feat_override if feat_override is not None else nd["feats"]
        logits = nd["model"](f)
        w = F.softmax(logits, dim=0)
        q = torch.einsum("k,knc->nc", w, nd["P"])
        q_y = q[torch.arange(q.size(0), device=device), nd["y"]]
        return -q_y.clamp(min=1e-8).log().mean().item()

    # Baseline loss
    baseline = np.mean([_compute_loss(nd) for nd in node_data.values()])

    # Permutation importance per feature
    n_feat = len(feat_names)
    importances = np.zeros((n_feat, n_repeats))
    rng = np.random.default_rng(42)

    for d in range(n_feat):
        for r in range(n_repeats):
            losses = []
            for nd in node_data.values():
                feat_perm = nd["feats"].clone()
                perm_idx = torch.tensor(
                    rng.permutation(feat_perm.size(0)),
                    device=device)
                feat_perm[:, d] = feat_perm[perm_idx, d]
                losses.append(_compute_loss(nd, feat_perm))
            importances[d, r] = np.mean(losses) - baseline

    results = []
    for d in range(n_feat):
        results.append((
            feat_names[d],
            float(importances[d].mean()),
            float(importances[d].std()),
        ))

    # Sort by importance descending
    results.sort(key=lambda x: -x[1])
    return results


# =====================================================================
#  Helpers (from ensemble_experiment.py)
# =====================================================================

def global_test_accuracy(system):
    acc = system.evaluate_all_nodes_on_test()
    return float(np.mean(list(acc.values()))) if acc else 0.0

def global_train_accuracy(system):
    accs = [n.evaluate_accuracy(n.train_eval_loader)
            for n in system.nodes.values()]
    return float(np.mean(accs)) if accs else 0.0


def _unmerge_val_from_train(system):
    """Rebuild train loaders WITHOUT val data for Stage 2.

    Stage 1 used baseline_merge_val=True, so train included val.
    Now remove val so that:
      1) Probe responses evaluate on data not recently trained on
      2) Stage 2 supervised steps use only train (the fair penalty)
      3) Val loaders (already created at init) remain for trust
    """
    for i, node in system.nodes.items():
        arch = system._node_arch_map.get(i, system.arch)
        cache = system._feat_cache_by_arch.get(arch, {})
        ft_full = cache.get("feats_train")
        lb_full = cache.get("labs_train")
        if ft_full is None or lb_full is None:
            continue

        train_idx = system._stored_train_idx.get(i, [])
        if not train_idx:
            continue

        train_t = torch.as_tensor(train_idx, dtype=torch.long)
        train_ds = exp.FeatureTensorDataset(
            ft_full[train_t], lb_full[train_t])

        node.train_loader = DataLoader(
            train_ds, batch_size=system.batch_size, shuffle=True,
            num_workers=0, pin_memory=True)
        node.train_eval_loader = DataLoader(
            train_ds, batch_size=system.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)
        node._train_iter = None  # force iterator reset

    exp._pprint(f"[VAL_UNMERGE] Rebuilt train loaders without val "
                f"for {len(system.nodes)} nodes")


def _draw_fresh_val(system, val_fraction, min_val_size=200):
    """Draw fresh val from unused pool — never trained on."""
    vf = float(val_fraction)
    if vf <= 0:
        return
    n_drawn = 0
    for i, node in system.nodes.items():
        arch = system._node_arch_map.get(i, system.arch)
        cache = system._feat_cache_by_arch.get(arch, {})
        ft_full = cache.get("feats_train")
        lb_full = cache.get("labs_train")
        if ft_full is None or lb_full is None:
            continue
        train_idx = system._stored_train_idx.get(i, [])
        val_idx = system._stored_val_idx.get(i, [])
        test_idx = system._stored_test_idx.get(i, [])
        used = set(train_idx) | set(val_idx) | set(test_idx)
        unlab_idx = system._unlabeled_idx_per_node.get(i)
        if unlab_idx is not None:
            used |= set(unlab_idx.tolist())
        pool_size = len(ft_full)
        available = np.array([idx for idx in range(pool_size)
                              if idx not in used])
        if len(available) < 2:
            continue
        n_train = len(train_idx) + len(val_idx)
        n_val = max(min_val_size, int(round(n_train * vf)))
        n_val = min(n_val, len(available))
        targets = lb_full.numpy()
        class_probs = system._node_skew_weights[i]
        idx_probs = class_probs[targets[available]]
        idx_probs = idx_probs / idx_probs.sum()
        rng = np.random.default_rng(system.seed * 10_000 + i + 999_999)
        chosen = rng.choice(available, size=n_val, replace=False, p=idx_probs)
        chosen_t = torch.as_tensor(chosen, dtype=torch.long)
        val_ds = exp.FeatureTensorDataset(ft_full[chosen_t], lb_full[chosen_t])
        node.val_loader = DataLoader(
            val_ds, batch_size=system.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)
        system._val_idx_per_node[i] = chosen_t
        system._stored_val_idx[i] = chosen.tolist()
        n_drawn += 1
    exp._pprint(
        f"[FRESH_VAL] Drew fresh val for {n_drawn} nodes "
        f"(min_size={min_val_size})")


@torch.no_grad()
def compute_cn_local(system):
    oracle_map = {}
    for i, node in system.nodes.items():
        loader = system._node_test_loaders.get(i)
        if loader is None:
            oracle_map[i] = (i, 0.0)
            continue
        hood = [i] + (list(node.neighbor_ids) if node.neighbor_ids else [])
        best_j, best_acc = i, 0.0
        for j in hood:
            j_arch = system._node_arch_map.get(j, system.arch)
            i_arch = system._node_arch_map.get(i, system.arch)
            if j == i or j_arch == i_arch:
                acc = system.nodes[j].evaluate_accuracy(loader)
            else:
                acc = system.evaluate_j_on_i_loader(
                    j, i, loader, system._test_idx_per_node)
            if acc > best_acc:
                best_acc = acc
                best_j = j
        oracle_map[i] = (best_j, best_acc)
    cn_local = float(np.mean([v[1] for v in oracle_map.values()]))
    return cn_local, oracle_map


def _top_k_weights(weights, k):
    """Keep only the top-k entries by weight and renormalize."""
    if k <= 0 or len(weights) <= k:
        return weights
    sorted_arms = sorted(weights.items(), key=lambda x: -x[1])
    top = dict(sorted_arms[:k])
    total = sum(top.values())
    if total < 1e-8:
        return top
    return {j: w / total for j, w in top.items()}


@torch.no_grad()
def deployed_ensemble_accuracy(system, trust_model, probe, estimated_dists,
                               device, include_self=True, hedge_boost=None,
                               coarse_trust=False, trust_gate=False,
                               top_k_teachers=0, deploy_log=None,
                               val_labels=None):
    """Test accuracy using trust-weighted ensemble at deployment.
    If trust_gate=True, nodes where the ensemble can't beat self on val
    fall back to self-only predictions.
    If deploy_log is a list, per-node deployment info is appended."""
    C = system.num_classes
    accs = []
    for i, node in system.nodes.items():
        test_idx = system._test_idx_per_node.get(i)
        if test_idx is None or test_idx.numel() == 0:
            continue

        weights = get_trust_weights(
            system, trust_model, probe, estimated_dists, i, device,
            include_self=include_self, hedge_boost=hedge_boost,
            coarse_trust=coarse_trust, val_labels=val_labels)
        if not weights:
            self_test = node.evaluate_accuracy(
                system._node_test_loaders.get(i))
            accs.append(self_test)
            if deploy_log is not None:
                deploy_log.append(dict(node=i, mode="SELF(no_wt)",
                    a_self=0, a_ens=0, test_acc=self_test))
            continue

        if top_k_teachers > 0:
            weights = _top_k_weights(weights, top_k_teachers)

        # Trust gate: if ensemble can't beat self on val (weighted by node's
        # class distribution), use self only
        a_self_val = 0.0
        a_ens_val = 0.0
        gate_closed = False
        if trust_gate:
            w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            self_probe = probe.get(i, {}).get(i, np.zeros(C))
            a_self_val = float((w_i * self_probe).sum())
            a_ens_val = sum(
                w_j * float((w_i * probe.get(i, {}).get(j, np.zeros(C))).sum())
                for j, w_j in weights.items())
            if a_ens_val < a_self_val:
                self_test = node.evaluate_accuracy(
                    system._node_test_loaders.get(i))
                accs.append(self_test)
                if deploy_log is not None:
                    deploy_log.append(dict(node=i, mode="SELF",
                        a_self=a_self_val, a_ens=a_ens_val,
                        test_acc=self_test))
                gate_closed = True
        if gate_closed:
            continue

        any_cache = next(iter(system._feat_cache_by_arch.values()))
        y_test = any_cache["labs_train"][test_idx].to(device)

        p_ensemble = torch.zeros(len(test_idx), system.num_classes,
                                 device=device)
        for j, w_j in weights.items():
            if w_j < 1e-6:
                continue
            j_arch = system._node_arch_map.get(j, system.arch)
            j_cache = system._feat_cache_by_arch.get(j_arch, {})
            ft_full = j_cache.get("feats_train")
            if ft_full is None:
                continue
            z_j = ft_full[test_idx].to(device)
            system.nodes[j].model.eval()
            for s in range(0, z_j.size(0), system.batch_size):
                zb = z_j[s:s + system.batch_size]
                p_j = F.softmax(
                    system.nodes[j].model.forward_head(zb), dim=-1)
                p_ensemble[s:s + zb.size(0)] += w_j * p_j

        p_ensemble /= p_ensemble.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        correct = (p_ensemble.argmax(dim=1) == y_test).sum().item()
        ens_test = correct / max(1, len(y_test))
        accs.append(ens_test)
        if deploy_log is not None:
            self_w = weights.get(i, 0.0)
            deploy_log.append(dict(node=i, mode="ENS",
                a_self=a_self_val, a_ens=a_ens_val,
                test_acc=ens_test, self_w=self_w))

    return float(np.mean(accs)) if accs else 0.0


# =====================================================================
#  Validation-optimized collaborative deployment (Algorithm 3)
# =====================================================================

@torch.no_grad()
def _collect_val_preds_for_deploy(system, val_preds, val_labels, i, device):
    """Gather cached val predictions for all j in closed neighborhood."""
    hood = [i] + [j for j in (system.nodes[i].neighbor_ids or [])]
    preds = {}
    for j in hood:
        if j in val_preds.get(i, {}):
            preds[j] = val_preds[i][j].to(device)
    y = val_labels[i].to(device) if i in val_labels else None
    return preds, y, hood


@torch.enable_grad()
def val_optimized_deploy_weights(
    val_preds_i, y_val, hood, device,
    lr=0.05, steps=300, eps_dep=1e-8, tau=0.01,
):
    """Solve for optimal simplex weights by minimizing clipped CE on val.

    Args:
        val_preds_i: dict {j: Tensor (N, C)} softmax predictions
        y_val: Tensor (N,) ground truth labels
        hood: list of node ids in closed neighborhood
        eps_dep: clipping floor for log
        tau: L2 regularization on weights
    Returns:
        dict {j: float} optimized deployment weights
    """
    arms = [j for j in hood if j in val_preds_i]
    if not arms or y_val is None or len(y_val) == 0:
        return {}

    # Stack predictions: (K, N, C)
    K = len(arms)
    P = torch.stack([val_preds_i[j] for j in arms], dim=0)  # (K, N, C)
    N = P.size(1)

    # Learnable logits
    logits = torch.zeros(K, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    best_loss = float("inf")
    best_logits = logits.detach().clone()

    for step in range(steps):
        optimizer.zero_grad()
        alpha = F.softmax(logits, dim=0)  # (K,)

        # Weighted ensemble: (N, C)
        q = torch.einsum("k,knc->nc", alpha, P)

        # Clipped log-likelihood
        q_y = q[torch.arange(N, device=device), y_val]  # (N,)
        q_y_clipped = q_y.clamp(min=eps_dep)
        nll = -q_y_clipped.log().mean()

        # L2 regularization
        reg = tau * (alpha * alpha).sum()

        loss = nll + reg
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_logits = logits.detach().clone()

    # Extract final weights
    with torch.no_grad():
        alpha_final = F.softmax(best_logits, dim=0)
    return {arms[k]: float(alpha_final[k]) for k in range(K)}


@torch.no_grad()
def val_optimized_deploy_accuracy(
    system, val_preds, val_labels, device,
    lr=0.05, steps=300, eps_dep=1e-8, tau=0.01,
    trust_gate=False, probe=None,
):
    """Test accuracy using validation-optimized ensemble weights."""
    C = system.num_classes
    accs = []
    for i, node in system.nodes.items():
        test_idx = system._test_idx_per_node.get(i)
        if test_idx is None or test_idx.numel() == 0:
            continue

        preds_i, y_val, hood = _collect_val_preds_for_deploy(
            system, val_preds, val_labels, i, device)
        if not preds_i or y_val is None:
            accs.append(node.evaluate_accuracy(
                system._node_test_loaders.get(i)))
            continue

        weights = val_optimized_deploy_weights(
            preds_i, y_val, hood, device,
            lr=lr, steps=steps, eps_dep=eps_dep, tau=tau)
        if not weights:
            accs.append(node.evaluate_accuracy(
                system._node_test_loaders.get(i)))
            continue

        # Trust gate: compare val-optimized ensemble vs self
        if trust_gate and probe is not None:
            w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            self_probe = probe.get(i, {}).get(i, np.zeros(C))
            self_acc = float((w_i * self_probe).sum())
            ens_acc = sum(
                w_j * float((w_i * probe.get(i, {}).get(j, np.zeros(C))).sum())
                for j, w_j in weights.items())
            if ens_acc < self_acc:
                accs.append(node.evaluate_accuracy(
                    system._node_test_loaders.get(i)))
                continue

        # Compute test accuracy with optimized weights
        any_cache = next(iter(system._feat_cache_by_arch.values()))
        y_test = any_cache["labs_train"][test_idx].to(device)

        p_ensemble = torch.zeros(len(test_idx), C, device=device)
        for j, w_j in weights.items():
            if w_j < 1e-6:
                continue
            j_arch = system._node_arch_map.get(j, system.arch)
            j_cache = system._feat_cache_by_arch.get(j_arch, {})
            ft_full = j_cache.get("feats_train")
            if ft_full is None:
                continue
            z_j = ft_full[test_idx].to(device)
            system.nodes[j].model.eval()
            for s in range(0, z_j.size(0), system.batch_size):
                zb = z_j[s:s + system.batch_size]
                p_j = F.softmax(
                    system.nodes[j].model.forward_head(zb), dim=-1)
                p_ensemble[s:s + zb.size(0)] += w_j * p_j

        p_ensemble /= p_ensemble.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        correct = (p_ensemble.argmax(dim=1) == y_test).sum().item()
        accs.append(correct / max(1, len(y_test)))

    return float(np.mean(accs)) if accs else 0.0


# =====================================================================
#  Pseudo-label round with hard labels
# =====================================================================

def trust_pseudo_round(
    system, trust_model, probe, estimated_dists, round_idx,
    examples_per_round=1000, pseudo_weight=0.5,
    sup_steps_per_node=5, trust_update_freq=10,
    val_preds=None, val_labels=None,
    trust_lr=0.01, trust_steps=50,
    include_self_pseudo=False,
    soft_distil=False, soft_distil_alpha=0.3,
    hedge_boost=None,
    trust_gate=False,
    coarse_trust=False,
    top_k_teachers=0,
    # 2026-04-24: new kwarg for the tau ablation. None => preserve legacy
    # class-count-dependent rule exactly; a float overrides the rule.
    # See the Confidence filter block below for semantics.
    conf_threshold=None,
):
    """One round of hard-label pseudo-learning with learned trust."""
    device = system.device
    bufs = system._build_unlabeled_bufs_if_needed()
    stats = {}

    # ── Re-train trust model periodically ────────────────────────────
    if round_idx % trust_update_freq == 0:
        # Refresh probe responses (models have improved)
        probe_new, vp_new, vl_new = compute_probe_responses(
            system, include_self=True)
        probe.update(probe_new)
        if vp_new:
            val_preds.update(vp_new)
        if vl_new:
            val_labels.update(vl_new)

        # Refresh estimated distributions
        est_new = estimate_neighbor_distributions(system)
        estimated_dists.update(est_new)

        if isinstance(trust_model, HedgeWeights):
            for i in system.nodes:
                trust_model.update_on_val(i, probe)
            stats["_trust_loss"] = 0.0
        elif isinstance(trust_model, UniformWeights):
            stats["_trust_loss"] = 0.0
        elif isinstance(trust_model, ValOptWeights):
            trust_model.val_preds = val_preds
            trust_model.val_labels = val_labels
            trust_model.update_all()
            stats["_trust_loss"] = 0.0
        else:
            trust_loss = (train_trust_models_per_node if isinstance(trust_model, dict)
                          else train_trust_model)(
                system, trust_model, probe, estimated_dists,
                val_preds, val_labels,
                device, lr=trust_lr, steps=trust_steps,
                include_self=True, coarse_trust=coarse_trust)
            stats["_trust_loss"] = trust_loss

        if hedge_boost is not None:
            for i in system.nodes:
                hedge_boost.update_on_val(i, probe)

    # ── Supervised steps ─────────────────────────────────────────────
    if sup_steps_per_node > 0:
        for node in system.nodes.values():
            node._sup_loss_sum = 0.0
            node._sup_loss_n = 0

        node_ids = list(system.nodes.keys())
        n_total = sup_steps_per_node * len(node_ids)
        rng = random.Random(system.seed * 1_000_003 + 1337 + round_idx)
        for _ in range(n_total):
            system.nodes[node_ids[rng.randrange(len(node_ids))]].supervised_step()

        _sup_losses = {}
        for i, node in system.nodes.items():
            n = getattr(node, '_sup_loss_n', 0)
            _sup_losses[i] = getattr(node, '_sup_loss_sum', 0.0) / max(1, n) if n > 0 else 0.0
    else:
        _sup_losses = {i: 0.0 for i in system.nodes}

    # ── Hard pseudo-labels with learned trust ────────────────────────
    for i, node in system.nodes.items():
        if not node.neighbor_ids:
            continue
        buf = bufs.get(i)
        if buf is None or buf.n <= 0:
            continue

        # Get trust weights (neighbors only for pseudo-labels)
        weights = get_trust_weights(
            system, trust_model, probe, estimated_dists, i, device,
            include_self=include_self_pseudo, hedge_boost=hedge_boost,
            coarse_trust=coarse_trust, val_labels=val_labels)

        if not weights or sum(weights.values()) < 1e-8:
            stats[i] = {"examples": 0, "ce_loss": 0.0, "kl_loss": 0.0,
                        "sup_loss": _sup_losses.get(i, 0.0)}
            continue

        if top_k_teachers > 0:
            weights = _top_k_weights(weights, top_k_teachers)

        # ── Trust gate: scale pseudo_weight by ensemble vs self quality ──
        # Weighted by node's class distribution so self's strength on its
        # favored class is properly valued
        eff_pw = pseudo_weight
        if trust_gate:
            C = system.num_classes
            w_i = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            self_probe = probe.get(i, {}).get(i, np.zeros(C))
            self_acc = float((w_i * self_probe).sum())
            ens_acc = sum(
                w_j * float((w_i * probe.get(i, {}).get(j, np.zeros(C))).sum())
                for j, w_j in weights.items())
            ratio = ens_acc / max(self_acc, 1e-8)
            eff_pw = pseudo_weight * min(1.0, ratio)

        # Sample unlabeled data
        n_ex = min(examples_per_round, buf.n)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(system.seed * 100_000 + round_idx * 1000 + i)
        perm = buf.make_perm(generator=gen)[:n_ex]
        z_self = buf.x[perm].to(device)

        unlab_idx = system._unlabeled_idx_per_node.get(i)
        batch_idx = unlab_idx[perm] if unlab_idx is not None else None

        # Build weighted ensemble prediction
        p_ensemble = torch.zeros(n_ex, system.num_classes, device=device)
        for j, w_j in weights.items():
            if w_j < 1e-6:
                continue
            nbr = system.nodes[j]
            nbr.model.eval()
            z_j = system._feats_for_teacher(j, batch_idx, z_self)
            with torch.no_grad():
                if system.cache_features:
                    p_j = F.softmax(nbr.model.forward_head(z_j), dim=-1)
                else:
                    p_j = F.softmax(nbr.model(z_j), dim=-1)
            p_ensemble += w_j * p_j

        p_ensemble /= p_ensemble.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Confidence filter
        # 2026-04-24: allow override via --conf_threshold for the tau
        # ablation. When conf_threshold is None we use the original
        # class-count-dependent rule so legacy runs reproduce. When set
        # to a float, that value is used directly (0.0 disables the
        # filter, since ensemble top-probability is always > 0).
        conf = p_ensemble.max(dim=-1).values
        C = system.num_classes
        if conf_threshold is not None:
            conf_thresh = float(conf_threshold)
        else:
            conf_thresh = max(0.2, 2.0 / C) if C > 10 else (1.0 / C + 0.1)
        keep = conf > conf_thresh
        if keep.sum() == 0:
            stats[i] = {"examples": 0, "ce_loss": 0.0, "kl_loss": 0.0,
                        "sup_loss": _sup_losses.get(i, 0.0)}
            continue
        z_keep = z_self[keep]
        p_keep = p_ensemble[keep]

        node.model.train()
        if system.cache_features:
            logits_s = node.model.forward_head(z_keep)
        else:
            logits_s = node.model(z_keep)

        if soft_distil:
            # ── Soft distillation: KL to ensemble soft targets ───────
            C = system.num_classes
            log_probs_s = F.log_softmax(logits_s, dim=-1)
            kl_per_ex = F.kl_div(log_probs_s, p_keep.detach(),
                                 reduction="none").sum(dim=-1) / float(C)

            # Importance weighting
            node_skew = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            iw = torch.tensor(node_skew, dtype=torch.float32, device=device)
            pred_classes = p_keep.argmax(dim=-1)
            per_ex_w = iw[pred_classes] * float(C)
            per_ex_w = per_ex_w / per_ex_w.sum().clamp(min=1e-8) * float(p_keep.size(0))

            kl_loss = (kl_per_ex * per_ex_w).mean()
            loss = eff_pw * soft_distil_alpha * kl_loss

            node.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            node.optimizer.step()

            stats[i] = {"examples": int(p_keep.size(0)),
                        "ce_loss": 0.0,
                        "kl_loss": float(kl_loss.item()),
                        "sup_loss": _sup_losses.get(i, 0.0)}
        else:
            # ── Hard labels: plain CE ────────────────────────────────
            y_keep = p_keep.argmax(dim=-1)

            # Importance weighting for class balance
            node_skew = _get_w_i(system, i, val_labels)  # 2026-04-30: empirical
            C = system.num_classes
            iw = torch.tensor(node_skew, dtype=torch.float32, device=device)
            per_ex_w = iw[y_keep] * float(C)
            per_ex_w = per_ex_w / per_ex_w.sum().clamp(min=1e-8) * float(y_keep.size(0))

            per_ex_ce = F.cross_entropy(logits_s, y_keep, reduction="none")
            loss = eff_pw * (per_ex_ce * per_ex_w).mean()

            node.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            node.optimizer.step()

            stats[i] = {"examples": int(y_keep.size(0)),
                        "ce_loss": float(loss.item()),
                        "kl_loss": 0.0,
                        "sup_loss": _sup_losses.get(i, 0.0)}

    return stats


# =====================================================================
#  Geographic cache override
# =====================================================================

def _apply_geo_cache(system, geo_cache_path, val_fraction=0.2,
                     test_fraction=0.15, seed=0, _log=None):
    """Override system internals with geographic data from precomputed cache.

    Replaces the synthetic data partition, graph topology, architecture
    assignments, and class distributions with real geographic data from
    the EuroSAT preprocessing script.

    Args:
        system: DecentralizedPseudoLabelSystem (already constructed)
        geo_cache_path: path to .pt file from precompute_eurosat_features.py
        val_fraction: fraction of each node's data for validation
        test_fraction: fraction of each node's data for test
        seed: random seed for train/val/test splits
        _log: logging function
    """
    if _log is None:
        _log = exp._pprint

    blob = torch.load(geo_cache_path, map_location="cpu")
    node_assign = blob["node_assign"].numpy()       # [N_total]
    adj_list = blob["adj_list"]                      # {i: [j, ...]}
    arch_map = blob.get("arch_map", {})              # {i: str}
    labs = blob["labs_train"].numpy()                 # [N_total]
    meta = blob.get("meta", {})
    C = meta.get("num_classes", system.num_classes)
    num_nodes = meta.get("num_nodes", len(adj_list))
    class_names = blob.get("class_names", [])

    # 2026-04-22: sync num_classes into the system. Previously the cache's
    # class count was read into local C but never written back, meaning
    # system.num_classes kept whatever value --dataset had inferred. For
    # EuroSAT (10 classes) loaded into a --dataset cifar10 pipeline this
    # was silently correct by coincidence, but loading it into cifar100
    # (100 classes) would leave system.num_classes=100 and produce silent
    # mis-sized tensors in eval (one-hot encodings, per-class accuracy
    # arrays, skew weight vectors). Write it back explicitly with a log.
    if C != system.num_classes:
        _log(f"[GEO] Updating num_classes from {system.num_classes} to {C} "
             f"(cache metadata)")
        system.num_classes = C

    _log(f"[GEO] Loading geographic cache: {geo_cache_path}")
    _log(f"[GEO] {len(labs)} images, {num_nodes} nodes, {C} classes")

    # ── 1. Override graph topology ───────────────────────────────────
    for i, node in system.nodes.items():
        nbrs = adj_list.get(i, adj_list.get(str(i), []))
        # adj_list keys might be int or str depending on torch.save
        if isinstance(nbrs, list):
            node.neighbor_ids = set(int(j) for j in nbrs)
        else:
            node.neighbor_ids = set()

    degrees = [len(system.nodes[i].neighbor_ids) for i in range(num_nodes)]
    n_edges = sum(degrees) // 2
    _log(f"[GEO] Graph: {n_edges} edges, degree min={min(degrees)} "
         f"mean={np.mean(degrees):.1f} max={max(degrees)}")

    # ── 2. Override architecture map ─────────────────────────────────
    # 2026-04-22: gate on system.random_models. The EuroSAT precompute
    # populates arch_map for a heterogeneous setup (top 10% degree hubs
    # → efficientnet_b0, rest → mobilenet_v2) that mirrors
    # --random_models. Without --random_models, only self.arch's feature
    # cache is populated in _feat_cache_by_arch, so applying the override
    # would leave the "other arch" nodes with no loader to slice from —
    # producing the [GEO] WARNING: No feature cache for node X messages
    # and leaving those nodes unable to train. For homogeneous runs the
    # arch_map in the cache is ignored (not an error); for heterogeneous
    # runs it's respected as before.
    if arch_map and getattr(system, "random_models", False):
        for i_key, arch in arch_map.items():
            i = int(i_key)
            if i in system._node_arch_map:
                system._node_arch_map[i] = arch
        n_mn = sum(1 for v in system._node_arch_map.values()
                   if v == "mobilenet_v2")
        n_ef = sum(1 for v in system._node_arch_map.values()
                   if v == "efficientnet_b0")
        _log(f"[GEO] Architectures (from geo cache, random_models=True): "
             f"MobileNetV2={n_mn} EfficientNet-B0={n_ef}")
    elif arch_map:
        _log(f"[GEO] arch_map present in cache but random_models is off; "
             f"keeping all nodes on self.arch={system.arch} "
             f"(arch_map ignored).")

    # ── 3. Partition each node's images into train/val/test ──────────
    for i in range(num_nodes):
        node_indices = np.where(node_assign == i)[0]
        rng_i = np.random.default_rng(seed * 10_000 + i)
        rng_i.shuffle(node_indices)

        n = len(node_indices)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - n_train) // 2)
            n_test = n - n_train - n_val

        test_idx = node_indices[:n_test]
        val_idx = node_indices[n_test:n_test + n_val]
        train_idx = node_indices[n_test + n_val:]

        # Store index lists
        system._stored_train_idx[i] = train_idx.tolist()
        system._stored_val_idx[i] = val_idx.tolist()
        system._stored_test_idx[i] = test_idx.tolist()

        # Store as tensors
        train_t = torch.as_tensor(train_idx, dtype=torch.long)
        val_t = torch.as_tensor(val_idx, dtype=torch.long)
        test_t = torch.as_tensor(test_idx, dtype=torch.long)
        system._val_idx_per_node[i] = val_t
        system._test_idx_per_node[i] = test_t

        # Clear unlabeled pool for this node
        system._unlabeled_idx_per_node[i] = torch.tensor([], dtype=torch.long)

        # Get feature cache for this node's architecture
        arch = system._node_arch_map.get(i, system.arch)
        cache = system._feat_cache_by_arch.get(arch, {})
        ft = cache.get("feats_train")
        lb = cache.get("labs_train")
        if ft is None or lb is None:
            # 2026-04-22: upgraded from silent-skip to hard error. With the
            # random_models gate above, reaching this branch means either
            # (a) --random_models is on but a required per-arch cache was
            # never loaded, or (b) the loaded cache is missing train
            # tensors. Leaving the node with null loaders would defer the
            # crash to training time. Fail fast with guidance.
            available = sorted(a for a, c in system._feat_cache_by_arch.items()
                               if c.get("feats_train") is not None)
            raise RuntimeError(
                f"[GEO] No feature cache for node {i} arch={arch!r}. "
                f"Available arches in system._feat_cache_by_arch: {available}. "
                f"If you passed --random_models, make sure BOTH "
                f"--mobilenet_cache_path and --efficientnet_cache_path "
                f"point at real files containing feats_train/labs_train. "
                f"If you are NOT using heterogeneous models, drop "
                f"--random_models (arch_map from the cache will then be "
                f"ignored and every node will use --arch={system.arch!r})."
            )

        node = system.nodes[i]

        # Build train loaders
        train_ds = exp.FeatureTensorDataset(ft[train_t], lb[train_t])
        node.train_loader = DataLoader(
            train_ds, batch_size=system.batch_size, shuffle=True,
            num_workers=0, pin_memory=True)
        node.train_eval_loader = DataLoader(
            train_ds, batch_size=system.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)
        node._train_iter = None

        # Build val loader
        val_ds = exp.FeatureTensorDataset(ft[val_t], lb[val_t])
        node.val_loader = DataLoader(
            val_ds, batch_size=system.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

        # Build test loader
        test_ds = exp.FeatureTensorDataset(ft[test_t], lb[test_t])
        system._node_test_loaders[i] = DataLoader(
            test_ds, batch_size=system.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

        # ── 4. Empirical class distribution (no synthetic skew) ──────
        node_labels = labs[node_assign == i]
        counts = np.bincount(node_labels, minlength=C).astype(np.float64)
        counts += 1e-8
        system._node_skew_weights[i] = counts / counts.sum()

        # Favored class = most common
        system.favored_class_map[i] = int(np.argmax(counts))

    # ── 5. Print node statistics ─────────────────────────────────────
    sizes = [int((node_assign == i).sum()) for i in range(num_nodes)]
    _log(f"[GEO] Node sizes: min={min(sizes)} mean={np.mean(sizes):.0f} "
         f"max={max(sizes)}")

    # Show top-5 nodes by size with class distribution
    top5 = sorted(range(num_nodes), key=lambda x: -sizes[x])[:5]
    for i in top5:
        node_labels = labs[node_assign == i]
        counts = np.bincount(node_labels, minlength=C)
        top3c = np.argsort(-counts)[:3]
        if class_names:
            desc = ", ".join(f"{class_names[c]}={counts[c]}" for c in top3c)
        else:
            desc = ", ".join(f"c{c}={counts[c]}" for c in top3c)
        _log(f"  node {i:2d} (n={sizes[i]:4d}, deg={degrees[i]:2d}): {desc}")

    _log(f"[GEO] Override complete: {num_nodes} nodes, "
         f"{sum(sizes)} images partitioned")


# =====================================================================
#  Main experiment loop
# =====================================================================

def run_experiment(
    seed=0, num_nodes=50, connection_model="barabasi_albert",
    ba_m=2, p=0.1, skew_factor=10.0, per_node_sample_size=300,
    val_fraction=0.2, test_fraction=0.15,
    dataset="cifar10", training_data_mode="skewed",
    unlabeled_pool_skew="iid",
    no_unlabeled_pool=False,
    cache_features=True, random_models=False,
    random_models_mnv2_frac=0.9, baseline_non_linearity=True,
    dropout_p=0.5,
    pretrain_max_rounds=50, max_rounds=200,
    sup_steps_per_node=5, pseudo_examples_per_round=1000,
    pseudo_weight=0.5, pseudo_warmup_rounds=5,
    trust_lr=0.01, trust_steps=200, trust_update_freq=10,
    trust_hidden=32, per_node_trust=False,
    trust_method="learned", hedge_eta=0.5,
    hedge_boost=False, hedge_boost_eta=0.5,
    include_self_pseudo=False,
    trust_gate=False,
    coarse_trust=False,
    val_opt_deploy=False, val_opt_tau=0.01, val_opt_steps=300,
    top_k_teachers=0,
    # 2026-04-24: plumb conf_threshold through run_experiment so the CLI
    # flag can reach trust_pseudo_round. None keeps the legacy rule.
    conf_threshold=None,
    soft_distil=False, soft_distil_alpha=0.3,
    min_per_node_size=100, min_val_size=200,
    min_classes_per_node=2,
    eval_freq=10, print_every=10,
    mobilenet_cache_path="", efficientnet_cache_path="",
    out_dir="./trust_results", run_id="",
    save_pretrain="", load_pretrain="",
    geo_cache="",
):
    p10 = int(round(p * 10))
    log_path = os.path.join(out_dir, f"{run_id}_seed{seed}_p{p10}.txt")
    log_fh = open(log_path, "w", encoding="utf-8")
    exp._pprint(f"[LOG] Writing to: {log_path}")

    def _log(msg):
        exp._pprint(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"\n{'='*70}")
    _log(f"  LEARNED TRUST  seed={seed}  p={p}  nodes={num_nodes}")
    _log(f"  pseudo_weight={pseudo_weight}  trust_lr={trust_lr}  "
         f"trust_steps={trust_steps}  update_freq={trust_update_freq}")
    _log(f"  per_node_trust={per_node_trust}  soft_distil={soft_distil}  "
         f"alpha={soft_distil_alpha}  training={training_data_mode}  "
         f"trust_method={trust_method}")
    # 2026-04-24: surface conf_threshold in the run banner so the tau
    # ablation logs are self-identifying.
    _log(f"  conf_threshold={conf_threshold}  "
         f"(None => legacy rule max(0.2,2/C) or 1/C+0.1)")
    _log(f"{'='*70}\n")

    # ── Init system ──────────────────────────────────────────────────
    system = exp.DecentralizedPseudoLabelSystem(
        num_nodes=num_nodes, batch_size=64,
        val_fraction=float(val_fraction), test_fraction=test_fraction,
        unlabeled_fraction=0.0 if no_unlabeled_pool else 0.4,
        unlabeled_per_node=0 if no_unlabeled_pool else 2000,
        unlabeled_pool_skew=unlabeled_pool_skew,
        per_node_sample_size=per_node_sample_size,
        lr=0.01, momentum=0.9, weight_decay=1e-4,
        dropout_p=dropout_p, dropout_feat=0.0, kl_weight=1.0,
        seed=seed, hub=0, num_workers_train=0, num_workers_eval=0,
        network_connection_p=p, connection_model=connection_model,
        pseudo_conf_threshold=0.0, pseudo_entropy_threshold=99.0,
        pseudo_warmup_rounds=pseudo_warmup_rounds,
        kl_ramp_rounds=0,
        pseudo_disable_patience=0, pseudo_disable_delta=0.0,
        training_data_mode=training_data_mode, skew_factor=skew_factor,
        skew_strategy="round_robin", skew_seed=0,
        skew_min_other_frac=0.02, min_classes_per_node=min_classes_per_node,
        arch="mobilenet_v2", cache_features=cache_features,
        baseline_non_linearity=baseline_non_linearity,
        feature_cache_path=mobilenet_cache_path,
        size_skew_mode="dirichlet", min_per_node_size=min_per_node_size,
        size_dirichlet_alpha=0.05,
        neighbor_weighting="none", bandit_type="none",
        entropy_gate_tau=0.0, entropy_ucb_align_tau=0.0,
        pseudo_teacher_mode="avg", ba_m=ba_m,
        random_models=random_models,
        random_models_mnv2_frac=random_models_mnv2_frac,
        pseudo_examples_per_round=pseudo_examples_per_round,
        teacher_ema=False,
        mobilenet_cache_path=mobilenet_cache_path,
        efficientnet_cache_path=efficientnet_cache_path,
        _out_dir=out_dir, baseline_merge_val=False,
        dataset=dataset,
        # 2026-04-22: pass geo_cache so the system's internal
        # _apply_geo_cache runs at the end of __init__ (the right time to
        # avoid the CIFAR-indices-vs-EuroSAT-features IndexError).
        geo_cache=geo_cache,
    )

    # 2026-04-22: The system's internal _apply_geo_cache (called at end
    # of __init__ when geo_cache is set) already did the override. The
    # standalone _apply_geo_cache function below is kept for
    # backwards-compat but no longer invoked — the inline system version
    # runs at the correct point in __init__ and does the same work.

    # ── Stage 1: Supervised pretrain (or load checkpoint) ──────────
    _log(f"[STAGE 1] pretrain cap={pretrain_max_rounds}")
    best_test = 0.0

    # Path helper matching experiment.py convention
    def _ckpt_path(base, s):
        base = str(base).strip()
        if not base:
            return ""
        root, ext = os.path.splitext(base)
        if ext.lower() in (".pt", ".pth"):
            return f"{root}_seed{s}{ext}"
        return f"{base}_seed{s}.pt"

    if load_pretrain:
        ckpt_path = load_pretrain
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=system.device)
            # Support both experiment.py format (node_states) and our format (node_models)
            if "node_states" in ckpt:
                node_states = ckpt["node_states"]
                for i, ns in node_states.items():
                    if i not in system.nodes:
                        raise KeyError(
                            f"Checkpoint has node {i} but system only has "
                            f"nodes {sorted(system.nodes.keys())}. "
                            f"Checkpoint params don't match current run.")
                    system.nodes[i].model.load_state_dict(ns["model"])
                    if "optimizer" in ns:
                        system.nodes[i].optimizer.load_state_dict(ns["optimizer"])
                best_test = ckpt.get("pre_best", ckpt.get("best_test", 0.0))
            elif "node_models" in ckpt:
                node_models = ckpt["node_models"]
                for i, sd in node_models.items():
                    if i not in system.nodes:
                        raise KeyError(
                            f"Checkpoint has node {i} but system only has "
                            f"nodes {sorted(system.nodes.keys())}. "
                            f"Checkpoint params don't match current run.")
                    system.nodes[i].model.load_state_dict(sd)
                best_test = ckpt.get("best_test", 0.0)
            _log(f"  [LOADED] {ckpt_path}  best_test={best_test:.4f}")
        else:
            _log(f"  [WARN] Checkpoint not found: {ckpt_path}, training from scratch")
            load_pretrain = ""

    if not load_pretrain:
        for t in range(1, pretrain_max_rounds + 1):
            system.supervised_steps_synchronous(
                sup_steps_total=sup_steps_per_node * num_nodes,
                sup_steps_per_node=sup_steps_per_node, round_idx=t)
            if t % eval_freq == 0 or t == pretrain_max_rounds:
                test_acc = global_test_accuracy(system)
                train_acc = global_train_accuracy(system)
                cn, _ = compute_cn_local(system)
                if test_acc > best_test:
                    best_test = test_acc
                _log(
                    f"  [PRE {t:03d}/{pretrain_max_rounds}] "
                    f"test={test_acc:.4f}  cn_local={cn:.4f}  "
                    f"train={train_acc:.4f}")

        if save_pretrain:
            ckpt_path = _ckpt_path(save_pretrain, seed)
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            # Save in experiment.py-compatible format
            node_states = {}
            for i, node in system.nodes.items():
                node_states[i] = {
                    "model": node.model.state_dict(),
                    "optimizer": node.optimizer.state_dict(),
                }
            torch.save({
                "node_states": node_states,
                "pre_best": best_test,
                "seed": seed,
                "num_nodes": num_nodes,
            }, ckpt_path)
            _log(f"  [SAVED] {ckpt_path}")

    # Val data comes from each node's own allocation (never trained on,
    # baseline_merge_val=False keeps val separate throughout Stage 1)

    # ── Compute initial probe responses ──────────────────────────────
    _log(f"[TRUST] Computing probe responses...")
    probe, val_preds, val_labels = compute_probe_responses(
        system, include_self=True)

    # ── Estimate neighbor distributions from predictions ─────────────
    _log(f"[TRUST] Estimating neighbor distributions from predictions...")
    estimated_dists = estimate_neighbor_distributions(system)

    # ── Init and train trust model ───────────────────────────────────
    C = system.num_classes
    feat_dim = (6 if _DROP_SELF_ENT_DEG else 8) if coarse_trust else (4 * C + 3)
    device = system.device
    neighbor_map = {i: list(n.neighbor_ids)
                    for i, n in system.nodes.items()}

    # Auto-scale for large C
    if C > 10 and trust_hidden <= 32:
        trust_hidden = max(64, feat_dim // 3)
        _log(f"[TRUST] Auto-scaled trust_hidden to {trust_hidden} for C={C}")
    if C > 10 and trust_steps <= 200:
        trust_steps = max(trust_steps, 400)
        _log(f"[TRUST] Auto-scaled trust_steps to {trust_steps} for C={C}")

    if trust_method == "hedge":
        trust_model = HedgeWeights(neighbor_map, eta=hedge_eta, include_self=True)
        for i in system.nodes:
            trust_model.update_on_val(i, probe)
        init_loss = 0.0
        _log(f"[TRUST] Hedge weights (eta={hedge_eta})")
    elif trust_method == "uniform":
        trust_model = UniformWeights(neighbor_map, include_self=True)
        init_loss = 0.0
        _log(f"[TRUST] Uniform weights")
    elif trust_method == "val_opt":
        trust_model = ValOptWeights(
            neighbor_map, val_preds, val_labels, device,
            lr=0.05, steps=val_opt_steps, tau=val_opt_tau, include_self=True)
        trust_model.update_all()
        init_loss = 0.0
        val_opt_deploy = True  # auto-enable for deployment too
        _log(f"[TRUST] Val-optimized weights (steps={val_opt_steps}, tau={val_opt_tau})")
    elif per_node_trust:
        trust_model = {i: TrustModel(feat_dim, hidden_dim=trust_hidden).to(device)
                       for i in system.nodes}
        _log(f"[TRUST] Training {len(trust_model)} per-node trust models "
                    f"(dim={feat_dim}, hidden={trust_hidden}, steps={trust_steps})...")
        init_loss = train_trust_models_per_node(
            system, trust_model, probe, estimated_dists, val_preds, val_labels,
            device, lr=trust_lr, steps=trust_steps, include_self=True,
            coarse_trust=coarse_trust)
    else:
        trust_model = TrustModel(feat_dim, hidden_dim=trust_hidden).to(device)
        _log(f"[TRUST] Training shared trust model (dim={feat_dim}, "
                    f"hidden={trust_hidden}, steps={trust_steps})...")
        init_loss = train_trust_model(
            system, trust_model, probe, estimated_dists, val_preds, val_labels,
            device, lr=trust_lr, steps=trust_steps, include_self=True,
            coarse_trust=coarse_trust)
    _log(f"[TRUST] Initial trust loss: {init_loss:.4f}")

    # Init hedge boost if requested
    _hedge_boost = None
    if hedge_boost and trust_method == "learned":
        _hedge_boost = HedgeWeights(neighbor_map, eta=hedge_boost_eta,
                                    include_self=True)
        for i in system.nodes:
            _hedge_boost.update_on_val(i, probe)
        _log(f"[HEDGE_BOOST] eta={hedge_boost_eta}")

    # Show learned weights for a few nodes
    for i in list(system.nodes.keys())[:5]:
        w = get_trust_weights(system, trust_model, probe, estimated_dists,
                              i, device, hedge_boost=_hedge_boost,
                              coarse_trust=coarse_trust, val_labels=val_labels)
        top = sorted(w.items(), key=lambda x: -x[1])[:5]
        wstr = "  ".join(f"n{j}={v:.3f}" for j, v in top)
        _log(f"  n{i:02d} trust: {wstr}")

    # Feature importance (permutation-based)
    if coarse_trust and isinstance(trust_model, dict):
        fi = trust_feature_importance(
            system, trust_model, probe, estimated_dists,
            val_preds, val_labels, device,
            coarse_trust=coarse_trust, n_repeats=10)
        if fi:
            _log("[TRUST] Feature importance (permutation, ΔCE):")
            for name, imp, std in fi:
                bar = "█" * max(1, int(imp * 50 / max(fi[0][1], 1e-8)))
                _log(f"  {name:<16s}  {imp:+.4f} ± {std:.4f}  {bar}")

    # ── Stage 2: Hard pseudo-labels with learned trust ───────────────
    _log(f"\n[STAGE 2] learned trust pseudo-labels: "
                f"{max_rounds - pretrain_max_rounds} rounds")

    best_self_s2 = best_test
    best_deploy = best_test
    no_improve = 0
    patience = 500

    for t in range(pretrain_max_rounds + 1, max_rounds + 1):
        s2r = t - pretrain_max_rounds
        eff_ex = pseudo_examples_per_round if s2r > pseudo_warmup_rounds else 0

        rstats = trust_pseudo_round(
            system, trust_model, probe, estimated_dists, t,
            examples_per_round=eff_ex,
            pseudo_weight=pseudo_weight,
            sup_steps_per_node=sup_steps_per_node,
            trust_update_freq=trust_update_freq,
            val_preds=val_preds, val_labels=val_labels,
            trust_lr=trust_lr, trust_steps=50,
            include_self_pseudo=include_self_pseudo,
            soft_distil=soft_distil, soft_distil_alpha=soft_distil_alpha,
            hedge_boost=_hedge_boost,
            trust_gate=trust_gate,
            coarse_trust=coarse_trust,
            top_k_teachers=top_k_teachers,
            # 2026-04-24: forward the tau override every round.
            conf_threshold=conf_threshold)

        if t % eval_freq == 0 or t == max_rounds:
            self_acc = global_test_accuracy(system)
            train_acc = global_train_accuracy(system)
            cn, oracle_map = compute_cn_local(system)

            # Ensemble deploy with trust weights
            ens_acc = deployed_ensemble_accuracy(
                system, trust_model, probe, estimated_dists,
                device, include_self=True, hedge_boost=_hedge_boost,
                coarse_trust=coarse_trust, trust_gate=trust_gate,
                top_k_teachers=top_k_teachers, val_labels=val_labels)

            if self_acc > best_self_s2:
                best_self_s2 = self_acc
                no_improve = 0
            else:
                no_improve += 1
            if ens_acc > best_deploy:
                best_deploy = ens_acc

            total_ex = sum(s.get("examples", 0) for s in rstats.values()
                          if isinstance(s, dict) and "examples" in s)
            ce_vals = [s["ce_loss"] for s in rstats.values()
                      if isinstance(s, dict) and s.get("ce_loss", 0) > 0]
            kl_vals = [s["kl_loss"] for s in rstats.values()
                      if isinstance(s, dict) and s.get("kl_loss", 0) > 0]
            sup_vals = [s["sup_loss"] for s in rstats.values()
                       if isinstance(s, dict) and s.get("sup_loss", 0) > 0]
            avg_ce = float(np.mean(ce_vals)) if ce_vals else 0.0
            avg_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
            avg_sup = float(np.mean(sup_vals)) if sup_vals else 0.0
            trust_loss = rstats.get("_trust_loss", "")
            tl_str = f"  tl={trust_loss:.4f}" if isinstance(trust_loss, float) else ""

            if soft_distil:
                _log(
                    f"  [LT {t:03d}/{max_rounds}] "
                    f"test={ens_acc:.4f}  cn_local={cn:.4f}  "
                    f"self={self_acc:.4f}  train={train_acc:.4f}  "
                    f"ex={total_ex}  sup={avg_sup:.4f}  kl={avg_kl:.4f}{tl_str}  "
                    f"no_improve={no_improve}/{patience}")
            else:
                _log(
                    f"  [LT {t:03d}/{max_rounds}] "
                    f"test={ens_acc:.4f}  cn_local={cn:.4f}  "
                    f"self={self_acc:.4f}  train={train_acc:.4f}  "
                    f"ex={total_ex}  sup={avg_sup:.4f}  ce={avg_ce:.4f}{tl_str}  "
                    f"no_improve={no_improve}/{patience}")

            # Per-node detail
            if t % (eval_freq * max(1, print_every)) == 0 or t == max_rounds:
                test_pn = system.evaluate_all_nodes_on_test()
                for i, node in system.nodes.items():
                    w = get_trust_weights(
                        system, trust_model, probe, estimated_dists,
                        i, device, hedge_boost=_hedge_boost,
                        coarse_trust=coarse_trust, val_labels=val_labels)
                    top_j = max(w, key=w.get) if w else i
                    top_w = w.get(top_j, 0)
                    self_w = w.get(i, 0)
                    s = rstats.get(i, {})
                    arch = ("eff" if system._node_arch_map.get(
                        i, system.arch) == "efficientnet_b0" else "mob")
                    deg = len(node.neighbor_ids) if node.neighbor_ids else 0
                    ns = test_pn.get(i, 0.0)
                    ora_j, ora_acc = oracle_map.get(i, (i, 0.0))
                    fav_i = system.favored_class_map.get(i, -1)

                    nbr_parts = []
                    for j_id, j_w in sorted(w.items(), key=lambda x: -x[1]):
                        if j_id == i:
                            continue
                        fav_j = system.favored_class_map.get(j_id, -1)
                        nbr_parts.append(f"n{j_id:02d}(c{fav_j},{j_w:.2f})")
                    nbr_str = " ".join(nbr_parts[:5])

                    _log(
                        f"    n{i:02d} {arch} d{deg:02d} c{fav_i} | "
                        f"self={ns:.3f}  self_w={self_w:.2f}  "
                        f"oracle->{'self' if ora_j==i else f'n{ora_j:02d}'}"
                        f"({ora_acc:.3f})  "
                        f"ex={s.get('examples',0)}  "
                        f"sup={s.get('sup_loss',0):.4f}  "
                        f"ce={s.get('ce_loss',0):.4f}  "
                        f"nbrs=[{nbr_str}]")

            if no_improve >= patience:
                _log(f"  Early stop at round {t}")
                break

    final_self = global_test_accuracy(system)
    deploy_log = []
    final_ens = deployed_ensemble_accuracy(
        system, trust_model, probe, estimated_dists,
        device, include_self=True, hedge_boost=_hedge_boost,
        coarse_trust=coarse_trust, trust_gate=trust_gate,
        top_k_teachers=top_k_teachers, deploy_log=deploy_log,
        val_labels=val_labels)
    final_cn, _ = compute_cn_local(system)

    final_vopt = None
    if val_opt_deploy:
        final_vopt = val_optimized_deploy_accuracy(
            system, val_preds, val_labels, device,
            lr=0.05, steps=val_opt_steps, eps_dep=1e-8, tau=val_opt_tau,
            trust_gate=trust_gate, probe=probe)

    # Per-node deployment decisions
    n_ens = sum(1 for d in deploy_log if d["mode"] == "ENS")
    n_self = sum(1 for d in deploy_log if d["mode"].startswith("SELF"))
    _log(f"\n[DEPLOY] {n_ens} nodes use ensemble, {n_self} nodes fall back to self")
    for d in sorted(deploy_log, key=lambda x: x["node"]):
        i = d["node"]
        fav = system.favored_class_map.get(i, -1)
        sw = f"  self_w={d.get('self_w',0):.2f}" if d["mode"] == "ENS" else ""
        _log(f"    n{i:02d} c{fav} | deploy={d['mode']:<4}  "
             f"a_self={d['a_self']:.3f}  a_ens={d['a_ens']:.3f}  "
             f"test={d['test_acc']:.3f}{sw}")

    # Final feature importance
    if coarse_trust and isinstance(trust_model, dict):
        fi = trust_feature_importance(
            system, trust_model, probe, estimated_dists,
            val_preds, val_labels, device,
            coarse_trust=coarse_trust, n_repeats=10)
        if fi:
            _log("\n[TRUST] Final feature importance (permutation, ΔCE):")
            for name, imp, std in fi:
                bar = "█" * max(1, int(imp * 50 / max(fi[0][1], 1e-8)))
                _log(f"  {name:<16s}  {imp:+.4f} ± {std:.4f}  {bar}")

    _log(f"\n{'='*70}")
    _log(f"  DONE  seed={seed}")
    _log(f"  stage1:      {best_test:.4f}")
    _log(f"  self:        {final_self:.4f}  (best {best_self_s2:.4f})")
    _log(f"  deploy_ens:  {final_ens:.4f}  (best {best_deploy:.4f})")
    if final_vopt is not None:
        _log(f"  val_opt_dep: {final_vopt:.4f}")
    _log(f"  cn_local:    {final_cn:.4f}")
    _log(f"{'='*70}\n")

    _log(f"[DEBUG FILE] {log_path}")
    log_fh.close()

    result = {"seed": seed, "stage1": best_test,
              "best_self": best_self_s2, "best_deploy": best_deploy,
              "final_cn": final_cn}
    if final_vopt is not None:
        result["val_opt_deploy"] = final_vopt
    return result


# =====================================================================
#  CLI
# =====================================================================

def main():
    P = argparse.ArgumentParser(
        description="Learned Trust decentralized learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = P.add_argument_group("System")
    g.add_argument("--seed_list", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--num_nodes", type=int, default=50)
    g.add_argument("--connection_model", type=str, default="barabasi_albert")
    g.add_argument("--ba_m", type=int, default=2)
    g.add_argument("--p", type=float, default=0.1)
    g.add_argument("--skew_factor", type=float, default=10.0)
    g.add_argument("--per_node_sample_size", type=int, default=300)
    g.add_argument("--val_fraction", type=float, default=0.2)
    g.add_argument("--test_fraction", type=float, default=0.15)
    g.add_argument("--random_models_mnv2_frac", type=float, default=0.9)
    g.add_argument("--random_models", action="store_true", default=False,
                   help="Use heterogeneous architectures (MobileNet + EfficientNet)")
    g.add_argument("--min_per_node_size", type=int, default=100)
    g.add_argument("--min_val_size", type=int, default=200)
    g.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "eurosat"],
                   help="cifar10/cifar100: standard CIFAR pipeline. "
                        "eurosat: requires --geo_cache pointing at a .pt "
                        "produced by precompute_eurosat_features.py; "
                        "_apply_geo_cache replaces the synthetic BA graph "
                        "and skewed partition with the real Sentinel-2 "
                        "geographic topology and per-region data split.")
    g.add_argument("--training_data_mode", type=str, default="skewed",
                   choices=["skewed", "iid"])
    g.add_argument("--unlabeled_pool_skew", type=str, default="iid",
                   choices=["iid", "skewed"],
                   help="Distribution of unlabeled pool: 'iid' (class-balanced) or 'skewed' (matches node distribution)")
    g.add_argument("--no_unlabeled_pool", action="store_true", default=False,
                   help="Disable unlabeled pool entirely. Trust features fall back to "
                        "probe accuracy only; no distillation possible.")

    g = P.add_argument_group("Training")
    g.add_argument("--pretrain_max_rounds", type=int, default=50)
    g.add_argument("--max_rounds", type=int, default=200)
    g.add_argument("--sup_steps_per_node", type=int, default=5)
    g.add_argument("--pseudo_examples_per_round", type=int, default=1000)
    g.add_argument("--pseudo_weight", type=float, default=0.5,
                   help="Weight of pseudo-label CE loss relative to supervised")
    g.add_argument("--pseudo_warmup_rounds", type=int, default=5)

    g = P.add_argument_group("Trust Model")
    g.add_argument("--trust_lr", type=float, default=0.01)
    g.add_argument("--trust_steps", type=int, default=200,
                   help="Gradient steps for initial trust training")
    g.add_argument("--trust_update_freq", type=int, default=10,
                   help="Re-train trust model every N rounds")
    g.add_argument("--trust_hidden", type=int, default=32)
    g.add_argument("--per_node_trust", action="store_true", default=False,
                   help="Train one trust model per node (fully decentralized)")
    g.add_argument("--soft_distil", action="store_true", default=False)
    g.add_argument("--soft_distil_alpha", type=float, default=0.3)
    g.add_argument("--trust_method", type=str, default="learned",
                   choices=["learned", "hedge", "uniform", "val_opt"])
    g.add_argument("--hedge_eta", type=float, default=0.5)
    g.add_argument("--hedge_boost", action="store_true", default=False)
    g.add_argument("--hedge_boost_eta", type=float, default=0.5)
    g.add_argument("--include_self_pseudo", action="store_true", default=False)
    g.add_argument("--trust_gate", action="store_true", default=False,
                   help="Skip distillation for nodes where ensemble can't beat self on val")
    g.add_argument("--coarse_trust", action="store_true", default=True,
                   help="Use 8-dim aggregate trust features instead of 4C+3 per-class (default: True)")
    g.add_argument("--no_coarse_trust", dest="coarse_trust", action="store_false",
                   help="Use full 4C+3 per-class trust features")
    g.add_argument("--drop_self_ent_deg", action="store_true", default=True,
                   help="Drop entropy(w_i) and deg_i from coarse trust features (constant across arms, "
                        "zero importance in softmax). Reduces from 8 to 6 features. (default: True)")
    g.add_argument("--no_drop_self_ent_deg", dest="drop_self_ent_deg", action="store_false",
                   help="Keep all 8 coarse trust features")
    g.add_argument("--val_opt_deploy", action="store_true", default=False,
                   help="Use validation-optimized simplex weights for deployment instead of trust model")
    g.add_argument("--val_opt_tau", type=float, default=0.01,
                   help="L2 regularization for val-optimized deployment weights")
    g.add_argument("--val_opt_steps", type=int, default=300,
                   help="Optimization steps for val-optimized deployment")
    g.add_argument("--top_k_teachers", type=int, default=0,
                   help="Keep only top-K trusted neighbors for distillation (0=use all)")
    # 2026-04-24: new flag for the tau ablation. Default None preserves
    # the legacy class-count-dependent rule bit-for-bit.
    g.add_argument("--conf_threshold", type=float, default=None,
                   help="Confidence filter threshold tau for pseudo-label "
                        "selection. If unset, use the default rule "
                        "max(0.2, 2/C) for C>10 else (1/C + 0.1) "
                        "(equals 0.2 for both CIFAR-10 and CIFAR-100). "
                        "Set to 0.0 to disable filtering entirely. Added "
                        "2026-04-24 for the tau ablation (Table 8).")
    g.add_argument("--baseline_non_linearity", action="store_true", default=False)
    g.add_argument("--dropout_p", type=float, default=0.5)
    g.add_argument("--min_classes_per_node", type=int, default=2)

    g = P.add_argument_group("Output")
    g.add_argument("--eval_freq", type=int, default=10)
    g.add_argument("--print_every", type=int, default=10)
    g.add_argument("--out_dir", type=str, default="./trust_results")

    g = P.add_argument_group("Paths")
    g.add_argument("--mobilenet_cache_path", type=str, default="")
    g.add_argument("--efficientnet_cache_path", type=str, default="")
    g.add_argument("--save_pretrain", type=str, default="",
                   help="Directory to save Stage 1 pretrained models")
    g.add_argument("--load_pretrain", type=str, default="",
                   help="Directory to load Stage 1 pretrained models (skip Stage 1)")
    # 2026-04-24: fix latent type-correctness bug. --geo_cache is declared
    # type=str but default was the Python bool False, so args.geo_cache
    # came back as False when the flag wasn't passed. That forwarded into
    # experiment.py's _apply_geo_cache, whose "is geo_cache set?" check is
    # `if self.geo_cache is not None` — False is not None, so the load
    # branch fires and torch.load(False) stringifies to 'False', giving
    # FileNotFoundError: 'False'. The bug was silent until the 2026-04-22
    # geo-cache plumbing started actually reading the value. Matching
    # run_experiment's signature default ("") keeps the CLI type-correct
    # and preserves the no-op-when-unset behavior.
    g.add_argument("--geo_cache", type=str, default="",
                   help="Path to geographic .pt cache from precompute_eurosat_features.py. "
                        "Overrides graph topology, data partition, and architecture "
                        "assignments with real geographic data.")

    args = P.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global _DROP_SELF_ENT_DEG
    _DROP_SELF_ENT_DEG = args.drop_self_ent_deg

    # Print reproducible command
    import hashlib, datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + \
             hashlib.md5(str(vars(args)).encode()).hexdigest()[:6].upper()
    exp._pprint(f"[RUN_ID] {run_id}")
    cmd_parts = ["python3", "learned_trust.py"]
    for k, v in sorted(vars(args).items()):
        if isinstance(v, list):
            cmd_parts += [f"--{k}"] + [str(x) for x in v]
        elif isinstance(v, bool):
            if v:
                cmd_parts.append(f"--{k}")
        else:
            cmd_parts += [f"--{k}", str(v)]
    exp._pprint("[CMD] " + " \\\n  ".join(cmd_parts))
    exp._pprint(
        f"[CONFIG] dataset={args.dataset} nodes={args.num_nodes} "
        f"conn={args.connection_model} random_models={args.random_models} "
        f"training={args.training_data_mode} skew={args.skew_factor} "
        f"seeds={args.seed_list} max_rounds={args.max_rounds} "
        f"pseudo_weight={args.pseudo_weight} trust_lr={args.trust_lr}"
    )

    results = []
    for seed in args.seed_list:
        r = run_experiment(
            seed=seed, num_nodes=args.num_nodes,
            connection_model=args.connection_model,
            ba_m=args.ba_m, p=args.p,
            skew_factor=args.skew_factor,
            per_node_sample_size=args.per_node_sample_size,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            dataset=args.dataset,
            unlabeled_pool_skew=args.unlabeled_pool_skew,
            no_unlabeled_pool=args.no_unlabeled_pool,
            training_data_mode=args.training_data_mode,
            random_models_mnv2_frac=args.random_models_mnv2_frac,
            random_models=args.random_models,
            pretrain_max_rounds=args.pretrain_max_rounds,
            max_rounds=args.max_rounds,
            sup_steps_per_node=args.sup_steps_per_node,
            pseudo_examples_per_round=args.pseudo_examples_per_round,
            pseudo_weight=args.pseudo_weight,
            pseudo_warmup_rounds=args.pseudo_warmup_rounds,
            trust_lr=args.trust_lr,
            trust_steps=args.trust_steps,
            trust_update_freq=args.trust_update_freq,
            trust_hidden=args.trust_hidden,
            per_node_trust=args.per_node_trust,
            trust_method=args.trust_method,
            hedge_eta=args.hedge_eta,
            hedge_boost=args.hedge_boost,
            hedge_boost_eta=args.hedge_boost_eta,
            include_self_pseudo=args.include_self_pseudo,
            trust_gate=args.trust_gate,
            coarse_trust=args.coarse_trust,
            val_opt_deploy=args.val_opt_deploy,
            val_opt_tau=args.val_opt_tau,
            val_opt_steps=args.val_opt_steps,
            top_k_teachers=args.top_k_teachers,
            # 2026-04-24: forward the new CLI flag.
            conf_threshold=args.conf_threshold,
            soft_distil=args.soft_distil,
            soft_distil_alpha=args.soft_distil_alpha,
            baseline_non_linearity=args.baseline_non_linearity,
            dropout_p=args.dropout_p,
            min_per_node_size=args.min_per_node_size,
            min_val_size=args.min_val_size,
            min_classes_per_node=args.min_classes_per_node,
            eval_freq=args.eval_freq,
            print_every=args.print_every,
            mobilenet_cache_path=args.mobilenet_cache_path,
            efficientnet_cache_path=args.efficientnet_cache_path,
            out_dir=args.out_dir,
            run_id=run_id,
            save_pretrain=args.save_pretrain,
            load_pretrain=args.load_pretrain,
            geo_cache=args.geo_cache)
        results.append(r)

    if len(results) > 1:
        n = len(results)
        s1 = [r["stage1"] for r in results]
        bs = [r["best_self"] for r in results]
        bd = [r["best_deploy"] for r in results]
        cn = [r["final_cn"] for r in results]
        exp._pprint(f"\n[CI] Final metrics across seeds:")
        exp._pprint(
            f"  exp      budget         p    type   n"
            f"      test (primary)                self"
            f"       cn_local (oracle)")
        exp._pprint(
            f"  ours     {args.per_node_sample_size:>6}"
            f"      {args.p:.2f}  normal"
            f"   {n}"
            f"  test={np.mean(bd):.4f}\u00b1{np.std(bd):.4f}"
            f"  self={np.mean(bs):.4f}\u00b1{np.std(bs):.4f}"
            f"  cn_local={np.mean(cn):.4f}\u00b1{np.std(cn):.4f}")
        exp._pprint(f"")
        exp._pprint(f"  stage1:      {np.mean(s1):.4f}\u00b1{np.std(s1):.4f}")
        exp._pprint(f"  self  gain:  +{np.mean(bs)-np.mean(s1):.4f}")
        exp._pprint(f"  deploy gain: +{np.mean(bd)-np.mean(s1):.4f}")


if __name__ == "__main__":
    main()