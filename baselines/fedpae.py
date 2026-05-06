"""
FedPAE Baseline — Peer-Adaptive Ensemble Learning
Mueller et al., 2024.  arXiv: 2410.14075

Faithful reimplementation of the core mechanism:
  1. Each node trains independently (no collaborative distillation).
  2. Models are shared peer-to-peer (in our cached-features setup,
     we simulate this by accessing neighbor predictions).
  3. Each node runs NSGA-II (pop=100, gen=100) to find the best
     binary subset of neighbor models for deployment ensemble.
  4. Selected models are combined with equal weights.

Two objectives for NSGA-II:
  (a) Maximize validation accuracy of the equal-weight ensemble
  (b) Minimize ensemble size (sparsity pressure)

Communication cost: O(|θ|) per peer (full model sharing), same class
as D-PSGD/Gossip-FedAvg.  In our framework we simulate this via
cached predictions — the communication cost comparison is noted in
the paper.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
#  NSGA-II for binary subset selection
# =====================================================================

def _nsga2_select(
    val_preds: Dict[int, torch.Tensor],   # {j: (N_val, C)} softmax preds
    y_val: torch.Tensor,                   # (N_val,) ground truth
    arms: List[int],                        # node ids in closed neighborhood
    pop_size: int = 100,
    n_gen: int = 100,
    seed: int = 0,
) -> List[int]:
    """Run NSGA-II to find the best subset of arms for ensemble.

    Returns list of selected node ids (equal-weight ensemble).
    """
    K = len(arms)
    N = len(y_val)
    if K == 0 or N == 0:
        return []
    if K == 1:
        return list(arms)

    # Stack predictions: (K, N, C)
    P = torch.stack([val_preds[j] for j in arms], dim=0)
    rng = np.random.default_rng(seed)

    # ── Fitness evaluation ────────────────────────────────────────────
    def evaluate(individual: np.ndarray) -> Tuple[float, float]:
        """Returns (neg_accuracy, ensemble_size) — both minimized."""
        selected = np.where(individual)[0]
        if len(selected) == 0:
            return (1.0, 0.0)
        p_ens = P[selected].mean(dim=0)
        acc = (p_ens.argmax(dim=1) == y_val).float().mean().item()
        return (-acc, float(len(selected)))

    # ── Initialize population ─────────────────────────────────────────
    pop = np.zeros((pop_size, K), dtype=bool)
    for i in range(pop_size):
        pop[i] = rng.random(K) < 0.5
        if not pop[i].any():
            pop[i, rng.integers(K)] = True

    # ── Non-dominated sorting ─────────────────────────────────────────
    def fast_non_dominated_sort(fits):
        n_pop = len(fits)
        dom_count = np.zeros(n_pop, dtype=int)
        dom_set = [[] for _ in range(n_pop)]
        fronts = [[]]

        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                i_le = all(fits[i][k] <= fits[j][k] for k in range(2))
                i_lt = any(fits[i][k] < fits[j][k] for k in range(2))
                j_le = all(fits[j][k] <= fits[i][k] for k in range(2))
                j_lt = any(fits[j][k] < fits[i][k] for k in range(2))
                if i_le and i_lt:
                    dom_set[i].append(j)
                    dom_count[j] += 1
                elif j_le and j_lt:
                    dom_set[j].append(i)
                    dom_count[i] += 1

            if dom_count[i] == 0:
                fronts[0].append(i)

        fi = 0
        while fronts[fi]:
            nxt = []
            for i in fronts[fi]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt.append(j)
            fi += 1
            fronts.append(nxt)
        return fronts[:-1]

    # ── Crowding distance ─────────────────────────────────────────────
    def crowding_distance(fits, front):
        n = len(front)
        if n <= 2:
            return [float('inf')] * n
        dists = [0.0] * n
        for obj in range(2):
            order = sorted(range(n), key=lambda x: fits[front[x]][obj])
            dists[order[0]] = float('inf')
            dists[order[-1]] = float('inf')
            rng_obj = fits[front[order[-1]]][obj] - fits[front[order[0]]][obj]
            if rng_obj < 1e-12:
                continue
            for i in range(1, n - 1):
                dists[order[i]] += (
                    fits[front[order[i + 1]]][obj] -
                    fits[front[order[i - 1]]][obj]
                ) / rng_obj
        return dists

    # ── Assign rank and crowding to full population ───────────────────
    def assign_rank_crowd(fits):
        fronts = fast_non_dominated_sort(fits)
        rank = np.zeros(len(fits), dtype=int)
        crowd = np.zeros(len(fits), dtype=float)
        for fi, front in enumerate(fronts):
            cd = crowding_distance(fits, front)
            for idx_in_front, pop_idx in enumerate(front):
                rank[pop_idx] = fi
                crowd[pop_idx] = cd[idx_in_front]
        return rank, crowd

    # ── Main NSGA-II loop ─────────────────────────────────────────────
    fits = [evaluate(ind) for ind in pop]
    rank, crowd = assign_rank_crowd(fits)

    for gen in range(n_gen):
        # Create offspring
        offspring = np.zeros((pop_size, K), dtype=bool)
        for oi in range(pop_size):
            # Binary tournament selection (twice)
            def tournament():
                a, b = rng.integers(pop_size, size=2)
                if rank[a] < rank[b]:
                    return a
                elif rank[b] < rank[a]:
                    return b
                else:
                    return a if crowd[a] > crowd[b] else b
            p1, p2 = tournament(), tournament()

            # Uniform crossover
            child = np.where(rng.random(K) < 0.5, pop[p1], pop[p2])

            # Bit-flip mutation (rate = 1/K)
            mask = rng.random(K) < (1.0 / max(K, 1))
            child = np.logical_xor(child, mask)

            # Ensure at least one selected
            if not child.any():
                child[rng.integers(K)] = True
            offspring[oi] = child

        # Evaluate offspring
        off_fits = [evaluate(ind) for ind in offspring]

        # Combine parent + offspring
        combined = np.concatenate([pop, offspring], axis=0)
        combined_fits = fits + off_fits

        # Select next generation via non-dominated sorting + crowding
        fronts = fast_non_dominated_sort(combined_fits)
        next_pop = []
        next_fits = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(front)
                next_fits.extend(combined_fits[i] for i in front)
            else:
                # Fill remaining slots by crowding distance
                cd = crowding_distance(combined_fits, front)
                remaining = pop_size - len(next_pop)
                sorted_by_cd = sorted(
                    range(len(front)),
                    key=lambda x: -cd[x]
                )
                for idx in sorted_by_cd[:remaining]:
                    next_pop.append(front[idx])
                    next_fits.append(combined_fits[front[idx]])
                break

        pop = combined[next_pop]
        fits = next_fits
        rank, crowd = assign_rank_crowd(fits)

    # ── Select best solution from Pareto front ────────────────────────
    # Pick the solution with highest accuracy (lowest -acc)
    best_idx = min(range(len(fits)), key=lambda i: fits[i][0])
    selected_mask = pop[best_idx]
    return [arms[k] for k in range(K) if selected_mask[k]]


# =====================================================================
#  Baseline class
# =====================================================================

class FedPAEBaseline:
    """FedPAE: no training-time collaboration, NSGA-II deployment ensemble.

    Usage in experiment.py:
        baseline_obj = FedPAEBaseline(system)
        # ... training loop ...
        baseline_obj.post_round_hook(t)  # no-op
        # ... after training ...
        test_acc = baseline_obj.deployed_accuracy(device)
    """

    def __init__(self, system, pop_size: int = 100, n_gen: int = 100):
        self.system = system
        self.pop_size = pop_size
        self.n_gen = n_gen

    def post_round_hook(self, round_idx: int) -> None:
        """No training-time collaboration — FedPAE is deployment-only."""
        pass

    @torch.no_grad()
    def deployed_accuracy(self, device: torch.device) -> float:
        """Compute test accuracy using NSGA-II ensemble selection.

        For each node:
          1. Collect all neighbor (+ self) predictions on val set
          2. Run NSGA-II to find best subset
          3. Evaluate equal-weight ensemble of selected subset on test
        """
        system = self.system
        C = system.num_classes
        accs = []

        for i, node in system.nodes.items():
            test_idx = system._test_idx_per_node.get(i)
            if test_idx is None or test_idx.numel() == 0:
                continue

            val_idx = system._val_idx_per_node.get(i)
            if val_idx is None or val_idx.numel() == 0:
                # No val data — fall back to self
                accs.append(node.evaluate_accuracy(
                    system._node_test_loaders.get(i)))
                continue

            # Closed neighborhood
            hood = [i] + [j for j in (node.neighbor_ids or [])]

            # Collect val predictions for all neighbors
            any_cache = next(iter(system._feat_cache_by_arch.values()))
            y_val = any_cache["labs_train"][val_idx].to(device)

            val_preds = {}
            for j in hood:
                j_arch = system._node_arch_map.get(j, system.arch)
                j_cache = system._feat_cache_by_arch.get(j_arch, {})
                ft = j_cache.get("feats_train")
                if ft is None:
                    continue
                z_j = ft[val_idx].to(device)
                system.nodes[j].model.eval()
                preds_list = []
                for s in range(0, z_j.size(0), system.batch_size):
                    zb = z_j[s:s + system.batch_size]
                    p_j = F.softmax(
                        system.nodes[j].model.forward_head(zb), dim=-1)
                    preds_list.append(p_j)
                val_preds[j] = torch.cat(preds_list, dim=0)

            arms = [j for j in hood if j in val_preds]
            if not arms:
                accs.append(node.evaluate_accuracy(
                    system._node_test_loaders.get(i)))
                continue

            # Run NSGA-II
            selected = _nsga2_select(
                val_preds, y_val, arms,
                pop_size=self.pop_size,
                n_gen=self.n_gen,
                seed=system.seed * 10_000 + i,
            )

            if not selected:
                accs.append(node.evaluate_accuracy(
                    system._node_test_loaders.get(i)))
                continue

            # Evaluate selected ensemble on test set
            y_test = any_cache["labs_train"][test_idx].to(device)
            p_ensemble = torch.zeros(len(test_idx), C, device=device)
            for j in selected:
                j_arch = system._node_arch_map.get(j, system.arch)
                j_cache = system._feat_cache_by_arch.get(j_arch, {})
                ft = j_cache.get("feats_train")
                if ft is None:
                    continue
                z_j = ft[test_idx].to(device)
                system.nodes[j].model.eval()
                for s in range(0, z_j.size(0), system.batch_size):
                    zb = z_j[s:s + system.batch_size]
                    p_j = F.softmax(
                        system.nodes[j].model.forward_head(zb), dim=-1)
                    p_ensemble[s:s + zb.size(0)] += p_j

            p_ensemble /= len(selected)
            correct = (p_ensemble.argmax(dim=1) == y_test).sum().item()
            accs.append(correct / max(1, len(y_test)))

        return float(np.mean(accs)) if accs else 0.0