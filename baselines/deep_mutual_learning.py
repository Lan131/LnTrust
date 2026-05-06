"""
Deep Mutual Learning Baseline  (Zhang et al., CVPR 2018)
arXiv: 1706.00384

Fully decentralised, model-agnostic, peer-to-peer knowledge distillation
on *labeled* training data.  Each round, every node draws a labeled batch
from its own training set, gathers the current soft predictions from all
its neighbours on that same batch, and takes one extra KL step toward the
neighbour average.

Ablation purpose
----------------
Answers: "Does using unlabelled data with IS-weighting matter, or is
peer-to-peer distillation on labeled data sufficient?"
The key privacy distinction: DML implicitly leaks that the labeled samples
belong to a node's local dataset, violating the data-privacy constraint our
method respects.

Interface
---------
    obj = DeepMutualLearningBaseline(system, kl_weight=1.0, temperature=3.0)
    obj.post_round_hook(round_idx)

The system already ran supervised_steps_synchronous before this hook is
called; the DML step is additive.

Notes
-----
- In cache_features=True mode, Node._next_train_batch() returns
  (feat_tensor, label_tensor) where feat_tensor is already in feature space.
  forward_head(feat) is used directly.
- In image mode, the backbone is called with torch.no_grad() to extract
  features before forwarding through the head.
- Temperature-squared rescaling follows the Hinton et al. convention so
  the gradient magnitude is independent of T.
"""

from __future__ import annotations

import random
from typing import List

import torch
import torch.nn.functional as F


class DeepMutualLearningBaseline:
    """
    Parameters
    ----------
    kl_weight        : Weight on the KL loss term (default 1.0).
    temperature      : Softmax temperature for soft targets (default 3.0).
                       T=1 is vanilla DML; T>1 softens early noisy predictions.
    use_all_neighbors: If True (default) average over all neighbours.
                       If False, sample one neighbour per round.
    """

    def __init__(
        self,
        system,
        kl_weight:         float = 1.0,
        temperature:       float = 3.0,
        use_all_neighbors: bool  = True,
    ):
        self.system            = system
        self.kl_weight         = float(kl_weight)
        self.temperature       = max(float(temperature), 1e-3)
        self.use_all_neighbors = bool(use_all_neighbors)
        self._rng              = random.Random(system.seed ^ 0xDEAD_BEEF)

    # ------------------------------------------------------------------
    def post_round_hook(self, round_idx: int) -> None:
        system = self.system
        T      = self.temperature

        # 2026-04-23 DIAGNOSTIC: unconditional heartbeat on every 25th round
        # so we can confirm the hook is being called throughout training —
        # not just at the start. If the log shows no [DML-HEARTBEAT] lines
        # at all, the call site in experiment.py is the bug.
        if round_idx % 25 == 0:
            print(f"[DML-HEARTBEAT] round_idx={round_idx}", flush=True)

        # 2026-04-23 DIAGNOSTIC: on first Stage-2 round only, record whether
        # this hook is actually doing work. Logs will show (i) that the
        # hook was entered, (ii) loss values seen, (iii) whether a node's
        # head params actually changed after optimizer.step().
        _diag = (round_idx == 0) or (round_idx == 1)
        if _diag:
            print(f"[DML-DIAG] post_round_hook called, round_idx={round_idx}, "
                  f"n_nodes={len(system.nodes)}, kl_weight={self.kl_weight}",
                  flush=True)

        for i, node in system.nodes.items():
            nbrs: List[int] = node.neighbor_ids
            if not nbrs:
                continue

            active_nbrs = (
                nbrs if self.use_all_neighbors
                else [self._rng.choice(nbrs)]
            )

            # --- draw one labeled batch from node i's training loader ---
            a_batch, y_batch = node._next_train_batch()
            a_batch = a_batch.to(system.device, non_blocking=True)
            y_batch = y_batch.to(system.device, non_blocking=True)

            # Obtain feature vectors.
            if system.cache_features:
                # a_batch is already a pre-extracted feature tensor.
                z = a_batch
            else:
                # Extract features with frozen backbone (no gradient needed).
                node.model.eval()
                with torch.no_grad():
                    z = node.model.forward_features(a_batch)

            # --- build average soft-target from active neighbours (no grad) ---
            sum_p = torch.zeros(z.size(0), 10, device=system.device)
            n_active = 0
            for j in active_nbrs:
                system.nodes[j].model.eval()
                with torch.no_grad():
                    p_j = F.softmax(
                        system.nodes[j].model.forward_head(z) / T, dim=-1
                    )
                sum_p  += p_j
                n_active += 1

            if n_active == 0:
                continue
            p_avg = sum_p / n_active          # [B, C]

            # --- supervised CE + mutual KL on the same batch ---
            node.model.train()
            logits_i = node.model.forward_head(z)

            loss_ce  = F.cross_entropy(logits_i, y_batch)

            log_p_i  = F.log_softmax(logits_i / T, dim=-1)
            # F.kl_div: input=log-probabilities, target=probabilities
            loss_kl  = F.kl_div(log_p_i, p_avg.detach(), reduction="batchmean")
            loss_kl  = loss_kl * (T ** 2)    # temperature-squared rescaling

            loss = loss_ce + self.kl_weight * loss_kl

            # 2026-04-23 DIAGNOSTIC: capture parameter-sum BEFORE step so
            # we can detect whether optimizer.step() actually mutated anything.
            if _diag and i < 2:
                _param_sum_before = sum(
                    float(p.detach().sum().item())
                    for p in node.model.parameters()
                    if p.requires_grad
                )
                _n_req_grad = sum(1 for p in node.model.parameters() if p.requires_grad)

            node.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            node.optimizer.step()

            if _diag and i < 2:
                _param_sum_after = sum(
                    float(p.detach().sum().item())
                    for p in node.model.parameters()
                    if p.requires_grad
                )
                _changed = abs(_param_sum_after - _param_sum_before) > 1e-8
                print(f"[DML-DIAG] node={i}  loss_ce={float(loss_ce):.4f}  "
                      f"loss_kl={float(loss_kl):.4f}  total={float(loss):.4f}  "
                      f"n_req_grad_tensors={_n_req_grad}  "
                      f"param_sum_before={_param_sum_before:.6f}  "
                      f"param_sum_after={_param_sum_after:.6f}  "
                      f"CHANGED={_changed}",
                      flush=True)