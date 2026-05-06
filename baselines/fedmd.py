"""
FedMD baseline.

Black-box output sharing via a shared public dataset.

Reference: Li & Wang, "FedMD: Heterogeneous Federated Learning via Model
Distillation", NeurIPS 2019.

Protocol each round:
  1. All nodes forward-pass the public dataset (no labels needed).
  2. Per-sample predictions are averaged across all nodes.
  3. Each node does one distillation step: KL(node || global_avg) on
     the public data.

Public dataset: carved from the union of all nodes' unlabeled feature
buffers on first call. This avoids needing a truly separate dataset and
keeps the comparison within the same data budget.

For a stronger FedMD comparison, pass --fedmd_public_size N to control
how many public samples are used. Default: 2000.

Caller must set kl_weight=0 on all nodes before constructing this baseline
(handled automatically by make_baseline in __init__.py).
"""

import torch
import torch.nn.functional as F
from typing import Optional


class FedMDBaseline:
    """
    FedMD: each round, every node distills from globally averaged predictions
    on a shared public dataset sampled from the combined unlabeled pool.
    """

    def __init__(self, system, public_size: int = 2000):
        self.system = system
        self.public_size = int(public_size)
        self._public_z: Optional[torch.Tensor] = None   # [P, D] on CPU
        self._bufs: Optional[dict] = None

    # ------------------------------------------------------------------

    def _build_public_dataset(self) -> torch.Tensor:
        """Sample from union of unlabeled feature buffers (CPU tensors)."""
        bufs = self.system._build_unlabeled_bufs_if_needed()
        self._bufs = bufs

        parts = [buf.x for buf in bufs.values() if buf.n > 0]
        if not parts:
            feat_dim = self.system.feat_dim if self.system.cache_features else 3 * 224 * 224
            return torch.zeros((0, feat_dim), dtype=torch.float32)

        combined = torch.cat(parts, dim=0)      # [N_total, D]
        n = combined.size(0)

        if n <= self.public_size:
            return combined.contiguous()

        perm = torch.randperm(n)[: self.public_size]
        return combined[perm].contiguous()

    @torch.no_grad()
    def _global_avg_predictions(self, z_cpu: torch.Tensor) -> torch.Tensor:
        """
        Forward all nodes on public data (in batches), return averaged
        softmax predictions [P, C] on CPU.
        """
        device = self.system.device
        bs = self.system.batch_size
        n = z_cpu.size(0)
        num_nodes = len(self.system.nodes)

        sum_probs_cpu = torch.zeros((n, 10), dtype=torch.float32)

        for start in range(0, n, bs):
            end = min(start + bs, n)
            z_batch = z_cpu[start:end].to(device, non_blocking=True)

            batch_sum = torch.zeros((end - start, 10), device=device, dtype=torch.float32)
            for node in self.system.nodes.values():
                node.model.eval()
                logits = node.model.forward_head(z_batch)
                batch_sum += F.softmax(logits, dim=-1)

            sum_probs_cpu[start:end] = batch_sum.cpu()

        return (sum_probs_cpu / max(1, num_nodes)).contiguous()

    def _distill_from_global(self, z_cpu: torch.Tensor, p_avg_cpu: torch.Tensor) -> None:
        """
        One distillation pass for every node: KL(log_student || p_avg) on
        the full public dataset, processed in batches.
        """
        device = self.system.device
        bs = self.system.batch_size
        n = z_cpu.size(0)

        for node in self.system.nodes.values():
            node.model.train()

        for start in range(0, n, bs):
            end = min(start + bs, n)
            z_batch = z_cpu[start:end].to(device, non_blocking=True)
            p_batch = p_avg_cpu[start:end].to(device, non_blocking=True)

            for node in self.system.nodes.values():
                logits = node.model.forward_head(z_batch)
                loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    p_batch,
                    reduction="batchmean",
                )
                node.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                node.optimizer.step()

    # ------------------------------------------------------------------

    def post_round_hook(self, round_idx: int) -> None:
        # Build public dataset on first call (one-time cost)
        if self._public_z is None:
            self._public_z = self._build_public_dataset()

        if self._public_z.size(0) == 0:
            return

        # For non-cached case: extract features from images using any node's backbone
        if not self.system.cache_features:
            # Convert stored image tensors to features (reuse node 0's frozen backbone)
            node0 = self.system.nodes[0]
            device = self.system.device
            bs = self.system.batch_size
            n = self._public_z.size(0)
            feat_parts = []
            with torch.no_grad():
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    x = self._public_z[start:end].to(device, non_blocking=True)
                    feat_parts.append(node0.model.forward_features(x).cpu())
            # Replace stored images with cached features for subsequent rounds
            self._public_z = torch.cat(feat_parts, dim=0).contiguous()

        # --- FedMD protocol ---
        p_avg_cpu = self._global_avg_predictions(self._public_z)  # no grad
        self._distill_from_global(self._public_z, p_avg_cpu)
