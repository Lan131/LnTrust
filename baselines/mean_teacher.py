"""
Mean Teacher Baseline  (Tarvainen & Valpola, NeurIPS 2017)
arXiv: 1703.01780

Each node keeps an EMA copy of its own model parameters ("teacher").
After each round the EMA is updated, then `pseudo_steps` self-distillation
steps are run using the EMA teacher to produce soft labels on the local
unlabelled buffer.  No cross-node communication occurs.

Ablation purpose
----------------
Answers: "Does neighbour collaboration help beyond a smooth local teacher?"
A win over Mean Teacher means the gain is attributable to cross-node
knowledge, not just EMA smoothing.

Notes
-----
- Works in both cache_features=True (stage 1) and cache_features=False
  (stage 2 / full backbone).
- In cache_features mode the unlabelled buffer contains pre-extracted
  feature tensors — forward_head(z) is used.
- In image mode the unlabelled buffer contains raw images — the full
  model forward node.model(z) is used, with gradients flowing through
  the backbone so it trains end-to-end.
- The EMA tracks the FULL model state dict (backbone + head) so the
  teacher reflects the evolving backbone in stage 2, not just the head.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
def _ema_update(ema_sd: dict, student_sd: dict, decay: float) -> None:
    """In-place EMA on a state-dict (both on CPU, float32)."""
    with torch.no_grad():
        for k in ema_sd:
            pe, ps = ema_sd[k], student_sd[k]
            if pe.dtype.is_floating_point:
                pe.mul_(decay).add_(ps.float(), alpha=1.0 - decay)
            else:
                pe.copy_(ps)   # integer buffers — hard copy


# -----------------------------------------------------------------------
class MeanTeacherBaseline:
    """
    Parameters
    ----------
    ema_decay       : EMA momentum (default 0.999).
    pseudo_steps    : Extra KL steps per round on unlabelled data (default 4).
    conf_threshold  : Min teacher max-prob to keep a sample (default 0 = off).
    """

    def __init__(
        self,
        system,
        ema_decay:      float = 0.999,
        pseudo_steps:   int   = 4,
        conf_threshold: float = 0.0,
    ):
        self.system         = system
        self.ema_decay      = float(ema_decay)
        self.pseudo_steps   = int(pseudo_steps)
        self.conf_threshold = float(conf_threshold)
        self._step          = 0

        # Snapshot the full model state dict for every node (CPU float32).
        # Tracks backbone + head so teacher evolves correctly in stage 2.
        self.ema_sd: Dict[int, dict] = {}
        for i, node in system.nodes.items():
            self.ema_sd[i] = {
                k: v.detach().cpu().float().clone()
                for k, v in node.model.state_dict().items()
            }

    # ------------------------------------------------------------------
    def post_round_hook(self, round_idx: int) -> None:
        self._step += 1
        # Warm-up: ramp decay from 0 → ema_decay over the first ~100 steps.
        decay = min(self.ema_decay,
                    (1.0 + self._step) / (10.0 + self._step))

        for i, node in self.system.nodes.items():
            student_sd = {
                k: v.detach().cpu()
                for k, v in node.model.state_dict().items()
            }
            # Reinitialize EMA if model structure changed (stage 2 switch adds backbone keys)
            if set(self.ema_sd[i].keys()) != set(student_sd.keys()):
                self.ema_sd[i] = {
                    k: v.detach().cpu().float().clone()
                    for k, v in node.model.state_dict().items()
                }
            else:
                _ema_update(self.ema_sd[i], student_sd, decay)

        if self.pseudo_steps > 0:
            self._pseudo_distil(round_idx)

    # ------------------------------------------------------------------
    def _pseudo_distil(self, round_idx: int) -> None:
        """Run pseudo-label self-distillation using the EMA teacher."""
        system = self.system
        bufs   = system._build_unlabeled_bufs_if_needed()

        for i, node in system.nodes.items():
            buf = bufs.get(i)
            if buf is None or buf.n == 0:
                continue

            dev = system.device
            # EMA full model weights on device (for teacher inference).
            ema_sd_dev = {k: v.to(dev) for k, v in self.ema_sd[i].items()}

            # Snapshot current student weights to restore after teacher use.
            student_snap = {
                k: v.clone() for k, v in node.model.state_dict().items()
            }

            g = torch.Generator(device="cpu")
            g.manual_seed(system.seed * 10_000 + i + round_idx * 997)

            for _ in range(self.pseudo_steps):
                xb = buf.sample_with_replacement(system.batch_size, generator=g)
                if xb is None:
                    continue
                z = xb.to(dev, non_blocking=True)

                # --- teacher inference with EMA weights ---
                node.model.load_state_dict(ema_sd_dev, strict=False)
                node.model.eval()
                with torch.no_grad():
                    if system.cache_features:
                        probs = F.softmax(node.model.forward_head(z), dim=-1)
                    else:
                        probs = F.softmax(node.model(z), dim=-1)

                # Optional confidence gate.
                if self.conf_threshold > 0.0:
                    keep = probs.max(dim=-1).values >= self.conf_threshold
                    if not keep.any():
                        continue
                    z     = z[keep]
                    probs = probs[keep]

                # --- student gradient step ---
                node.model.load_state_dict(student_snap, strict=False)
                node.model.train()
                if system.cache_features:
                    log_ps = F.log_softmax(node.model.forward_head(z), dim=-1)
                else:
                    log_ps = F.log_softmax(node.model(z), dim=-1)

                loss = F.kl_div(log_ps, probs.detach(), reduction="batchmean")
                node.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                node.optimizer.step()

                # Refresh student snapshot after the step.
                student_snap = {
                    k: v.clone() for k, v in node.model.state_dict().items()
                }

            # Always leave the model with the latest student weights.
            node.model.load_state_dict(student_snap, strict=False)
            node.model.train()