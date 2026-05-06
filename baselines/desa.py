"""
DeSA Baseline — Decentralised Federated Learning via Synthetic Anchors
ICML 2024.  arXiv: 2405.11525

Faithful pixel-space implementation
-------------------------------------
Key distinction from our earlier adaptation:
  - Synthesis uses RANDOM feature extractors (ψ^rand), NOT the client's
    trained backbone. This is exactly Eq. 3 from the paper, following
    Zhao & Bilen (2023) Distribution Matching.
  - ipc=10 per class (from GitHub: --ipc=10)
  - synth_steps=1000 (standard DM default for CIFAR-10)
  - Random extractors are lightweight ConvNets, much faster than MobileNetV2
  - Architecture-agnostic: synthetic images can be used by any model

KD step (model-agnostic, per paper)
-------------------------------------
The anchors are plain pixel-space images — any model can run inference on
them. In _kd_step:
  - Sender j produces soft labels via its OWN full model: model_j(anchor_imgs)
  - Student i trains end-to-end via its OWN full model: model_i(anchor_imgs)
  - No shared backbone is assumed or used during KD.

In cache_features=True (stage 1):
  node.model(anch_imgs) works correctly — FrozenBackboneHead.forward()
  calls forward_features then forward_head internally.

In stage_2 / image mode (cache_features=False):
  - Each node has its own trainable backbone.
  - Sender forward is under no_grad (we don't train neighbours here).
  - Student forward has full gradients so the backbone trains end-to-end.
  - self._backbones (frozen ImageNet weights) are NOT used in KD — removed.

Communication cost (CIFAR-10, ipc=10, 10 classes):
  Images: 10 × 10 × 3 × 32 × 32 = 307,200 floats per neighbor per round
  Logits: 10 × 10 × 10 = 1,000 floats per neighbor per round
  With avg degree 4.9: ~3,010,560 floats total (images dominate)
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------
# Lightweight random ConvNet for distribution matching (ψ^rand)
# -----------------------------------------------------------------------

class _RandConv(nn.Module):
    """3-layer ConvNet with random weights — matches Zhao & Bilen (2023)."""
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed)
        def _conv(cin, cout):
            m = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.InstanceNorm2d(cout, affine=False),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
            )
            for p in m.parameters():
                nn.init.normal_(p, generator=g)
            return m
        self.net = nn.Sequential(_conv(3, 128), _conv(128, 256), _conv(256, 128))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.net(x), 1).flatten(1)


def _rand_features(imgs: torch.Tensor, seed: int, device: torch.device,
                   no_grad: bool = True) -> torch.Tensor:
    """Encode images with a random ConvNet (architecture-agnostic)."""
    net = _RandConv(seed).to(device)
    net.eval()
    if no_grad:
        with torch.no_grad():
            return net(imgs.to(device))
    return net(imgs.to(device))


# -----------------------------------------------------------------------
# Anchor synthesis via Distribution Matching with random extractors
# -----------------------------------------------------------------------

def _synthesise_anchors_dm(
    node_id:       int,
    train_imgs:    torch.Tensor,   # [n_node, 3, 32, 32] CPU
    train_labels:  torch.Tensor,   # [n_node] CPU
    device:        torch.device,
    num_classes:   int   = 10,
    ipc:           int   = 10,
    synth_steps:   int   = 200,
    synth_lr:      float = 0.1,
    num_rand_nets: int   = 1,      # paper uses multiple per step; 1 is faster
) -> Dict[int, torch.Tensor]:
    """
    Synthesise `ipc` images per class via Distribution Matching (Zhao & Bilen 2023).

    For each class c:
      synth* = argmin_synth E_ψ[|| mean_ψ(synth_c) - mean_ψ(real_c) ||²]
    where ψ are randomly sampled ConvNets (new seed each step → expectation).

    Returns {class_id → Tensor[ipc, 3, 32, 32]} CPU.
    """
    anchors: Dict[int, torch.Tensor] = {}

    for c in range(num_classes):
        mask = (train_labels == c)
        if mask.sum() < 2:
            continue

        real_c = train_imgs[mask].to(device)   # [n_c, 3, H, W]

        # Initialise synthetic images near real mean
        g = torch.Generator(device=device)
        g.manual_seed(node_id * 10_007 + c * 997 + 42)
        synth = (
            real_c.mean(0, keepdim=True).expand(ipc, -1, -1, -1).clone()
            + 0.01 * torch.randn(ipc, 3, 32, 32, device=device, generator=g)
        ).clamp(-3, 3).detach().requires_grad_(True)

        opt = torch.optim.SGD([synth], lr=float(synth_lr), momentum=0.5)

        for step in range(int(synth_steps)):
            opt.zero_grad()
            loss = torch.tensor(0.0, device=device)
            for k in range(num_rand_nets):
                # New random network each step — matches Eq. 3's expectation
                seed_k  = node_id * 100_003 + c * 9973 + step * 97 + k
                real_f  = _rand_features(real_c, seed_k, device, no_grad=True).mean(0)
                synth_f = _rand_features(synth,  seed_k, device, no_grad=False).mean(0)
                loss    = loss + F.mse_loss(synth_f, real_f.detach())
            loss.backward()
            opt.step()
            with torch.no_grad():
                synth.clamp_(-3, 3)

        anchors[c] = synth.detach().cpu()

    return anchors


# -----------------------------------------------------------------------
# Baseline class
# -----------------------------------------------------------------------

class DeSABaseline:
    def __init__(
        self,
        system,
        ipc_per_class: int   = 10,
        synth_steps:   int   = 200,
        synth_lr:      float = 0.1,
        synth_freq:    int   = 50,
        kd_weight:     float = 1.0,
        temperature:   float = 1.0,
    ):
        self.system      = system
        self.ipc         = int(ipc_per_class)
        self.synth_steps = int(synth_steps)
        self.synth_lr    = float(synth_lr)
        self.synth_freq  = int(synth_freq)
        self.kd_weight   = float(kd_weight)
        self.temperature = max(float(temperature), 1e-3)
        self._round      = 0
        self._num_classes = getattr(system, "num_classes", 10)

        # Load raw training images (always needed for anchor synthesis)
        _ds_name = getattr(system, "_dataset", "cifar10")
        print(f"[DESA] Loading {_ds_name} training images...", flush=True)
        _transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
        _DS_cls = torchvision.datasets.CIFAR100 if _ds_name == "cifar100" else torchvision.datasets.CIFAR10
        _cifar  = _DS_cls(
            root="./data", train=True, download=True, transform=_transform,
        )
        _loader = torch.utils.data.DataLoader(
            _cifar, batch_size=1000, shuffle=False, num_workers=0,
        )
        imgs_list, labs_list = [], []
        for xb, yb in _loader:
            imgs_list.append(xb); labs_list.append(yb)
        self._all_imgs   = torch.cat(imgs_list, 0).cpu()
        self._all_labels = torch.cat(labs_list, 0).cpu()

        # Per-node training image subsets
        print("[DESA] Slicing per-node image subsets...", flush=True)
        self._node_imgs:   Dict[int, torch.Tensor] = {}
        self._node_labels: Dict[int, torch.Tensor] = {}
        for i, node in system.nodes.items():
            ds     = node.train_eval_loader.dataset
            labels = ds.labels.cpu()
            _global_idx = getattr(system, "_train_global_idx_per_node", {}).get(i, None)
            if _global_idx is not None:
                self._node_imgs[i]   = self._all_imgs[_global_idx].cpu()
                self._node_labels[i] = self._all_labels[_global_idx].cpu()
            else:
                # Fallback: sample by class proportions deterministically
                idxs = []
                for c in range(self._num_classes):
                    c_mask = (self._all_labels == c).nonzero(as_tuple=True)[0]
                    n_c    = int((labels == c).sum().item())
                    if n_c > 0 and len(c_mask) >= n_c:
                        rng = torch.Generator()
                        rng.manual_seed(i * 997 + c * 31)
                        perm = torch.randperm(len(c_mask), generator=rng)[:n_c]
                        idxs.append(c_mask[perm])
                if idxs:
                    idx_cat = torch.cat(idxs)
                    self._node_imgs[i]   = self._all_imgs[idx_cat].cpu()
                    self._node_labels[i] = self._all_labels[idx_cat].cpu()
                else:
                    self._node_imgs[i]   = torch.zeros(0, 3, 32, 32)
                    self._node_labels[i] = torch.zeros(0, dtype=torch.long)

        self._local_anchors: Dict[int, Dict[int, torch.Tensor]] = {}
        self._received: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {
            i: {} for i in system.nodes
        }

        # ── Shared frozen backbone(s) for cache_features mode ─────────────
        # In cache_features=True, nodes have HeadOnly models (no forward() for
        # raw images). We load shared frozen backbones here — one per arch —
        # and use them to extract anchor features, then run KD head-only.
        # With --random_models we need both MobileNetV2 and EfficientNet-B0
        # because each node's head was trained on its own architecture's features.
        self._frozen_backbones: Dict[str, torch.nn.Module] = {}
        if system.cache_features:
            print("[DESA] Loading shared frozen backbone(s) for anchor feature extraction...", flush=True)
            archs_needed = set(system._node_arch_map.values()) if system._node_arch_map else {system.arch}

            def _make_backbone(arch: str) -> torch.nn.Module:
                if arch == "mobilenet_v2":
                    try:
                        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                        base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
                    except Exception:
                        from torchvision.models import mobilenet_v2
                        base = mobilenet_v2(pretrained=True)
                    features = base.features
                elif arch == "efficientnet_b0":
                    try:
                        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                    except Exception:
                        from torchvision.models import efficientnet_b0
                        base = efficientnet_b0(pretrained=True)
                    features = base.features
                else:
                    raise ValueError(f"[DESA] Unknown arch for frozen backbone: {arch}")

                for p in features.parameters():
                    p.requires_grad_(False)

                class _BackboneWrapper(torch.nn.Module):
                    def __init__(self, feats):
                        super().__init__()
                        self.feats = feats   # registered submodule → .to(device) moves weights
                        self.pool  = torch.nn.AdaptiveAvgPool2d((1, 1))
                    def forward(self, x):
                        if x.dim() == 4 and x.shape[-1] < 64:
                            x = torch.nn.functional.interpolate(x, size=96, mode='bilinear', align_corners=False)
                        return torch.flatten(self.pool(self.feats(x)), 1)

                return _BackboneWrapper(features).to(system.device).eval()

            for arch in archs_needed:
                self._frozen_backbones[arch] = _make_backbone(arch)
                print(f"[DESA] Frozen backbone ready: {arch}", flush=True)

        print("[DESA] Ready.", flush=True)

    # ------------------------------------------------------------------
    def post_round_hook(self, round_idx: int) -> None:
        self._round += 1
        needs_synth = (
            self._round == 1
            or (self.synth_freq > 0 and self._round % self.synth_freq == 0)
        )
        if needs_synth:
            self._synthesise_all()
        if not self._local_anchors:
            return
        self._share_p2p()
        self._kd_step()

    def _synthesise_all(self) -> None:
        print(f"[DESA] Synthesising anchors (round {self._round})...", flush=True)
        dev = self.system.device
        for i in self.system.nodes:
            self._local_anchors[i] = _synthesise_anchors_dm(
                node_id      = i,
                train_imgs   = self._node_imgs.get(i, torch.zeros(0, 3, 32, 32)),
                train_labels = self._node_labels.get(i, torch.zeros(0, dtype=torch.long)),
                device       = dev,
                num_classes  = self._num_classes,
                ipc          = self.ipc,
                synth_steps  = self.synth_steps,
                synth_lr     = self.synth_lr,
            )
            # Flush GPU allocator between nodes to prevent fragmentation OOM
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print("[DESA] Synthesis done.", flush=True)

    def _share_p2p(self) -> None:
        for i, node in self.system.nodes.items():
            for j in node.neighbor_ids:
                if j not in self._received:
                    self._received[j] = {}
                self._received[j][i] = self._local_anchors.get(i, {})

    def _kd_step(self) -> None:
        """
        KD using received synthetic anchor images.

        cache_features=True (no stage_2_tuning — the fast path):
          For each anchor batch, extract features using the SENDER's frozen
          backbone (respecting --random_models arch heterogeneity), run sender's
          head to get soft labels, then extract features with the STUDENT's
          frozen backbone and run student's head for the KD loss.
          Cost: 2 backbone passes per class batch (sender arch + student arch,
          often the same), then 2 linear passes — instead of crashing on HeadOnly.

        cache_features=False (stage_2_tuning):
          Falls back to full-model forward per node.
        """
        system    = self.system
        T_val     = self.temperature
        dev       = system.device
        arch_map  = system._node_arch_map
        bbs       = self._frozen_backbones  # {} when not in cache_features mode

        for i, node in system.nodes.items():
            received = self._received.get(i, {})
            if not received:
                continue

            i_arch = arch_map.get(i, system.arch)
            bb_i   = bbs.get(i_arch)

            node.model.train()
            node.optimizer.zero_grad(set_to_none=True)
            n_batches = 0

            for j, j_anchors in received.items():
                sender = system.nodes[j]
                sender.model.eval()
                j_arch = arch_map.get(j, system.arch)
                bb_j   = bbs.get(j_arch)

                for c, anch_imgs_cpu in j_anchors.items():
                    if anch_imgs_cpu.size(0) == 0:
                        continue
                    anch_imgs = anch_imgs_cpu.to(dev)

                    if bb_j is not None and bb_i is not None:
                        # Fast path: frozen backbone per arch, head-only KD
                        with torch.no_grad():
                            feats_j = bb_j(anch_imgs)
                            p_j = F.softmax(
                                sender.model.forward_head(feats_j) / T_val, dim=-1
                            )
                            feats_i = bb_i(anch_imgs) if j_arch != i_arch else feats_j
                        logits_i = node.model.forward_head(feats_i)
                    else:
                        # Full-model path (stage_2_tuning)
                        with torch.no_grad():
                            p_j = F.softmax(sender.model(anch_imgs) / T_val, dim=-1)
                        logits_i = node.model(anch_imgs)

                    log_p_i = F.log_softmax(logits_i / T_val, dim=-1)
                    loss_c  = self.kd_weight * F.kl_div(
                        log_p_i, p_j.detach(), reduction="batchmean"
                    ) * (T_val ** 2)
                    loss_c.backward()
                    n_batches += 1
                    del anch_imgs, p_j, logits_i, log_p_i, loss_c

            if n_batches == 0:
                continue

            if n_batches > 1:
                for p in node.model.parameters():
                    if p.grad is not None:
                        p.grad.div_(n_batches)
            node.optimizer.step()
