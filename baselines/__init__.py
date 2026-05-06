"""
Baseline methods for decentralized semi-supervised learning.

Each baseline exposes a post_round_hook(round_idx) that is called after
system.run_round() in both the pretrain and pseudo-enabled stages.

Baselines that replace the pseudo-label KL mechanism:
  - gossip_fedavg  : head param averaging (MH weights)
  - d_psgd         : head param averaging (lazy random walk weights)
  - mean_teacher   : local EMA teacher, self-distillation on unlabelled data
                     (no cross-node communication; ablates value of collaboration)
  - desa           : synthetic feature anchor sharing + KD  (ICML 2024)
                     (data sharing baseline; ablates lightweight pseudo-labels)

Baselines that extend the pseudo-label mechanism:
  - decdiff_vt     : keep pseudo KL, additionally average head params

Baselines that use a shared public dataset:
  - fedmd          : distill from globally averaged predictions on public data

Baselines that use labeled data for peer distillation:
  - dml            : Deep Mutual Learning — KL toward neighbour predictions
                     on local labeled batches (Zhang et al., CVPR 2018)
                     (ablates need for unlabelled data + IS-weighting)

Usage in run_one_setting():
    baseline_obj = make_baseline(
        baseline_name, system,
        baseline_rho=0.05,       # SAM rho for fl2
        fedmd_public_size=2000,  # public samples for fedmd
    )
    # After each system.run_round(...):
    if baseline_obj is not None:
        baseline_obj.post_round_hook(round_idx)
"""

from typing import Optional


_PARAM_REPLACE = {"gossip_fedavg", "d_psgd", "fedmd", "desa"}
"""Baselines that override pseudo-KL -- kl_weight must be zeroed out."""


def make_baseline(
    name: str,
    system,
    baseline_rho: float = 0.05,
    fedmd_public_size: int = 2000,
) -> Optional[object]:
    """
    Factory: create and return the requested baseline object, or None for 'none'.

    Also zeroes kl_weight on all nodes when the baseline replaces the native
    pseudo-label KL mechanism (gossip_fedavg, d_psgd, fl2, fedmd,
    mean_teacher, desa).
    DecDiff-VT keeps native pseudo-KL and adds param averaging on top.
    DML keeps native pseudo-KL and adds labeled-data peer distillation on top.
    """
    name = str(name).strip().lower()

    if name == "none":
        return None

    # Baselines that fully replace pseudo-KL communication
    if name in _PARAM_REPLACE:
        for node in system.nodes.values():
            node.kl_weight = 0.0

    if name == "gossip_fedavg":
        from .gossip_fedavg import GossipFedAvgBaseline
        return GossipFedAvgBaseline(system)

    elif name == "d_psgd":
        from .d_psgd import DPSGDBaseline
        return DPSGDBaseline(system)

    elif name == "decdiff_vt":
        # Keeps native pseudo-KL; adds param averaging
        from .decdiff_vt import DecDiffVTBaseline
        return DecDiffVTBaseline(system)

    elif name == "fedmd":
        from .fedmd import FedMDBaseline
        return FedMDBaseline(system, public_size=int(fedmd_public_size))

    elif name == "mean_teacher":
        from .mean_teacher import MeanTeacherBaseline
        return MeanTeacherBaseline(system)

    elif name == "dml":
        # Keeps native pseudo-KL; adds labeled-data peer distillation on top
        from .deep_mutual_learning import DeepMutualLearningBaseline
        return DeepMutualLearningBaseline(system)

    elif name == "desa":
        from .desa import DeSABaseline
        return DeSABaseline(system)
    
    elif name == "fedpae":                          
        from .fedpae import FedPAEBaseline          
        return FedPAEBaseline(system)               

    else:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Choose from: none, gossip_fedavg, d_psgd, decdiff_vt, fl2, fedmd, "
            f"mean_teacher, dml, desa"
        )