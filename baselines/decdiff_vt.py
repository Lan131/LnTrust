"""
DecDiff-VT baseline (approximate implementation).

Combines soft-label KL distillation (handled by the system's pseudo-label
mechanism) with parameter averaging via Metropolis-Hastings weights.

Stage 1 (cache_features=True): head parameters only (backbone frozen).
Stage 2 (cache_features=False): full model — backbone mixed among
    same-architecture neighbors only; head mixed across all neighbors.

Keep kl_weight at its normal value; do NOT zero it out for this baseline.

Reference: Boldrini et al. 2023, "DecDiff-VT: Decentralized Diffusion-based
Knowledge Transfer for Heterogeneous Federated Learning."
"""

from .param_mixing import metropolis_hastings_weights, mix_parameters_auto


class DecDiffVTBaseline:
    def __init__(self, system):
        self.system = system
        self.W = metropolis_hastings_weights(system.neighbor_map)

    def post_round_hook(self, round_idx: int) -> None:
        s = self.system
        mix_parameters_auto(
            nodes           = s.nodes,
            neighbor_map    = s.neighbor_map,
            W               = self.W,
            cache_features  = s.cache_features,
            node_arch_map   = getattr(s, "_node_arch_map", {}),
            default_arch    = getattr(s, "arch", "mobilenet_v2"),
        )