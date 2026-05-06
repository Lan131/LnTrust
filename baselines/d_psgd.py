"""
D-PSGD baseline.

Decentralized Parallel SGD with lazy random walk mixing matrix.
Each round: local supervised SGD step, then gossip parameter averaging.

Stage 1 (cache_features=True): head parameters only (backbone frozen).
Stage 2 (cache_features=False): full model — backbone mixed among
    same-architecture neighbors only; head mixed across all neighbors.

Reference: Lian et al., "Can Decentralized Algorithms Outperform Centralized
Algorithms?" NeurIPS 2017.
"""

from .param_mixing import lazy_random_walk_weights, mix_parameters_auto


class DPSGDBaseline:
    def __init__(self, system):
        self.system = system
        self.W = lazy_random_walk_weights(system.neighbor_map)

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