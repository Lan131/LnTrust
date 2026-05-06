"""
Gossip FedAvg baseline.

Replaces pseudo-label exchange with parameter averaging each round,
using Metropolis-Hastings mixing weights over the graph topology.

Stage 1 (cache_features=True): head parameters only (backbone frozen).
Stage 2 (cache_features=False): full model — backbone mixed among
    same-architecture neighbors only; head mixed across all neighbors.

Reference: McMahan et al. 2017 adapted to gossip/decentralized setting.
"""

from .param_mixing import metropolis_hastings_weights, mix_parameters_auto


class GossipFedAvgBaseline:
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