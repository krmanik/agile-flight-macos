"""ACMPC policies for stock stable-baselines3 2.x.

ACMPC (uzh-rpg/acmpc_public) ships a stable-baselines3 1.7 fork that threads a
13-dim drone state [p, q, v, w] next to the observation through the policy,
rollout buffer and PPO, and adds ``distr_identity`` (the MPC output is the
Gaussian mean, with no extra Linear layer). The fork pins gym 0.21.

Here the raw drone state is appended to the observation instead (see
acmpc_env.py), and the policy splits it off again. Stock PPO, RolloutBuffer and
gymnasium then work unchanged, and the upstream networks (CustomNetwork,
CustomNetworkMlp) are used as they are.
"""
import os

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

# Length of the drone state appended to each observation: p(3) q(4) v(3) w(3)
STATE_DIM = 13


class StateSplitExtractor(nn.Module):
    """Wraps an ACMPC network whose forward() takes (features, states)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.latent_dim_pi = net.latent_dim_pi
        self.latent_dim_vf = net.latent_dim_vf

    @staticmethod
    def split(features: th.Tensor):
        return features[:, :-STATE_DIM], features[:, -STATE_DIM:]

    def forward(self, features: th.Tensor):
        obs, states = self.split(features)
        return self.net.forward_actor(obs, states), self.net.forward_critic(obs)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        obs, states = self.split(features)
        return self.net.forward_actor(obs, states)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        obs, _ = self.split(features)
        return self.net.forward_critic(obs)


class _ACMPCBasePolicy(ActorCriticPolicy):
    # The fork's distr_identity: use the network output as the Gaussian mean
    distr_identity = False

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        if self.distr_identity:
            self.action_net = nn.Identity()
            # Rebuild the optimizer without the dropped Linear layer
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

    def _make_net(self, obs_dim: int) -> nn.Module:
        raise NotImplementedError

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = StateSplitExtractor(
            self._make_net(self.features_dim - STATE_DIM)
        )


class MlpMpcPolicy(_ACMPCBasePolicy):
    """Actor: MLP -> MPC cost (Q, p) -> differentiable MPC -> CTBR command."""

    distr_identity = True

    def _make_net(self, obs_dim: int) -> nn.Module:
        from mlp_mpc_policy import CustomNetwork

        os.environ.setdefault("ACMPC_T", "5")
        return CustomNetwork(obs_dim)


class MlpOnlyPolicy(_ACMPCBasePolicy):
    """ACMPC's model-free baseline with the same network sizes."""

    def _make_net(self, obs_dim: int) -> nn.Module:
        from mlp_only_policy import CustomNetworkMlp

        return CustomNetworkMlp(obs_dim)
