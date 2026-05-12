"""Residual action correction network for pi05.

This module implements a small network that predicts a residual correction
to the base actions produced by pi05. It takes three inputs:
  - proprioception (robot state)
  - base actions from pi05
  - intermediate features from pi05's action expert

Each input is encoded through a linear layer, concatenated, and fed through
an MLP fusion backbone with two output heads (mean and log_std) for a
Gaussian policy with tanh squashing.

Architecture reference: policy_decorator/online/pi_dec_bet_maniskill2.py Actor class.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ResidualActorConfig:
    """Configuration for the residual action correction network.

    All dimensions are configurable to support different robot setups and
    pi05 model variants.
    """

    # Input dimensions
    proprio_dim: int = 7  # ee_xyz(3) + ee_euler(3) + gripper(1)
    base_action_horizon: int = 10  # pi05 action horizon
    base_action_dim: int = 32  # pi05 action dim (padded)
    feature_dim: int = 1024  # action expert hidden width (gemma_300m)
    feature_horizon: int = 10  # typically same as base_action_horizon

    # Network architecture
    encoded_dim: int = 256  # each encoder projects to this dim
    hidden_dims: tuple[int, ...] = (256, 256, 256)  # fusion MLP layers

    # Output
    action_horizon: int = 10
    action_dim: int = 7  # actual robot DOF

    # Gaussian policy
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_scale: float = 1.0  # (high - low) / 2
    action_bias: float = 0.0  # (high + low) / 2

    # Pi05 feature extraction
    feature_layer_idx: int = 12  # which transformer layer to extract from


class ResidualActor(nnx.Module):
    """Residual action correction network.

    Takes proprio, base actions, and pi05 intermediate features as inputs,
    predicts a residual action chunk via a Gaussian policy with tanh squashing.
    """

    def __init__(self, config: ResidualActorConfig, rngs: nnx.Rngs):
        self.config = config

        # Input encoders: each compresses its input to encoded_dim
        self.proprio_encoder = nnx.Linear(config.proprio_dim, config.encoded_dim, rngs=rngs)
        self.action_encoder = nnx.Linear(
            config.base_action_horizon * config.base_action_dim, config.encoded_dim, rngs=rngs
        )
        self.feature_encoder = nnx.Linear(
            config.feature_horizon * config.feature_dim, config.encoded_dim, rngs=rngs
        )

        # Fusion MLP backbone
        self.fusion_layers = []
        in_dim = config.encoded_dim * 3
        for i, h_dim in enumerate(config.hidden_dims):
            layer = nnx.Linear(in_dim, h_dim, rngs=rngs)
            setattr(self, f"fusion_{i}", layer)
            self.fusion_layers.append(layer)
            in_dim = h_dim

        # Output heads (small init so residual starts near zero)
        self.fc_mean = nnx.Linear(
            config.hidden_dims[-1],
            config.action_horizon * config.action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )
        self.fc_logstd = nnx.Linear(
            config.hidden_dims[-1],
            config.action_horizon * config.action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )

    def forward(self, proprio, base_action, pi05_feature):
        """Encode inputs, fuse, and produce mean/log_std.

        Args:
            proprio: [B, proprio_dim]
            base_action: [B, base_action_horizon, base_action_dim]
            pi05_feature: [B, feature_horizon, feature_dim]

        Returns:
            (mean, log_std) each shape [B, action_horizon * action_dim]
        """
        h_proprio = nnx.relu(self.proprio_encoder(proprio))
        h_action = nnx.relu(self.action_encoder(base_action.reshape(base_action.shape[0], -1)))
        h_feature = nnx.relu(self.feature_encoder(pi05_feature.reshape(pi05_feature.shape[0], -1)))

        h = jnp.concatenate([h_proprio, h_action, h_feature], axis=-1)
        for layer in self.fusion_layers:
            h = nnx.relu(layer(h))

        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        return mean, log_std

    def get_action(self, rng, proprio, base_action, pi05_feature):
        """Sample action for training using reparameterization trick.

        Returns:
            (action, log_prob, mean_action)
            action: [B, action_horizon, action_dim]
            log_prob: [B, 1]
            mean_action: [B, action_horizon, action_dim]
        """
        mean, log_std = self.forward(proprio, base_action, pi05_feature)

        # Tanh-squash log_std into [log_std_min, log_std_max]
        log_std = jnp.tanh(log_std)
        log_std = (
            self.config.log_std_min
            + 0.5 * (self.config.log_std_max - self.config.log_std_min) * (log_std + 1)
        )
        std = jnp.exp(log_std)

        # Reparameterization trick
        noise = jax.random.normal(rng, mean.shape)
        x_t = mean + std * noise
        y_t = jnp.tanh(x_t)
        action = y_t * self.config.action_scale + self.config.action_bias

        # Log probability with tanh correction
        log_prob = jax.scipy.stats.norm.logpdf(x_t, mean, std)
        log_prob -= jnp.log(self.config.action_scale * (1 - y_t**2) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)

        mean_action = jnp.tanh(mean) * self.config.action_scale + self.config.action_bias
        H, D = self.config.action_horizon, self.config.action_dim
        return action.reshape(-1, H, D), log_prob, mean_action.reshape(-1, H, D)

    def get_eval_action(self, proprio, base_action, pi05_feature):
        """Deterministic eval action: tanh(mean).

        Returns:
            action: [B, action_horizon, action_dim]
        """
        mean, _ = self.forward(proprio, base_action, pi05_feature)
        action = jnp.tanh(mean) * self.config.action_scale + self.config.action_bias
        H, D = self.config.action_horizon, self.config.action_dim
        return action.reshape(-1, H, D)
