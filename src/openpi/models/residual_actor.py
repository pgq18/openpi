"""Residual action correction network for pi05.

This module implements a small network that predicts a one-step residual
correction to the base actions produced by pi05. It takes five inputs:
  - chunk-start proprioception
  - current-step proprioception
  - remaining skeleton actions from pi05
  - remaining-action mask
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


@dataclasses.dataclass(frozen=True)
class ResidualActorConfig:
    """Configuration for the residual action correction network.

    All dimensions are configurable to support different robot setups and
    pi05 model variants.
    """

    # Input dimensions
    proprio_dim: int = 7  # ee_xyz(3) + ee_euler(3) + gripper(1)
    base_action_horizon: int = 10  # pi05 action horizon
    base_action_dim: int = 7  # unpadded skeleton action dim
    feature_dim: int = 1024  # action expert hidden width (gemma_300m)
    feature_horizon: int = 10  # typically same as base_action_horizon

    # Network architecture
    encoded_dim: int = 256  # each encoder projects to this dim
    hidden_dims: tuple[int, ...] = (256, 256, 256)  # fusion MLP layers

    # Output
    action_horizon: int = 10
    action_dim: int = 6  # one-step 6-DoF pose residual; gripper uses base policy directly

    # Gaussian policy
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_scale: float = 1.0  # (high - low) / 2
    action_bias: float = 0.0  # (high + low) / 2

    # Pi05 feature extraction
    feature_layer_idx: int = 12  # which transformer layer to extract from
    feature_capture_step: int = 0  # which denoising step to capture features from


class ResidualActor(nnx.Module):
    """Residual action correction network.

    Takes proprio, remaining skeleton actions, and pi05 features as inputs,
    predicts a one-step residual action via a Gaussian policy with tanh squashing.
    """

    def __init__(self, config: ResidualActorConfig, rngs: nnx.Rngs):
        self.config = config

        # Input encoders: each compresses its input to encoded_dim
        self.chunk_start_proprio_encoder = nnx.Linear(config.proprio_dim, config.encoded_dim, rngs=rngs)
        self.current_proprio_encoder = nnx.Linear(config.proprio_dim, config.encoded_dim, rngs=rngs)
        self.action_encoder = nnx.Linear(
            config.base_action_horizon * config.base_action_dim, config.encoded_dim, rngs=rngs
        )
        self.mask_encoder = nnx.Linear(config.base_action_horizon, config.encoded_dim, rngs=rngs)
        self.feature_encoder = nnx.Linear(
            config.feature_horizon * config.feature_dim, config.encoded_dim, rngs=rngs
        )

        # Fusion MLP backbone
        self.fusion_layers = []
        in_dim = config.encoded_dim * 5
        for i, h_dim in enumerate(config.hidden_dims):
            layer = nnx.Linear(in_dim, h_dim, rngs=rngs)
            setattr(self, f"fusion_{i}", layer)
            self.fusion_layers.append(layer)
            in_dim = h_dim

        # Output heads (small init so residual starts near zero)
        self.fc_mean = nnx.Linear(
            config.hidden_dims[-1],
            config.action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )
        self.fc_logstd = nnx.Linear(
            config.hidden_dims[-1],
            config.action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.01),
        )

    def forward(self, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature):
        """Encode inputs, fuse, and produce mean/log_std.

        Args:
            chunk_start_proprio: [B, proprio_dim]
            current_proprio: [B, proprio_dim]
            remaining_skeleton: [B, base_action_horizon, base_action_dim]
            remaining_mask: [B, base_action_horizon]
            pi05_feature: [B, feature_horizon, feature_dim]

        Returns:
            (mean, log_std) each shape [B, action_dim]
        """
        h_chunk_start = nnx.relu(self.chunk_start_proprio_encoder(chunk_start_proprio))
        h_current = nnx.relu(self.current_proprio_encoder(current_proprio))
        h_action = nnx.relu(self.action_encoder(remaining_skeleton.reshape(remaining_skeleton.shape[0], -1)))
        h_mask = nnx.relu(self.mask_encoder(remaining_mask.astype(jnp.float32)))
        h_feature = nnx.relu(self.feature_encoder(pi05_feature.reshape(pi05_feature.shape[0], -1)))

        h = jnp.concatenate([h_chunk_start, h_current, h_action, h_mask, h_feature], axis=-1)
        for layer in self.fusion_layers:
            h = nnx.relu(layer(h))

        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        return mean, log_std

    def _squashed_distribution(self, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature):
        """Return Gaussian parameters and tanh-squashed deterministic action."""
        mean, log_std = self.forward(
            chunk_start_proprio,
            current_proprio,
            remaining_skeleton,
            remaining_mask,
            pi05_feature,
        )

        log_std = jnp.tanh(log_std)
        log_std = (
            self.config.log_std_min
            + 0.5 * (self.config.log_std_max - self.config.log_std_min) * (log_std + 1)
        )
        std = jnp.exp(log_std)
        mean_action = jnp.tanh(mean) * self.config.action_scale + self.config.action_bias
        return mean, log_std, std, mean_action

    def _squashed_log_prob(self, x_t, y_t, mean, std):
        log_prob = jax.scipy.stats.norm.logpdf(x_t, mean, std)
        log_prob -= jnp.log(self.config.action_scale * (1 - y_t**2) + 1e-6)
        return log_prob.sum(axis=-1, keepdims=True)

    def get_action(self, rng, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature):
        """Sample action for training using reparameterization trick.

        Returns:
            (action, log_prob, mean_action)
            action: [B, action_dim]
            log_prob: [B, 1]
            mean_action: [B, action_dim]
        """
        mean, _, std, mean_action = self._squashed_distribution(
            chunk_start_proprio,
            current_proprio,
            remaining_skeleton,
            remaining_mask,
            pi05_feature,
        )

        # Reparameterization trick
        noise = jax.random.normal(rng, mean.shape)
        x_t = mean + std * noise
        y_t = jnp.tanh(x_t)
        action = y_t * self.config.action_scale + self.config.action_bias

        log_prob = self._squashed_log_prob(x_t, y_t, mean, std)
        return action, log_prob, mean_action

    def log_prob(self, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature, action):
        """Log probability of a tanh-squashed residual action, for PPO-style updates."""
        mean, _, std, _ = self._squashed_distribution(
            chunk_start_proprio,
            current_proprio,
            remaining_skeleton,
            remaining_mask,
            pi05_feature,
        )
        flat_action = action.reshape(action.shape[0], -1)
        y_t = (flat_action - self.config.action_bias) / self.config.action_scale
        y_t = jnp.clip(y_t, -1.0 + 1e-6, 1.0 - 1e-6)
        x_t = jnp.arctanh(y_t)
        return self._squashed_log_prob(x_t, y_t, mean, std)

    def entropy(self, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature):
        """Pre-squash Gaussian entropy approximation, useful as a PPO regularizer."""
        _, log_std, _, _ = self._squashed_distribution(
            chunk_start_proprio,
            current_proprio,
            remaining_skeleton,
            remaining_mask,
            pi05_feature,
        )
        entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std
        return entropy.sum(axis=-1, keepdims=True)

    def get_eval_action(self, chunk_start_proprio, current_proprio, remaining_skeleton, remaining_mask, pi05_feature):
        """Deterministic eval action: tanh(mean).

        Returns:
            action: [B, action_dim]
        """
        _, _, _, action = self._squashed_distribution(
            chunk_start_proprio,
            current_proprio,
            remaining_skeleton,
            remaining_mask,
            pi05_feature,
        )
        return action
