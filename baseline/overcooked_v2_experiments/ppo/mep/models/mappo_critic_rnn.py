"""
MAPPO Centralized Critic Network.

Takes global observations (concatenation of ALL agents' observations along the
channel dimension) and produces a value estimate per environment.

This enables centralized training with decentralized execution (CTDE).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from overcooked_v2_experiments.ppo.models.common import CNN
from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN


class MAPPOCentralCriticRNN(nn.Module):
    """Centralized critic for MAPPO.

    Receives global_obs = concat(obs_agent_0, obs_agent_1, ...) along the channel
    axis and outputs one scalar value per environment.

    Args:
        config: Model config dict with keys GRU_HIDDEN_DIM, FC_DIM_SIZE, ACTIVATION.
    """

    config: dict

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass.

        Args:
            hidden: RNN carry, shape (batch, hidden_size).
            x: Tuple of (global_obs, dones).
               global_obs shape: (time, batch, H, W, num_agents*C)
               dones shape:      (time, batch)

        Returns:
            new_hidden: Updated RNN carry (batch, hidden_size).
            value:      Scalar value per (time, batch), shape (time, batch).
        """
        rnn_state = hidden
        global_obs, dones = x

        if self.config.get("ACTIVATION", "tanh") == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embedding over global obs (vmap over batch dimension inside scan)
        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(global_obs)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # Critic value head
        value = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        value = activation(value)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(value)

        return rnn_state, jnp.squeeze(value, axis=-1)
