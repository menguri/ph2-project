"""
MAPPO Actor Network (decentralized execution).

Takes per-agent observations and produces an action distribution.
No critic head — value estimation is handled by MAPPOCentralCriticRNN.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal

from overcooked_v2_experiments.ppo.models.common import CNN
from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN


class MAPPOActorRNN(nn.Module):
    """Actor-only RNN network for MAPPO.

    Interface mirrors ActorCriticRNN but returns only (rnn_state, pi).

    Args:
        action_dim: Number of discrete actions.
        config: Model config dict with keys GRU_HIDDEN_DIM, FC_DIM_SIZE, ACTIVATION.
    """

    action_dim: int
    config: dict

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass.

        Args:
            hidden: RNN carry, shape (batch, hidden_size).
            x: Tuple of (obs, dones).
               obs shape:   (time, batch, H, W, C)
               dones shape: (time, batch)

        Returns:
            new_hidden: Updated RNN carry (batch, hidden_size).
            pi: Categorical action distribution over (time, batch).
        """
        rnn_state = hidden
        obs, dones = x

        if self.config.get("ACTIVATION", "tanh") == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embedding: vmap over time × batch dimension (first axis after scan)
        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)  # (time*batch, embed_dim) via scan
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # Actor head
        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)
        return rnn_state, pi
