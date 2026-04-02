import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import remat
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, MLP
from .e3t import PartnerPredictor


class _FlattenMLP(nn.Module):
    """obs를 flatten 후 MLP에 통과시키는 래퍼. ToyCoop 등 작은 환경용."""
    hidden_size: int = 128
    output_size: int = 128
    activation: type = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        return x


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry            # rnn_state.shape == (batch_size, hidden_size)
        ins, resets = x              # ins는 scan 한 step에 대한 입력

        batch_size, hidden_size = rnn_state.shape

        new_carry = self.initialize_carry(batch_size, hidden_size)

        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            new_carry,
            rnn_state,
        )

        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(ActorCriticBase):

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)

    @nn.compact
    def __call__(self, hidden, x, train=False, use_prediction=False, actor_only=False, encode_only=False):
        rnn_state = hidden

        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # OBS_ENCODER: "MLP" 이면 _FlattenMLP 사용 (ToyCoop 등), 기본값 "CNN"
        encoder_type = self.config.get("OBS_ENCODER", "CNN").upper()
        if encoder_type == "MLP":
            embed_model = _FlattenMLP(
                hidden_size=self.config.get("FC_DIM_SIZE", 128),
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
            )
        else:
            embed_model = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
            )

        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # encode_only: GRU embedding만 반환 (GAMMA VAE obs_feat용)
        if encode_only:
            return rnn_state, embedding

        # E3T Prediction: use GRU output with stop_gradient as predictor input
        # num_partners: 2-agent=1, 3-agent=2 → pred_logits 차원이 달라짐
        pred_logits = None
        if use_prediction:
            num_partners = self.config.get("NUM_PARTNERS", 1)
            pred_logits = PartnerPredictor(
                action_dim=self.action_dim, num_partners=num_partners, name="predictor"
            )(jax.lax.stop_gradient(embedding))
            embedding = jnp.concatenate([embedding, pred_logits], axis=-1)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        if actor_only:
            dummy_value = jnp.zeros(embedding.shape[:-1])
            return rnn_state, pi, dummy_value, pred_logits

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return rnn_state, pi, jnp.squeeze(critic, axis=-1), pred_logits

    def encode(self, hidden, x):
        """
        관측을 GRU까지 인코딩하여 embedding만 반환 (actor/critic head 미포함).
        GAMMA VAE obs_feat용. __call__(encode_only=True) 위임으로 params 공유 보장.
        """
        return self.__call__(hidden, x, encode_only=True)
