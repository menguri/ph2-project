import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import remat
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, CNNSimple, CNNGamma, MLP
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
    def __call__(self, hidden, x, train=False, use_prediction=False, actor_only=False, encode_only=False, avail_actions=None, global_obs=None):
        # MAPPO_MODE: global_obs 가 주어지면 critic head 는 기존 GRU embedding 대신
        # 별도 MLP 를 통해 global state 에서 value 를 산출한다 (centralized critic).
        # global_obs=None 이면 기존 IPPO 경로와 완전히 동일 (파라미터 키 불변).
        rnn_state = hidden

        obs, dones = x

        # NORMALIZE_OBS_DIVISOR: GridSpread 등 bounded coord obs를 [0,1] 근처로 normalize.
        # ref MAPPO 의 use_feature_normalization 대체 (running stats 대신 환경 known bound 사용).
        _norm_div = self.config.get("NORMALIZE_OBS_DIVISOR", 0.0)
        if _norm_div and _norm_div > 0:
            obs = obs.astype(jnp.float32) / jnp.float32(_norm_div)

        act_name = self.config.get("ACTIVATION", "relu")
        if act_name == "leaky_relu":
            activation = nn.leaky_relu
        elif act_name == "tanh":
            activation = nn.tanh
        else:
            activation = nn.relu

        # OBS_ENCODER: "MLP" | "CNNSimple" | "CNN" (기본값)
        encoder_type = self.config.get("OBS_ENCODER", "CNN").upper()
        if encoder_type == "MLP":
            embed_model = _FlattenMLP(
                hidden_size=self.config.get("FC_DIM_SIZE", 128),
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
            )
        elif encoder_type == "CNNSIMPLE":
            # 원본 E3T-Overcooked와 동일 구조 (5x5/3x3/3x3)
            embed_model = CNNSimple(
                output_size=self.config["GRU_HIDDEN_DIM"],
                features=self.config.get("CNN_FEATURES", 32),
                activation=activation,
            )
        elif encoder_type == "CNNGAMMA":
            # GAMMA 논문 Cooperator policy CNN: [32,64,32] 3×3 SAME
            embed_model = CNNGamma(
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

        # USE_RNN=False (예: GridSpread) → GRU 우회. fully-observable 환경에서 MLP-only 학습.
        if self.config.get("USE_RNN", True):
            rnn_in = (embedding, dones)
            rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # encode_only: GRU embedding만 반환 (GAMMA VAE obs_feat용)
        if encode_only:
            return rnn_state, embedding

        # E3T Prediction: GRU output → PartnerPredictor → pred_logits
        # 원본 E3T 논문과 동일하게 stop_gradient 없이 gradient 양방향 흐름
        # num_partners: 2-agent=1, 3-agent=2 → pred_logits 차원이 달라짐
        pred_logits = None
        if use_prediction:
            num_partners = self.config.get("NUM_PARTNERS", 1)
            pred_logits = PartnerPredictor(
                action_dim=self.action_dim, num_partners=num_partners, name="predictor"
            )(embedding)
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

        # GridSpread 전용 action masking: invalid action logit을 -inf로.
        # avail_actions is None이면 no-op이라 Overcooked 등 다른 환경엔 영향 없음.
        if avail_actions is not None:
            actor_mean = actor_mean + jnp.where(
                avail_actions == 1, 0.0, -jnp.inf
            )

        pi = distrax.Categorical(logits=actor_mean)

        if actor_only:
            dummy_value = jnp.zeros(embedding.shape[:-1])
            return rnn_state, pi, dummy_value, pred_logits

        if global_obs is not None:
            # Centralized critic: global_obs 를 별도 MLP 로 통과 (param 이름 명시 → 기존 IPPO ckpt 키와 충돌 X)
            # global_obs shape 가 (T, B, D) 또는 임의 leading dims 라도 마지막 축이 feature.
            g = global_obs
            if g.ndim > 2:
                g = g.reshape((-1, g.shape[-1]))
            critic = nn.Dense(
                self.config["FC_DIM_SIZE"],
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="critic_global_dense",
            )(g)
            critic = activation(critic)
            critic = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="critic_global_out",
            )(critic)
            # embedding leading dims 에 맞춰 reshape (마지막 축은 1 → squeeze 후 매칭)
            critic = critic.reshape(embedding.shape[:-1] + (1,))
        else:
            # SPLIT_OPTIMIZER (GS 전용) 에서만 critic Dense 에 명시적 name 부여 →
            # multi_transform label_fn 이 path 에서 "critic" 검출 가능.
            # 다른 환경은 기존 자동명 (Dense_X) 유지 → 기존 ckpt 호환 보존.
            _split_opt = bool(self.config.get("SPLIT_OPTIMIZER", False))
            critic = nn.Dense(
                self.config["FC_DIM_SIZE"],
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=("critic_dense" if _split_opt else None),
            )(embedding)
            critic = activation(critic)
            critic = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name=("critic_out" if _split_opt else None),
            )(critic)

        return rnn_state, pi, jnp.squeeze(critic, axis=-1), pred_logits

    def encode(self, hidden, x):
        """
        관측을 GRU까지 인코딩하여 embedding만 반환 (actor/critic head 미포함).
        GAMMA VAE obs_feat용. __call__(encode_only=True) 위임으로 params 공유 보장.
        """
        return self.__call__(hidden, x, encode_only=True)
