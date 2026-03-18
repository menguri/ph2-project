import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN
from .e3t import PartnerPredictor


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

        # --- 여기 수정 ---
        # ins.shape를 쓰지 말고, 현재 hidden state에서 batch / hidden을 읽어온다.
        batch_size, hidden_size = rnn_state.shape

        new_carry = self.initialize_carry(batch_size, hidden_size)

        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            new_carry,
            rnn_state,
        )

        # GRUCell도 hidden_size를 기준으로 정의
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

    def _action_prediction_enabled(self) -> bool:
        return bool(self.config.get("ACTION_PREDICTION", True))

    @nn.compact
    def encode_obs(self, obs):
        """
        Helper method to get observation embedding without running the full RNN/Partner prediction.
        Useful for PH-1 penalty calculation or simple inference.
        """
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="shared_encoder",
        )
        shared_ln = nn.LayerNorm(name="shared_encoder_ln")

        # Encode current observation
        # If input has Time dimension (T, B, ...), vmap over it.
        if obs.ndim == 5:
            obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        else:
            obs_emb = shared_ln(embed_model(obs))

        return obs_emb

    @nn.compact
    def encode_blocked(self, blocked_states):
        """Encode blocked target ($\
        tilde{s}$) into latent space.

        This encoder is intentionally *separate* from `encode_obs`, because the
        execution observation may have different channel semantics (and even
        different channel count) from the global full state used for PH1.

        Args:
            blocked_states: (B, H, W, C_full) or (T, B, H, W, C_full)
        Returns:
            blocked_emb: (B, D) or (T, B, D)
        """
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        blocked_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="blocked_encoder",
        )
        blocked_ln = nn.LayerNorm(name="blocked_encoder_ln")

        if blocked_states.ndim == 5:
            return blocked_ln(jax.vmap(blocked_model)(blocked_states))
        return blocked_ln(blocked_model(blocked_states))

    @nn.compact
    def get_obs_embedding(self, obs):
        # Keep old method alias just in case, but redirect to encode_obs
        return self.encode_obs(obs)

    @nn.compact
    def __call__(
        self,
        hidden,
        x,
        train=False,
        partner_prediction=None,
        blocked_states=None,
        agent_idx=None,
    ):
        # NOTE: `agent_idx` is accepted for backward compatibility with older
        # training scripts, but is intentionally ignored (PH1 agent index
        # conditioning was removed).
        # Unpack hidden state
        if isinstance(hidden, tuple):
            rnn_state = hidden[0]
        else:
            rnn_state = hidden

        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="shared_encoder",
        )
        shared_ln = nn.LayerNorm(name="shared_encoder_ln")

        # Encode current observation
        # embedding shape: (T, B, H, W, C) -> (T, B, D)
        obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        embedding = obs_emb

        # [STA-PH1] blocked_states가 이미지(상태/관측)인 경우 인코딩
        # NOTE: blocked target is expected to be a *global full* state, which may
        # have different channel count from the execution observation.
        blocked_emb = None
        blocked_emb_slots = None
        if blocked_states is not None:
            blocked_states_in = blocked_states.astype(jnp.float32)

            # 이미지 형태 판별: (B,H,W,C) 또는 (T,B,H,W,C)
            # NOTE:
            #   PH1의 blocked target(tilde{s})은 full state일 수 있어
            #   execution obs(agent_view_size 적용)와 spatial shape가 달라도 정상이다.
            #   따라서 obs와의 H/W shape 비교를 하지 않는다.
            #   (좌표 기반 stablock은 ndim<4 이므로 아래 else 경로로 감)
            is_image_like = blocked_states_in.ndim >= 4

            if is_image_like:
                blocked_single = None
                blocked_multi = None

                # Single target path:
                #  - (B,H,W,C) -> (T,B,H,W,C)
                #  - (T,B,H,W,C) -> 그대로
                if blocked_states_in.ndim == obs.ndim - 1:
                    blocked_single = jnp.broadcast_to(
                        blocked_states_in[jnp.newaxis, ...],
                        (obs.shape[0],) + blocked_states_in.shape,
                    )
                elif blocked_states_in.ndim == obs.ndim:
                    if (
                        blocked_states_in.shape[0] == obs.shape[0]
                        and blocked_states_in.shape[1] == obs.shape[1]
                    ):
                        blocked_single = blocked_states_in
                    elif blocked_states_in.shape[0] == obs.shape[1]:
                        # (B,K,H,W,C) with missing time -> multi target
                        blocked_multi = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (obs.shape[0],) + blocked_states_in.shape,
                        )
                elif blocked_states_in.ndim == obs.ndim + 1:
                    # Multi target:
                    #  - (B,K,H,W,C) -> (T,B,K,H,W,C)
                    #  - (T,B,K,H,W,C) -> 그대로
                    if (
                        blocked_states_in.shape[0] == obs.shape[0]
                        and blocked_states_in.shape[1] == obs.shape[1]
                    ):
                        blocked_multi = blocked_states_in
                    elif blocked_states_in.shape[0] == obs.shape[1]:
                        blocked_multi = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (obs.shape[0],) + blocked_states_in.shape,
                        )

                if blocked_multi is not None:
                    # Encode each slot independently, then concatenate slot embeddings.
                    t_dim, b_dim, k_dim = blocked_multi.shape[:3]
                    flat_multi = blocked_multi.reshape(
                        (t_dim, b_dim * k_dim) + blocked_multi.shape[3:]
                    )
                    blocked_emb_flat = self.encode_blocked(flat_multi)
                    blocked_emb_slots = blocked_emb_flat.reshape(
                        (t_dim, b_dim, k_dim, blocked_emb_flat.shape[-1])
                    )
                    blocked_emb = blocked_emb_slots.reshape(
                        (t_dim, b_dim, k_dim * blocked_emb_slots.shape[-1])
                    )
                    embedding = jnp.concatenate([embedding, blocked_emb], axis=-1)
                elif blocked_single is not None:
                    blocked_emb = self.encode_blocked(blocked_single)
                    embedding = jnp.concatenate([embedding, blocked_emb], axis=-1)
            else:
                # 좌표 기반(stablock 기존) 경로는 유지 (Dense input 등)
                if blocked_states_in.ndim == 2:
                    blocked_states_in = blocked_states_in[jnp.newaxis, ...]
                elif blocked_states_in.ndim != embedding.ndim:
                    # 차원 불일치 시 예외 처리 보다는 브로드캐스팅 시도
                    if blocked_states_in.shape[0] == embedding.shape[1]:
                         # (B, D) -> (1, B, D) -> (T, B, D)
                         blocked_states_in = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (embedding.shape[0],) + blocked_states_in.shape
                        )

                embedding = jnp.concatenate([embedding, blocked_states_in], axis=-1)

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # Action-prediction: compute pred_logits from GRU output with stop_gradient
        pred_logits = None
        if self._action_prediction_enabled():
            pred_logits = PartnerPredictor(action_dim=self.action_dim, name="predictor")(
                jax.lax.stop_gradient(embedding)
            )
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

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # [STA-PH1] Return extras
        extras = {
            "obs_emb": obs_emb,
            "blocked_emb": blocked_emb,
            "blocked_emb_slots": blocked_emb_slots,
            "pred_logits": pred_logits,
        }

        return rnn_state, pi, jnp.squeeze(critic, axis=-1), extras
