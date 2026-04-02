import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, MLP


class _FlattenMLP(nn.Module):
    """obs를 flatten 후 MLP에 통과시키는 래퍼. ToyCoop 등 작은 환경용."""
    hidden_size: int = 128
    output_size: int = 128
    activation: type = nn.relu
    name: str = "shared_encoder"

    @nn.compact
    def __call__(self, x, train=False):
        # (B, H, W, C) → (B, H*W*C) 또는 이미 flat인 경우 그대로
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
from .e3t import PartnerPredictor, ZPredictor, CycleDecoder
from .cycle_transformer import CycleTransformerModule, CycleTransformerModuleV3


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

    def _build_shared_encoder(self, activation):
        """CNN 또는 MLP 인코더를 config["OBS_ENCODER"]에 따라 생성.
        기본값 "CNN" — 기존 OvercookedV2 코드에 영향 없음."""
        encoder_type = self.config.get("OBS_ENCODER", "CNN").upper()
        if encoder_type == "MLP":
            return _FlattenMLP(
                hidden_size=self.config["FC_DIM_SIZE"],
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="shared_encoder",
            )
        else:
            return CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="shared_encoder",
            )

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

        embed_model = self._build_shared_encoder(activation)
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

        encoder_type = self.config.get("OBS_ENCODER", "CNN").upper()
        if encoder_type == "MLP":
            blocked_model = _FlattenMLP(
                hidden_size=self.config.get("FC_DIM_SIZE", 128),
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="blocked_encoder",
            )
        else:
            blocked_model = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="blocked_encoder",
            )
        blocked_ln = nn.LayerNorm(name="blocked_encoder_ln")

        if encoder_type == "MLP":
            # MPE (1D obs): forward pass ndim==3 (T,B,D) → vmap, direct call ndim==2 (B,D) → 직접
            if blocked_states.ndim >= 3:
                return blocked_ln(jax.vmap(blocked_model)(blocked_states))
            return blocked_ln(blocked_model(blocked_states))
        else:
            # Overcooked (grid obs): forward pass ndim==5 (T,B,H,W,C) → vmap, direct call ndim==4 (B,H,W,C) → 직접
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

        embed_model = self._build_shared_encoder(activation)
        shared_ln = nn.LayerNorm(name="shared_encoder_ln")

        # Encode current observation
        # embedding shape: CNN → (T, B, H, W, C) -> (T, B, D)
        #                  MLP → (T, B, D_flat) -> (T, B, D)
        if obs.ndim == 5:
            # (T, B, H, W, C) — CNN or MLP with flatten
            obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        else:
            # (T, B, D_flat) — already flat
            obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        embedding = obs_emb

        # -------------------------------------------------------------------
        # CycleTransformer branch  (transformer_action=True only)
        # When False, this block is fully skipped → zero impact on existing code.
        # -------------------------------------------------------------------
        transformer_action = self.config.get("TRANSFORMER_ACTION", False)
        _ct_z_hat_sg = None   # (T, B, D_obs)  — set below if enabled
        _ct_a_hat_sg = None   # (T, B, A)
        _new_obs_window = None
        _new_step_idx = None
        ct_obs_emb_raw = None  # (T, B, D_obs) CT obs emb for _loss_fn

        if transformer_action:
            W = int(self.config.get("TRANSFORMER_WINDOW_SIZE", 16))
            D_c = int(self.config.get("TRANSFORMER_D_C", 128))
            D_obs = int(self.config["GRU_HIDDEN_DIM"])

            # Unpack window state from hidden (added by initialize_carry when CT enabled)
            if isinstance(hidden, tuple) and len(hidden) == 3:
                _, obs_window_init, step_idx_init = hidden
            else:
                # Fallback: start from zeros (e.g. fine-tuning from old checkpoint)
                batch_size = obs.shape[1]
                obs_window_init = jnp.zeros((batch_size, W, D_obs))
                step_idx_init = jnp.zeros(batch_size, dtype=jnp.int32)

            T_dim, B_dim = obs.shape[:2]
            _transformer_v2 = bool(self.config.get("TRANSFORMER_V2", False))
            _transformer_v3 = bool(self.config.get("TRANSFORMER_V3", False))
            _state_shape = tuple(self.config.get("TRANSFORMER_STATE_SHAPE", []))

            if _transformer_v3:
                # CT v3: partner GRU z 복원 기반
                cycle_module = CycleTransformerModuleV3(
                    d_model=D_c,
                    d_obs=D_obs,
                    d_gru=int(self.config["GRU_HIDDEN_DIM"]),
                    action_dim=self.action_dim,
                    window_size=W,
                    n_heads=int(self.config.get("TRANSFORMER_N_HEADS", 4)),
                    n_layers=int(self.config.get("TRANSFORMER_N_LAYERS", 1)),
                    activation=self.config.get("ACTIVATION", "relu"),
                    obs_encoder_type=self.config.get("OBS_ENCODER", "CNN"),
                    name="cycle_transformer",
                )
            else:
                # CT v1/v2
                cycle_module = CycleTransformerModule(
                    d_model=D_c,
                    d_obs=D_obs,
                    action_dim=self.action_dim,
                    window_size=W,
                    n_heads=int(self.config.get("TRANSFORMER_N_HEADS", 4)),
                    n_layers=int(self.config.get("TRANSFORMER_N_LAYERS", 1)),
                    activation=self.config.get("ACTIVATION", "relu"),
                    v2=_transformer_v2,
                    state_shape=_state_shape if _transformer_v2 else (),
                    obs_encoder_type=self.config.get("OBS_ENCODER", "CNN"),
                    name="cycle_transformer",
                )

            # CT obs encoder: independent from shared_encoder, trainable via CT losses.
            # rollout window 빌드에는 stop_gradient 적용 → policy gradient가 ct_obs_encoder에 흐르지 않음.
            # _loss_fn의 window 재구성 경로에서는 stop_gradient 없이 CT loss gradient가 흐름.
            ct_obs_emb_raw = jax.vmap(cycle_module.encode_obs)(obs)  # (T, B, D_obs)
            sg_ct_obs_emb = jax.lax.stop_gradient(ct_obs_emb_raw)

            def _window_step(carry, inputs):
                obs_window, step_idx = carry
                ct_obs_emb_t, done_t = inputs

                # Reset window and counter at episode boundaries
                obs_window = jnp.where(
                    done_t[:, None, None], jnp.zeros_like(obs_window), obs_window
                )
                step_idx = jnp.where(done_t, jnp.zeros_like(step_idx), step_idx)

                # Shift window left and write current CT obs embedding at the end
                obs_window = jnp.roll(obs_window, shift=-1, axis=1)
                obs_window = obs_window.at[:, -1, :].set(ct_obs_emb_t)

                # Padding mask: True = valid slot
                slots = jnp.arange(W)
                valid_from = W - 1 - jnp.minimum(step_idx, W - 1)
                padding_mask = slots[None, :] >= valid_from[:, None]  # (B, W)

                return (obs_window, step_idx + 1), (obs_window, padding_mask)

            (
                _new_obs_window,
                _new_step_idx,
            ), (obs_windows_seq, pad_masks_seq) = jax.lax.scan(
                _window_step,
                (obs_window_init, step_idx_init),
                (sg_ct_obs_emb, dones),
            )
            # obs_windows_seq: (T, B, W, D_obs),  pad_masks_seq: (T, B, W)

            z_hat_sg_flat, a_hat_sg_flat = cycle_module.encode_only(
                obs_windows_seq.reshape(T_dim * B_dim, W, D_obs),
                pad_masks_seq.reshape(T_dim * B_dim, W),
            )
            _ct_z_hat_sg = z_hat_sg_flat.reshape(T_dim, B_dim, D_obs)
            _ct_a_hat_sg = a_hat_sg_flat.reshape(T_dim, B_dim, self.action_dim)

        # [STA-PH1] blocked_states가 이미지(상태/관측)인 경우 인코딩
        # NOTE: blocked target is expected to be a *global full* state, which may
        # have different channel count from the execution observation.
        blocked_emb = None
        blocked_emb_slots = None
        if blocked_states is not None:
            blocked_states_in = blocked_states.astype(jnp.float32)

            # 인코딩 대상 판별:
            #   좌표 기반 blocked target (B, 2) → ndim=2 → 인코딩 불필요
            #   1D 관측 환경(MPE) (T,B,D) 또는 (B,K,D) → ndim=3 → 인코딩 필요
            #   이미지 환경(Overcooked) (T,B,H,W,C) 등 → ndim≥4 → 인코딩 필요
            # NOTE:
            #   PH1의 blocked target(tilde{s})은 full state일 수 있어
            #   execution obs(agent_view_size 적용)와 spatial shape가 달라도 정상이다.
            is_image_like = blocked_states_in.ndim >= 3

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

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        # Action-prediction: compute pred_logits from GRU output
        # When transformer_action=True, CycleTransformer replaces PartnerPredictor.
        pred_logits = None
        z_partner_hat = None
        gru_output = embedding  # GRU 직후 출력 (cycle loss target용으로 extras에 저장)

        z_prediction_enabled = bool(self.config.get("Z_PREDICTION_ENABLED", False))

        if transformer_action:
            # Concatenate z + sg(ẑ) + sg(â) as policy input
            embedding = jnp.concatenate(
                [embedding, _ct_z_hat_sg, _ct_a_hat_sg], axis=-1
            )
            pred_logits = _ct_a_hat_sg  # used for logging (already stopped)
        else:
            z_sg = jax.lax.stop_gradient(embedding)

            # Action prediction (PartnerPredictor / E3T)
            # num_partners: 2-agent=1, 3-agent=2 → pred_logits 차원이 달라짐
            if self._action_prediction_enabled():
                num_partners = self.config.get("NUM_PARTNERS", 1)
                pred_logits = PartnerPredictor(
                    action_dim=self.action_dim, num_partners=num_partners, name="predictor"
                )(z_sg)

            # Partner z prediction (ZPredictor) — sg 입력, PPO gradient 차단
            if z_prediction_enabled:
                z_partner_hat = ZPredictor(
                    hidden_dim=self.config["GRU_HIDDEN_DIM"],
                    output_dim=self.config["GRU_HIDDEN_DIM"],
                    name="z_predictor",
                )(z_sg)

            # Policy input 구성: [z_GRU, (sg(z_partner_hat)), (pred_logits)]
            parts = [embedding]
            if z_partner_hat is not None:
                parts.append(jax.lax.stop_gradient(z_partner_hat))
            if pred_logits is not None:
                parts.append(pred_logits)
            if len(parts) > 1:
                embedding = jnp.concatenate(parts, axis=-1)

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
            # CT obs embedding (CT-own encoder output, used in _loss_fn for window rebuild)
            "ct_obs_emb": ct_obs_emb_raw if transformer_action else None,
            "blocked_emb": blocked_emb,
            "blocked_emb_slots": blocked_emb_slots,
            "pred_logits": pred_logits,
            "z_partner_hat": z_partner_hat,  # (T, B, D) or None — Z_PREDICTION용
            "gru_output": gru_output,        # (T, B, D) — cycle loss target용
        }

        # Build new hidden state
        if transformer_action:
            new_hidden = (rnn_state, _new_obs_window, _new_step_idx)
        else:
            new_hidden = rnn_state

        return new_hidden, pi, jnp.squeeze(critic, axis=-1), extras

    @nn.compact
    def cycle_decode(self, x):
        """CycleDecoder forward — CT OFF cycle loss 계산용.

        _loss_fn에서 method=network.cycle_decode 로 호출.
        입력: sg(pred_logits)(6) 또는 concat(sg(z_partner_hat)(128), sg(pred_logits)(6))
        출력: z_hat (GRU_HIDDEN_DIM,) — ego z_GRU 복원

        Args:
            x: (N, D_in) where D_in varies by config (6, 128, 134)
        Returns:
            (N, GRU_HIDDEN_DIM)
        """
        D = int(self.config["GRU_HIDDEN_DIM"])
        decoder = CycleDecoder(
            hidden_dim=D,
            output_dim=D,
            name="cycle_decoder",
        )
        return decoder(x)

    @nn.compact
    def ct_encode_state(self, obs):
        """CT state encoder forward — v1 전용 recon target 계산용.

        v2에서는 recon target이 raw global_obs (pixel space)이므로 이 메서드를 호출하지 않음.
        Uses CT-own ct_state_encoder (independent of blocked_encoder).
        Parameters shared via name="cycle_transformer".

        Args:
            obs: (T, B, H, W, C) or (B, H, W, C) raw observations (full global state)
        Returns:
            (T, B, D_obs) or (B, D_obs) state embedding
        """
        D_c = int(self.config.get("TRANSFORMER_D_C", 128))
        D_obs = int(self.config["GRU_HIDDEN_DIM"])
        _transformer_v2 = bool(self.config.get("TRANSFORMER_V2", False))
        _state_shape = tuple(self.config.get("TRANSFORMER_STATE_SHAPE", []))
        ct_module = CycleTransformerModule(
            d_model=D_c,
            d_obs=D_obs,
            action_dim=self.action_dim,
            window_size=int(self.config.get("TRANSFORMER_WINDOW_SIZE", 16)),
            n_heads=int(self.config.get("TRANSFORMER_N_HEADS", 4)),
            n_layers=int(self.config.get("TRANSFORMER_N_LAYERS", 1)),
            activation=self.config.get("ACTIVATION", "relu"),
            v2=_transformer_v2,
            state_shape=_state_shape if _transformer_v2 else (),
            obs_encoder_type=self.config.get("OBS_ENCODER", "CNN"),
            name="cycle_transformer",  # must match name in __call__
        )
        if obs.ndim == 5:
            return jax.vmap(ct_module.encode_state)(obs)
        return ct_module.encode_state(obs)

    @nn.compact
    def cycle_transformer_forward(self, obs_windows, padding_masks=None):
        """Full CycleTransformer forward for auxiliary loss computation in _loss_fn.

        Uses the same module name as in __call__ so parameters are shared.
        Only called when transformer_action=True.

        Args:
            obs_windows:   (N, W, D_obs) where N = T*B (flattened time*batch)
            padding_masks: (N, W) bool, True = valid slot (may be None)
        Returns:
            v1/v2: (C, state_out, a_hat, C_prime) each of shape (N, ...)
            v3:    (C, z_partner_hat, a_hat, C_prime) each of shape (N, ...)
        """
        W = int(self.config.get("TRANSFORMER_WINDOW_SIZE", 16))
        D_c = int(self.config.get("TRANSFORMER_D_C", 128))
        D_obs = int(self.config["GRU_HIDDEN_DIM"])
        _transformer_v2 = bool(self.config.get("TRANSFORMER_V2", False))
        _transformer_v3 = bool(self.config.get("TRANSFORMER_V3", False))
        _state_shape = tuple(self.config.get("TRANSFORMER_STATE_SHAPE", []))

        if _transformer_v3:
            module = CycleTransformerModuleV3(
                d_model=D_c,
                d_obs=D_obs,
                d_gru=int(self.config["GRU_HIDDEN_DIM"]),
                action_dim=self.action_dim,
                window_size=W,
                n_heads=int(self.config.get("TRANSFORMER_N_HEADS", 4)),
                n_layers=int(self.config.get("TRANSFORMER_N_LAYERS", 1)),
                activation=self.config.get("ACTIVATION", "relu"),
                obs_encoder_type=self.config.get("OBS_ENCODER", "CNN"),
                name="cycle_transformer",
            )
        else:
            module = CycleTransformerModule(
                d_model=D_c,
                d_obs=D_obs,
                action_dim=self.action_dim,
                window_size=W,
                n_heads=int(self.config.get("TRANSFORMER_N_HEADS", 4)),
                n_layers=int(self.config.get("TRANSFORMER_N_LAYERS", 1)),
                activation=self.config.get("ACTIVATION", "relu"),
                v2=_transformer_v2,
                state_shape=_state_shape if _transformer_v2 else (),
                obs_encoder_type=self.config.get("OBS_ENCODER", "CNN"),
                name="cycle_transformer",
            )
        return module(obs_windows, padding_masks)
