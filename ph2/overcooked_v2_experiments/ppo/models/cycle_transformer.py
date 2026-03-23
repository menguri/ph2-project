"""CycleTransformer: Causal-Transformer-based context module.

Architecture (transformer_action=True):
  obs → ct_obs_encoder → ct_obs_emb history → Window Stack → CausalTransformerEncoder → C
  C → StateDecoder → ẑ  (state latent reconstruction)
  C → ActionDecoder → â  (partner action prediction)
  sg(ẑ), sg(â) → CycleEncoder → C'  (cycle-consistency loss)

  Recon target: obs → ct_state_encoder → recon_target  (CT-own encoder, independent)

--- CT v1 (default) ---
  StateDecoder: C(D_c) → z_hat(D_obs)  — latent space reconstruction
  Recon loss:   MSE(z_hat, sg(ct_state_encoder(global_obs)))  — latent space
  Policy input: concat([z_GRU, sg(z_hat), sg(a_hat)])  — 128+128+6=262

--- CT v2 (TRANSFORMER_V2=True) ---
  StateDecoder: C(D_c) → s_hat(H,W,C_full)  — pixel space reconstruction
  Recon loss:   MSE(s_hat, global_obs)  — pixel space (per-channel logging 포함)
  z_from_s = ct_state_encoder(sg(s_hat))  — frozen CNN으로 s_hat 임베딩
  Policy input: concat([z_GRU, sg(z_from_s), sg(a_hat)])  — 128+128+6=262 동일
  Cycle input:  concat([sg(z_from_s), sg(a_hat)])  — v1과 동일 dim

Gradient flow:
  obs → ct_obs_encoder → window:  rollout은 stop_gradient, _loss_fn 경로는 gradient 흐름
  obs → ct_state_encoder → recon_target:  stop_gradient (v1, fixed random projection)
  v2: s_hat → ct_state_encoder(frozen) → z_from_s: stop_gradient on s_hat 입력
  C → StateDecoder, ActionDecoder:  gradient flows
  ẑ/s_hat, â → CycleEncoder:   stop_gradient
  C' target = sg(C):      C encoder does not receive cycle loss gradient
  ẑ, â → Policy:         stop_gradient

When transformer_action=False the module is never instantiated and existing
code paths are completely unchanged.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from .common import CNN
from flax.linen.initializers import constant, orthogonal


class _CTFlattenMLP(nn.Module):
    """CT 전용 FlattenMLP — obs를 flatten 후 Dense 통과."""
    hidden_size: int = 128
    output_size: int = 128
    activation: type = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        return x


class CausalTransformerEncoder(nn.Module):
    d_model: int
    n_heads: int = 4
    n_layers: int = 1

    @nn.compact
    def __call__(self, obs_window, padding_mask=None):
        """
        Args:
            obs_window:    (B, W, D_obs)
            padding_mask:  (B, W) bool, True = valid slot (may be None)
        Returns:
            (B, D_c) – representation of the current (last) position
        """
        B, W, _ = obs_window.shape

        x = nn.Dense(self.d_model)(obs_window)  # (B, W, D_c)

        # Lower-triangular causal mask: (W, W)
        causal_mask = jnp.where(jnp.tril(jnp.ones((W, W))), 0.0, -1e9)

        for i in range(self.n_layers):
            mask = causal_mask  # (W, W) or broadcastable
            if padding_mask is not None:
                # padding_mask: (B, W) True=valid → add -inf to padded keys
                pad = jnp.where(padding_mask, 0.0, -1e9)[:, None, None, :]  # (B,1,1,W)
                mask = causal_mask + pad  # (B, W, W) via broadcast

            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, name=f"attn_{i}"
            )(x, x, mask=mask)
            x = nn.LayerNorm(name=f"ln1_{i}")(x + attn_out)
            h = nn.relu(nn.Dense(self.d_model * 4, name=f"ff1_{i}")(x))
            x = nn.LayerNorm(name=f"ln2_{i}")(x + nn.Dense(self.d_model, name=f"ff2_{i}")(h))

        return x[:, -1, :]  # (B, D_c) – last (current) timestep


class CycleTransformerModule(nn.Module):
    """Full CycleTransformer used as a submodule of ActorCriticRNN.

    Contains two independent CNNs:
      ct_obs_encoder:   raw obs → ct_obs_emb  (window input for transformer, trainable)
      ct_state_encoder: raw obs → recon_target (reconstruction target, fixed random projection)
    Both are CT-internal and completely independent of shared_encoder / blocked_encoder.

    v1 (default): latent state reconstruction (C → z_hat)
    v2:           pixel state reconstruction  (C → s_hat → ct_state_encoder(frozen) → z_from_s)
    """

    d_model: int       # D_c: transformer hidden dimension
    d_obs: int         # D_obs = GRU_HIDDEN_DIM (obs/state encoder output size)
    action_dim: int    # number of discrete actions
    window_size: int   # W: observation history window size
    n_heads: int = 4
    n_layers: int = 1
    activation: str = "relu"  # matches global ACTIVATION config
    v2: bool = False           # CT v2: pixel space state reconstruction
    state_shape: tuple = ()    # (H, W, C_full) required when v2=True
    obs_encoder_type: str = "CNN"  # "CNN" or "MLP" — ToyCoop 등 작은 환경용

    def setup(self):
        act = nn.relu if self.activation == "relu" else nn.tanh

        def _make_encoder(name_suffix=""):
            if self.obs_encoder_type.upper() == "MLP":
                return _CTFlattenMLP(hidden_size=self.d_obs, output_size=self.d_obs, activation=act)
            return CNN(output_size=self.d_obs, activation=act)

        # CT-own obs encoder: raw obs → obs_emb (used as window input, trainable via CT losses)
        self.ct_obs_encoder = _make_encoder()
        self.ct_obs_encoder_ln = nn.LayerNorm()
        # CT-own state encoder: raw obs/s_hat → state_emb (fixed random projection)
        self.ct_state_encoder = _make_encoder()
        self.ct_state_encoder_ln = nn.LayerNorm()
        # Causal Transformer
        self.encoder = CausalTransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        )

        if self.v2:
            # v2 전용: pixel space decoder (C → s_hat)
            H, W, C_full = self.state_shape
            self.pixel_fc1 = nn.Dense(self.d_model * 2)
            self.pixel_fc2 = nn.Dense(H * W * C_full)
        else:
            # v1 전용: latent state decoder (C → z_hat)
            self.state_fc = nn.Dense(self.d_obs)
            self.state_ln = nn.LayerNorm()

        # shared: ActionDecoder, CycleEncoder
        self.action_hidden = nn.Dense(64)
        self.action_out = nn.Dense(self.action_dim)
        self.cycle_hidden = nn.Dense(self.d_model)
        self.cycle_out = nn.Dense(self.d_model)

    def encode_obs(self, obs_t):
        """Encode a single-step raw obs for the window buffer.

        Args:
            obs_t: (B, H, W, C) raw observation
        Returns:
            (B, D_obs) CT obs embedding
        """
        return self.ct_obs_encoder_ln(self.ct_obs_encoder(obs_t))

    def encode_state(self, obs_t):
        """Encode a single-step raw obs as reconstruction target (v1).

        Args:
            obs_t: (B, H, W, C) raw observation (full global state)
        Returns:
            (B, D_obs) CT state embedding (stop_gradient applied by caller)
        """
        return self.ct_state_encoder_ln(self.ct_state_encoder(obs_t))

    def _decode_state(self, C):
        """State decode: C → state_out.

        v1: C(D_c) → z_hat(D_obs)        — latent
        v2: C(D_c) → s_hat(H, W, C_full) — pixel space

        Args:
            C: (B, D_c)
        Returns:
            v1: z_hat  (B, D_obs)
            v2: s_hat  (B, H, W, C_full)
        """
        if self.v2:
            H, W, C_full = self.state_shape
            s_flat = self.pixel_fc2(nn.relu(self.pixel_fc1(C)))  # (B, H*W*C_full)
            return s_flat.reshape((-1, H, W, C_full))            # (B, H, W, C_full)
        else:
            return self.state_ln(self.state_fc(C))               # (B, D_obs)

    def _state_to_z(self, state_out):
        """state_out → z (D_obs) for CycleEncoder / policy input.

        v1: state_out is z_hat (D_obs) — already latent, use directly
        v2: state_out is s_hat (H,W,C_full) → ct_state_encoder(frozen) → z_from_s (D_obs)
            ct_state_encoder의 gradient는 흐르지 않음 (stop_gradient on s_hat input).

        Args:
            state_out: v1=(B,D_obs), v2=(B,H,W,C_full)
        Returns:
            z (B, D_obs)
        """
        if self.v2:
            s_hat_sg = jax.lax.stop_gradient(state_out)
            return self.ct_state_encoder_ln(self.ct_state_encoder(s_hat_sg))
        else:
            return state_out  # z_hat already

    def _encode(self, obs_window, padding_mask=None):
        """Shared forward: transformer → decoders.

        Args:
            obs_window: (B, W, D_obs) pre-computed CT obs embeddings
        Returns:
            C:         (B, D_c)
            state_out: v1=(B,D_obs), v2=(B,H,W,C_full)
            a_hat:     (B, A)
        """
        C = self.encoder(obs_window, padding_mask)                        # (B, D_c)
        state_out = self._decode_state(C)                                 # v1/v2 분기
        a_hat = self.action_out(nn.relu(self.action_hidden(C)))           # (B, A)
        return C, state_out, a_hat

    def __call__(self, obs_window, padding_mask=None):
        """Full forward for auxiliary loss computation (includes CycleEncoder).

        Returns:
            C:         (B, D_c)   context vector
            state_out: v1=(B,D_obs) z_hat / v2=(B,H,W,C_full) s_hat
            a_hat:     (B, A)     partner action logits
            C_prime:   (B, D_c)   cycle-reconstructed context
        """
        C, state_out, a_hat = self._encode(obs_window, padding_mask)

        # CycleEncoder: sg(z) + sg(a_hat) → C_prime
        # v1: z = z_hat (already D_obs)
        # v2: z = ct_state_encoder(sg(s_hat)) (frozen CNN)
        z = self._state_to_z(state_out)  # (B, D_obs)
        z_sg = jax.lax.stop_gradient(z)
        a_sg = jax.lax.stop_gradient(a_hat)
        cycle_in = jnp.concatenate([z_sg, a_sg], axis=-1)
        C_prime = self.cycle_out(nn.relu(self.cycle_hidden(cycle_in)))   # (B, D_c)

        return C, state_out, a_hat, C_prime

    def encode_only(self, obs_window, padding_mask=None):
        """Rollout / eval forward – stop_gradient on outputs for policy use.

        Returns:
            z_sg:     (B, D_obs) policy용 state latent (stop_gradient)
            a_hat_sg: (B, A)     policy용 action prediction (stop_gradient)

        Both v1/v2 return same shapes → policy head dim 동일 (262).
        """
        _, state_out, a_hat = self._encode(obs_window, padding_mask)
        z = self._state_to_z(state_out)  # v1: z_hat / v2: ct_state_encoder(s_hat)
        return jax.lax.stop_gradient(z), jax.lax.stop_gradient(a_hat)
