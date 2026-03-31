"""
GAMMA VAE 모델 (Flax/JAX).

원본: mapbt/scripts/overcooked_population/vae_constructor/vae_model.py
구조: ObsEncoder + AddZMLP + VAEEncoder(obs+action→z) + VAEDecoder(obs+z→action)

원본과의 핵심 차이점 해결:
- VAE가 자체 obs encoder를 보유 (원본의 Flatten_Z_Actor.base() 대응)
- obs_feat는 VAE 내부에서 추출 — policy network params와 독립
"""

import functools
import jax
import jax.numpy as jnp
import flax.linen as nn


class ObsEncoder(nn.Module):
    """
    VAE 자체 obs encoder (원본 Flatten_Z_Actor.base() 대응).
    policy network와 독립된 가중치로, VAE 학습 시 같이 학습됨.
    """
    hidden_dim: int
    obs_encoder_type: str = "CNN"

    @nn.compact
    def __call__(self, obs):
        if self.obs_encoder_type == "MLP":
            x = obs.reshape(obs.shape[:-3] + (-1,)) if obs.ndim > 2 else obs
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        else:
            x = nn.Conv(32, (3, 3), padding="SAME")(obs)
            x = nn.relu(x)
            x = nn.Conv(64, (3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.Conv(32, (3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = x.reshape(x.shape[:-3] + (-1,))
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return x


class AddZMLP(nn.Module):
    """z 벡터를 feature에 결합하는 MLP. 원본 Add_Z_MLP 포팅."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, feature, z):
        z_feat = nn.Dense(self.hidden_dim)(z)
        z_feat = nn.relu(z_feat)
        z_feat = nn.Dense(self.hidden_dim)(z_feat)
        z_feat = nn.relu(z_feat)
        z_feat = nn.LayerNorm()(z_feat)

        merged = jnp.concatenate([feature, z_feat], axis=-1)
        out = nn.Dense(self.output_dim)(merged)
        out = nn.relu(out)
        out = nn.LayerNorm()(out)
        return out


class ScannedGRU(nn.Module):
    """Flax nn.scan 기반 GRU. ScannedRNN과 동일 패턴 (reset 없음)."""
    features: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        new_carry, y = nn.GRUCell(features=self.features)(carry, x)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size, features):
        cell = nn.GRUCell(features=features)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, features))


class VAEEncoder(nn.Module):
    """(obs, action) → ObsEncoder → AddZMLP → ScannedGRU → z_mean, z_logvar"""
    hidden_dim: int
    z_dim: int
    action_dim: int
    obs_encoder_type: str = "CNN"

    @nn.compact
    def __call__(self, obs_seq, action_seq, carry):
        """
        Args:
            obs_seq: (T, B, *obs_shape) — raw observation
            action_seq: (T, B) — integer actions
            carry: (B, hidden_dim)
        """
        # 자체 obs encoder (원본 base(obs))
        obs_feat = jax.vmap(jax.vmap(ObsEncoder(self.hidden_dim, self.obs_encoder_type)))(obs_seq)

        action_onehot = jax.nn.one_hot(action_seq, self.action_dim)
        add_z = AddZMLP(self.hidden_dim, self.hidden_dim)
        fused = jax.vmap(jax.vmap(add_z))(obs_feat, action_onehot)

        final_carry, all_hidden = ScannedGRU(features=self.hidden_dim)(carry, fused)
        last_hidden = all_hidden[-1]
        z_mean = nn.Dense(self.z_dim)(last_hidden)
        z_logvar = nn.Dense(self.z_dim)(last_hidden)

        return z_mean, z_logvar, final_carry


class VAEDecoder(nn.Module):
    """(obs, z) → ObsEncoder → AddZMLP → ScannedGRU → action_logits"""
    hidden_dim: int
    z_dim: int
    action_dim: int
    obs_encoder_type: str = "CNN"

    @nn.compact
    def __call__(self, obs_seq, z, carry):
        """
        Args:
            obs_seq: (T, B, *obs_shape) — raw observation
            z: (B, z_dim)
            carry: (B, hidden_dim)
        """
        T = obs_seq.shape[0]
        # 자체 obs encoder
        obs_feat = jax.vmap(jax.vmap(ObsEncoder(self.hidden_dim, self.obs_encoder_type)))(obs_seq)

        z_expanded = jnp.broadcast_to(z[jnp.newaxis], (T,) + z.shape)
        add_z = AddZMLP(self.hidden_dim, self.hidden_dim)
        fused = jax.vmap(jax.vmap(add_z))(obs_feat, z_expanded)

        final_carry, all_hidden = ScannedGRU(features=self.hidden_dim)(carry, fused)
        action_logits = nn.Dense(self.action_dim)(all_hidden)

        return action_logits, final_carry


class GAMMAVAE(nn.Module):
    """
    완전한 VAE (encoder + decoder).
    자체 ObsEncoder 내장 — raw observation을 직접 처리.
    원본의 VAEModel 구조와 동일.
    """
    hidden_dim: int = 256
    z_dim: int = 16
    action_dim: int = 6
    obs_encoder_type: str = "CNN"

    def setup(self):
        self.encoder = VAEEncoder(
            self.hidden_dim, self.z_dim, self.action_dim, self.obs_encoder_type,
        )
        self.decoder = VAEDecoder(
            self.hidden_dim, self.z_dim, self.action_dim, self.obs_encoder_type,
        )

    def __call__(self, obs_seq, action_seq, enc_carry, dec_carry, rng):
        """obs_seq: (T, B, *obs_shape) — raw observation (not pre-encoded feat)."""
        z_mean, z_logvar, enc_carry = self.encoder(obs_seq, action_seq, enc_carry)
        z = self.reparameterize(z_mean, z_logvar, rng)
        action_logits, dec_carry = self.decoder(obs_seq, z, dec_carry)
        return action_logits, z_mean, z_logvar, z, enc_carry, dec_carry

    def decode(self, obs_seq, z, carry):
        """Decoder만 호출 (S2 RL에서 partner action 생성용)."""
        return self.decoder(obs_seq, z, carry)

    @staticmethod
    def reparameterize(mean, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mean.shape)
        return mean + std * eps

    @staticmethod
    def kl_divergence(mean, logvar):
        return -0.5 * jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=-1)
