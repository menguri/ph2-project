import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class PartnerPredictor(nn.Module):
    """num_partners=1: 2-agent (출력 action_dim), num_partners=2: 3-agent (출력 action_dim*2)"""
    action_dim: int = 6
    num_partners: int = 1

    @nn.compact
    def __call__(self, embedding):
        output_dim = self.action_dim * self.num_partners
        x = embedding
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-6)
        return x


class ZPredictor(nn.Module):
    """Partner GRU hidden state 복원. sg(z_ego) → z_partner_hat.

    입력: sg(z_GRU) (D,)
    출력: z_partner_hat (output_dim,) — partner의 GRU hidden state 추정
    """
    hidden_dim: int = 128
    output_dim: int = 128

    @nn.compact
    def __call__(self, embedding):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return nn.LayerNorm()(x)


class CycleDecoder(nn.Module):
    """Auxiliary prediction → z_GRU 복원 (cycle consistency).

    입력: sg(pred_logits)(6) 또는 concat(sg(z_partner_hat)(128), sg(pred_logits)(6)) = 134
    출력: z_hat (output_dim,) — ego z_GRU 복원
    """
    hidden_dim: int = 128
    output_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return nn.LayerNorm()(x)
