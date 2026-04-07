import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class PartnerPredictor(nn.Module):
    """
    GRU hidden state를 입력으로 받아 파트너의 다음 행동을 예측합니다.
    CEC-zero-shot 구현 기준 5층 구조.

    num_partners=1: 기존 2-agent (출력 action_dim)
    num_partners=2: 3-agent (출력 action_dim * 2, obs 순서대로 partner 0/1)
    """
    action_dim: int = 6
    num_partners: int = 1

    @nn.compact
    def __call__(self, embedding):
        """
        Args:
            embedding: (Batch, hidden_dim) - GRU output
        Returns:
            (Batch, action_dim * num_partners) - L2 normalized prediction logits
        """
        output_dim = self.action_dim * self.num_partners
        x = embedding

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.leaky_relu(x)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.leaky_relu(x)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.leaky_relu(x)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)

        x = nn.Dense(output_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)

        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-10)
        x = x / norm

        return x
