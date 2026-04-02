import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class PartnerPredictor(nn.Module):
    """
    GRU hidden state를 입력으로 받아 파트너의 다음 행동을 예측합니다.
    stop_gradient를 호출하는 측에서 적용하며, 이 모듈 자체는 순수 Dense 네트워크입니다.

    num_partners=1: 기존 2-agent (출력 action_dim)
    num_partners=2: 3-agent (출력 action_dim * 2, obs 순서대로 partner 0/1)
    """
    action_dim: int = 6
    num_partners: int = 1

    @nn.compact
    def __call__(self, embedding):
        """
        Args:
            embedding: (Batch, hidden_dim) - GRU output (stop_gradient applied by caller)
        Returns:
            (Batch, action_dim * num_partners) - L2 normalized prediction logits
        """
        output_dim = self.action_dim * self.num_partners
        x = embedding

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        x = nn.Dense(
            output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)

        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-6)

        return x
