import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class PartnerPredictor(nn.Module):
    """
    GRU hidden state를 입력으로 받아 파트너의 다음 행동을 예측합니다.

    구조: num_hidden_layers 개의 Dense(64) 중간층 + Dense(output_dim) 출력층.
      - Modern (default, CEC-zero-shot 기준): num_hidden_layers=4 → 총 5 Dense.
      - Legacy (초기 체크포인트): num_hidden_layers=2 → 총 3 Dense.

    중간층 activation:
      - 마지막 중간층: tanh (modern 은 Dense_3 에 tanh, legacy 는 Dense_1 에 tanh)
      - 그 외: leaky_relu

    num_partners=1: 기존 2-agent (출력 action_dim)
    num_partners=2: 3-agent (출력 action_dim * 2)
    """
    action_dim: int = 6
    num_partners: int = 1
    num_hidden_layers: int = 4     # modern default. legacy ckpt 는 2.

    @nn.compact
    def __call__(self, embedding):
        output_dim = self.action_dim * self.num_partners
        x = embedding

        # 중간 Dense(64) 층들. 마지막 중간층은 tanh, 나머지는 leaky_relu.
        for i in range(self.num_hidden_layers):
            x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
            if i == self.num_hidden_layers - 1:
                x = nn.tanh(x)
            else:
                x = nn.leaky_relu(x)

        # 출력 Dense
        x = nn.Dense(output_dim, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x)

        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-10)
        x = x / norm
        return x
