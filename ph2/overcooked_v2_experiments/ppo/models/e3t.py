import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class PartnerPredictor(nn.Module):
    """
    Simple MLP that takes a flat embedding (GRU hidden state) and outputs
    L2-normalised action logits used for partner action prediction.
    """
    action_dim: int = 6

    @nn.compact
    def __call__(self, embedding):
        x = embedding
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-6)
        return x
