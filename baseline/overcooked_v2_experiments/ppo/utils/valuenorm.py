"""
ValueNorm: JAX 구현.
HSP 참조 코드(PyTorch)의 ValueNorm을 JAX/jnp로 포팅.
EMA(Exponential Moving Average) 기반 running mean/var로 value target을 정규화.

사용법:
    state = ValueNormState.create()
    state = valuenorm_update(state, targets)                 # 통계 업데이트
    normalized_targets = valuenorm_normalize(state, targets)  # 정규화
    denorm_values = valuenorm_denormalize(state, values)      # 역정규화 (로깅용)
"""

import jax.numpy as jnp
from typing import NamedTuple


class ValueNormState(NamedTuple):
    """ValueNorm running statistics. jax.lax.scan에서 carry로 전달."""
    running_mean: jnp.ndarray       # scalar
    running_mean_sq: jnp.ndarray    # scalar
    debiasing_term: jnp.ndarray     # scalar

    @staticmethod
    def create():
        return ValueNormState(
            running_mean=jnp.float32(0.0),
            running_mean_sq=jnp.float32(0.0),
            debiasing_term=jnp.float32(0.0),
        )


# 원본 HSP 참조 코드: beta=0.99999
VALUENORM_BETA = 0.99999
VALUENORM_EPS = 1e-5


def valuenorm_update(state: ValueNormState, values: jnp.ndarray, beta=VALUENORM_BETA) -> ValueNormState:
    """EMA 통계 업데이트. values: 임의 shape (평균은 전체 원소에 대해 계산)."""
    batch_mean = values.mean()
    batch_mean_sq = (values ** 2).mean()
    new_running_mean = state.running_mean * beta + batch_mean * (1.0 - beta)
    new_running_mean_sq = state.running_mean_sq * beta + batch_mean_sq * (1.0 - beta)
    new_debiasing = state.debiasing_term * beta + (1.0 - beta)
    return ValueNormState(
        running_mean=new_running_mean,
        running_mean_sq=new_running_mean_sq,
        debiasing_term=new_debiasing,
    )


def _get_mean_var(state: ValueNormState):
    """Debiased mean과 variance 계산."""
    debiased_mean = state.running_mean / jnp.maximum(state.debiasing_term, VALUENORM_EPS)
    debiased_mean_sq = state.running_mean_sq / jnp.maximum(state.debiasing_term, VALUENORM_EPS)
    debiased_var = jnp.maximum(debiased_mean_sq - debiased_mean ** 2, 1e-2)
    return debiased_mean, debiased_var


def valuenorm_normalize(state: ValueNormState, values: jnp.ndarray) -> jnp.ndarray:
    """Value target을 정규화: (x - mean) / std."""
    mean, var = _get_mean_var(state)
    return (values - mean) / jnp.sqrt(var)


def valuenorm_denormalize(state: ValueNormState, values: jnp.ndarray) -> jnp.ndarray:
    """정규화된 value를 원래 스케일로 복원 (로깅용)."""
    mean, var = _get_mean_var(state)
    return values * jnp.sqrt(var) + mean
