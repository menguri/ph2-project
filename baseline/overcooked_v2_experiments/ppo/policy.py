from functools import partial
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)
import jax.numpy as jnp
from overcooked_v2_experiments.eval.policy import PolicyPairing
import jax
from flax import core
from typing import Any
import chex


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]


class PPOPolicy(AbstractPolicy):
    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: core.FrozenDict[str, Any]
    stochastic: bool = True
    with_batching: bool = False

    def __init__(self, params, config, stochastic=True, with_batching=False):
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching

        self.network = get_actor_critic(config)

        self.params = params

    def compute_action(self, obs, done, hstate, key, params=None, obs_history=None, act_history=None):
        if params is None:
            params = self.params
        assert params is not None

        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)

        # E3T prediction path must be enabled during eval as well, otherwise
        # parameter shapes mismatch (e.g., Dense expecting 128+action_dim input).
        alg_name = self.config.get("ALG_NAME", "")
        if "alg" in self.config and isinstance(self.config["alg"], dict):
            alg_name = self.config["alg"].get("ALG_NAME", alg_name)
        use_prediction = bool(self.config.get("USE_PREDICTION", True))
        model_type = self.config.get("model", {}).get("TYPE", "")
        use_pred_flag = (alg_name == "E3T") and use_prediction and (model_type == "RNN")
        actor_only_flag = alg_name in ("MEP_S1", "MEP_S2", "GAMMA_S1", "GAMMA_S2", "HSP_S1", "HSP_S2",
                                       "MEP", "GAMMA", "HSP")
        # MAPPO_MODE 로 학습된 ckpt 는 critic 이 centralized 라 critic_global_* 만 존재하고
        # 기존 IPPO critic Dense 파라미터(Dense_2/Dense_3)가 없음 → 일반 critic 호출 시 ScopeParamNotFoundError.
        # Eval 경로는 value 가 필요 없으므로 actor_only=True 로 우회.
        try:
            _params_root = params["params"] if "params" in params else params
            if any(str(k).startswith("critic_global") for k in _params_root.keys()):
                actor_only_flag = True
        except Exception:
            pass

        if use_pred_flag:
            outputs = self.network.apply(
                params,
                hstate,
                ac_in,
                use_prediction=True,
            )
        elif actor_only_flag:
            outputs = self.network.apply(
                params,
                hstate,
                ac_in,
                actor_only=True,
            )
        else:
            outputs = self.network.apply(
                params,
                hstate,
                ac_in,
            )

        if len(outputs) == 4:
            next_hstate, pi, value, pred_logits = outputs
        else:
            next_hstate, pi, value = outputs
            pred_logits = None

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]

        extras = {}

        return action, next_hstate, extras

    def init_hstate(self, batch_size, key=None):
        # assert batch_size == 1 or self.with_batching
        return initialize_carry(self.config, batch_size)


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(PPOPolicy(checkpoint.params, config))

    return PolicyPairing(*policies)
