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
import copy
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
        config = copy.deepcopy(config)
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching

        self.network = get_actor_critic(config)

        self.params = params

    def compute_action(
        self,
        obs,
        done,
        hstate,
        key,
        params=None,
        blocked_states=None,
        **kwargs,  # ignored: obs_history, act_history (GRU carries history implicitly)
    ):
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

        alg_name = self.config.get("ALG_NAME", "")
        if "alg" in self.config:
            alg_name = self.config["alg"].get("ALG_NAME", alg_name)

        ph1_enabled = "PH1" in alg_name or bool(self.config.get("PH1_ENABLED", False))
        if "alg" in self.config:
            ph1_enabled = ph1_enabled or bool(self.config["alg"].get("PH1_ENABLED", False))

        e3t_like = ("E3T" in alg_name) or ("STL" in alg_name)
        learner_use_blocked_input = bool(self.config.get("LEARNER_USE_BLOCKED_INPUT", True))
        if "alg" in self.config:
            learner_use_blocked_input = bool(
                self.config["alg"].get("LEARNER_USE_BLOCKED_INPUT", learner_use_blocked_input)
            )

        blocked_states_in = None
        if blocked_states is not None and learner_use_blocked_input and (e3t_like or ph1_enabled):
            blocked_states = jnp.array(blocked_states)
            if not self.with_batching:
                blocked_states_in = _add_dim(blocked_states)
            else:
                blocked_states_in = blocked_states
            if blocked_states_in.ndim == 2:
                blocked_states_in = blocked_states_in[jnp.newaxis, ...]

        next_hstate, pi, value, net_extras = self.network.apply(
            params,
            hstate,
            ac_in,
            blocked_states=blocked_states_in,
        )

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]
        # action = action[0, 0]
        # action = action.flatten()
        # print("Action", action)

        # Extract scalar value for this agent
        if self.with_batching:
            value_scalar = value[0]
        else:
            value_scalar = value[0, 0]

        extras = {"value": value_scalar}

        if net_extras is not None and isinstance(net_extras, dict):
            blocked_emb = net_extras.get("blocked_emb", None)
            if blocked_emb is not None:
                if self.with_batching:
                    extras["blocked_emb"] = blocked_emb[0]
                else:
                    extras["blocked_emb"] = blocked_emb[0, 0]
            blocked_emb_slots = net_extras.get("blocked_emb_slots", None)
            if blocked_emb_slots is not None:
                if self.with_batching:
                    extras["blocked_emb_slots"] = blocked_emb_slots[0]
                else:
                    extras["blocked_emb_slots"] = blocked_emb_slots[0, 0]

        return action, next_hstate, extras

    def init_hstate(self, batch_size, key=None):
        # assert batch_size == 1 or self.with_batching
        return initialize_carry(self.config, batch_size)


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(PPOPolicy(checkpoint.params, config))

    return PolicyPairing(*policies)
