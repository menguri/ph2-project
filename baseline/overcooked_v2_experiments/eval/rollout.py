from typing import List
import jax
import jax.numpy as jnp
import chex
from .policy import AbstractPolicy, PolicyPairing


@chex.dataclass
class PolicyRollout:
    state_seq: chex.Array
    actions_seq: chex.Array
    total_reward: chex.Scalar
    prediction_accuracy: chex.Array = None  # (num_agents,)


def init_rollout(policies: List[AbstractPolicy], env):
    num_agents = env.num_agents

    assert len(policies) == num_agents

    init_hstate = {f"agent_{i}": policies[i].init_hstate(1) for i in range(num_agents)}

    @jax.jit
    def _get_actions(obs, done, hstate, key):
        sample_keys = jax.random.split(key, num_agents)

        actions = {}
        next_hstates = {}
        all_extras = {}

        for i, policy in enumerate(policies):
            agent_id = f"agent_{i}"

            obs_agent, done_agent, hstate_agent = (
                obs[agent_id],
                done[agent_id],
                hstate[agent_id],
            )

            action, next_hstate, extras = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i]
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate
            all_extras[agent_id] = extras

        return actions, next_hstates, all_extras

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key, algorithm="PPO") -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)

    key, key_r = jax.random.split(key, 2)
    obs, state = env.reset(key_r)

    e3t_like = algorithm in ("E3T", "STL") or "E3T" in algorithm or "STL" in algorithm

    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry

        key_sample, key_step = jax.random.split(key, 2)

        actions, next_hstate, extras = _get_actions(obs, done, hstate, key_sample)

        # Calculate prediction accuracy
        prediction_correct = jnp.zeros(env.num_agents, dtype=jnp.float32)
        prediction_mask = jnp.zeros(env.num_agents, dtype=jnp.float32)

        if e3t_like:
            for i in range(env.num_agents):
                agent_id = f"agent_{i}"
                partner_idx = (i + 1) % env.num_agents
                partner_id = f"agent_{partner_idx}"

                if agent_id in extras and "partner_prediction" in extras[agent_id]:
                    pred_logits = extras[agent_id]["partner_prediction"]
                    pred_action = jnp.argmax(pred_logits)
                    true_action = actions[partner_id]

                    is_correct = (pred_action == true_action).astype(jnp.float32)
                    prediction_correct = prediction_correct.at[i].set(is_correct)
                    prediction_mask = prediction_mask.at[i].set(1.0)

        # STEP ENV
        next_obs, next_state, reward, next_done, info = env.step(
            key_step, state, actions
        )

        new_total_reward = total_reward + reward["agent_0"]

        carry = (next_obs, next_state, next_done, new_total_reward, next_hstate)
        return carry, (next_state, actions, prediction_correct, prediction_mask)

    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False

    keys = jax.random.split(key, env.max_steps)
    carry = (
        obs,
        state,
        init_done,
        0.0,
        init_hstate,
    )
    carry, (state_seq, actions_seq, prediction_correct_seq, prediction_mask_seq) = jax.lax.scan(
        _perform_step, carry, keys
    )

    total_reward = carry[3]

    # Calculate mean accuracy per agent
    total_correct = jnp.sum(prediction_correct_seq, axis=0)
    total_count = jnp.sum(prediction_mask_seq, axis=0)
    prediction_accuracy = jnp.where(total_count > 0, total_correct / total_count, 0.0)

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=total_reward,
        prediction_accuracy=prediction_accuracy,
    )
