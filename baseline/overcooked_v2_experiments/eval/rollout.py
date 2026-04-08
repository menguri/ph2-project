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
    total_reward_combined: chex.Scalar = None
    prediction_accuracy: chex.Array = None  # (num_agents,)


def init_rollout(policies: List[AbstractPolicy], env, use_jit=True):
    num_agents = env.num_agents

    assert len(policies) == num_agents

    init_hstate = {f"agent_{i}": policies[i].init_hstate(1) for i in range(num_agents)}

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

    # When use_jit=True, lax.scan handles JIT compilation of the scan body.
    # Wrapping _get_actions in jax.jit here would create a new JIT object per
    # init_rollout call (new local function = new id(f) = cache miss), causing
    # per-pairing recompilation in viz mode.
    # Only JIT _get_actions when use_jit=False (Python loop mode) to speed up
    # individual steps.
    if not use_jit:
        _get_actions = jax.jit(_get_actions)

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key, algorithm="PPO", use_jit=True, env_device=None, eval_reward="sparse") -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env, use_jit=use_jit)

    # Optional backend pinning for environment interaction.
    reset_fn = env.reset
    step_fn = env.step
    if env_device is not None:
        dev = str(env_device).strip().lower()
        backend = "cpu" if dev == "cpu" else ("gpu" if dev in ("gpu", "cuda") else None)
        if backend is not None:
            try:
                reset_fn = jax.jit(env.reset, backend=backend)
                step_fn = jax.jit(env.step, backend=backend)
            except Exception:
                pass  # backend not available; fall back to default

    key, key_r = jax.random.split(key, 2)
    obs, state = reset_fn(key_r)

    e3t_like = algorithm in ("E3T", "STL") or "E3T" in algorithm or "STL" in algorithm

    def _perform_step(carry, key):
        obs, state, done, total_reward, total_reward_combined, episode_done, hstate = carry

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
        next_obs, next_state, reward, next_done, info = step_fn(
            key_step, state, actions
        )

        # eval 기록용: sparse/combined 둘 다 누적 (step_cost 미포함)
        _sparse_r = info["sparse_reward"]["agent_0"] if isinstance(info, dict) and "sparse_reward" in info else reward["agent_0"]
        _combined_r = info["combined_reward"]["agent_0"] if isinstance(info, dict) and "combined_reward" in info else reward["agent_0"]
        # 첫 episode 종료(LogWrapper auto-reset) 이후는 reward 누적 차단 → episode 1회분만 기록.
        _alive = (1.0 - episode_done.astype(jnp.float32))
        new_total_reward = total_reward + _sparse_r * _alive
        new_total_reward_combined = total_reward_combined + _combined_r * _alive
        new_episode_done = episode_done | next_done["__all__"]

        carry = (next_obs, next_state, next_done, new_total_reward, new_total_reward_combined, new_episode_done, next_hstate)
        return carry, (next_state, actions, prediction_correct, prediction_mask)

    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False

    keys = jax.random.split(key, env.max_steps)
    carry = (
        obs,
        state,
        init_done,
        jnp.float32(0.0),  # total_reward (sparse)
        jnp.float32(0.0),  # total_reward_combined
        jnp.bool_(False),  # episode_done latch
        init_hstate,
    )

    if use_jit:
        carry, (state_seq, actions_seq, prediction_correct_seq, prediction_mask_seq) = jax.lax.scan(
            _perform_step, carry, keys
        )
    else:
        state_seqs, actions_seqs, pred_correct_seqs, pred_mask_seqs = [], [], [], []
        for k in keys:
            carry, (s, a, pc, pm) = _perform_step(carry, k)
            state_seqs.append(s)
            actions_seqs.append(a)
            pred_correct_seqs.append(pc)
            pred_mask_seqs.append(pm)
        state_seq = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *state_seqs)
        actions_seq = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *actions_seqs)
        prediction_correct_seq = jnp.stack(pred_correct_seqs)
        prediction_mask_seq = jnp.stack(pred_mask_seqs)

    total_reward = carry[3]  # sparse
    total_reward_combined = carry[4]  # combined

    # Calculate mean accuracy per agent
    total_correct = jnp.sum(prediction_correct_seq, axis=0)
    total_count = jnp.sum(prediction_mask_seq, axis=0)
    safe_count = jnp.maximum(total_count, 1.0)  # 0 → 1로 대체하여 NaN 방지
    prediction_accuracy = jnp.where(total_count > 0, total_correct / safe_count, 0.0)

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=total_reward,
        total_reward_combined=total_reward_combined,
        prediction_accuracy=prediction_accuracy,
    )
