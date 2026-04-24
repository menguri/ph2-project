#!/usr/bin/env python3
"""Fast BC × RL / BC × CEC cross-play — JIT + vmap 기반.

기존 `evaluate.py::run_crossplay` 는 (bc_seed × rl_seed × pos × eval_seed) 마다
새 `PolicyPairing` 을 만들어 `get_rollout` 에 넘긴다. 이 때 `init_rollout` 내부의
`_get_actions` 클로저가 Python 함수 identity 가 매 호출마다 달라서
`jax.lax.scan` 이 매번 재컴파일. pairing 1개당 ~10-30s 컴파일 × 100-200 pair =
대부분 시간을 컴파일에 소모.

여기서는:
1. **Params 를 인자로 lifting**: 네트워크는 한 번만 인스턴스화하고, `params` 만
   runtime arg 로 전달. → rollout 함수 JIT 1회 컴파일.
2. **Vmap (bc_seed × rl_seed × eval_seed)**: 모든 pairing 을 배치로 한 번에 실행.

평가 의미는 동일 — 같은 params / 같은 key 로 호출하면 기존 경로와 동일한 reward
(floating-point ULP 수준 차이만 가능).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# human-proxy/code 경로
sys.path.insert(0, str(Path(__file__).parent))
from policy import BCPolicy  # noqa


# ══════════════════════════════════════════════════════════════════
#   BC apply (stateless — params 를 arg 로)
# ══════════════════════════════════════════════════════════════════

def _build_bc_apply(bc_policy: BCPolicy):
    """BC forward 를 pure function 으로 추출.

    Returns: apply(params, obs, hstate, done, key) → (action, next_hstate)
      - obs: (H, W, C) uint8/int
      - hstate: unblock_if_stuck 시 (1, BCHState.flat_dim)  그 외 None
    """
    from policy import BCHState, _remove_and_renormalize

    network = bc_policy.network
    action_dim = network.action_dim
    stochastic = bc_policy.stochastic
    unblock = bc_policy.unblock_if_stuck

    def apply(params, obs, hstate, done, key):
        obs_float = obs.astype(jnp.float32) / 255.0
        obs_batch = obs_float[jnp.newaxis, ...]
        logits = network.apply({"params": params}, obs_batch)
        action_probs = jax.nn.softmax(logits[0])

        if unblock:
            h_flat = hstate[0] if hstate.ndim == 2 else hstate
            h_flat = jnp.where(done, BCHState.init_empty().to_flat(), h_flat)
            bc_h = BCHState.from_flat(h_flat)
            is_stuck = bc_h.is_stuck()
            unstuck_probs = _remove_and_renormalize(action_probs, bc_h.actions)
            action_probs = jnp.where(is_stuck, unstuck_probs, action_probs)

        if stochastic:
            action = jax.random.choice(key, action_dim, p=action_probs)
        else:
            action = jnp.argmax(action_probs)

        if unblock:
            bc_h = BCHState.from_flat(h_flat).append(action, None)
            next_hstate = bc_h.to_flat()[jnp.newaxis, ...]
        else:
            next_hstate = hstate
        return action, next_hstate

    return apply


def _bc_init_hstate(bc_policy: BCPolicy):
    return bc_policy.init_hstate(batch_size=1)


# ══════════════════════════════════════════════════════════════════
#   RL (PPO-family) apply
# ══════════════════════════════════════════════════════════════════

def _build_ppo_apply(ppo_policy):
    """PPOPolicy forward 를 pure function 으로 추출. branch 선택은 config 기준 1회.

    Returns: apply(params, obs, hstate, done, key) → (action, next_hstate)
    """
    network = ppo_policy.network
    config = ppo_policy.config
    stochastic = ppo_policy.stochastic

    alg_name = config.get("ALG_NAME", "")
    if "alg" in config and isinstance(config["alg"], dict):
        alg_name = config["alg"].get("ALG_NAME", alg_name)
    use_prediction = bool(config.get("USE_PREDICTION", True))
    model_type = config.get("model", {}).get("TYPE", "")
    use_pred_flag = (alg_name == "E3T") and use_prediction and (model_type == "RNN")
    actor_only_flag = alg_name in ("MEP_S1", "MEP_S2", "GAMMA_S1", "GAMMA_S2", "HSP_S1",
                                   "HSP_S2", "MEP", "GAMMA", "HSP")

    # MAPPO: critic params 가 centralized 라 일반 critic 호출 시 missing → actor_only
    try:
        params_root = ppo_policy.params["params"] if "params" in ppo_policy.params else ppo_policy.params
        if any(str(k).startswith("critic_global") for k in params_root.keys()):
            actor_only_flag = True
    except Exception:
        pass

    def apply(params, obs, hstate, done, key):
        done_arr = jnp.array(done)
        # ac_in dims: (T=1, B=1, ...)
        obs_dim = obs[jnp.newaxis, jnp.newaxis, ...]
        done_dim = done_arr[jnp.newaxis, jnp.newaxis, ...]
        ac_in = (obs_dim, done_dim)

        if use_pred_flag:
            outputs = network.apply(params, hstate, ac_in, use_prediction=True)
        elif actor_only_flag:
            outputs = network.apply(params, hstate, ac_in, actor_only=True)
        else:
            outputs = network.apply(params, hstate, ac_in)

        if len(outputs) == 4:
            next_hstate, pi, value, pred_logits = outputs
        else:
            next_hstate, pi, value = outputs

        if stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)
        action = action[0, 0]
        return action, next_hstate

    return apply


def _ppo_init_hstate(ppo_policy):
    return ppo_policy.init_hstate(batch_size=1)


# ══════════════════════════════════════════════════════════════════
#   Params stacking (vmap 대비)
# ══════════════════════════════════════════════════════════════════

def _stack_params(params_list):
    """pytree list → leading-dim stacked pytree."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params_list)


# ══════════════════════════════════════════════════════════════════
#   BC × RL fast rollout
# ══════════════════════════════════════════════════════════════════

def build_bc_rl_rollout(env, bc_policy, rl_policy, bc_slot: int, max_steps: int = 400):
    """(bc_params, rl_params, key) → total_reward 반환하는 JIT rollout 생성.

    bc_slot: 0 이면 BC 가 agent_0 / 1 이면 agent_1.
    """
    bc_apply = _build_bc_apply(bc_policy)
    rl_apply = _build_ppo_apply(rl_policy)

    bc_init_h = _bc_init_hstate(bc_policy)
    rl_init_h = _ppo_init_hstate(rl_policy)

    bc_agent = f"agent_{bc_slot}"
    rl_agent = f"agent_{1 - bc_slot}"

    def _rollout(bc_params, rl_params, key):
        key, k_reset = jax.random.split(key)
        obs, state = env.reset(k_reset)

        def _step(carry, k):
            obs, state, bc_h, rl_h, total_reward, episode_done = carry
            k_bc, k_rl, k_env = jax.random.split(k, 3)

            bc_action, new_bc_h = bc_apply(bc_params, obs[bc_agent], bc_h,
                                           jnp.bool_(False), k_bc)
            rl_action, new_rl_h = rl_apply(rl_params, obs[rl_agent], rl_h,
                                           jnp.bool_(False), k_rl)

            actions = {bc_agent: bc_action, rl_agent: rl_action}
            next_obs, next_state, reward, next_done, info = env.step(k_env, state, actions)

            sparse = (info["sparse_reward"]["agent_0"] if isinstance(info, dict) and "sparse_reward" in info
                      else reward["agent_0"])
            alive = 1.0 - episode_done.astype(jnp.float32)
            new_total = total_reward + sparse * alive
            new_episode_done = episode_done | next_done["__all__"]

            new_carry = (next_obs, next_state, new_bc_h, new_rl_h, new_total, new_episode_done)
            return new_carry, None

        keys = jax.random.split(key, max_steps)
        carry = (obs, state, bc_init_h, rl_init_h,
                 jnp.float32(0.0), jnp.bool_(False))
        carry, _ = jax.lax.scan(_step, carry, keys)
        return carry[4]  # total_reward

    return jax.jit(_rollout)


# ══════════════════════════════════════════════════════════════════
#   CEC × BC fast rollout (OV2 engine + JIT)
# ══════════════════════════════════════════════════════════════════

def _build_cec_apply(cec_policy, bc_slot: int, layout: str):
    """CEC forward (OV2 obs → CEC action) 를 pure function 으로 추출.

    bc_slot 으로 CEC 슬롯 (1 - bc_slot) 을 고정 — 기존 `CECPolicy` 의 runtime 좌표
    매칭은 OV2 grid 크기가 9×9 가 아닌 layout 에서 실패해서 0 reward 원인이 됨.
    여기서는 BC_POS_TO_V1_SLOT 에 따라 CEC slot 을 정적으로 결정.

    Returns: apply(params, obs, hstate, done, key) → (action, next_hstate)
    """
    # cec_integration path insert
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from cec_integration.obs_adapter_v2 import ov2_obs_to_cec  # noqa
    from cec_integration.webapp_v1_engine_helpers import (  # noqa
        ACTION_REMAP_OV2_TO_V1, BC_POS_TO_V1_SLOT,
    )

    network = cec_policy._runtime.network
    beta = cec_policy._runtime.beta
    argmax = cec_policy._runtime.argmax

    # layout 별 CEC slot 고정 (BC_POS_TO_V1_SLOT: BC pos → V1 slot; CEC 는 반대)
    slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
    bc_slot_idx = slot_map[bc_slot]
    cec_slot_idx = 1 - bc_slot_idx  # 0 또는 1 (Python int)

    # V1 action remap table: CEC output (V1 action semantic) → OV2 engine action
    # OV2 engine 은 OV2 action 을 받으므로 V1 → OV2 변환 필요 (같은 테이블이라 involution)
    remap_table = jnp.asarray(ACTION_REMAP_OV2_TO_V1, dtype=jnp.int32)

    import distrax

    def apply(params, obs, hstate, done, key):
        # hstate: CEC LSTM carry (num_agents=2)
        cec_obs = ov2_obs_to_cec(obs, layout, 0, 400)  # step 0 — CEC 는 graph_net time feature 쓰지만 여기선 dummy
        dummy = jnp.zeros_like(cec_obs)
        # CEC 는 [slot_0_obs, slot_1_obs] 배치를 기대
        if cec_slot_idx == 0:
            obs_arr = jnp.stack([cec_obs, dummy])
        else:
            obs_arr = jnp.stack([dummy, cec_obs])

        done_arr = jnp.array([done, done])
        agent_positions = jnp.zeros((2, 2, 2), dtype=jnp.int32)

        obs_batch = obs_arr.reshape(2, -1)
        ac_in = (obs_batch[jnp.newaxis, :], done_arr[jnp.newaxis, :],
                 agent_positions[jnp.newaxis, :])
        new_hidden, pi, _value = network.apply(params, hstate, ac_in)
        logits = pi.logits * beta
        scaled = distrax.Categorical(logits=logits)
        if argmax:
            actions = jnp.argmax(scaled.probs, axis=-1)[0]
        else:
            actions = scaled.sample(seed=key)[0]
        cec_action_v1 = actions[cec_slot_idx].astype(jnp.int32)
        # V1 → OV2 action remap (engine 이 OV2 이므로)
        cec_action_ov2 = remap_table[cec_action_v1]
        return cec_action_ov2, new_hidden

    return apply


def _cec_init_hstate(cec_policy):
    return cec_policy._runtime.init_hidden(2)


def build_cec_bc_rollout(env, bc_policy, cec_policy, layout: str, bc_slot: int,
                         max_steps: int = 400):
    """CEC × BC cross-play — OV2 engine + JIT scan."""
    bc_apply = _build_bc_apply(bc_policy)
    cec_apply = _build_cec_apply(cec_policy, bc_slot=bc_slot, layout=layout)

    bc_init_h = _bc_init_hstate(bc_policy)
    cec_init_h = _cec_init_hstate(cec_policy)

    bc_agent = f"agent_{bc_slot}"
    cec_agent = f"agent_{1 - bc_slot}"

    def _rollout(bc_params, cec_params, key):
        key, k_reset = jax.random.split(key)
        obs, state = env.reset(k_reset)

        def _step(carry, k):
            obs, state, bc_h, cec_h, total_reward, episode_done = carry
            k_bc, k_cec, k_env = jax.random.split(k, 3)
            bc_action, new_bc_h = bc_apply(bc_params, obs[bc_agent], bc_h,
                                           jnp.bool_(False), k_bc)
            cec_action, new_cec_h = cec_apply(cec_params, obs[cec_agent], cec_h,
                                              jnp.bool_(False), k_cec)
            actions = {bc_agent: bc_action, cec_agent: cec_action}
            next_obs, next_state, reward, next_done, info = env.step(k_env, state, actions)
            sparse = (info["sparse_reward"]["agent_0"] if isinstance(info, dict)
                      and "sparse_reward" in info else reward["agent_0"])
            alive = 1.0 - episode_done.astype(jnp.float32)
            new_total = total_reward + sparse * alive
            new_episode_done = episode_done | next_done["__all__"]
            return (next_obs, next_state, new_bc_h, new_cec_h, new_total, new_episode_done), None

        keys = jax.random.split(key, max_steps)
        carry = (obs, state, bc_init_h, cec_init_h,
                 jnp.float32(0.0), jnp.bool_(False))
        carry, _ = jax.lax.scan(_step, carry, keys)
        return carry[4]

    return jax.jit(_rollout)


# ══════════════════════════════════════════════════════════════════
#   CEC × BC fast rollout — V1 engine 경로 (JIT + vmap)
#
#   CEC 는 V1 engine 에서 학습됐으므로 V1 dynamics 에 붙여야 의미 있는 reward.
#   obs_adapter_v1_to_ov2.get_ov2_obs_jit (JAX-native) + V1 env.step (jittable) +
#   ov2_action_remap 를 lax.scan 으로 감싸고 (bc, cec, eval_seed) 3차원 vmap.
# ══════════════════════════════════════════════════════════════════

def _build_cec_runtime_apply(cec_policy):
    """CEC network forward. V1 engine 경로 용. (actions, new_hidden) 반환."""
    network = cec_policy._runtime.network
    beta = cec_policy._runtime.beta
    argmax = cec_policy._runtime.argmax
    import distrax

    def apply(params, obs_batch, hidden, done_arr, key):
        """obs_batch: (num_agents, flat_dim). hidden: scanned RNN carry. done_arr: (num_agents,)."""
        num_agents = obs_batch.shape[0]
        agent_positions = jnp.zeros((num_agents, 2, 2), dtype=jnp.int32)
        ac_in = (obs_batch[jnp.newaxis, :], done_arr[jnp.newaxis, :],
                 agent_positions[jnp.newaxis, :])
        new_hidden, pi, _v = network.apply(params, hidden, ac_in)
        logits = pi.logits * beta
        scaled = distrax.Categorical(logits=logits)
        if argmax:
            actions = jnp.argmax(scaled.probs, axis=-1)[0]
        else:
            actions = scaled.sample(seed=key)[0]
        return actions, new_hidden

    return apply


def build_cec_bc_rollout_v1(v1_env, adapter, bc_policy, cec_policy, layout: str,
                            bc_pos: int, max_steps: int = 400):
    """V1 engine 기반 CEC × BC rollout (JIT).

    CEC: V1 native obs (flat) 직접 받음 (ckpt 학습 분포 그대로).
    BC : adapter.get_ov2_obs_jit → OV2 (H,W,30) → BC forward → OV2 action →
         V1 action 으로 remap (engine 은 V1).
    """
    # path import
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from cec_integration.webapp_v1_engine_helpers import (  # noqa
        ACTION_REMAP_OV2_TO_V1, BC_POS_TO_V1_SLOT,
    )

    bc_apply = _build_bc_apply(bc_policy)
    cec_apply = _build_cec_runtime_apply(cec_policy)
    bc_init_h = _bc_init_hstate(bc_policy)
    cec_init_h = _cec_init_hstate(cec_policy)

    slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
    bc_slot_idx = slot_map[bc_pos]            # 0 또는 1 (Python int, static)
    cec_slot_idx = 1 - bc_slot_idx
    remap_table = jnp.asarray(ACTION_REMAP_OV2_TO_V1, dtype=jnp.int32)
    num_agents = v1_env.num_agents

    agents = [f"agent_{i}" for i in range(num_agents)]
    bc_agent_name = f"agent_{bc_slot_idx}"
    cec_agent_name = f"agent_{cec_slot_idx}"

    def _rollout(bc_params, cec_params, key):
        key, k_reset = jax.random.split(key)
        obs_dict, state = v1_env.reset(k_reset)

        def _step(carry, packed):
            state, bc_h, cec_h, total_reward, episode_done, step_t = carry
            k_bc, k_cec, k_env = jax.random.split(packed, 3)

            # ── CEC: V1 native obs (9,9,26) 직접 (flatten) ──
            v1_obs_dict = v1_env.get_obs(state)
            v1_obs_flat = jnp.stack([v1_obs_dict[a].flatten() for a in agents])
            done_arr = jnp.zeros((num_agents,), dtype=bool)  # 단일 episode — done 은 latch 로 처리
            cec_actions, next_cec_h = cec_apply(cec_params, v1_obs_flat, cec_h,
                                                done_arr, k_cec)
            cec_action_v1 = cec_actions[cec_slot_idx].astype(jnp.int32)

            # ── BC: V1 state → OV2 obs (JIT adapter) ──
            ov2_obs_dict = adapter.get_ov2_obs_jit(state, step_t)
            bc_obs = ov2_obs_dict[bc_agent_name].astype(jnp.uint8)
            bc_action_ov2, new_bc_h = bc_apply(bc_params, bc_obs, bc_h,
                                               jnp.bool_(False), k_bc)
            bc_action_v1 = remap_table[bc_action_ov2].astype(jnp.int32)

            # ── V1 env step ──
            actions = {bc_agent_name: bc_action_v1, cec_agent_name: cec_action_v1}
            next_obs_dict, next_state, reward, next_done, info = v1_env.step(
                k_env, state, actions,
            )
            # reward: V1 env 은 info 없이 dict("agent_0", "agent_1") 반환 가정
            r_agent0 = reward["agent_0"]
            alive = 1.0 - episode_done.astype(jnp.float32)
            new_total = total_reward + r_agent0 * alive
            new_episode_done = episode_done | next_done["__all__"]
            return (next_state, new_bc_h, next_cec_h, new_total, new_episode_done,
                    step_t + jnp.int32(1)), None

        keys = jax.random.split(key, max_steps)
        carry = (state, bc_init_h, cec_init_h,
                 jnp.float32(0.0), jnp.bool_(False), jnp.int32(0))
        carry, _ = jax.lax.scan(_step, carry, keys)
        return carry[3]  # total_reward

    return jax.jit(_rollout)


def run_cec_bc_crossplay_v1_fast(bc_policies_dict, cec_policies, v1_env, adapter,
                                 layout: str, num_eval_seeds=5, max_steps=400):
    """V1 engine 기반 CEC × BC cross-play (JIT + vmap)."""
    results = []
    for bc_pos, bc_list in bc_policies_dict.items():
        if not bc_list or not cec_policies:
            continue
        bc0 = bc_list[0]
        cec0 = cec_policies[0]
        rollout_fn = build_cec_bc_rollout_v1(v1_env, adapter, bc0, cec0,
                                             layout=layout, bc_pos=bc_pos,
                                             max_steps=max_steps)

        bc_params_stacked = _stack_params([p.params for p in bc_list])
        cec_params_stacked = _stack_params([p._runtime.params for p in cec_policies])

        eval_keys = jax.random.split(jax.random.PRNGKey(0), num_eval_seeds)
        vm_eval = jax.vmap(rollout_fn, in_axes=(None, None, 0))
        vm_cec = jax.vmap(vm_eval, in_axes=(None, 0, None))
        vm_all = jax.vmap(vm_cec, in_axes=(0, None, None))
        rewards = vm_all(bc_params_stacked, cec_params_stacked, eval_keys)
        rewards_np = np.asarray(rewards)

        for bi in range(len(bc_list)):
            for ci in range(len(cec_policies)):
                r = rewards_np[bi, ci]
                results.append({
                    "bc_pos": bc_pos, "bc_seed": bi, "cec_seed": ci,
                    "rewards": r.tolist(),
                    "mean_reward": float(np.mean(r)),
                    "std_reward": float(np.std(r)),
                })
                print(f"    BC(pos{bc_pos},s{bi}) × CEC(s{ci}): "
                      f"{float(np.mean(r)):6.1f} ± {float(np.std(r)):5.1f}  [V1+jit]",
                      flush=True)
    return results


def run_cec_bc_crossplay_fast(bc_policies_dict, cec_policies, env, layout: str,
                              num_eval_seeds=5, max_steps=400):
    """CEC × BC cross-play — vmap 기반 (OV2 engine)."""
    results = []
    for bc_pos, bc_list in bc_policies_dict.items():
        if not bc_list or not cec_policies:
            continue
        bc0 = bc_list[0]
        cec0 = cec_policies[0]
        rollout_fn = build_cec_bc_rollout(env, bc0, cec0, layout=layout,
                                          bc_slot=bc_pos, max_steps=max_steps)

        bc_params_stacked = _stack_params([p.params for p in bc_list])
        # CECPolicy 의 params 는 _runtime.params 에 있음
        cec_params_stacked = _stack_params([p._runtime.params for p in cec_policies])

        eval_keys = jax.random.split(jax.random.PRNGKey(0), num_eval_seeds)
        vm_eval = jax.vmap(rollout_fn, in_axes=(None, None, 0))
        vm_cec = jax.vmap(vm_eval, in_axes=(None, 0, None))
        vm_all = jax.vmap(vm_cec, in_axes=(0, None, None))
        rewards = vm_all(bc_params_stacked, cec_params_stacked, eval_keys)
        rewards_np = np.asarray(rewards)

        for bi in range(len(bc_list)):
            for ci in range(len(cec_policies)):
                r = rewards_np[bi, ci]
                results.append({
                    "bc_pos": bc_pos, "bc_seed": bi, "cec_seed": ci,
                    "rewards": r.tolist(),
                    "mean_reward": float(np.mean(r)),
                    "std_reward": float(np.std(r)),
                })
                print(f"    BC(pos{bc_pos},s{bi}) × CEC(s{ci}): "
                      f"{float(np.mean(r)):6.1f} ± {float(np.std(r)):5.1f}  [fast]",
                      flush=True)
    return results


def run_bc_rl_crossplay_fast(bc_policies_dict, rl_policies, env, num_eval_seeds=5,
                             max_steps=400):
    """BC × RL cross-play — vmap 기반.

    Args:
        bc_policies_dict: {pos: [BCPolicy, ...]} pos 는 0/1.
        rl_policies: [PPOPolicy, ...] — 각 seed.
        env: jaxmarl env (make_eval_env 결과).
        num_eval_seeds: eval seed 수.
    Returns: list[dict] (bc_pos, bc_seed, rl_seed, mean_reward, std_reward, rewards)
    """
    results = []
    # 대표 policy 로 rollout fn 만들기 (모든 seeds 동일 network/config 전제)
    for bc_pos, bc_list in bc_policies_dict.items():
        if not bc_list or not rl_policies:
            continue
        bc0 = bc_list[0]
        rl0 = rl_policies[0]

        # pos 별로 rollout fn 이 다름 (bc_slot 다름) → 2회 컴파일
        rollout_fn = build_bc_rl_rollout(env, bc0, rl0, bc_slot=bc_pos,
                                         max_steps=max_steps)

        # 모든 seed 의 params 를 vmap 용으로 stack
        bc_params_stacked = _stack_params([p.params for p in bc_list])
        rl_params_stacked = _stack_params([p.params for p in rl_policies])

        # 3차원 vmap: (bc_seed, rl_seed, eval_seed)
        eval_keys = jax.random.split(jax.random.PRNGKey(0), num_eval_seeds)
        # in_axes: bc_params (0, None, None), rl_params (None, 0, None), key (None, None, 0)
        vm_eval = jax.vmap(rollout_fn, in_axes=(None, None, 0))
        vm_rl = jax.vmap(vm_eval, in_axes=(None, 0, None))
        vm_all = jax.vmap(vm_rl, in_axes=(0, None, None))
        rewards = vm_all(bc_params_stacked, rl_params_stacked, eval_keys)
        rewards_np = np.asarray(rewards)  # (num_bc, num_rl, num_eval)

        num_bc = len(bc_list)
        num_rl = len(rl_policies)
        for bi in range(num_bc):
            for ri in range(num_rl):
                r = rewards_np[bi, ri]
                results.append({
                    "bc_pos": bc_pos, "bc_seed": bi, "rl_seed": ri,
                    "rewards": r.tolist(),
                    "mean_reward": float(np.mean(r)),
                    "std_reward": float(np.std(r)),
                })
                print(f"    BC(pos{bc_pos},s{bi}) × RL(s{ri}): "
                      f"{float(np.mean(r)):6.1f} ± {float(np.std(r)):5.1f}  [fast]",
                      flush=True)
    return results
