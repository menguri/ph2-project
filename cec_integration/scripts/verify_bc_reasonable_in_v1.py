"""V1 engine 에서 BC 가 의미있는 행동을 하는지 검증.

비교:
  - Random × Random (baseline)
  - BC(pos_0) × BC(pos_1)       — BC self-play
  - BC × CEC                     — 이전 측정값
  - CEC × CEC                    — 이전 측정값 (sanity)

BC×BC >> Random×Random 이면 BC 가 실제 delivery 로직을 학습했다는 증거.
BC×CEC 가 BC×BC 보다 높으면 CEC 가 BC 를 보조한다는 증거.

Run (GPU 0 권장 — 5 layout 여러 조합):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:human-proxy/code PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/verify_bc_reasonable_in_v1.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = "/home/mlic/mingukang/ph2-project"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "human-proxy", "code"))

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter
from cec_integration.webapp_v1_engine_helpers import BC_POS_TO_V1_SLOT, ACTION_REMAP_OV2_TO_V1

from policy import BCPolicy


LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "counter_circuit"]
NUM_STEPS = 200
NUM_EPISODES = 2
BC_MODELS_ROOT = os.path.join(PROJECT_ROOT, "human-proxy", "models")
MAX_BC_SEEDS = 2  # per pos

# 커맨드라인 --layout 로 단일 레이아웃만 빠르게 테스트 가능
if len(sys.argv) > 1 and sys.argv[1] == "--layout":
    LAYOUTS = [sys.argv[2]]


def load_bc(layout, pos, seed_idx=0):
    pos_dir = Path(BC_MODELS_ROOT) / layout / f"pos_{pos}"
    seeds = sorted([d for d in pos_dir.iterdir()
                    if d.is_dir() and d.name.startswith("seed_") and "tmp" not in d.name])
    if seed_idx >= len(seeds):
        return None
    return BCPolicy.from_pretrained(seeds[seed_idx])


def run_random_episode(env, rng):
    _, env_state = env.reset(rng)
    done_arr = jnp.zeros(env.num_agents, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k = jax.random.split(rng)
        actions = jax.random.randint(k, (env.num_agents,), 0, 6)
        env_act = {env.agents[i]: jnp.int32(int(actions[i])) for i in range(env.num_agents)}
        rng, ke = jax.random.split(rng)
        _, env_state, reward, done_dict, _ = env.step(ke, env_state, env_act)
        done_arr = jnp.array([done_dict[a] for a in env.agents])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total


def run_bc_bc_episode(env, adapter, bc_for_slot, rng):
    """bc_for_slot: {0: bc_policy, 1: bc_policy} — V1 slot 에 할당된 BC.

    BC 는 OV2 action semantic 출력 → V1 action 으로 remap 후 env.step.
    """
    _, env_state = env.reset(rng)
    h = {
        0: bc_for_slot[0].init_hstate(batch_size=1) if hasattr(bc_for_slot[0], "init_hstate") else None,
        1: bc_for_slot[1].init_hstate(batch_size=1) if hasattr(bc_for_slot[1], "init_hstate") else None,
    }
    done_arr = jnp.zeros(env.num_agents, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k0, k1, ke = jax.random.split(rng, 4)
        ov2_obs = adapter.get_ov2_obs(env_state, current_step=t)
        keys = [k0, k1]
        acts = {}
        for s in [0, 1]:
            obs = jnp.asarray(np.asarray(ov2_obs[f"agent_{s}"]).astype(np.uint8))
            a, h[s], _ = bc_for_slot[s].compute_action(obs, jnp.array(bool(done_arr[s])), h[s], keys[s])
            a_v1 = int(ACTION_REMAP_OV2_TO_V1[int(a)])
            acts[f"agent_{s}"] = jnp.int32(a_v1)
        _, env_state, reward, done_dict, _ = env.step(ke, env_state, acts)
        done_arr = jnp.array([done_dict[a] for a in env.agents])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total


def run_bc_bc_ov2_episode(ov2_env, bc0, bc1, rng):
    """OV2 engine 에서 BC×BC self-play (native OV2 obs 직접 사용)."""
    obs_dict, env_state = ov2_env.reset(rng)
    h0 = bc0.init_hstate(batch_size=1) if hasattr(bc0, "init_hstate") else None
    h1 = bc1.init_hstate(batch_size=1) if hasattr(bc1, "init_hstate") else None
    done_arr = jnp.zeros(2, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k0, k1, ke = jax.random.split(rng, 4)
        o0 = jnp.asarray(np.asarray(obs_dict["agent_0"]).astype(np.uint8))
        o1 = jnp.asarray(np.asarray(obs_dict["agent_1"]).astype(np.uint8))
        a0, h0, _ = bc0.compute_action(o0, jnp.array(bool(done_arr[0])), h0, k0)
        a1, h1, _ = bc1.compute_action(o1, jnp.array(bool(done_arr[1])), h1, k1)
        env_act = {"agent_0": jnp.int32(int(a0)), "agent_1": jnp.int32(int(a1))}
        obs_dict, env_state, reward, done_dict, _ = ov2_env.step(ke, env_state, env_act)
        done_arr = jnp.array([done_dict["agent_0"], done_dict["agent_1"]])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total


def run_random_ov2_episode(ov2_env, rng):
    obs_dict, env_state = ov2_env.reset(rng)
    done_arr = jnp.zeros(2, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k, ke = jax.random.split(rng, 3)
        actions = jax.random.randint(k, (2,), 0, 6)
        env_act = {"agent_0": jnp.int32(int(actions[0])), "agent_1": jnp.int32(int(actions[1]))}
        obs_dict, env_state, reward, done_dict, _ = ov2_env.step(ke, env_state, env_act)
        done_arr = jnp.array([done_dict["agent_0"], done_dict["agent_1"]])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total


def evaluate_layout(layout):
    print(f"\n=== {layout} ===", flush=True)
    v1_env = V1Overcooked(layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False, max_steps=NUM_STEPS)
    ov2_env = jaxmarl.make("overcooked_v2", layout=layout, max_steps=NUM_STEPS,
                            random_reset=False, random_agent_positions=False)
    adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=NUM_STEPS)

    # Random × Random — V1 and OV2
    rng = jax.random.PRNGKey(42)
    r_v1 = []
    r_ov2 = []
    for ep in range(NUM_EPISODES):
        rng, s1, s2 = jax.random.split(rng, 3)
        r_v1.append(run_random_episode(v1_env, s1))
        r_ov2.append(run_random_ov2_episode(ov2_env, s2))
    rv1_mean = float(np.mean(r_v1))
    rov2_mean = float(np.mean(r_ov2))
    print(f"  Random × Random V1 : mean={rv1_mean:.1f}  eps={r_v1}", flush=True)
    print(f"  Random × Random OV2: mean={rov2_mean:.1f}  eps={r_ov2}", flush=True)

    # BC 를 V1 slot 에 매핑 (pos→slot)
    slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
    print(f"  slot map: pos_0 → V1 slot {slot_map[0]}, pos_1 → V1 slot {slot_map[1]}", flush=True)

    # BC(pos_0) × BC(pos_1) — V1 and OV2
    bc_v1_results = []
    bc_ov2_results = []
    for s0 in range(MAX_BC_SEEDS):
        bc0 = load_bc(layout, pos=0, seed_idx=s0)  # BC trained as OV2 agent_0
        if bc0 is None:
            continue
        for s1 in range(MAX_BC_SEEDS):
            bc1 = load_bc(layout, pos=1, seed_idx=s1)  # BC trained as OV2 agent_1
            if bc1 is None:
                continue
            # V1 engine: pos-based slot 매핑 적용
            bc_for_slot = {slot_map[0]: bc0, slot_map[1]: bc1}
            rng = jax.random.PRNGKey(42)
            rw_v1 = []
            for ep in range(NUM_EPISODES):
                rng, sub = jax.random.split(rng)
                rw_v1.append(run_bc_bc_episode(v1_env, adapter, bc_for_slot, sub))
            mv1 = float(np.mean(rw_v1))
            bc_v1_results.append(mv1)
            # OV2 engine (native: pos == slot, BC 학습 그대로)
            rng = jax.random.PRNGKey(42)
            rw_ov2 = []
            for ep in range(NUM_EPISODES):
                rng, sub = jax.random.split(rng)
                rw_ov2.append(run_bc_bc_ov2_episode(ov2_env, bc0, bc1, sub))
            mov2 = float(np.mean(rw_ov2))
            bc_ov2_results.append(mov2)
            print(f"  BC(pos0,s{s0}) × BC(pos1,s{s1}): V1={mv1:6.1f} eps={rw_v1}  |  OV2={mov2:6.1f} eps={rw_ov2}",
                  flush=True)

    bc_v1_avg = float(np.mean(bc_v1_results)) if bc_v1_results else 0.0
    bc_ov2_avg = float(np.mean(bc_ov2_results)) if bc_ov2_results else 0.0
    print(f"  BC×BC avg over combos: V1={bc_v1_avg:.1f}  OV2={bc_ov2_avg:.1f}", flush=True)
    return {"random_v1": rv1_mean, "random_ov2": rov2_mean,
            "bc_v1": bc_v1_avg, "bc_ov2": bc_ov2_avg}


def main():
    print("=" * 72, flush=True)
    print("BC 행동 검증 — Random baseline vs BC×BC vs CEC (이전 결과 참조)", flush=True)
    print(f"  {NUM_EPISODES} episodes × {NUM_STEPS} steps, V1 Overcooked engine", flush=True)
    print("=" * 72, flush=True)
    results = {}
    for layout in LAYOUTS:
        results[layout] = evaluate_layout(layout)

    print("\n" + "=" * 80)
    print(f"{'layout':20s} {'Rand V1':>8s} {'Rand OV2':>9s} {'BC×BC V1':>9s} {'BC×BC OV2':>10s} {'CEC×CEC V1*':>12s}")
    print("=" * 80)
    prior_cec_cec = {"cramped_room": 220.0, "coord_ring": 220.0,
                      "forced_coord": 6.67, "counter_circuit": 100.0}
    for layout, r in results.items():
        cc = prior_cec_cec.get(layout, 0)
        print(f"{layout:20s} {r['random_v1']:8.1f} {r['random_ov2']:9.1f} "
              f"{r['bc_v1']:9.1f} {r['bc_ov2']:10.1f} {cc:12.1f}")
    print("=" * 80)
    print("* CEC×CEC V1 는 별도 측정 (eval_cec_v1_engine_selfplay.py 결과)")


if __name__ == "__main__":
    sys.exit(main() or 0)
