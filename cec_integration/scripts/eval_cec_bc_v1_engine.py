"""V1 Overcooked engine 에서 CEC × BC cross-play — 4 layout.

A1 구현: V1 engine + CEC_LAYOUTS 로 primary engine 이용.
  - CEC: V1 native obs (9×9×26) 직접 받음.
  - BC:  V1 state → OV2 obs (H×W×30) via V1StateToOV2ObsAdapter → BC 학습 분포와 일치.

slot 매핑:
  - V1 agent_0 = CEC (CEC 학습 time ego slot 가정)
  - V1 agent_1 = BC
  - BC 가 받는 OV2 obs 는 agent_1 의 시점 (self=V1 agent_1 위치)
  - BC pos_0 와 pos_1 seeds 모두 시도 (각 seed 가 어느 slot 에 trained 됐는지에 따라 결과 다를 수 있음)

Run (GPU 0):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:human-proxy/code PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/eval_cec_bc_v1_engine.py
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

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter
from cec_integration.webapp_v1_engine_helpers import ACTION_REMAP_OV2_TO_V1, BC_POS_TO_V1_SLOT

from policy import BCPolicy  # human-proxy/code/policy.py


LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "counter_circuit"]
NUM_STEPS = 400
NUM_EPISODES = 3
BC_MODELS_ROOT = os.path.join(PROJECT_ROOT, "human-proxy", "models")
CEC_MODELS_ROOT = os.path.join(PROJECT_ROOT, "webapp", "models")
MAX_BC_SEEDS = 3  # per pos
MAX_CEC_RUNS = 2


def load_bc_list(layout, pos):
    pos_dir = Path(BC_MODELS_ROOT) / layout / f"pos_{pos}"
    if not pos_dir.is_dir():
        return []
    pols = []
    for d in sorted(pos_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("seed_"):
            continue
        if "tmp" in d.name:
            continue
        try:
            pols.append((d.name, BCPolicy.from_pretrained(d)))
            if len(pols) >= MAX_BC_SEEDS:
                break
        except Exception as e:
            print(f"    [WARN] {d}: {e}", flush=True)
    return pols


def list_cec_runs(layout):
    root = Path(CEC_MODELS_ROOT) / layout / "cec"
    if not root.is_dir():
        return []
    runs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "ckpt_final").is_dir():
            runs.append((d.name, str(d / "ckpt_final")))
            if len(runs) >= MAX_CEC_RUNS:
                break
    return runs


def run_crossplay_episode(cec_rt, bc_policy, ov2_adapter, env, cec_slot, bc_slot, rng):
    """One episode. cec_slot/bc_slot ∈ {"agent_0", "agent_1"}, no overlap."""
    obs_dict, env_state = env.reset(rng)
    cec_h = cec_rt.init_hidden(env.num_agents)
    bc_h = bc_policy.init_hstate(batch_size=1) if hasattr(bc_policy, "init_hstate") else None
    done_arr = jnp.zeros(env.num_agents, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k_cec, k_bc, k_env = jax.random.split(rng, 4)

        # CEC: V1 obs 직접
        v1_obs_flat = jnp.stack([obs_dict[a].flatten() for a in env.agents])
        cec_actions, cec_h, _ = cec_rt.step(v1_obs_flat, cec_h, done_arr, k_cec)
        cec_action = int(cec_actions[0 if cec_slot == "agent_0" else 1])

        # BC: V1 state → OV2 obs (BC slot)
        ov2_obs_dict = ov2_adapter.get_ov2_obs(env_state, current_step=t)
        bc_obs = np.asarray(ov2_obs_dict[bc_slot]).astype(np.uint8)
        bc_done_scalar = jnp.array(bool(done_arr[0 if bc_slot == "agent_0" else 1]))
        bc_action, bc_h, _ = bc_policy.compute_action(
            jnp.array(bc_obs), bc_done_scalar, bc_h, k_bc,
        )
        # BC 가 OV2 semantic 출력 → V1 action 으로 remap
        bc_action_v1 = int(ACTION_REMAP_OV2_TO_V1[int(bc_action)])

        env_act = {cec_slot: jnp.int32(cec_action), bc_slot: jnp.int32(bc_action_v1)}
        obs_dict, env_state, reward, done_dict, _ = env.step(k_env, env_state, env_act)
        done_arr = jnp.array([done_dict[a] for a in env.agents])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total, t + 1


def evaluate_layout(layout):
    print(f"\n{'='*72}\nLayout: {layout}\n{'='*72}", flush=True)
    cec_runs = list_cec_runs(layout)
    if not cec_runs:
        print(f"  [SKIP] no CEC runs at {CEC_MODELS_ROOT}/{layout}/cec")
        return None
    print(f"  CEC runs: {[r[0] for r in cec_runs]}", flush=True)

    env = V1Overcooked(layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False,
                       max_steps=NUM_STEPS)
    ov2_adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=NUM_STEPS)

    layout_results = []
    # BC slot 배치: pos_0 → V1 slot 0, pos_1 → V1 slot 1
    for bc_pos in [0, 1]:
        bc_list = load_bc_list(layout, bc_pos)
        if not bc_list:
            print(f"  [SKIP] BC pos_{bc_pos} no seeds", flush=True)
            continue
        slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
        bc_slot_idx = slot_map[bc_pos]
        bc_slot = f"agent_{bc_slot_idx}"
        cec_slot = f"agent_{1 - bc_slot_idx}"
        print(f"  BC pos_{bc_pos} → V1 {bc_slot}, CEC → V1 {cec_slot} (seeds: {[n for n,_ in bc_list]})", flush=True)

        for cec_name, cec_ckpt in cec_runs:
            try:
                cec_rt = CECRuntime(cec_ckpt)
            except Exception as e:
                print(f"    CEC {cec_name} load fail: {e}", flush=True)
                continue
            for bc_seed_name, bc_policy in bc_list:
                rewards = []
                rng = jax.random.PRNGKey(42)
                for ep in range(NUM_EPISODES):
                    rng, sub = jax.random.split(rng)
                    try:
                        r, _ = run_crossplay_episode(
                            cec_rt, bc_policy, ov2_adapter, env,
                            cec_slot=cec_slot, bc_slot=bc_slot, rng=sub,
                        )
                    except Exception as e:
                        print(f"    {cec_name}×{bc_seed_name} ep{ep} err: {e}", flush=True)
                        r = 0.0
                    rewards.append(r)
                mean = float(np.mean(rewards))
                std = float(np.std(rewards))
                print(f"    {cec_name} × BC(pos_{bc_pos},{bc_seed_name}): mean={mean:6.1f} std={std:5.1f}  eps={rewards}",
                      flush=True)
                layout_results.append({
                    "bc_pos": bc_pos, "bc_seed": bc_seed_name,
                    "cec_run": cec_name, "mean": mean, "std": std,
                })
    return layout_results


def main():
    print("=" * 72, flush=True)
    print("CEC × BC cross-play on V1 Overcooked engine (A1 구현)", flush=True)
    print(f"  env = V1Overcooked + CEC_LAYOUTS[_9]", flush=True)
    print(f"  BC obs = V1 state → V1StateToOV2ObsAdapter.get_ov2_obs(slot)", flush=True)
    print(f"  {NUM_EPISODES} episodes × {NUM_STEPS} steps", flush=True)
    print(f"  per layout: up to {MAX_BC_SEEDS} BC seeds × {MAX_CEC_RUNS} CEC runs × 2 pos", flush=True)
    print("=" * 72, flush=True)
    all_results = {}
    for layout in LAYOUTS:
        all_results[layout] = evaluate_layout(layout)

    # Summary: best mean per layout per BC pos
    print("\n" + "=" * 72)
    print(f"{'layout':20s} {'pos':>4s} {'best BC seed':>14s} {'best CEC':>10s} {'mean':>8s} {'std':>6s}")
    print("=" * 72)
    for layout, res in all_results.items():
        if not res:
            print(f"{layout:20s} --")
            continue
        for pos in [0, 1]:
            pos_res = [r for r in res if r["bc_pos"] == pos]
            if not pos_res:
                continue
            best = max(pos_res, key=lambda r: r["mean"])
            print(f"{layout:20s} {pos:>4d} {best['bc_seed']:>14s} {best['cec_run']:>10s} "
                  f"{best['mean']:8.2f} {best['std']:6.2f}")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main() or 0)
