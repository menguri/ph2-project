"""V1→OV2 obs adapter round-trip 검증 (5 layout).

파이프라인:
  OV2 env self-play → ov2_state
  (A) ov2_env.get_obs(ov2_state)                                    → orig_obs
  (B) state_direct: ov2_state → v1_state
      v1_to_ov2: v1_state → synth_ov2_state
      ov2_env.get_obs(synth_ov2_state)                               → roundtrip_obs

orig_obs 와 roundtrip_obs 를 byte-exact 비교.
static 은 OV2 그대로라 보존되고, dyn/extra/agent 정보만 state_direct→v1_to_ov2 을 거친다.
따라서 이 round-trip 이 통과하면 두 adapter 가 정보를 손실없이 전달한다는 증명.

synthetic 케이스도 포함: pot filling / cooking / ready, inventory, loose items.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. JAX_PLATFORMS=cpu ./overcooked_v2/bin/python \
        cec_integration/scripts/verify_v1_to_ov2_roundtrip.py
"""
import os
import sys

sys.path.insert(0, "/home/mlic/mingukang/ph2-project")

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from jaxmarl.environments.overcooked_v2.common import StaticObject

from cec_integration.obs_adapter_v2_state_direct import OV2StateToCECDirectAdapter
from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter


# asymm_advantages 는 CEC `_9x9` 템플릿과 OV2 native layout 이 y-shift 뿐 아니라
# agent_1 의 x 좌표도 달라서 (CEC x=6 vs OV2 x=5) 동일한 physical layout 이 아님.
# state_direct / v1_to_ov2 만으로 byte-exact round-trip 이 불가 → 이 경로에서는 제외.
LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "counter_circuit"]
NUM_ROLLOUT_STEPS = 40


def _compare(orig_obs, rt_obs, ctx):
    matched = True
    details = []
    for key in ["agent_0", "agent_1"]:
        a = np.asarray(orig_obs[key])
        b = np.asarray(rt_obs[key])
        if not np.array_equal(a, b):
            matched = False
            d = np.abs(a.astype(np.int32) - b.astype(np.int32))
            for ch in range(a.shape[-1]):
                if d[:, :, ch].max() > 0:
                    coords = np.argwhere(d[:, :, ch] > 0)
                    y, x = coords[0]
                    details.append(f"{ctx} {key} ch{ch} n={len(coords)} "
                                   f"({y},{x})A={a[y,x,ch]}/B={b[y,x,ch]}")
    return matched, details[:6]


def modify_pot(state, pot_xy, dyn, extra):
    x, y = pot_xy
    grid = state.grid.at[y, x, 1].set(dyn).at[y, x, 2].set(extra)
    return state.replace(grid=grid)


def modify_loose(state, xy, dyn):
    x, y = xy
    grid = state.grid.at[y, x, 1].set(dyn)
    return state.replace(grid=grid)


def modify_inv(state, agent_idx, inv):
    agents = state.agents
    inventory = agents.inventory.at[agent_idx].set(inv)
    return state.replace(agents=agents.replace(inventory=inventory))


def first_pot_xy(state):
    grid = np.asarray(state.grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if int(grid[y, x, 0]) == StaticObject.POT:
                return (x, y)
    return None


def first_empty_xy(state):
    grid = np.asarray(state.grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if int(grid[y, x, 0]) == StaticObject.EMPTY:
                return (x, y)
    return None


def run_layout(layout):
    print(f"\n{'='*72}\nLayout: {layout}\n{'='*72}", flush=True)
    env = jaxmarl.make("overcooked_v2", layout=layout, max_steps=NUM_ROLLOUT_STEPS,
                       random_reset=False, random_agent_positions=False)
    to_v1 = OV2StateToCECDirectAdapter(target_layout=layout, max_steps=NUM_ROLLOUT_STEPS)
    to_ov2 = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=NUM_ROLLOUT_STEPS)

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    all_pass = True

    # --- (1) random rollout round-trip ---
    rng = np.random.RandomState(42)
    for t in range(NUM_ROLLOUT_STEPS):
        a0 = int(rng.randint(0, 6))
        a1 = int(rng.randint(0, 6))
        key, sub = jax.random.split(key)
        _, state, _, done, _ = env.step(sub, state, {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)})

        orig = env.get_obs(state)
        v1_state = to_v1.build_v1_state(state, current_step=t + 1)
        synth_ov2 = to_ov2.build_ov2_state(v1_state, current_step=t + 1)
        rt = env.get_obs(synth_ov2)

        matched, details = _compare(orig, rt, f"t={t+1}")
        if not matched:
            all_pass = False
            print(f"  t={t+1} FAIL")
            for d in details:
                print(f"    {d}")
        if bool(done["__all__"]):
            break

    if all_pass:
        print(f"  rollout {NUM_ROLLOUT_STEPS}step: PASS", flush=True)

    # --- (2) synthetic states ---
    _, s0 = env.reset(jax.random.PRNGKey(0))
    pot_xy = first_pot_xy(s0)
    loose_xy = first_empty_xy(s0)
    if pot_xy is None:
        print(f"  no pot in OV2 layout — skipping synthetic")
        return all_pass

    ONION_1 = 1 << 2
    ONION_2 = 2 << 2
    ONION_3 = 3 << 2                     # dyn during cooking (no cooked bit yet)
    POT_READY_DYN = ONION_3 | 0x2        # dyn when cooking completed (cooked bit set)
    INV_ONION = ONION_1
    INV_PLATE = 0x1
    INV_DISH = 0x1 | 0x2 | (3 << 2)

    # OV2 실제 의미적 상태: cooking 중에는 dyn=12 (no cooked bit), extra>0.
    # Ready 시 dyn=14 (cooked bit), extra=0.
    # Ready 직전 (cooking 마지막 tick) dyn=12, extra=1.
    cases = [
        ("pot_1_onion",       lambda s: modify_pot(s, pot_xy, ONION_1, 0)),
        ("pot_2_onion",       lambda s: modify_pot(s, pot_xy, ONION_2, 0)),
        ("pot_cooking_19",    lambda s: modify_pot(s, pot_xy, ONION_3, 19)),  # 방금 auto_cook 1 tick 경과
        ("pot_cooking_10",    lambda s: modify_pot(s, pot_xy, ONION_3, 10)),
        ("pot_cooking_1",     lambda s: modify_pot(s, pot_xy, ONION_3, 1)),
        ("pot_ready",         lambda s: modify_pot(s, pot_xy, POT_READY_DYN, 0)),
        ("inv0_onion",        lambda s: modify_inv(s, 0, INV_ONION)),
        ("inv0_plate",        lambda s: modify_inv(s, 0, INV_PLATE)),
        ("inv0_dish",         lambda s: modify_inv(s, 0, INV_DISH)),
        ("inv1_onion",        lambda s: modify_inv(s, 1, INV_ONION)),
    ]
    if loose_xy is not None:
        cases.extend([
            ("loose_onion",   lambda s: modify_loose(s, loose_xy, INV_ONION)),
            ("loose_plate",   lambda s: modify_loose(s, loose_xy, INV_PLATE)),
            ("loose_dish",    lambda s: modify_loose(s, loose_xy, INV_DISH)),
        ])

    for name, fn in cases:
        s = fn(s0)
        orig = env.get_obs(s)
        v1_state = to_v1.build_v1_state(s, current_step=0)
        synth_ov2 = to_ov2.build_ov2_state(v1_state, current_step=0)
        rt = env.get_obs(synth_ov2)
        matched, details = _compare(orig, rt, name)
        flag = "PASS" if matched else "FAIL"
        print(f"  [{flag}] {name}")
        for d in details:
            print(f"      {d}")
        all_pass = all_pass and matched

    return all_pass


def main():
    print("V1 ↔ OV2 obs adapter round-trip (5 layouts)", flush=True)
    results = {}
    for layout in LAYOUTS:
        try:
            results[layout] = run_layout(layout)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[layout] = False

    print(f"\n{'='*72}\nSummary:\n{'='*72}")
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
