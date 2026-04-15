"""레이아웃별로 CEC native eval 의 obs 와 webapp/v2 adapter 의 obs 를 비교.

CEC native eval 환경 (ph2 의 V1Overcooked + CEC_LAYOUTS) 을 reset 해서 매 step 의
(9,9,26) obs 를 기록하고, 동일한 동작을 OV2 env + obs_adapter_v2 경로에서 돌렸을 때
나오는 (9,9,26) obs 와 채널별로 비교한다.

두 환경은 reset 시 agent 위치가 다를 수 있으므로:
  1. 정적 채널 (pot/plate/goal/onion_pile, wall) → 정확히 일치해야 함
  2. 동적 채널 (agent pos, dir, inventory, pot state, urgency) → agent 가 동일 좌표·동일 inventory
     상태일 때만 일치 검증 (agent 위치 자체는 reset 마다 다를 수 있음)

레포트 형식: 채널별 max_diff, 다른 셀 수, 가장 다른 셀 좌표.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. ./overcooked_v2/bin/python \
        cec_integration/scripts/compare_obs_native_vs_v2.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec, LAYOUT_PADDING, LAYOUT_OV2_CROP


# CEC 채널 라벨 (26ch)
CH_NAMES = [
    "self_pos", "other_pos",
    "self_dir_E", "self_dir_S", "self_dir_W", "self_dir_N",
    "other_dir_E", "other_dir_S", "other_dir_W", "other_dir_N",
    "pot", "wall", "onion_pile", "tomato_pile", "plate_pile", "goal",
    "onions_in_pot", "tomato_in_pot", "onions_in_soup", "tomato_in_soup",
    "cook_time", "soup_ready", "plate_on_grid", "onion_on_grid",
    "tomato_on_grid", "urgency",
]

# CEC native eval 시 사용하는 v1 env config (test_general.py:67-72 와 동일 의도)
NUM_STEPS = 400


def _native_obs_at_reset(cec_layout_name):
    """ph2 의 v1 Overcooked env 를 CEC_LAYOUTS 로 build, reset → CEC 26ch obs.

    이게 CEC 가 eval 시 실제로 보던 obs 의 ground truth.
    """
    from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
    layout = CEC_LAYOUTS[cec_layout_name]
    env = V1Overcooked(layout=layout, random_reset=False, max_steps=NUM_STEPS)
    obs_dict, env_state = env.reset(jax.random.PRNGKey(0))
    return np.asarray(obs_dict["agent_0"]), np.asarray(obs_dict["agent_1"]), env_state


def _v2_obs_at_reset(ov2_layout_name):
    """webapp 경로 — OV2 env reset → ov2_obs_to_cec → (9,9,26)."""
    env = jaxmarl.make("overcooked_v2", layout=ov2_layout_name, max_steps=NUM_STEPS,
                       random_reset=False, random_agent_positions=False)
    obs_dict, env_state = env.reset(jax.random.PRNGKey(0))
    a0_cec = np.asarray(ov2_obs_to_cec(jnp.array(obs_dict["agent_0"], dtype=jnp.float32),
                                        ov2_layout_name, 0, NUM_STEPS))
    a1_cec = np.asarray(ov2_obs_to_cec(jnp.array(obs_dict["agent_1"], dtype=jnp.float32),
                                        ov2_layout_name, 0, NUM_STEPS))
    return a0_cec, a1_cec, env_state


def _diff_summary(native_obs, v2_obs):
    """채널별 diff 요약 (max_diff, n_diff_cells, top-3 diff coords)."""
    diffs = []
    for ch in range(26):
        d = np.abs(native_obs[:, :, ch] - v2_obs[:, :, ch])
        max_d = float(d.max())
        n_cells = int((d > 0.01).sum())
        if n_cells > 0:
            top = np.argwhere(d > 0.01)[:3]
            top_str = " ".join(f"({y},{x})n={native_obs[y,x,ch]:.1f}/v2={v2_obs[y,x,ch]:.1f}"
                               for y, x in top)
            diffs.append((ch, CH_NAMES[ch], max_d, n_cells, top_str))
    return diffs


def compare_one(ov2_name, cec_name):
    print(f"\n=== [{ov2_name}] vs CEC native [{cec_name}] ===")
    nat_a0, nat_a1, nat_state = _native_obs_at_reset(cec_name)
    v2_a0, v2_a1, v2_state = _v2_obs_at_reset(ov2_name)

    # 두 env 의 agent 위치 추출 (CEC obs ch0/ch1)
    n_self = np.argwhere(nat_a0[:, :, 0] > 0.5)
    n_other = np.argwhere(nat_a0[:, :, 1] > 0.5)
    v_self = np.argwhere(v2_a0[:, :, 0] > 0.5)
    v_other = np.argwhere(v2_a0[:, :, 1] > 0.5)
    print(f"  native  agent_0 self={tuple(n_self[0])} other={tuple(n_other[0])}")
    print(f"  v2-adp  agent_0 self={tuple(v_self[0])} other={tuple(v_other[0])}")

    # agent_0 obs 비교
    diffs0 = _diff_summary(nat_a0, v2_a0)
    if not diffs0:
        print(f"  [agent_0] PERFECT MATCH (모든 26 channels 일치)")
    else:
        print(f"  [agent_0] {len(diffs0)} channels differ:")
        for ch, name, max_d, n, top in diffs0:
            print(f"    ch{ch:2d} {name:18s} max_diff={max_d:.2f} n_cells={n} top: {top}")

    diffs1 = _diff_summary(nat_a1, v2_a1)
    if not diffs1:
        print(f"  [agent_1] PERFECT MATCH")
    else:
        print(f"  [agent_1] {len(diffs1)} channels differ:")
        for ch, name, max_d, n, top in diffs1:
            print(f"    ch{ch:2d} {name:18s} max_diff={max_d:.2f} n_cells={n} top: {top}")

    # 정적 채널만 비교 (agent 위치/방향과 무관해야 일치해야 함)
    static_chs = {10: "pot", 11: "wall", 12: "onion_pile", 14: "plate_pile", 15: "goal"}
    print(f"  --- 정적 채널만 비교 (pot/wall/onion_pile/plate_pile/goal) ---")
    static_pass = True
    for ch, name in static_chs.items():
        d = np.abs(nat_a0[:, :, ch] - v2_a0[:, :, ch])
        n = int((d > 0.01).sum())
        if n > 0:
            static_pass = False
            print(f"    ch{ch} ({name}) DIFFER: {n} cells")
        else:
            print(f"    ch{ch} ({name}) match")
    return static_pass


def main():
    print("=" * 70)
    print("CEC native obs vs webapp/v2 adapter obs — per-layout 비교")
    print("=" * 70)

    pairs = [
        ("cramped_room",     "cramped_room_9"),
        ("coord_ring",       "coord_ring_9"),
        ("counter_circuit",  "counter_circuit_9"),
        ("forced_coord",     "forced_coord_9"),
        ("asymm_advantages", "asymm_advantages_9"),
    ]
    results = {}
    for ov2_name, cec_name in pairs:
        try:
            results[ov2_name] = compare_one(ov2_name, cec_name)
        except Exception as e:
            print(f"  [{ov2_name}] FAILED: {e}")
            results[ov2_name] = False

    print("\n" + "=" * 70)
    print("Static-channel comparison summary:")
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
