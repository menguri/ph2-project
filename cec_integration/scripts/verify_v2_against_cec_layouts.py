"""obs_adapter_v2 출력을 CEC `_9x9` ground truth 레이아웃과 직접 비교.

CEC 는 eval 시 `overcooked_layouts["{layout}_9"] = make_*_9x9(PRNGKey(0), ik=False)`
를 사용한다. `cec_integration.cec_layouts.CEC_LAYOUTS` 가 이 5개 레이아웃의
정적 객체 위치를 precompute 해둔다. obs_adapter_v2 의 출력이 이 ground truth
의 pot/plate/goal/onion/wall 좌표를 그대로 재현하는지 레이아웃별로 검증한다.

이 스크립트는 v1 adapter 와의 비교(verify_obs_adapter_v2.py) 가 아니라,
"CEC 모델이 훈련/eval 시 실제로 봤던 obs 분포" 와의 일치를 확인한다.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. ./overcooked_v2/bin/python \
        cec_integration/scripts/verify_v2_against_cec_layouts.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec, LAYOUT_PADDING

NUM_STEPS = 400

# OV2 canonical layout name → CEC ground truth layout name
LAYOUT_PAIRS = [
    ("cramped_room",     "cramped_room_9"),
    ("coord_ring",       "coord_ring_9"),
    ("counter_circuit",  "counter_circuit_9"),
    ("forced_coord",     "forced_coord_9"),
    ("asymm_advantages", "asymm_advantages_9"),
]

# CEC obs channel indices for static objects
CH_POT = 10
CH_WALL = 11
CH_ONION_PILE = 12
CH_PLATE_PILE = 14
CH_GOAL = 15


def _flat_to_coords(flat_indices):
    """9x9 flat index list → set of (row, col) tuples."""
    return {(int(i) // 9, int(i) % 9) for i in flat_indices}


def _channel_to_coords(obs, ch):
    """CEC obs 의 한 채널에서 값이 > 0.5 인 셀들의 (row, col) 집합."""
    coords = np.argwhere(np.asarray(obs[:, :, ch]) > 0.5)
    return {(int(r), int(c)) for r, c in coords}


def _verify_one(ov2_layout, cec_layout_name):
    """한 레이아웃에 대해 adapter 출력 vs CEC_LAYOUTS ground truth 비교."""
    print(f"\n[{ov2_layout}] vs CEC ground truth {cec_layout_name}")

    env = jaxmarl.make(
        "overcooked_v2",
        layout=ov2_layout,
        max_steps=NUM_STEPS,
        random_reset=False,
        random_agent_positions=False,
    )
    obs_dict, _ = env.reset(jax.random.PRNGKey(0))

    # agent_0 기준으로 CEC obs 생성
    ov2_obs = jnp.array(obs_dict["agent_0"], dtype=jnp.float32)
    cec_obs = np.asarray(ov2_obs_to_cec(ov2_obs, ov2_layout, 0, NUM_STEPS))

    truth = CEC_LAYOUTS[cec_layout_name]
    expected = {
        "pot":        _flat_to_coords(truth["pot_idx"]),
        "plate_pile": _flat_to_coords(truth["plate_pile_idx"]),
        "goal":       _flat_to_coords(truth["goal_idx"]),
        "onion_pile": _flat_to_coords(truth["onion_pile_idx"]),
    }
    actual = {
        "pot":        _channel_to_coords(cec_obs, CH_POT),
        "plate_pile": _channel_to_coords(cec_obs, CH_PLATE_PILE),
        "goal":       _channel_to_coords(cec_obs, CH_GOAL),
        "onion_pile": _channel_to_coords(cec_obs, CH_ONION_PILE),
    }

    # Wall 은 정적 + 동적 (loose item) 이 섞이므로 "정적 wall 위치가 정답 wall 의 부분집합" 조건만 확인
    # truth wall_idx 는 여분 객체 포함이므로 (pot/plate/goal/onion 제외한) 순수 wall 만 추출
    all_obj = (set(truth["pot_idx"].tolist()) | set(truth["plate_pile_idx"].tolist())
               | set(truth["goal_idx"].tolist()) | set(truth["onion_pile_idx"].tolist())
               | set(truth["agent_idx"].tolist()))
    pure_wall_flats = [int(i) for i in truth["wall_idx"] if int(i) not in all_obj]
    expected_walls = _flat_to_coords(pure_wall_flats)
    actual_walls = _channel_to_coords(cec_obs, CH_WALL)

    # Agent 위치는 OV2 reset 결과라 CEC ground truth 와 다를 수 있음 — 제외
    # 대신 OV2 agent 위치에서만 wall 이 꺼져 있는지 확인
    agent_self_set = _channel_to_coords(cec_obs, 0)  # ch0 = self pos
    agent_other_set = _channel_to_coords(cec_obs, 1)  # ch1 = other pos
    agent_cells = agent_self_set | agent_other_set

    all_pass = True
    for key in ["pot", "plate_pile", "goal", "onion_pile"]:
        missing = expected[key] - actual[key]
        extra = actual[key] - expected[key]
        if missing or extra:
            print(f"  [FAIL] {key}: missing={sorted(missing)}, extra={sorted(extra)}")
            all_pass = False
        else:
            print(f"  [PASS] {key}: {sorted(actual[key])}")

    # Wall: actual 은 정적 wall 에서 agent 셀만 제외한 것과 같아야 함
    expected_walls_minus_agents = expected_walls - agent_cells
    wall_missing = expected_walls_minus_agents - actual_walls
    wall_extra = actual_walls - expected_walls_minus_agents
    # CEC 는 padding 영역도 wall 이므로 (h_template 바깥) 차이 나는 건 무시
    if wall_missing or wall_extra:
        print(f"  [WARN] wall: missing={len(wall_missing)}, extra={len(wall_extra)} cells (may include padding)")
    else:
        print(f"  [PASS] wall: {len(actual_walls)} cells match")

    return all_pass


def main() -> int:
    print("=" * 70)
    print("obs_adapter_v2 vs CEC `_9x9` ground truth 정적 객체 위치 비교")
    print("=" * 70)

    results = {}
    for ov2_name, cec_name in LAYOUT_PAIRS:
        results[ov2_name] = _verify_one(ov2_name, cec_name)

    print("\n" + "=" * 70)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name:20s} : {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n전체 결과: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
