"""CEC obs sync 시나리오 테스트.

overcooked_v2 env에서 cooking 사이클 (양파 픽업 → pot → 요리 → plate → soup → 배달) 을
스크립트된 action 으로 돌리면서, 각 step 에서 `ov2_obs_to_cec` 가 만드는 CEC (9,9,26) obs
가 실제 env 상태와 동기화되는지 검증한다.

검증 항목:
1. 정적 객체 (pot/plate_pile/goal/onion_pile/wall) 위치는 에피소드 내 불변
2. Agent 위치 채널 (ch0 self, ch1 other) 합이 각각 1
3. 방향 one-hot: agent 위치에서 ch2-5 (self dir) 또는 ch6-9 (other dir) 합이 1
4. Inventory: agent 가 아이템을 들고 있으면, 해당 위치에 (CEC 관습에 따라) grid item 채널이 반영
5. Pot 상태: 요리 중일 때 ch20 (cook_time) > 0, ch18 (onions_in_soup) = 3, 완성 시 ch21 (soup_ready) = 1
6. Urgency (ch25): 남은 스텝 < 40 일 때만 1

사용:
    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. ./overcooked_v2/bin/python \
        cec_integration/scripts/test_cec_scenarios.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.obs_adapter_v2 import ov2_obs_to_cec, LAYOUT_PADDING, LAYOUT_OV2_CROP
from cec_integration.cec_layouts import CEC_LAYOUTS


# CEC obs 채널 인덱스
CH = {
    "self_pos": 0, "other_pos": 1,
    "self_dir": slice(2, 6), "other_dir": slice(6, 10),
    "pot": 10, "wall": 11, "onion_pile": 12, "plate_pile": 14, "goal": 15,
    "onions_in_pot": 16, "onions_in_soup": 18,
    "cook_time": 20, "soup_ready": 21,
    "plate_on_grid": 22, "onion_on_grid": 23, "urgency": 25,
}

# OV2 action: right=0, down=1, left=2, up=3, stay=4, interact=5
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = range(6)


def _check_invariants(cec_obs, layout, step, max_steps):
    """에피소드 내내 지켜져야 할 invariant 체크."""
    errs = []

    # (1) self/other pos 각각 정확히 1개
    s_sum = float(cec_obs[:, :, CH["self_pos"]].sum())
    o_sum = float(cec_obs[:, :, CH["other_pos"]].sum())
    if not np.isclose(s_sum, 1.0):
        errs.append(f"self_pos sum={s_sum} (expected 1)")
    if not np.isclose(o_sum, 1.0):
        errs.append(f"other_pos sum={o_sum} (expected 1)")

    # (2) 방향 one-hot
    self_yx = np.argwhere(cec_obs[:, :, CH["self_pos"]] > 0.5)
    other_yx = np.argwhere(cec_obs[:, :, CH["other_pos"]] > 0.5)
    if len(self_yx) == 1:
        sy, sx = self_yx[0]
        d = float(cec_obs[sy, sx, CH["self_dir"]].sum())
        if not np.isclose(d, 1.0):
            errs.append(f"self_dir at ({sy},{sx}) sum={d}")
    if len(other_yx) == 1:
        oy, ox = other_yx[0]
        d = float(cec_obs[oy, ox, CH["other_dir"]].sum())
        if not np.isclose(d, 1.0):
            errs.append(f"other_dir at ({oy},{ox}) sum={d}")

    # (3) 정적 객체 합이 CEC ground truth 와 동일
    truth = CEC_LAYOUTS[f"{layout}_9"]
    for ch_key, truth_key in [("pot", "pot_idx"), ("plate_pile", "plate_pile_idx"),
                              ("goal", "goal_idx"), ("onion_pile", "onion_pile_idx")]:
        expected = len(truth[truth_key])
        actual = float(cec_obs[:, :, CH[ch_key]].sum())
        # plate_on_grid/onion_on_grid 는 held item 때문에 pile 외에도 증가 가능 —
        # pile 자체는 고정이어야 하므로 ch 값 대신 개수만 체크
        if not np.isclose(actual, expected):
            errs.append(f"{ch_key} count={actual}, expected={expected}")

    # (4) Urgency: 남은 스텝 < 40 이면 모든 셀이 1
    remaining = max_steps - step
    u_vals = np.unique(cec_obs[:, :, CH["urgency"]])
    if remaining < 40:
        if not (len(u_vals) == 1 and u_vals[0] == 1.0):
            errs.append(f"urgency should be all 1 at step={step}, got unique={u_vals}")
    else:
        if not (len(u_vals) == 1 and u_vals[0] == 0.0):
            errs.append(f"urgency should be all 0 at step={step}, got unique={u_vals}")

    # (5) NaN/Inf 체크
    if not np.all(np.isfinite(cec_obs)):
        errs.append("NaN/Inf in obs")

    return errs


def _describe_pots(cec_obs):
    """각 pot 위치의 상태를 문자열로 요약."""
    lines = []
    pot_coords = np.argwhere(cec_obs[:, :, CH["pot"]] > 0.5)
    for (py, px) in pot_coords:
        in_pot = float(cec_obs[py, px, CH["onions_in_pot"]])
        in_soup = float(cec_obs[py, px, CH["onions_in_soup"]])
        cook = float(cec_obs[py, px, CH["cook_time"]])
        ready = float(cec_obs[py, px, CH["soup_ready"]])
        lines.append(f"pot@({py},{px}): in_pot={in_pot:.0f} in_soup={in_soup:.0f} cook={cook:.0f} ready={ready:.0f}")
    return " | ".join(lines)


def _describe_inventory(cec_obs, label):
    """agent 위치에서 inventory (plate/cooked/ing0) 근사 추출."""
    pos_ch = CH["self_pos"] if label == "self" else CH["other_pos"]
    yx = np.argwhere(cec_obs[:, :, pos_ch] > 0.5)
    if len(yx) != 1:
        return f"{label}: ?"
    y, x = yx[0]
    plate = float(cec_obs[y, x, CH["plate_on_grid"]])
    ready = float(cec_obs[y, x, CH["soup_ready"]])
    onion = float(cec_obs[y, x, CH["onion_on_grid"]])
    return f"{label}@({y},{x}) plate={plate:.0f} ready={ready:.0f} onion={onion:.0f}"


def test_cooking_cycle(layout, max_steps=100):
    """cramped_room 에서 간단한 cooking 사이클을 돌리며 obs 검증."""
    print(f"\n=== [{layout}] cooking cycle ===")

    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=max_steps,
        random_reset=False,
        random_agent_positions=False,
    )
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # 스크립트된 action 시퀀스 — cramped_room 전용 (다른 레이아웃은 invariant 만 검증)
    # cramped_room OV2: pot@(0,2), onion@(1,0)(1,4), plate@(3,1), goal@(3,3)
    # agent_0 @ (1,1), agent_1 @ (1,3) 시작
    if layout == "cramped_room":
        # agent_0: 왼쪽 onion → pot (x3), plate → soup → deliver
        # agent_1: 그냥 stay
        a0_script = [
            A_LEFT, A_INTERACT,                    # 양파 1 pickup
            A_UP, A_INTERACT,                      # pot 에 put
            A_LEFT, A_INTERACT,                    # 양파 2 pickup
            A_UP, A_INTERACT,                      # pot 에 put
            A_LEFT, A_INTERACT,                    # 양파 3 pickup
            A_UP, A_INTERACT,                      # pot 에 put → cooking start
        ] + [A_STAY] * 20 + [                      # 요리 대기
            A_DOWN, A_DOWN, A_INTERACT,            # plate pickup (plate@(3,1))
            A_UP, A_UP, A_INTERACT,                # soup pickup from pot
            A_DOWN, A_RIGHT, A_DOWN, A_INTERACT,   # goal@(3,3) 배달 시도
        ]
    else:
        # 다른 레이아웃은 단순 stay 로 invariant 만 확인
        a0_script = [A_STAY] * 20

    n_fail = 0
    print(f"  초기 obs: {_describe_pots(np.asarray(ov2_obs_to_cec(jnp.array(obs['agent_0'], dtype=jnp.float32), layout, 0, max_steps)))}")
    for step_i, a0 in enumerate(a0_script):
        actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(A_STAY)}
        rng, k = jax.random.split(rng)
        obs, state, reward, done, info = env.step(k, state, actions)
        cec = np.asarray(ov2_obs_to_cec(
            jnp.array(obs["agent_0"], dtype=jnp.float32), layout, step_i + 1, max_steps
        ))
        errs = _check_invariants(cec, layout, step_i + 1, max_steps)
        if errs:
            n_fail += 1
            print(f"  step {step_i+1} a0={a0} r={float(reward['agent_0']):.1f}: FAIL {errs}")
        else:
            # 주요 전환점만 출력
            pot_desc = _describe_pots(cec)
            inv = _describe_inventory(cec, "self")
            if any(x in pot_desc for x in ["in_pot=1", "in_pot=2", "in_pot=3", "cook=", "ready=1"]) or "ready=1" in inv or "plate=1" in inv or float(reward["agent_0"]) > 0:
                print(f"  step {step_i+1:3d} a0={a0} r={float(reward['agent_0']):.1f}: {inv} | {pot_desc}")
    return n_fail == 0


def test_pot_state_encoding(layout="cramped_room", max_steps=400):
    """OV2 obs 의 pot-관련 채널을 직접 조작해서 CEC pot state 변환의 정확성을 확인.

    OV2 pot 위치 (ch18=POT) 에 대해:
        ch23 = grid_plate_bit (이 셀에서는 0)
        ch24 = grid_cooked_bit (cooking/done 일 때 1)
        ch25 = grid_ing0_count (양파 개수 0~3)
        ch29 = pot_timer (요리 중이면 카운트다운, 0=ready 또는 empty)

    각 stage 별로 CEC 채널 (ch16/18/20/21) 이 올바르게 인코딩되는지 검증.
    """
    print(f"\n=== [pot state encoding via synthetic OV2 obs] layout={layout} ===")

    env = jaxmarl.make("overcooked_v2", layout=layout, max_steps=max_steps,
                       random_reset=False, random_agent_positions=False)
    obs_dict, _ = env.reset(jax.random.PRNGKey(0))
    base = np.asarray(obs_dict["agent_0"], dtype=np.float32)

    # OV2 pot 위치 찾기 (ch18)
    pot_yx = np.argwhere(base[:, :, 18] > 0.5)
    assert len(pot_yx) >= 1, "no pot found in OV2 obs"
    py, px = pot_yx[0]  # 첫 번째 pot
    print(f"  OV2 pot @ ({py}, {px})")

    # Adapter 출력에서 같은 OV2 위치가 매핑되는 CEC 좌표 찾기
    src_y, src_x = LAYOUT_OV2_CROP.get(layout, (0, 0))
    dst_y, dst_x = LAYOUT_PADDING[layout][2:4]
    cec_py, cec_px = (py - src_y) + dst_y, (px - src_x) + dst_x

    stages = [
        # (label, ch24=cooked_bit, ch25=ing0_count, ch29=pot_timer,
        #  expected ch16, ch18, ch20, ch21)
        ("empty",          0, 0, 0,   0, 0, 0,  0),
        ("1 onion",        0, 1, 0,   1, 0, 0,  0),
        ("2 onions",       0, 2, 0,   2, 0, 0,  0),
        ("3 onions (cooking start)",   1, 3, 20,  0, 3, 19, 0),
        ("cooking step 10", 1, 3, 10,  0, 3, 10, 0),
        ("ready",          1, 3, 0,   0, 3, 0,  1),
    ]

    n_fail = 0
    for label, c24, c25, c29, e16, e18, e20, e21 in stages:
        synth = base.copy()
        # pot 위치만 수정
        synth[py, px, 23] = 0.0  # plate_bit
        synth[py, px, 24] = float(c24)
        synth[py, px, 25] = float(c25)
        synth[py, px, 29] = float(c29)
        cec = np.asarray(ov2_obs_to_cec(jnp.array(synth, dtype=jnp.float32),
                                         layout, 0, max_steps))
        a16 = float(cec[cec_py, cec_px, 16])
        a18 = float(cec[cec_py, cec_px, 18])
        a20 = float(cec[cec_py, cec_px, 20])
        a21 = float(cec[cec_py, cec_px, 21])
        ok = (a16 == e16 and a18 == e18 and a20 == e20 and a21 == e21)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label:30s} → CEC pot@({cec_py},{cec_px}) "
              f"in_pot={a16:.0f}/{e16} in_soup={a18:.0f}/{e18} "
              f"cook={a20:.0f}/{e20} ready={a21:.0f}/{e21}")
        if not ok:
            n_fail += 1

    return n_fail == 0


def test_inventory_encoding(layout="cramped_room", max_steps=400):
    """OV2 self/other inventory 채널을 조작해서 CEC plate_on_grid/onion_on_grid/soup_ready
    채널이 agent 위치에 올바르게 반영되는지 검증.

    CEC 관습: agent 가 들고 있는 item 은 해당 grid 셀의 채널에도 반영됨.
        - onion 들기 → ch23 (onion_on_grid) at agent pos += 1
        - plate(dish) 들기 → ch22 (plate_on_grid) at agent pos += 1
        - soup 들기 → ch21 (soup_ready) at agent pos += 1, ch22 += 1
    """
    print(f"\n=== [inventory encoding via synthetic OV2 obs] layout={layout} ===")

    env = jaxmarl.make("overcooked_v2", layout=layout, max_steps=max_steps,
                       random_reset=False, random_agent_positions=False)
    obs_dict, _ = env.reset(jax.random.PRNGKey(0))
    base = np.asarray(obs_dict["agent_0"], dtype=np.float32)

    # self pos (ch0)
    s_yx = np.argwhere(base[:, :, 0] > 0.5)
    assert len(s_yx) == 1
    sy, sx = s_yx[0]

    src_y, src_x = LAYOUT_OV2_CROP.get(layout, (0, 0))
    dst_y, dst_x = LAYOUT_PADDING[layout][2:4]
    # OV2 self 위치가 src crop 안에 있어야만 CEC 에 매핑됨 — 일반 cramped_room 은 src=(0,0)
    cec_sy, cec_sx = (sy - src_y) + dst_y, (sx - src_x) + dst_x

    items = [
        # (label, ch5=plate, ch6=cooked, ch7=ing0,
        #  expected plate_on_grid(ch22), onion_on_grid(ch23), soup_ready(ch21))
        ("empty",          0, 0, 0,   0, 0, 0),
        ("hold onion",     0, 0, 1,   0, 1, 0),
        ("hold plate(dish)", 1, 0, 0,   1, 0, 0),
        ("hold soup",      1, 1, 3,   1, 3, 1),
    ]

    n_fail = 0
    for label, c5, c6, c7, e22, e23, e21 in items:
        synth = base.copy()
        synth[sy, sx, 5] = float(c5)
        synth[sy, sx, 6] = float(c6)
        synth[sy, sx, 7] = float(c7)
        cec = np.asarray(ov2_obs_to_cec(jnp.array(synth, dtype=jnp.float32),
                                         layout, 0, max_steps))
        a22 = float(cec[cec_sy, cec_sx, 22])
        a23 = float(cec[cec_sy, cec_sx, 23])
        a21 = float(cec[cec_sy, cec_sx, 21])
        ok = (a22 == e22 and a23 == e23 and a21 == e21)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label:18s} → CEC self@({cec_sy},{cec_sx}) "
              f"plate_on_grid={a22:.0f}/{e22} onion_on_grid={a23:.0f}/{e23} "
              f"soup_ready={a21:.0f}/{e21}")
        if not ok:
            n_fail += 1

    return n_fail == 0


def main():
    print("=" * 70)
    print("CEC obs sync scenario test")
    print("=" * 70)

    results = {}
    for layout in ["cramped_room", "coord_ring", "counter_circuit", "forced_coord", "asymm_advantages"]:
        results[layout] = test_cooking_cycle(layout)

    # Synthetic-state cooking tests on cramped_room (가장 간단한 레이아웃)
    pot_ok = test_pot_state_encoding("cramped_room")
    inv_ok = test_inventory_encoding("cramped_room")
    results["pot_state_encoding"] = pot_ok
    results["inventory_encoding"] = inv_ok

    print("\n" + "=" * 70)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name:20s} : {'PASS' if ok else 'FAIL'}")
    ok = all(results.values())
    print(f"\n전체 결과: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
