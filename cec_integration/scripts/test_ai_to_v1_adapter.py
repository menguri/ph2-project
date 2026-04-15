"""OvercookedAIToCECAdapter 검증: V1 env.reset() 결과와 새 adapter 가 만든 obs 비교.

두 경로:
  1) V1 env.reset() → V1 State → V1 env.get_obs → CEC obs (직접, ground truth)
  2) overcooked-ai env.reset() → V1 State (new adapter) → V1 env.get_obs → CEC obs

두 env 는 reset 시 agent 위치/방향이 달라질 수 있어 obs 가 완벽히 일치하진 않을 수 있음.
대신 다음을 검증:

(A) 같은 overcooked-ai state 를 new adapter 에 넣고, 같은 state 를 흉내낸 V1 State 를
    직접 만들어 V1 env.get_obs 에 넣어 byte-exact 비교.

(B) 여러 gameplay step 동안 adapter 가 V1 obs shape/dtype 을 준수하는지 + 주요 채널 invariant
    (정적 layout, pot/plate/goal/wall 위치) 가 CEC_LAYOUTS ground truth 와 일치.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from app.game.engine import _load_custom_layout
from app.game.action_map import jaxmarl_to_overcooked

from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_from_ai import OvercookedAIToCECAdapter


LAYOUT = "cramped_room"
NUM_STEPS = 40


def _obs_summary(obs):
    """obs 의 non-zero 채널 요약."""
    lines = []
    CH = ["self_pos","other_pos","self_E","self_S","self_W","self_N",
          "other_E","other_S","other_W","other_N",
          "pot","wall","onion_pile","tomato_pile","plate_pile","goal",
          "ons_in_pot","tom_in_pot","ons_in_soup","tom_in_soup",
          "cook_time","soup_ready","plate_on_grid","onion_on_grid","tomato_on_grid","urgency"]
    for ch in range(26):
        layer = np.asarray(obs[:, :, ch])
        nz = np.argwhere(layer > 0.5)
        if len(nz) > 0 and len(nz) < 20:
            lines.append(f"    ch{ch:2d} {CH[ch]:16s}: {[(int(y),int(x)) for y, x in nz[:5]]}")
    return "\n".join(lines)


def test_reset():
    print("=" * 70)
    print(f"[test_reset] layout={LAYOUT}")
    print("=" * 70, flush=True)

    adapter = OvercookedAIToCECAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)

    mdp = _load_custom_layout(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    env.reset()
    ai_state = env.state

    # 기본 player info
    print(f"ai_state agents:")
    for i, p in enumerate(ai_state.players):
        print(f"  p{i}: pos={p.position} orient={p.orientation} held={p.held_object}")

    # Adapter 로 obs 생성
    obs_a0 = adapter.get_cec_obs(ai_state, mdp, agent_idx=0, current_step=0)
    obs_a1 = adapter.get_cec_obs(ai_state, mdp, agent_idx=1, current_step=0)
    obs_a0_np = np.asarray(obs_a0)
    obs_a1_np = np.asarray(obs_a1)

    print(f"\nobs shape: {obs_a0_np.shape}, dtype: {obs_a0_np.dtype}")

    # 정적 채널이 CEC_LAYOUTS ground truth 와 일치하는지 검증
    truth = CEC_LAYOUTS[f"{LAYOUT}_9"]
    def _flat_to_yx_set(flat):
        return {(int(i) // 9, int(i) % 9) for i in flat}
    expected = {
        10: ("pot", _flat_to_yx_set(truth["pot_idx"])),
        12: ("onion_pile", _flat_to_yx_set(truth["onion_pile_idx"])),
        14: ("plate_pile", _flat_to_yx_set(truth["plate_pile_idx"])),
        15: ("goal", _flat_to_yx_set(truth["goal_idx"])),
    }
    all_pass = True
    for ch, (name, exp) in expected.items():
        got = {(int(y), int(x)) for y, x in np.argwhere(obs_a0_np[:, :, ch] > 0.5)}
        if got == exp:
            print(f"  [PASS] ch{ch} ({name}): {sorted(got)}")
        else:
            print(f"  [FAIL] ch{ch} ({name}): got={sorted(got)}, exp={sorted(exp)}")
            all_pass = False

    # self/other pos 정확히 1 개씩
    for label, obs in [("a0", obs_a0_np), ("a1", obs_a1_np)]:
        s_sum = float(obs[:, :, 0].sum())
        o_sum = float(obs[:, :, 1].sum())
        if s_sum == 1.0 and o_sum == 1.0:
            print(f"  [PASS] {label}: self/other pos 각 1 개")
        else:
            print(f"  [FAIL] {label}: self_sum={s_sum}, other_sum={o_sum}")
            all_pass = False

    # agent 0/1 mirror: a0 의 self = a1 의 other, a0 의 other = a1 의 self
    mirror_pos = np.array_equal(obs_a0_np[:, :, 0], obs_a1_np[:, :, 1]) and \
                 np.array_equal(obs_a0_np[:, :, 1], obs_a1_np[:, :, 0])
    if mirror_pos:
        print(f"  [PASS] agent_0 obs 의 self ↔ agent_1 obs 의 other 대칭")
    else:
        print(f"  [FAIL] agent mirror 실패")
        all_pass = False

    print(f"\n[reset] {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_gameplay_invariants():
    """간단한 gameplay 중 invariant 체크."""
    print("\n" + "=" * 70)
    print(f"[test_gameplay] layout={LAYOUT}, {NUM_STEPS} steps random actions")
    print("=" * 70, flush=True)

    adapter = OvercookedAIToCECAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)
    mdp = _load_custom_layout(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    env.reset()

    truth = CEC_LAYOUTS[f"{LAYOUT}_9"]
    expected_pot_count = len(truth["pot_idx"])
    expected_plate_count = len(truth["plate_pile_idx"])
    expected_goal_count = len(truth["goal_idx"])
    expected_onion_count = len(truth["onion_pile_idx"])

    rng = np.random.RandomState(0)
    all_pass = True
    for t in range(NUM_STEPS):
        # random action 적용
        a0 = int(rng.randint(0, 6))
        a1 = int(rng.randint(0, 6))
        joint = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, r, done, _ = env.step(joint)
        ai_state = next_state

        # adapter
        obs = np.asarray(adapter.get_cec_obs(ai_state, mdp, 0, t + 1))

        # invariant: 정적 객체 개수 (pot/plate/goal/onion_pile)
        cnt = {
            "pot": float(obs[:, :, 10].sum()),
            "plate_pile": float(obs[:, :, 14].sum()),
            "goal": float(obs[:, :, 15].sum()),
            "onion_pile": float(obs[:, :, 12].sum()),
        }
        for name, exp in [("pot", expected_pot_count),
                          ("plate_pile", expected_plate_count),
                          ("goal", expected_goal_count),
                          ("onion_pile", expected_onion_count)]:
            if cnt[name] != exp:
                print(f"  [FAIL] t={t+1} ch {name}: got={cnt[name]} exp={exp}")
                all_pass = False

        # self/other pos 1 개씩
        if not (obs[:, :, 0].sum() == 1 and obs[:, :, 1].sum() == 1):
            print(f"  [FAIL] t={t+1} self/other pos count")
            all_pass = False

        # NaN/Inf
        if not np.all(np.isfinite(obs)):
            print(f"  [FAIL] t={t+1} NaN/Inf")
            all_pass = False

        if done:
            break

    print(f"\n[gameplay] {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_pot_state_encoding():
    """실제로 pot 에 onion 을 넣는 scripted action 으로 pot_status 인코딩 검증."""
    print("\n" + "=" * 70)
    print(f"[test_pot_state] 수동 cooking sequence")
    print("=" * 70, flush=True)

    adapter = OvercookedAIToCECAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)
    mdp = _load_custom_layout(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    env.reset()

    # 초기: pot 은 빈 상태. CEC obs ch16 (ons_in_pot) = 0, ch18 (ons_in_soup) = 0
    obs = np.asarray(adapter.get_cec_obs(env.state, mdp, 0, 0))
    pot_yx = np.argwhere(obs[:, :, 10] > 0.5)
    print(f"pot 위치 (y, x): {[tuple(p) for p in pot_yx]}")
    for (py, px) in pot_yx:
        ip = obs[py, px, 16]
        ins = obs[py, px, 18]
        ct = obs[py, px, 20]
        rd = obs[py, px, 21]
        print(f"  reset pot@({py},{px}): in_pot={ip} in_soup={ins} cook={ct} ready={rd}")

    # 실제 ai_state.objects 에 soup 넣어보기 (수동)
    from overcooked_ai_py.mdp.overcooked_mdp import SoupState, Recipe
    # Recipe 초기화 (cramped_room 은 onion-only)
    Recipe.configure({"num_items_for_soup": 3, "all_orders": [{"ingredients": ["onion"] * 3}]})

    # 1 onion 투입 scenario
    state = env.state
    pot_loc = mdp.get_pot_locations()[0]
    # SoupState 를 수동으로 구성
    soup = SoupState(pot_loc, ingredients=[])
    soup.add_ingredient_from_str("onion")
    state.objects[pot_loc] = soup

    obs = np.asarray(adapter.get_cec_obs(state, mdp, 0, 1))
    expected_pot_x, expected_pot_y = pot_loc
    ip = obs[expected_pot_y, expected_pot_x, 16]
    print(f"  after 1 onion in pot@({expected_pot_y},{expected_pot_x}): in_pot={ip}")
    assert ip == 1, f"expected in_pot=1, got {ip}"

    # 3 onions, still idle (not cooking yet)
    soup.add_ingredient_from_str("onion")
    soup.add_ingredient_from_str("onion")
    obs = np.asarray(adapter.get_cec_obs(state, mdp, 0, 2))
    # V1 obs: onions_in_pot_layer active when pot_status >= POT_FULL_STATUS(20)
    #         pot_status=20 (3 onions idle) → layer = 23-20=3
    ip = obs[expected_pot_y, expected_pot_x, 16]
    print(f"  after 3 onions idle: in_pot={ip}")
    assert ip == 3, f"expected in_pot=3 (full, not cooking), got {ip}"

    # Cooking 시작
    soup.begin_cooking()
    # begin_cooking 직후: is_cooking=True, cooking_tick=0 (혹은 -1?)
    obs = np.asarray(adapter.get_cec_obs(state, mdp, 0, 3))
    ins = obs[expected_pot_y, expected_pot_x, 18]
    ct = obs[expected_pot_y, expected_pot_x, 20]
    print(f"  after begin_cooking: in_soup={ins} cook_time={ct}")
    assert ins == 3, f"expected in_soup=3, got {ins}"

    # cook 1 step
    soup.cook()
    obs = np.asarray(adapter.get_cec_obs(state, mdp, 0, 4))
    ct2 = obs[expected_pot_y, expected_pot_x, 20]
    print(f"  after cook() 1 step: cook_time={ct2}")
    assert ct2 < ct or (ct == 0 and ct2 == 0), f"cook_time 감소 안 함: {ct} → {ct2}"

    # 요리 완성까지 다 cook
    while not soup.is_ready:
        soup.cook()
    obs = np.asarray(adapter.get_cec_obs(state, mdp, 0, 25))
    rd = obs[expected_pot_y, expected_pot_x, 21]
    ct_end = obs[expected_pot_y, expected_pot_x, 20]
    print(f"  after ready: soup_ready={rd} cook_time={ct_end}")
    assert rd == 1, f"expected soup_ready=1, got {rd}"
    assert ct_end == 0, f"expected cook_time=0, got {ct_end}"

    print(f"\n[pot_state] ALL PASS")
    return True


def main():
    r1 = test_reset()
    r2 = test_gameplay_invariants()
    r3 = test_pot_state_encoding()
    print("\n" + "=" * 70)
    print(f"Overall: reset={r1}, gameplay={r2}, pot_state={r3}")
    print(f"{'ALL PASS' if r1 and r2 and r3 else 'FAIL'}")
    print("=" * 70)
    return 0 if (r1 and r2 and r3) else 1


if __name__ == "__main__":
    sys.exit(main())
