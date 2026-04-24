"""V1 JaxMARL 과 overcooked-ai 를 **동일한 초기 상태** 로 맞추고 **동일 action** 적용 후
next state 가 어디에서 어떻게 달라지는지 실증.

직접 원인 추적용. 어느 step 처리에서 divergence 발생하는지 확인.

테스트 시나리오:
  A. 두 agent 가 모두 빈 손, 둘 다 STAY
  B. agent_0 이 앞이 onion pile 을 보고 있음, INTERACT
  C. 양파 쥐고 pot 앞 UP 향, INTERACT (pot 에 drop)
  D. pot 에 양파 3개 full, (next step 에) cooking 시작 timing
  E. cooking 중 pot 앞에서 plate 로 INT (soup pickup)
  F. 양 agent 가 같은 셀로 이동 시도 (collision)
  G. 양 agent 가 지나가며 swap 시도

각 시나리오에서:
  - V1 state: agent_pos, agent_dir_idx, agent_inv, pot_status 출력
  - overcooked-ai state: players.position/orientation/held, soup state 출력
  - 논리적으로 같은가? 다르면 어디서?

Run:
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python cec_integration/scripts/find_first_state_divergence.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
from jaxmarl.environments.overcooked.common import make_overcooked_map, DIR_TO_VEC, OBJECT_TO_INDEX
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.actions import Direction, Action

from app.game.engine import _load_custom_layout
from app.game.action_map import jaxmarl_to_overcooked
from cec_integration.cec_layouts import CEC_LAYOUTS


LAYOUT = "cramped_room"
MAX_STEPS = 400

V1_EMPTY = OBJECT_TO_INDEX["empty"]
V1_ONION = OBJECT_TO_INDEX["onion"]
V1_PLATE = OBJECT_TO_INDEX["plate"]
V1_DISH = OBJECT_TO_INDEX["dish"]


def _auto_cook(state, mdp):
    """webapp 의 _auto_cook_full_pots — V1 dynamics 모방용."""
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup" and not obj.is_cooking and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()


def _summarize_v1_state(state):
    agent_info = []
    for i in range(2):
        pos = (int(state.agent_pos[i, 0]), int(state.agent_pos[i, 1]))
        dir_idx = int(state.agent_dir_idx[i])
        inv = int(state.agent_inv[i])
        agent_info.append(f"p{i}@{pos} dir={dir_idx} inv={inv}")

    pot_info = []
    pot_pos_np = np.asarray(state.pot_pos)
    for i in range(len(pot_pos_np)):
        px, py = int(pot_pos_np[i, 0]), int(pot_pos_np[i, 1])
        # V1 maze_map[..., 2] 에서 pot_status 읽기
        padding = (state.maze_map.shape[0] - 9) // 2
        status = int(state.maze_map[padding + py, padding + px, 2])
        pot_info.append(f"pot@({px},{py})={status}")

    return " | ".join(agent_info) + "  " + " ".join(pot_info)


def _summarize_ai_state(state, mdp):
    ps = []
    for i, p in enumerate(state.players):
        held = p.held_object.name if p.held_object else "_"
        ps.append(f"p{i}@{p.position} face={p.orientation} hold={held}")
    pots = []
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if obj.name == "soup":
                pots.append(f"pot@{pos}=[{len(obj.ingredients)}ing,cook={obj.is_cooking},ready={obj.is_ready},tick={getattr(obj,'_cooking_tick',-1)}]")
        else:
            pots.append(f"pot@{pos}=empty")
    return " | ".join(ps) + "  " + " ".join(pots)


# ────────────────────────────────────────────────────────────────
# V1 state 강제 구성 (ai state 와 동일한 agent pos/dir/inv + pot state)
# ────────────────────────────────────────────────────────────────

def _build_v1_state_from_ai_like(ai_state, mdp, v1_env, step=0):
    """webapp 의 obs_adapter_from_ai 와 동일 로직으로 V1 State 를 ai state 와 맞춤."""
    layout = v1_env.layout
    height, width = v1_env.height, v1_env.width

    # agents
    agent_pos_list, dir_idx_list, inv_list = [], [], []
    ORI_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    for p in ai_state.players[:2]:
        agent_pos_list.append([int(p.position[0]), int(p.position[1])])
        dir_idx_list.append(ORI_TO_DIR.get(tuple(p.orientation), 0))
        # inventory
        held = p.held_object
        if held is None:
            inv_list.append(V1_EMPTY)
        elif held.name == "onion":
            inv_list.append(V1_ONION)
        elif held.name == "dish":
            inv_list.append(V1_PLATE)
        elif held.name == "soup":
            inv_list.append(V1_DISH)
        else:
            inv_list.append(V1_EMPTY)
    agent_pos = jnp.array(agent_pos_list, dtype=jnp.uint32)
    agent_dir_idx = jnp.array(dir_idx_list, dtype=jnp.int32)
    agent_dir = DIR_TO_VEC[agent_dir_idx].astype(jnp.int8)
    agent_inv = jnp.array(inv_list, dtype=jnp.int32)

    # static positions
    def _flat_to_xy(flat):
        arr = np.asarray(flat)
        return jnp.stack([arr % width, arr // width], axis=-1).astype(jnp.uint32)

    goal_pos = _flat_to_xy(layout["goal_idx"])
    pot_pos = _flat_to_xy(layout["pot_idx"])
    plate_pile_pos = _flat_to_xy(layout["plate_pile_idx"])
    onion_pile_pos = _flat_to_xy(layout["onion_pile_idx"])
    wall_flat = np.asarray(layout["wall_idx"])
    wall = np.zeros((height, width), dtype=bool)
    for idx in wall_flat:
        wall[int(idx) // width, int(idx) % width] = True
    wall_map = jnp.array(wall, dtype=jnp.bool_)

    # pot_status — ai_state 의 soup 로부터 유도
    pot_status_list = []
    ai_soups = {tuple(pos): obj for pos, obj in ai_state.objects.items()
                if getattr(obj, "name", None) == "soup"}
    for pxy in np.asarray(pot_pos):
        x, y = int(pxy[0]), int(pxy[1])
        s = ai_soups.get((x, y))
        if s is None:
            pot_status_list.append(23)
        elif s.is_ready:
            pot_status_list.append(0)
        elif s.is_cooking:
            tick = int(getattr(s, "_cooking_tick", 0) or 0)
            # V1 formula: pot_status 는 cooking 동안 19→1 카운트다운.
            # AI tick=1 (방금 시작 1 tick 경과) ↔ V1 pot_status=19.
            # 따라서 status = POT_FULL_STATUS - tick = 20 - tick.
            status = 20 - tick
            pot_status_list.append(max(0, min(19, status)))
        else:
            n = len(s.ingredients)
            pot_status_list.append(23 - min(n, 3))
    pot_status = jnp.array(pot_status_list, dtype=jnp.uint32)

    # loose items (non-pot cells)
    onion_pos, plate_pos, dish_pos = [], [], []
    pot_set = {(int(p[0]), int(p[1])) for p in np.asarray(pot_pos)}
    for pos, obj in ai_state.objects.items():
        if (int(pos[0]), int(pos[1])) in pot_set:
            continue
        xy = [int(pos[0]), int(pos[1])]
        if obj.name == "onion":
            onion_pos.append(xy)
        elif obj.name == "dish":
            plate_pos.append(xy)
        elif obj.name == "soup":
            dish_pos.append(xy)
    onion_arr = jnp.array(onion_pos, dtype=jnp.uint32) if onion_pos else jnp.zeros((0, 2), dtype=jnp.uint32)
    plate_arr = jnp.array(plate_pos, dtype=jnp.uint32) if plate_pos else jnp.zeros((0, 2), dtype=jnp.uint32)
    dish_arr = jnp.array(dish_pos, dtype=jnp.uint32) if dish_pos else jnp.zeros((0, 2), dtype=jnp.uint32)

    maze_map = make_overcooked_map(
        wall_map, goal_pos, agent_pos, agent_dir_idx,
        plate_pile_pos, onion_pile_pos, pot_pos, pot_status,
        onion_arr, plate_arr, dish_arr,
        pad_obs=True, num_agents=2, agent_view_size=v1_env.agent_view_size,
    )

    from jaxmarl.environments.overcooked.overcooked import State as V1State
    return V1State(
        agent_pos=agent_pos, agent_dir=agent_dir, agent_dir_idx=agent_dir_idx,
        agent_inv=agent_inv, goal_pos=goal_pos, pot_pos=pot_pos,
        wall_map=wall_map, maze_map=maze_map, time=step, terminal=False,
    )


# ────────────────────────────────────────────────────────────────
# 시나리오별 divergence 검사
# ────────────────────────────────────────────────────────────────

def _run_scenario(name, action_v1_pair, setup_ai_fn=None):
    """
    1. ai_env reset
    2. setup_ai_fn(ai_state, mdp) — ai state 특정 상태로 조정
    3. ai state 에서 동등한 V1 state 구성
    4. 양 env 에 같은 action 1 회 적용
    5. next state 비교
    """
    print(f"\n{'='*70}")
    print(f"[{name}] action = {action_v1_pair}")
    print('='*70)

    mdp = _load_custom_layout(LAYOUT)
    ai_env = OvercookedEnv.from_mdp(mdp, horizon=MAX_STEPS)
    ai_env.reset()
    ai_state = ai_env.state

    if setup_ai_fn:
        setup_ai_fn(ai_state, mdp)
        ai_env.state = ai_state

    # V1 env 초기화
    v1_env = V1Overcooked(layout=CEC_LAYOUTS[f"{LAYOUT}_9"], random_reset=False,
                           max_steps=MAX_STEPS)
    v1_state = _build_v1_state_from_ai_like(ai_state, mdp, v1_env, step=0)

    # Before
    print(f"BEFORE:")
    print(f"  ai : {_summarize_ai_state(ai_state, mdp)}")
    print(f"  v1 : {_summarize_v1_state(v1_state)}")

    # Apply action
    a0, a1 = action_v1_pair
    # ai step
    ai_act = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
    ai_next, ai_rew, ai_done, _ = ai_env.step(ai_act)
    _auto_cook(ai_next, mdp)
    # v1 step
    key = jax.random.PRNGKey(0)
    v1_act = {v1_env.agents[0]: a0, v1_env.agents[1]: a1}
    rng, k = jax.random.split(key)
    v1_obs, v1_next, v1_rew, v1_done, _ = v1_env.step(k, v1_state, v1_act)

    # After
    print(f"\nAFTER action ({a0},{a1}):")
    print(f"  ai : {_summarize_ai_state(ai_next, mdp)}  reward={float(ai_rew):.0f}")
    print(f"  v1 : {_summarize_v1_state(v1_next)}  reward={float(v1_rew['agent_0']):.0f}")

    # Re-build v1 state from ai_next; compare with v1_next
    v1_rebuild = _build_v1_state_from_ai_like(ai_next, mdp, v1_env, step=1)
    print(f"\nV1-equivalent of ai_next:")
    print(f"  {_summarize_v1_state(v1_rebuild)}")

    # Diff 분석
    diffs = []
    ap1 = np.asarray(v1_next.agent_pos)
    ap2 = np.asarray(v1_rebuild.agent_pos)
    ad1 = np.asarray(v1_next.agent_dir_idx)
    ad2 = np.asarray(v1_rebuild.agent_dir_idx)
    ai1 = np.asarray(v1_next.agent_inv)
    ai2 = np.asarray(v1_rebuild.agent_inv)
    for i in range(2):
        if tuple(ap1[i]) != tuple(ap2[i]):
            diffs.append(f"p{i} pos: v1={tuple(ap1[i])} ai={tuple(ap2[i])}")
        if ad1[i] != ad2[i]:
            diffs.append(f"p{i} dir: v1={int(ad1[i])} ai={int(ad2[i])}")
        if ai1[i] != ai2[i]:
            diffs.append(f"p{i} inv: v1={int(ai1[i])} ai={int(ai2[i])}")
    # pot_status
    padding = (v1_next.maze_map.shape[0] - 9) // 2
    for px, py in np.asarray(v1_next.pot_pos):
        s1 = int(v1_next.maze_map[padding + int(py), padding + int(px), 2])
        s2 = int(v1_rebuild.maze_map[padding + int(py), padding + int(px), 2])
        if s1 != s2:
            diffs.append(f"pot@({int(px)},{int(py)}) status: v1={s1} ai={s2}")

    if diffs:
        print(f"\n⚠️ STATE DIVERGENCE:")
        for d in diffs:
            print(f"  {d}")
    else:
        print(f"\n✅ NO DIVERGENCE")

    return len(diffs) == 0


def main():
    Recipe.configure({"num_items_for_soup": 3, "all_orders": [{"ingredients": ["onion"] * 3}]})
    results = {}

    # Scenario A: 둘 다 stay, 아무 일도 안 함
    results["A_stay"] = _run_scenario("A: both stay", (4, 4))

    # Scenario B: agent_0 at (1,1) 에서 LEFT INTERACT (onion pile at (0,1))
    # default cramped_room reset: p0=(3,1), p1=(1,1). p0 는 INT 시 바라볼 방향에 따라 다름.
    # setup: p0 를 (1,1) 에 두고 LEFT 향 → INT 하면 (0,1)=onion 픽업
    def setup_B(ai_state, mdp):
        ai_state.players[0].position = (1, 1)
        ai_state.players[0].orientation = (-1, 0)  # WEST
        ai_state.players[1].position = (3, 1)
        ai_state.players[1].orientation = (1, 0)   # EAST
    results["B_onion_pickup"] = _run_scenario("B: onion pickup (INT)", (5, 4), setup_B)

    # Scenario C: p0 holds onion at (2,1) facing UP, INT → pot drop
    def setup_C(ai_state, mdp):
        ai_state.players[0].position = (2, 1)
        ai_state.players[0].orientation = (0, -1)  # NORTH
        ai_state.players[0].held_object = ObjectState("onion", (2, 1))
        ai_state.players[1].position = (3, 1)
        ai_state.players[1].orientation = (1, 0)
    results["C_onion_to_pot"] = _run_scenario("C: onion into pot", (5, 4), setup_C)

    # Scenario D: pot has 3 onions idle, next step 어떻게 되나
    def setup_D(ai_state, mdp):
        pot_loc = mdp.get_pot_locations()[0]
        soup = SoupState(pot_loc, ingredients=[])
        for _ in range(3):
            soup.add_ingredient_from_str("onion")
        ai_state.objects[pot_loc] = soup
        ai_state.players[0].position = (1, 2)
        ai_state.players[0].orientation = (1, 0)
        ai_state.players[1].position = (3, 2)
        ai_state.players[1].orientation = (-1, 0)
    results["D_cooking_start"] = _run_scenario("D: 3-onion pot, next step cooking", (4, 4), setup_D)

    # Scenario E: cooking 중 pot 앞 plate → INT (soup pickup)
    def setup_E(ai_state, mdp):
        pot_loc = mdp.get_pot_locations()[0]
        soup = SoupState(pot_loc, ingredients=[])
        for _ in range(3):
            soup.add_ingredient_from_str("onion")
        soup.begin_cooking()
        while not soup.is_ready:
            soup.cook()
        ai_state.objects[pot_loc] = soup
        ai_state.players[0].position = (2, 1)
        ai_state.players[0].orientation = (0, -1)
        ai_state.players[0].held_object = ObjectState("dish", (2, 1))
        ai_state.players[1].position = (3, 1)
        ai_state.players[1].orientation = (1, 0)
    results["E_soup_pickup"] = _run_scenario("E: plate+INT on ready pot", (5, 4), setup_E)

    # Scenario F: 두 agent 가 같은 셀 이동 시도 (collision)
    def setup_F(ai_state, mdp):
        ai_state.players[0].position = (2, 1)
        ai_state.players[0].orientation = (0, 1)  # SOUTH
        ai_state.players[1].position = (2, 3)
        ai_state.players[1].orientation = (0, -1)  # NORTH
    # 둘 다 (2,2) 이동 시도 (p0 남쪽, p1 북쪽)
    results["F_collision"] = _run_scenario("F: two agents → same cell", (1, 3), setup_F)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name, ok in results.items():
        status = "✅ match" if ok else "⚠️ DIVERGE"
        print(f"  {name:25s}  {status}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
