#!/usr/bin/env python3
"""
JaxMARL vs overcooked-ai 환경 step 로직 side-by-side 비교.

양쪽 환경을 동일한 action sequence로 실행하고, 매 스텝마다:
  - 에이전트 위치, 방향, 인벤토리
  - 그리드 상 오브젝트 (팟 내용물, 카운터 아이템)
  - 팟 타이머
  - 리워드
를 비교하여 차이를 정량적으로 분석.

사용법:
    cd webapp && /path/to/venv/bin/python scripts/test_env_consistency.py
"""
import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

# JaxMARL
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2 as JaxOvercooked
from jaxmarl.environments.overcooked_v2.common import (
    Actions, Direction, DynamicObject, StaticObject, DIR_TO_VEC
)
from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME

# overcooked-ai
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.actions import Action

# webapp patches — _load_custom_layout이 JaxMARL 호환 레이아웃을 로드
from app.game.engine import _load_custom_layout


# ─── JaxMARL action → overcooked-ai action 변환 ──────────────────────
JAXMARL_TO_OVERCOOKED = {
    0: (1, 0),     # right → EAST
    1: (0, 1),     # down → SOUTH
    2: (-1, 0),    # left → WEST
    3: (0, -1),    # up → NORTH
    4: (0, 0),     # stay
    5: "interact",
}

# JaxMARL Direction enum → overcooked-ai orientation
JAXMARL_DIR_TO_ORIENT = {
    0: (0, -1),   # UP → NORTH
    1: (0, 1),    # DOWN → SOUTH
    2: (1, 0),    # RIGHT → EAST
    3: (-1, 0),   # LEFT → WEST
}


def extract_jaxmarl_state_info(env, state):
    """JaxMARL state에서 비교 가능한 정보 추출."""
    info = {"agents": [], "pots": [], "grid_objects": [], "reward": 0}

    for i in range(env.num_agents):
        agent = jax.tree.map(lambda x: x[i], state.agents)
        pos = (int(agent.pos.x), int(agent.pos.y))
        direction = int(agent.dir)
        inventory = int(agent.inventory)

        # inventory 디코딩
        inv_decoded = _decode_dynamic_object(inventory)

        info["agents"].append({
            "pos": pos,
            "dir": direction,
            "orient": JAXMARL_DIR_TO_ORIENT.get(direction, (0, 0)),
            "inventory_raw": inventory,
            "inventory": inv_decoded,
        })

    # 그리드 오브젝트 (팟, 카운터 위 아이템)
    grid = np.array(state.grid)
    for r in range(env.height):
        for c in range(env.width):
            static_obj = int(grid[r, c, 0])
            ingredients = int(grid[r, c, 1])
            extra = int(grid[r, c, 2])

            if static_obj == StaticObject.POT:
                pot_info = {
                    "pos": (c, r),
                    "ingredients_raw": ingredients,
                    "timer": extra,
                    "is_cooking": extra > 0 and not (ingredients & DynamicObject.COOKED),
                    "is_ready": bool(ingredients & DynamicObject.COOKED),
                    "ingredient_count": _count_ingredients(ingredients),
                }
                info["pots"].append(pot_info)

            elif static_obj == StaticObject.WALL and ingredients != 0:
                info["grid_objects"].append({
                    "pos": (c, r),
                    "type": "counter",
                    "ingredients_raw": ingredients,
                    "decoded": _decode_dynamic_object(ingredients),
                })

    return info


def _decode_dynamic_object(val):
    """JaxMARL bitpacked 값 → 사람이 읽을 수 있는 형태."""
    if val == 0:
        return "empty"
    parts = []
    if val & DynamicObject.PLATE:
        parts.append("plate")
    if val & DynamicObject.COOKED:
        parts.append("cooked")
    onion_count = (val >> 2) & 0x3
    if onion_count > 0:
        parts.append(f"onion×{onion_count}")
    return "+".join(parts) if parts else f"raw_{val}"


def _count_ingredients(val):
    """bitpacked 값에서 총 재료 개수."""
    count = 0
    v = val >> 2
    while v > 0:
        count += v & 0x3
        v >>= 2
    return count


def extract_overcooked_state_info(mdp, state):
    """overcooked-ai state에서 비교 가능한 정보 추출."""
    info = {"agents": [], "pots": [], "grid_objects": [], "reward": 0}

    for i, player in enumerate(state.players):
        held = player.held_object
        if held is None:
            inv = "empty"
        elif held.name == "dish":
            inv = "plate"
        elif held.name == "onion":
            inv = "onion×1"
        elif held.name == "soup":
            parts = []
            if held.is_ready:
                parts.append("plate+cooked")
            n_onion = held.ingredients.count("onion") if hasattr(held, 'ingredients') else 0
            if n_onion > 0:
                parts.append(f"onion×{n_onion}")
            inv = "+".join(parts) if parts else "soup"
        else:
            inv = held.name

        info["agents"].append({
            "pos": player.position,
            "orient": player.orientation,
            "inventory": inv,
        })

    # 팟 상태
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if obj.name == "soup":
                n_ing = len(obj.ingredients)
                is_cooking = getattr(obj, "is_cooking", False)
                is_ready = getattr(obj, "is_ready", False)
                cooking_tick = getattr(obj, "_cooking_tick", -1)

                # JaxMARL 카운트다운 타이머로 변환
                if is_cooking and not is_ready:
                    try:
                        cook_time = obj.cook_time
                    except (ValueError, AttributeError):
                        cook_time = 20
                    timer = cook_time - cooking_tick
                elif is_ready:
                    timer = 0
                else:
                    timer = 0

                info["pots"].append({
                    "pos": pos,
                    "ingredient_count": n_ing,
                    "timer": timer,
                    "is_cooking": is_cooking,
                    "is_ready": is_ready,
                    "cooking_tick": cooking_tick,
                })
        else:
            info["pots"].append({
                "pos": pos,
                "ingredient_count": 0,
                "timer": 0,
                "is_cooking": False,
                "is_ready": False,
                "cooking_tick": -1,
            })

    # 카운터 위 아이템
    for obj_pos, obj in state.objects.items():
        terrain = mdp.terrain_mtx
        r, c = obj_pos[1], obj_pos[0]
        if terrain[r][c] in ("X",):  # 카운터 위 아이템
            info["grid_objects"].append({
                "pos": obj_pos,
                "type": "counter",
                "name": obj.name,
            })

    return info


def compare_states(jax_info, oc_info, step_num, action_pair):
    """두 환경의 state 비교, 차이점 반환."""
    diffs = []

    # 에이전트 비교
    for i in range(len(jax_info["agents"])):
        ja = jax_info["agents"][i]
        oa = oc_info["agents"][i]

        # 위치
        if ja["pos"] != oa["pos"]:
            diffs.append(f"  agent{i} pos: jax={ja['pos']} oc={oa['pos']}")

        # 방향
        if ja["orient"] != oa["orient"]:
            diffs.append(f"  agent{i} dir: jax={ja['orient']} oc={oa['orient']}")

        # 인벤토리
        if ja["inventory"] != oa["inventory"]:
            diffs.append(f"  agent{i} inv: jax={ja['inventory']} oc={oa['inventory']}")

    # 팟 비교
    jax_pots = sorted(jax_info["pots"], key=lambda p: p["pos"])
    oc_pots = sorted(oc_info["pots"], key=lambda p: p["pos"])

    for jp, op in zip(jax_pots, oc_pots):
        if jp["ingredient_count"] != op["ingredient_count"]:
            diffs.append(f"  pot{jp['pos']} ingredients: jax={jp['ingredient_count']} oc={op['ingredient_count']}")
        if jp["timer"] != op["timer"]:
            diffs.append(f"  pot{jp['pos']} timer: jax={jp['timer']} oc={op['timer']}")
        if jp["is_cooking"] != op["is_cooking"]:
            diffs.append(f"  pot{jp['pos']} is_cooking: jax={jp['is_cooking']} oc={op['is_cooking']}")
        if jp["is_ready"] != op["is_ready"]:
            diffs.append(f"  pot{jp['pos']} is_ready: jax={jp['is_ready']} oc={op['is_ready']}")

    return diffs


def run_comparison(layout_name="cramped_room", num_steps=500, seed=42):
    """양쪽 환경을 동일한 action sequence로 실행하여 비교."""

    print(f"\n{'='*70}")
    print(f"Layout: {layout_name}, Steps: {num_steps}, Seed: {seed}")
    print(f"{'='*70}")

    # ─── JaxMARL 환경 초기화 ───
    jax_env = JaxOvercooked(layout=layout_name, max_steps=num_steps + 10)
    jax_key = jax.random.PRNGKey(seed)
    jax_key, reset_key = jax.random.split(jax_key)
    jax_obs, jax_state = jax_env.reset(reset_key)

    # ─── overcooked-ai 환경 초기화 ───
    # webapp의 _load_custom_layout 사용: JaxMARL 레이아웃 파일로부터 mdp 생성
    # (auto_cook patch 포함)
    oc_mdp = _load_custom_layout(layout_name)

    # overcooked-ai 초기 상태를 JaxMARL과 동기화
    oc_state = oc_mdp.get_standard_start_state()

    # JaxMARL 초기 에이전트 위치/방향 확인 및 동기화
    jax_info = extract_jaxmarl_state_info(jax_env, jax_state)
    for i, agent_info in enumerate(jax_info["agents"]):
        oc_state.players[i].position = agent_info["pos"]
        oc_state.players[i].orientation = agent_info["orient"]

    # 초기 상태 비교
    oc_info = extract_overcooked_state_info(oc_mdp, oc_state)
    init_diffs = compare_states(jax_info, oc_info, 0, None)
    if init_diffs:
        print(f"초기 상태 차이:")
        for d in init_diffs:
            print(d)
    else:
        print("초기 상태 일치 ✓")

    # ─── 동일한 action sequence로 실행 ───
    rng = np.random.RandomState(seed)
    total_diffs = 0
    diff_categories = {}
    first_diff_step = None

    for step in range(num_steps):
        # 무작위 action 생성
        a0 = rng.randint(0, 6)
        a1 = rng.randint(0, 6)

        # JaxMARL step
        jax_key, step_key = jax.random.split(jax_key)
        jax_actions = {
            "agent_0": jnp.int32(a0),
            "agent_1": jnp.int32(a1),
        }
        jax_obs, jax_state, jax_rewards, jax_dones, jax_infos = jax_env.step(
            step_key, jax_state, jax_actions
        )
        jax_reward = float(jax_rewards["agent_0"]) + float(jax_rewards["agent_1"])

        # overcooked-ai step
        oc_action_0 = JAXMARL_TO_OVERCOOKED[a0]
        oc_action_1 = JAXMARL_TO_OVERCOOKED[a1]
        joint_action = (oc_action_0, oc_action_1)

        oc_next_state, oc_mdp_infos = oc_mdp.get_state_transition(oc_state, joint_action)

        # webapp의 auto-cook 패치 적용 (engine.py의 _auto_cook_full_pots와 동일)
        for pos in oc_mdp.get_pot_locations():
            if oc_next_state.has_object(pos):
                obj = oc_next_state.get_object(pos)
                if (obj.name == "soup"
                    and not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                    obj.begin_cooking()
                    obj.cook()  # 1스텝 보정: JaxMARL과 타이밍 동기화

        oc_reward = sum(oc_mdp_infos.get("sparse_reward_by_agent", [0, 0]))
        oc_state = oc_next_state

        # 비교
        jax_info = extract_jaxmarl_state_info(jax_env, jax_state)
        oc_info = extract_overcooked_state_info(oc_mdp, oc_state)

        diffs = compare_states(jax_info, oc_info, step + 1, (a0, a1))

        # 리워드 비교
        if abs(jax_reward - oc_reward) > 0.01:
            diffs.append(f"  reward: jax={jax_reward} oc={oc_reward}")

        if diffs:
            total_diffs += 1
            if first_diff_step is None:
                first_diff_step = step + 1

            for d in diffs:
                # 카테고리별 집계
                cat = d.strip().split(":")[0]
                diff_categories[cat] = diff_categories.get(cat, 0) + 1

            if total_diffs <= 5:  # 처음 5건만 상세 출력
                action_names = {0: "R", 1: "D", 2: "L", 3: "U", 4: "S", 5: "I"}
                print(f"\nStep {step+1} (a0={action_names[a0]}, a1={action_names[a1]}):")
                for d in diffs:
                    print(d)

        # JaxMARL terminal 체크
        if bool(jax_dones["__all__"]):
            print(f"\n  JaxMARL episode 종료 at step {step+1}")
            break

    # ─── 결과 요약 ───
    print(f"\n{'='*70}")
    print(f"결과 요약:")
    print(f"  총 스텝: {step+1}")
    print(f"  불일치 스텝 수: {total_diffs}/{step+1} ({100*total_diffs/(step+1):.1f}%)")
    if first_diff_step:
        print(f"  첫 불일치 스텝: {first_diff_step}")

    if diff_categories:
        print(f"\n  카테고리별 불일치:")
        for cat, count in sorted(diff_categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}건")
    else:
        print("  모든 스텝 완벽 일치 ✓")

    return total_diffs == 0


if __name__ == "__main__":
    layouts = ["cramped_room", "asymm_advantages", "coord_ring",
               "forced_coord", "counter_circuit"]
    all_pass = True

    for layout in layouts:
        passed = run_comparison(layout, num_steps=1000, seed=42)
        all_pass = all_pass and passed

    print(f"\n{'='*70}")
    if all_pass:
        print("모든 레이아웃 PASS")
    else:
        print("불일치 발견 — 수정 필요")
