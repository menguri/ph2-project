#!/usr/bin/env python3
"""
obs_adapter 불일치 정량 분석.

overcooked-ai 환경에서 게임을 진행하며, 현재 obs_adapter의 출력과
JaxMARL의 인코딩 방식(ground truth)을 채널별로 비교.

사용법:
    cd webapp && python scripts/test_obs_consistency.py
"""
import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import defaultdict

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs, get_obs_shape


# ─── JaxMARL ground truth 인코딩 (numpy 재구현) ───────────────────────

# JaxMARL Direction enum: UP=0, DOWN=1, RIGHT=2, LEFT=3
JAXMARL_DIR_MAP = {
    (0, -1): 0,   # NORTH/UP
    (0, 1): 1,    # SOUTH/DOWN
    (1, 0): 2,    # EAST/RIGHT
    (-1, 0): 3,   # WEST/LEFT
}

# JaxMARL StaticObject enum
TERRAIN_TO_STATIC_GT = {
    "X": 1,   # WALL
    "S": 4,   # GOAL
    "D": 4,   # GOAL
    "P": 5,   # POT
    "B": 9,   # PLATE_PILE
    "O": 10,  # ONION_PILE (INGREDIENT_PILE_BASE + 0)
    " ": 0,   # EMPTY
}

# JaxMARL DynamicObject 비트 인코딩
# PLATE = bit 0, COOKED = bit 1, ingredient(i) = (1<<2) << (2*i)
# _ingridient_layers: shift=[0, 1, 2*(i+1)...], mask=[0x1, 0x1, 0x3...]
# 이 함수는 bitpacked 정수에서 채널별 값을 추출
def ingridient_layers_gt(value, num_ingredients=1):
    """JaxMARL의 _ingridient_layers를 numpy로 재현. 단일 셀 값 → 채널 리스트."""
    shifts = [0, 1] + [2 * (i + 1) for i in range(num_ingredients)]
    masks = [0x1, 0x1] + [0x3] * num_ingredients
    return [(int(value) >> s) & m for s, m in zip(shifts, masks)]


def encode_inventory_gt(held_obj, num_ingredients=1):
    """overcooked-ai held_object → JaxMARL bitpacked inventory 값."""
    if held_obj is None:
        return 0

    val = 0
    if held_obj.name == "dish":
        val = 1  # PLATE bit
    elif held_obj.name == "soup":
        val = 1 | 2  # PLATE | COOKED
        for ing in held_obj.ingredients:
            if ing == "onion":
                val += (1 << 2)  # ingredient(0) = 4, 누적
            elif ing == "tomato" and num_ingredients > 1:
                val += (1 << 4)  # ingredient(1) = 16, 누적
    elif held_obj.name == "onion":
        val = 1 << 2  # ingredient(0) = 4
    elif held_obj.name == "tomato" and num_ingredients > 1:
        val = 1 << 4  # ingredient(1) = 16

    return val


def encode_grid_object_gt(obj, is_cooking=False, is_ready=False, num_ingredients=1):
    """overcooked-ai grid object → JaxMARL bitpacked 값."""
    val = 0
    if obj.name == "dish":
        val = 1  # PLATE
    elif obj.name == "onion":
        val = 1 << 2  # ingredient(0)
    elif obj.name == "tomato" and num_ingredients > 1:
        val = 1 << 4  # ingredient(1)
    elif obj.name == "soup":
        # 팟 안의 수프: 요리 중이거나 완성이면 COOKED 비트 세팅
        if is_cooking or is_ready:
            val = 2  # COOKED bit (plate bit는 요리 중일 때 안 세팅)
            if is_ready:
                val = 1 | 2  # PLATE | COOKED (완성 시)
        # 재료 개수 누적
        for ing in obj.ingredients:
            if ing == "onion":
                val += (1 << 2)
            elif ing == "tomato" and num_ingredients > 1:
                val += (1 << 4)
    return val


def generate_ground_truth_obs(state, mdp, agent_idx, num_ingredients=1):
    """
    JaxMARL get_obs_default()와 동일한 obs를 numpy로 생성 (ground truth).
    """
    terrain = mdp.terrain_mtx
    height = len(terrain)
    width = len(terrain[0])

    num_channels = 18 + 4 * (num_ingredients + 2)
    obs = np.zeros((height, width, num_channels), dtype=np.uint8)

    other_idx = 1 - agent_idx
    players = state.players

    self_player = players[agent_idx]
    other_player = players[other_idx]

    # === Agent layers ===
    def _encode_agent_gt(player, start_ch):
        y, x = player.position[1], player.position[0]
        # position
        obs[y, x, start_ch] = 1
        # direction (JaxMARL 순서: UP=0, DOWN=1, RIGHT=2, LEFT=3)
        dir_idx = JAXMARL_DIR_MAP.get(player.orientation, 0)
        obs[y, x, start_ch + 1 + dir_idx] = 1
        # inventory (bitpacked → channel decomposition)
        inv_val = encode_inventory_gt(player.held_object, num_ingredients)
        inv_channels = ingridient_layers_gt(inv_val, num_ingredients)
        inv_ch = start_ch + 5
        for i, v in enumerate(inv_channels):
            obs[y, x, inv_ch + i] = v

    ch = 0
    agent_ch_size = 5 + 2 + num_ingredients
    _encode_agent_gt(self_player, ch)
    ch += agent_ch_size
    _encode_agent_gt(other_player, ch)
    ch += agent_ch_size

    # === Static object layers (6 channels) ===
    # JaxMARL 순서: WALL, GOAL, POT, RECIPE_INDICATOR, BUTTON_RECIPE_INDICATOR, PLATE_PILE
    static_encoding = [1, 4, 5, 6, 7, 9]
    for r in range(height):
        for c in range(width):
            terrain_char = terrain[r][c]
            static_val = TERRAIN_TO_STATIC_GT.get(terrain_char, 0)
            for i, enc in enumerate(static_encoding):
                if static_val == enc:
                    obs[r, c, ch + i] = 1
    ch += 6

    # === Ingredient pile layers ===
    for r in range(height):
        for c in range(width):
            terrain_char = terrain[r][c]
            if terrain_char == "O":
                obs[r, c, ch] = 1
            elif terrain_char == "T" and num_ingredients > 1:
                obs[r, c, ch + 1] = 1
    ch += num_ingredients

    # === Ingredients on grid ===
    for obj_pos, obj in state.objects.items():
        y, x = obj_pos[1], obj_pos[0]
        is_cooking = getattr(obj, "is_cooking", False)
        is_ready = getattr(obj, "is_ready", False)
        obj_val = encode_grid_object_gt(obj, is_cooking, is_ready, num_ingredients)
        obj_channels = ingridient_layers_gt(obj_val, num_ingredients)
        for i, v in enumerate(obj_channels):
            obs[y, x, ch + i] = v
    ch += 2 + num_ingredients

    # === Recipe layers ===
    # 표준 레이아웃에는 RECIPE_INDICATOR 셀이 없으므로 0으로 둠
    # (cramped_room, asymm_advantages, counter_circuit 등)
    ch += 2 + num_ingredients

    # === Pot timer layer ===
    # JaxMARL: 카운트다운 (POT_COOK_TIME에서 시작, 매 스텝 -1, 0이면 완성)
    # overcooked-ai: 카운트업 (0에서 시작, 매 스텝 +1, cook_time 도달 시 완성)
    POT_COOK_TIME = 20
    for obj_pos, obj in state.objects.items():
        if hasattr(obj, "name") and obj.name == "soup":
            y, x = obj_pos[1], obj_pos[0]
            # 팟 위치인지 확인
            if terrain[y][x] == "P":
                is_cooking = getattr(obj, "is_cooking", False)
                is_ready = getattr(obj, "is_ready", False)
                # overcooked-ai: _cooking_tick은 1부터 시작, 매 스텝 +1
                cooking_tick = getattr(obj, "_cooking_tick", 0) or 0
                try:
                    cook_time = obj.cook_time
                except (ValueError, AttributeError):
                    cook_time = POT_COOK_TIME

                if is_cooking and not is_ready:
                    # JaxMARL: 카운트다운 = cook_time - cooking_tick
                    obs[y, x, ch] = max(0, cook_time - cooking_tick)
                elif is_ready:
                    # 완성: 타이머 = 0
                    obs[y, x, ch] = 0
                # else: 아직 요리 시작 안 함 → 타이머 0

    return obs


# ─── 채널 이름 ─────────────────────────────────────────────────────────

def get_channel_names(num_ingredients=1):
    names = []
    agent_ch = 5 + 2 + num_ingredients
    # Self agent
    names.append("self_pos")
    names.extend([f"self_dir_{d}" for d in ["UP", "DOWN", "RIGHT", "LEFT"]])
    names.append("self_inv_plate")
    names.append("self_inv_cooked")
    for i in range(num_ingredients):
        names.append(f"self_inv_ing{i}")
    # Other agent
    names.append("other_pos")
    names.extend([f"other_dir_{d}" for d in ["UP", "DOWN", "RIGHT", "LEFT"]])
    names.append("other_inv_plate")
    names.append("other_inv_cooked")
    for i in range(num_ingredients):
        names.append(f"other_inv_ing{i}")
    # Static
    names.extend(["wall", "goal", "pot", "recipe_ind", "button_recipe", "plate_pile"])
    # Ingredient piles
    for i in range(num_ingredients):
        names.append(f"ing_pile_{i}")
    # Ingredients on grid
    names.append("grid_plate")
    names.append("grid_cooked")
    for i in range(num_ingredients):
        names.append(f"grid_ing{i}")
    # Recipe
    names.append("recipe_plate")
    names.append("recipe_cooked")
    for i in range(num_ingredients):
        names.append(f"recipe_ing{i}")
    # Extra
    names.append("pot_timer")
    return names


# ─── 분석 ──────────────────────────────────────────────────────────────

def run_analysis(layout_name="cramped_room", num_steps=200, num_ingredients=1):
    """무작위 행동으로 게임을 진행하며 obs_adapter vs ground truth 비교."""

    print(f"\n{'='*60}")
    print(f"Layout: {layout_name}, Steps: {num_steps}")
    print(f"{'='*60}")

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=400)

    obs_shape = get_obs_shape(mdp=mdp, num_ingredients=num_ingredients)
    channel_names = get_channel_names(num_ingredients)
    num_channels = obs_shape[2]

    print(f"Obs shape: {obs_shape}, Channels: {num_channels}")

    # 통계 수집
    total_cells = 0  # H * W per step
    channel_mismatch_count = np.zeros(num_channels, dtype=np.int64)
    channel_total_nonzero = np.zeros(num_channels, dtype=np.int64)
    mismatch_situations = defaultdict(list)  # 어떤 상황에서 차이가 나는지

    state = mdp.get_standard_start_state()
    rng = np.random.RandomState(42)

    all_actions = list(Action.ALL_ACTIONS)  # [(0,-1), (0,1), (1,0), (-1,0), (0,0), 'interact']

    for step in range(num_steps):
        for agent_idx in [0, 1]:
            # 현재 obs_adapter 출력
            obs_adapter = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx, num_ingredients)
            # Ground truth (JaxMARL 방식)
            obs_gt = generate_ground_truth_obs(state, mdp, agent_idx, num_ingredients)

            total_cells += obs_shape[0] * obs_shape[1]

            # 채널별 비교
            for ch_idx in range(num_channels):
                adapter_ch = obs_adapter[:, :, ch_idx]
                gt_ch = obs_gt[:, :, ch_idx]

                diff_mask = adapter_ch != gt_ch
                n_diff = np.sum(diff_mask)

                channel_mismatch_count[ch_idx] += n_diff
                channel_total_nonzero[ch_idx] += np.sum(gt_ch > 0)

                if n_diff > 0 and step < 10:  # 처음 10스텝만 상세 기록
                    for r in range(obs_shape[0]):
                        for c in range(obs_shape[1]):
                            if diff_mask[r, c]:
                                mismatch_situations[channel_names[ch_idx]].append({
                                    "step": step,
                                    "agent": agent_idx,
                                    "pos": (r, c),
                                    "adapter": int(adapter_ch[r, c]),
                                    "gt": int(gt_ch[r, c]),
                                })

        # 무작위 행동으로 step
        a1 = all_actions[rng.randint(len(all_actions))]
        a2 = all_actions[rng.randint(len(all_actions))]
        joint_action = (a1, a2)
        next_state, mdp_infos = mdp.get_state_transition(state, joint_action)
        done = mdp.is_terminal(next_state)

        if done:
            state = mdp.get_standard_start_state()
        else:
            state = next_state

    # ─── 결과 출력 ─────────────────────────────────────────────────
    print(f"\n총 비교: {num_steps} steps × 2 agents × {obs_shape[0]}×{obs_shape[1]} cells = {total_cells * 2} cells/channel")

    print(f"\n{'Channel':<25} {'Mismatches':>12} {'GT Nonzero':>12} {'Mismatch %':>12}")
    print("-" * 65)

    total_mismatches = 0
    for ch_idx in range(num_channels):
        name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch_{ch_idx}"
        n_mis = channel_mismatch_count[ch_idx]
        n_nz = channel_total_nonzero[ch_idx]
        total_checks = total_cells * 2
        pct = 100.0 * n_mis / total_checks if total_checks > 0 else 0

        marker = " *** BUG ***" if n_mis > 0 else ""
        print(f"{name:<25} {n_mis:>12} {n_nz:>12} {pct:>11.4f}%{marker}")
        total_mismatches += n_mis

    print("-" * 65)
    overall_pct = 100.0 * total_mismatches / (total_cells * 2 * num_channels)
    print(f"{'TOTAL':<25} {total_mismatches:>12} {'':>12} {overall_pct:>11.4f}%")

    # 상세 불일치 예시
    if mismatch_situations:
        print(f"\n{'='*60}")
        print("불일치 상세 예시 (처음 10스텝):")
        print(f"{'='*60}")
        for ch_name, examples in sorted(mismatch_situations.items()):
            print(f"\n  [{ch_name}] — {len(examples)}건")
            for ex in examples[:3]:  # 채널당 최대 3개
                print(f"    step={ex['step']}, agent={ex['agent']}, "
                      f"pos={ex['pos']}: adapter={ex['adapter']}, gt={ex['gt']}")

    return total_mismatches == 0


if __name__ == "__main__":
    # overcooked-ai 레이아웃 이름 사용
    layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
               "forced_coordination", "counter_circuit"]
    all_pass = True

    for layout in layouts:
        passed = run_analysis(layout, num_steps=200)
        if passed:
            print(f"\n  [PASS] {layout}: obs_adapter와 ground truth 완벽 일치")
        else:
            print(f"\n  [FAIL] {layout}: 불일치 발견!")
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("모든 레이아웃 PASS")
    else:
        print("불일치 발견됨 — obs_adapter.py 수정 필요")
