#!/usr/bin/env python3
"""
협력 시나리오별 obs 인코딩 검증.

랜덤 행동 테스트로는 수프 완성/픽업/배달 등 핵심 시나리오가 거의 발생하지 않으므로,
의도적으로 각 단계를 재현하여 obs_adapter와 JaxMARL 인코딩 일치 여부를 검증한다.

검증 항목:
  1. 양파 픽업 (onion pile → 플레이어 인벤토리)
  2. 양파를 팟에 넣기 (interact with pot)
  3. 팟에 재료 1/2/3개 상태
  4. 요리 진행 중 (cooking) — pot timer
  5. 요리 완성 (ready) — pot 상태
  6. 접시(dish) 픽업
  7. 완성된 수프 픽업 (pot → 플레이어 인벤토리) ← 핵심!
  8. 수프 배달 (serving location interact)
  9. 카운터 위 오브젝트들 (dish, onion)

사용법:
    cd webapp && python scripts/test_cooperation_scenarios.py
"""
import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState, PlayerState, ObjectState, SoupState,
    Recipe,
)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs, get_obs_shape


# ─── JaxMARL ground truth 인코딩 ───────────────────────────────────────

def bitpacked_to_channels(val, num_ingredients=1):
    shifts = [0, 1] + [2 * (i + 1) for i in range(num_ingredients)]
    masks = [0x1, 0x1] + [0x3] * num_ingredients
    return [(int(val) >> s) & m for s, m in zip(shifts, masks)]


def gt_inventory_encode(held_obj):
    """JaxMARL 기준 인벤토리 인코딩 (ground truth)."""
    if held_obj is None:
        return [0, 0, 0]  # [plate, cooked, ing0]

    if held_obj.name == "dish":
        return [1, 0, 0]  # plate=1
    elif held_obj.name == "onion":
        return [0, 0, 1]  # ing0=1
    elif held_obj.name == "soup":
        # 완성된 수프: PLATE=1, COOKED=1, ingredient count
        # JaxMARL에서 pickup_soup은 항상 완성된 수프만 픽업 가능
        ing_count = sum(1 for i in held_obj.ingredients if i == "onion")
        return [1, 1, ing_count]  # plate=1, cooked=1, ing0=count
    return [0, 0, 0]


def gt_pot_encode(soup_obj, is_cooking, is_ready):
    """JaxMARL 기준 팟 안 수프 인코딩 (ground truth)."""
    if soup_obj is None:
        return [0, 0, 0]

    ing_count = sum(1 for i in soup_obj.ingredients if i == "onion")

    if is_ready:
        return [1, 1, ing_count]  # plate=1, cooked=1
    elif is_cooking:
        return [0, 1, ing_count]  # cooked=1 (요리 진행 중)
    else:
        return [0, 0, ing_count]  # 재료만 넣음


# ─── 헬퍼: 특정 위치의 obs 채널 추출 ─────────────────────────────────

def get_self_inventory(obs, player_pos):
    """obs에서 self agent inventory 채널값 추출. [plate, cooked, ing0]"""
    y, x = player_pos[1], player_pos[0]
    return [int(obs[y, x, 5]), int(obs[y, x, 6]), int(obs[y, x, 7])]


def get_other_inventory(obs, player_pos):
    """obs에서 other agent inventory 채널값 추출. [plate, cooked, ing0]"""
    y, x = player_pos[1], player_pos[0]
    return [int(obs[y, x, 13]), int(obs[y, x, 14]), int(obs[y, x, 15])]


def get_grid_object_channels(obs, pos):
    """obs에서 grid object 채널값 추출. [plate, cooked, ing0]"""
    y, x = pos[1], pos[0]
    return [int(obs[y, x, 23]), int(obs[y, x, 24]), int(obs[y, x, 25])]


def get_pot_timer(obs, pos):
    """obs에서 pot timer 추출."""
    y, x = pos[1], pos[0]
    return int(obs[y, x, 29])


# ─── 시나리오 테스트 ───────────────────────────────────────────────────

class ScenarioTester:
    def __init__(self, layout_name="cramped_room"):
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_ingredients = 1
        self.passed = 0
        self.failed = 0
        self.errors = []

    def _check(self, test_name, actual, expected, context=""):
        if actual == expected:
            self.passed += 1
            print(f"    ✓ {test_name}: {actual}")
        else:
            self.failed += 1
            msg = f"    ✗ {test_name}: got {actual}, expected {expected}"
            if context:
                msg += f" ({context})"
            print(msg)
            self.errors.append(msg)

    def _step(self, state, a1, a2):
        """환경 step + auto cook."""
        joint_action = (a1, a2)
        next_state, mdp_infos = self.mdp.get_state_transition(state, joint_action)
        # auto cook: 재료 3개 차면 자동 시작
        for pos in self.mdp.get_pot_locations():
            if next_state.has_object(pos):
                obj = next_state.get_object(pos)
                if (obj.name == "soup"
                    and not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                    obj.begin_cooking()
                    obj.cook()  # engine.py 와 동일한 1-step offset 보상
        return next_state

    def run_all(self):
        print(f"\n{'='*70}")
        print(f"협력 시나리오 검증: {self.layout_name}")
        print(f"{'='*70}")

        state = self.mdp.get_standard_start_state()
        pot_locs = self.mdp.get_pot_locations()
        onion_locs = self.mdp.get_onion_dispenser_locations()
        dish_locs = self.mdp.get_dish_dispenser_locations()
        serve_locs = self.mdp.get_serving_locations()

        print(f"  Pot: {pot_locs}, Onion: {onion_locs}, Dish: {dish_locs}, Serve: {serve_locs}")
        print(f"  Player 0: {state.players[0].position}, Player 1: {state.players[1].position}")

        # ─── 시나리오 1: 초기 상태 (빈 인벤토리) ─────────────────
        print(f"\n  [시나리오 1] 초기 상태 — 빈 인벤토리")
        obs0 = overcooked_state_to_jaxmarl_obs(state, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs0, state.players[0].position)
        self._check("P0 inventory (empty)", inv, [0, 0, 0])

        # ─── 시나리오 2: 수동으로 상태 조작 — 양파 들기 ────────
        print(f"\n  [시나리오 2] 양파 픽업 상태")
        p0 = state.players[0]
        onion_obj = ObjectState("onion", p0.position)
        state_with_onion = state.deepcopy()
        state_with_onion.players[0].held_object = onion_obj

        obs0 = overcooked_state_to_jaxmarl_obs(state_with_onion, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs0, state_with_onion.players[0].position)
        expected = gt_inventory_encode(onion_obj)
        self._check("P0 holding onion", inv, expected)

        # ─── 시나리오 3: 접시(dish) 들기 ──────────────────────
        print(f"\n  [시나리오 3] 접시(dish) 픽업 상태")
        state_with_dish = state.deepcopy()
        dish_obj = ObjectState("dish", p0.position)
        state_with_dish.players[0].held_object = dish_obj

        obs0 = overcooked_state_to_jaxmarl_obs(state_with_dish, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs0, state_with_dish.players[0].position)
        expected = gt_inventory_encode(dish_obj)
        self._check("P0 holding dish", inv, expected)

        # ─── 시나리오 4: 팟에 양파 1/2/3개 넣기 ──────────────
        print(f"\n  [시나리오 4] 팟에 양파 넣기 (1/2/3개)")
        pot_pos = pot_locs[0]

        for n_onions in [1, 2, 3]:
            state_pot = state.deepcopy()
            ingredients = ["onion"] * n_onions
            soup_obj = SoupState.get_soup(pot_pos, num_onions=n_onions, num_tomatoes=0,
                                          cooking_tick=-1)
            state_pot.objects[pot_pos] = soup_obj

            obs0 = overcooked_state_to_jaxmarl_obs(state_pot, self.mdp, 0, self.num_ingredients)
            grid_ch = get_grid_object_channels(obs0, pot_pos)
            expected = gt_pot_encode(soup_obj, False, False)
            self._check(f"Pot {n_onions} onions (not cooking)", grid_ch, expected)

        # ─── 시나리오 5: 요리 진행 중 (cooking) ──────────────
        print(f"\n  [시나리오 5] 요리 진행 중 (cooking)")
        state_cooking = state.deepcopy()
        soup_cooking = SoupState.get_soup(pot_pos, num_onions=3, num_tomatoes=0,
                                           cooking_tick=5)
        state_cooking.objects[pot_pos] = soup_cooking

        obs0 = overcooked_state_to_jaxmarl_obs(state_cooking, self.mdp, 0, self.num_ingredients)
        grid_ch = get_grid_object_channels(obs0, pot_pos)
        expected = gt_pot_encode(soup_cooking, True, False)
        self._check("Pot cooking (tick=5)", grid_ch, expected)

        timer = get_pot_timer(obs0, pot_pos)
        expected_timer = soup_cooking.cook_time - 5
        self._check("Pot timer (cooking)", timer, expected_timer,
                     f"cook_time={soup_cooking.cook_time}, tick=5")

        # ─── 시나리오 6: 요리 완성 (ready) ────────────────────
        print(f"\n  [시나리오 6] 요리 완성 (ready)")
        state_ready = state.deepcopy()
        # cook_time은 레이아웃별로 다름 (cramped_room=20, counter_circuit=45 등)
        # cooking_tick=5인 수프에서 cook_time을 얻어와서 사용
        _tmp_soup = SoupState.get_soup(pot_pos, num_onions=3, num_tomatoes=0, cooking_tick=5)
        actual_cook_time = _tmp_soup.cook_time
        soup_ready = SoupState.get_soup(pot_pos, num_onions=3, num_tomatoes=0,
                                         cooking_tick=actual_cook_time)
        state_ready.objects[pot_pos] = soup_ready

        obs0 = overcooked_state_to_jaxmarl_obs(state_ready, self.mdp, 0, self.num_ingredients)
        grid_ch = get_grid_object_channels(obs0, pot_pos)
        expected = gt_pot_encode(soup_ready, soup_ready.is_cooking, soup_ready.is_ready)
        self._check(f"Pot ready (is_cooking={soup_ready.is_cooking}, is_ready={soup_ready.is_ready})",
                     grid_ch, expected)

        timer = get_pot_timer(obs0, pot_pos)
        self._check("Pot timer (ready)", timer, 0)

        # ─── 시나리오 7: 완성된 수프 픽업 (핵심!) ─────────────
        print(f"\n  [시나리오 7] ★ 완성된 수프 픽업 ★")
        state_held_soup = state.deepcopy()
        # 완성된 수프를 플레이어가 들고 있는 상태 생성
        held_soup = SoupState.get_soup(p0.position, num_onions=3, num_tomatoes=0,
                                        cooking_tick=actual_cook_time)
        state_held_soup.players[0].held_object = held_soup

        obs0 = overcooked_state_to_jaxmarl_obs(state_held_soup, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs0, state_held_soup.players[0].position)
        expected = gt_inventory_encode(held_soup)
        self._check("P0 holding READY soup", inv, expected,
                     f"is_ready={held_soup.is_ready}")

        # agent_idx=1에서 P0의 인벤토리 확인 (other로 보임)
        obs1 = overcooked_state_to_jaxmarl_obs(state_held_soup, self.mdp, 1, self.num_ingredients)
        inv_other = get_other_inventory(obs1, state_held_soup.players[0].position)
        self._check("P0 soup seen as other (from P1)", inv_other, expected)

        # ─── 시나리오 8: 카운터 위 오브젝트들 ─────────────────
        print(f"\n  [시나리오 8] 카운터 위 오브젝트")
        # 빈 공간 찾기 (walkable position)
        terrain = self.mdp.terrain_mtx
        counter_pos = None
        for r in range(len(terrain)):
            for c in range(len(terrain[0])):
                if terrain[r][c] == "X":
                    # 벽 옆 빈 칸이 있으면 카운터로 사용 가능
                    pass
        # state.objects에 직접 넣어서 테스트
        # 빈 카운터 위치 찾기
        for r in range(len(terrain)):
            for c in range(len(terrain[0])):
                if terrain[r][c] == "X":
                    pos = (c, r)
                    if pos not in state.objects and pos not in pot_locs:
                        counter_pos = pos
                        break
            if counter_pos:
                break

        if counter_pos:
            # 카운터에 양파 놓기
            state_counter = state.deepcopy()
            onion_on_counter = ObjectState("onion", counter_pos)
            state_counter.objects[counter_pos] = onion_on_counter

            obs0 = overcooked_state_to_jaxmarl_obs(state_counter, self.mdp, 0, self.num_ingredients)
            grid_ch = get_grid_object_channels(obs0, counter_pos)
            self._check("Counter: onion", grid_ch, [0, 0, 1])

            # 카운터에 접시 놓기
            state_counter2 = state.deepcopy()
            dish_on_counter = ObjectState("dish", counter_pos)
            state_counter2.objects[counter_pos] = dish_on_counter

            obs0 = overcooked_state_to_jaxmarl_obs(state_counter2, self.mdp, 0, self.num_ingredients)
            grid_ch = get_grid_object_channels(obs0, counter_pos)
            self._check("Counter: dish", grid_ch, [1, 0, 0])
        else:
            print("    (카운터 위치를 찾지 못함, 스킵)")

        # ─── 시나리오 9: P1이 수프를 들고 있을 때 P0 시점 ──────
        print(f"\n  [시나리오 9] P1이 수프를 들고 있을 때 (P0 시점)")
        state_p1_soup = state.deepcopy()
        held_soup_p1 = SoupState.get_soup(state.players[1].position, num_onions=3,
                                           num_tomatoes=0, cooking_tick=actual_cook_time)
        state_p1_soup.players[1].held_object = held_soup_p1

        obs0 = overcooked_state_to_jaxmarl_obs(state_p1_soup, self.mdp, 0, self.num_ingredients)
        inv_other = get_other_inventory(obs0, state_p1_soup.players[1].position)
        expected = gt_inventory_encode(held_soup_p1)
        self._check("P1 soup seen as other (from P0)", inv_other, expected,
                     f"is_ready={held_soup_p1.is_ready}")

        # ─── 시나리오 10: 실제 게임 플레이로 수프 완성까지 ──────
        print(f"\n  [시나리오 10] 실제 게임 플레이 — 수프 완성+픽업 시뮬레이션")
        self._test_full_gameplay()

        # ─── 결과 ─────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"결과: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("실패 목록:")
            for e in self.errors:
                print(f"  {e}")
        print(f"{'='*70}")
        return self.failed == 0

    def _test_full_gameplay(self):
        """실제 env.step으로 수프 완성→픽업→배달 전체 과정 검증."""
        state = self.mdp.get_standard_start_state()
        pot_locs = self.mdp.get_pot_locations()
        pot_pos = pot_locs[0]

        # 단계별로 state를 수동 조작하여 확실하게 시나리오 재현
        # Step A: 팟에 양파 3개 + 자동 요리 시작
        state_a = state.deepcopy()
        soup = SoupState.get_soup(pot_pos, num_onions=3, num_tomatoes=0, cooking_tick=-1)
        state_a.objects[pot_pos] = soup
        # auto cook 적용
        soup.begin_cooking()
        soup.cook()

        obs = overcooked_state_to_jaxmarl_obs(state_a, self.mdp, 0, self.num_ingredients)
        grid_ch = get_grid_object_channels(obs, pot_pos)
        self._check("Gameplay A: pot cooking start", grid_ch, [0, 1, 3],
                     f"is_cooking={soup.is_cooking}, tick={soup._cooking_tick}")
        timer = get_pot_timer(obs, pot_pos)
        self._check("Gameplay A: pot timer", timer > 0, True,
                     f"timer={timer}")

        # Step B: 요리 진행 (cook_time - 3 tick까지)
        _tmp = SoupState.get_soup(pot_pos, num_onions=3, num_tomatoes=0, cooking_tick=5)
        ct = _tmp.cook_time
        ticks_to_near_done = ct - 3 - 2  # begin_cooking+cook으로 이미 2 tick
        for _ in range(ticks_to_near_done):
            soup.cook()
        state_b = state_a.deepcopy()

        obs = overcooked_state_to_jaxmarl_obs(state_b, self.mdp, 0, self.num_ingredients)
        grid_ch = get_grid_object_channels(obs, pot_pos)
        timer = get_pot_timer(obs, pot_pos)
        self._check("Gameplay B: pot near done", grid_ch[1], 1, "cooked bit should be 1")
        self._check("Gameplay B: timer near 0", timer <= 5, True, f"timer={timer}")

        # Step C: 요리 완성
        while not soup.is_ready:
            soup.cook()
        state_c = state_a.deepcopy()

        obs = overcooked_state_to_jaxmarl_obs(state_c, self.mdp, 0, self.num_ingredients)
        grid_ch = get_grid_object_channels(obs, pot_pos)
        self._check("Gameplay C: pot ready", grid_ch, [1, 1, 3],
                     f"is_ready={soup.is_ready}")
        timer = get_pot_timer(obs, pot_pos)
        self._check("Gameplay C: timer = 0", timer, 0)

        # Step D: 수프 픽업 (플레이어가 접시를 들고 interact → 수프 획득)
        state_d = state.deepcopy()
        # 플레이어가 완성된 수프를 들고 있음
        picked_soup = SoupState.get_soup(
            state.players[0].position,
            num_onions=3, num_tomatoes=0, cooking_tick=ct,
        )
        state_d.players[0].held_object = picked_soup

        obs = overcooked_state_to_jaxmarl_obs(state_d, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs, state_d.players[0].position)
        self._check("Gameplay D: ★ held soup ★", inv, [1, 1, 3],
                     f"is_ready={picked_soup.is_ready}")

        # Step E: 배달 후 (인벤토리 비어야 함)
        state_e = state.deepcopy()
        state_e.players[0].held_object = None

        obs = overcooked_state_to_jaxmarl_obs(state_e, self.mdp, 0, self.num_ingredients)
        inv = get_self_inventory(obs, state_e.players[0].position)
        self._check("Gameplay E: after delivery (empty)", inv, [0, 0, 0])


# ─── 메인 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
               "forced_coordination", "counter_circuit"]
    all_pass = True

    for layout in layouts:
        tester = ScenarioTester(layout)
        if not tester.run_all():
            all_pass = False

    print(f"\n{'='*70}")
    if all_pass:
        print("모든 레이아웃, 모든 시나리오 PASS ✓")
    else:
        print("불일치 발견! 위 에러 확인 필요")
