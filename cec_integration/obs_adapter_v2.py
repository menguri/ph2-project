"""
OV2 obs (H, W, 30) → CEC obs (9, 9, 26) 직접 변환 adapter.

v1 State 재구성 없이, OV2 obs 채널을 CEC 채널로 직접 리매핑 + 9×9 패딩.
각 레이아웃별 패딩 오프셋을 정의하여 CEC 9×9 그리드 내 올바른 위치에 배치.

OV2 obs 채널 (30ch, num_ingredients=1):
  [0]     self pos
  [1-4]   self dir (UP=0, DOWN=1, RIGHT=2, LEFT=3)
  [5-7]   self inventory (plate_bit, cooked_bit, ing0_count)
  [8]     other pos
  [9-12]  other dir
  [13-15] other inventory
  [16-21] static (WALL, GOAL, POT, RECIPE_IND, BUTTON_RECIPE_IND, PLATE_PILE)
  [22]    ingredient_pile (onion pile)
  [23-25] grid ingredients (plate_bit, cooked_bit, ing0_count)
  [26-28] recipe ingredients
  [29]    pot_timer

CEC obs 채널 (26ch):
  [0]     self pos
  [1]     other pos
  [2-5]   self dir (EAST=0, SOUTH=1, WEST=2, NORTH=3) ← CEC v1 agent_dir_idx 순서
  [6-9]   other dir (EAST=0, SOUTH=1, WEST=2, NORTH=3)
  [10]    pot locations
  [11]    wall/counter
  [12]    onion pile
  [13]    tomato pile (zeros)
  [14]    plate pile
  [15]    goal/delivery
  [16]    onions in pot (0-3, unfilled pots only)
  [17]    tomatoes in pot (zeros)
  [18]    onions in soup (0 or 3, cooking/done pots)
  [19]    tomatoes in soup (zeros)
  [20]    pot cooking time remaining (19→0)
  [21]    soup ready (binary)
  [22]    plate on grid
  [23]    onion on grid
  [24]    tomato on grid (zeros)
  [25]    urgency (binary, ≤40 steps remaining)
"""

import jax.numpy as jnp
from typing import Dict


# 각 레이아웃의 CEC 9×9 배치 크기 및 오프셋
# (dst_h, dst_w, dst_y, dst_x) — CEC 9×9 내에서 OV2 데이터를 배치할 영역
# 이 값은 CEC의 make_*_9x9() 함수가 compact template을 9×9에 임베딩할 때와 일치해야 함
# 모든 CEC 템플릿은 9×9의 좌상단(0,0)에서 시작, 나머지는 wall 패딩
LAYOUT_PADDING = {
    # asymm_advantages 는 CEC 템플릿이 4×9 (OV2는 5×9 — 상단 wall row 1줄 추가).
    # dst_h=4 로 설정하고 LAYOUT_OV2_CROP 으로 OV2 row 1~4 만 사용하도록 보정.
    "cramped_room":     (4, 5, 0, 0),
    "forced_coord":     (5, 5, 0, 0),
    "counter_circuit":  (5, 8, 0, 0),
    "coord_ring":       (5, 5, 0, 0),
    "asymm_advantages": (4, 9, 0, 0),
}

# OV2 obs 에서 읽기 시작할 (row, col) 오프셋. 지정 안 된 레이아웃은 (0, 0).
# asymm_advantages 만 OV2 grid 상단에 wall 행이 1줄 추가돼 있어서 1-row 오프셋 필요.
LAYOUT_OV2_CROP = {
    "asymm_advantages": (1, 0),
}

# CEC `_9x9(ik=False)` 레이아웃과 OV2 canonical 레이아웃의 static object 배치 차이 보정.
# - CEC 는 레이아웃마다 pot/plate/goal 을 2개씩 가짐 (CEC `_9x9` 템플릿의 쌍 객체 convention).
# - OV2 canonical 은 1개만 가진 레이아웃이 많음 → 차이만큼 추가 객체를 주입해서
#   CEC 훈련/eval 시 봤던 분포와 동일하게 재구성.
# - counter_circuit 은 OV2 에서도 2 pot/2 onion 이지만 plate/goal 이 1개씩 부족.
# - swap=True 는 과거 코드의 잘못된 보정(OV1/OV2 plate↔goal 위치가 뒤바뀌었다는 오해)
#   이었기 때문에 전부 False 로 고정. OV2 plate/goal 위치는 CEC 와 동일하다.
# extra_pot/extra_plate/extra_goal: CEC `_9x9` 에만 있는 추가 객체 좌표 (y, x)
#   — OV2 에서는 wall 인 위치. 해당 셀의 wall 채널은 0 으로 해제된다.
LAYOUT_OV1_FIX = {
    "cramped_room":     {"swap": False, "extra_pot": [(0, 0)], "extra_plate": [(3, 0)], "extra_goal": [(3, 4)]},
    "coord_ring":       {"swap": False, "extra_plate": [(0, 0)], "extra_goal": [(4, 4)]},
    "forced_coord":     {"swap": False, "extra_plate": [(4, 0)], "extra_goal": [(4, 4)]},
    "counter_circuit":  {"swap": False, "extra_plate": [(0, 0)], "extra_goal": [(0, 7)]},
    "asymm_advantages": {"swap": False},
}

# OV2 pot_timer 인코딩:
#   0 = empty pot (no onions)
#   1-19 = cooking countdown (19=just started, 1=almost done)
#   20 = ready (soup done)
# CEC pot state 채널:
#   ch16 = onions_in_pot (0-3) — only when not cooking/done
#   ch18 = onions_in_soup (0 or 3) — when cooking or done
#   ch20 = cooking_time (19→0) — only while cooking
#   ch21 = soup_ready (1) — when done
OV2_POT_COOK_TIME = 20  # OV2의 pot cook time (기본값)


def ov2_obs_to_cec(
    ov2_obs: jnp.ndarray,       # (H, W, 30) — single agent's OV2 obs
    layout_name: str,
    current_step: int = 0,
    max_steps: int = 400,
) -> jnp.ndarray:
    """OV2 (H, W, 30) obs → CEC (9, 9, 26) obs 변환.

    패딩 영역은 wall(ch11=1)로 채움.
    """
    h, w = LAYOUT_PADDING[layout_name][:2]
    y_off, x_off = LAYOUT_PADDING[layout_name][2:]
    src_y, src_x = LAYOUT_OV2_CROP.get(layout_name, (0, 0))

    cec = jnp.zeros((9, 9, 26), dtype=jnp.float32)

    # --- 패딩 영역: 전체를 wall로 채움 (ch11) ---
    cec = cec.at[:, :, 11].set(1.0)

    # --- OV2 영역 추출 (src 오프셋 적용) ---
    # asymm_advantages 같이 OV2 grid 상단에 여분의 wall row 가 있는 레이아웃은
    # src_y/src_x 로 크롭해서 CEC 템플릿 좌표계와 정렬한다.
    ov2 = ov2_obs[src_y:src_y + h, src_x:src_x + w, :]

    # Agent pos
    self_pos = ov2[:, :, 0]        # (h, w)
    other_pos = ov2[:, :, 8]       # (h, w)

    # Agent dir
    # OV2: UP=0, DOWN=1, RIGHT=2, LEFT=3  (채널 순서)
    # CEC v1: RIGHT=0, DOWN=1, LEFT=2, UP=3  (agent_dir_idx 순서 = 채널 순서)
    # 따라서 리매핑 필요: OV2[0]→CEC[3], OV2[1]→CEC[1], OV2[2]→CEC[0], OV2[3]→CEC[2]
    self_dir_ov2 = ov2[:, :, 1:5]      # (h, w, 4) — OV2 순서
    other_dir_ov2 = ov2[:, :, 9:13]    # (h, w, 4) — OV2 순서
    # CEC 순서로 리매핑: [RIGHT, DOWN, LEFT, UP]
    self_dir = jnp.stack([self_dir_ov2[:,:,2], self_dir_ov2[:,:,1],
                          self_dir_ov2[:,:,3], self_dir_ov2[:,:,0]], axis=-1)
    other_dir = jnp.stack([other_dir_ov2[:,:,2], other_dir_ov2[:,:,1],
                           other_dir_ov2[:,:,3], other_dir_ov2[:,:,0]], axis=-1)

    # Static objects (OV2 ch16-21: WALL, GOAL, POT, RECIPE_IND, BUTTON_RECIPE_IND, PLATE_PILE)
    ov2_wall = ov2[:, :, 16]       # WALL
    ov2_goal = ov2[:, :, 17]       # GOAL
    ov2_pot = ov2[:, :, 18]        # POT
    ov2_plate_pile = ov2[:, :, 21] # PLATE_PILE

    # Ingredient pile (onion pile)
    ov2_onion_pile = ov2[:, :, 22] # ingredient_pile ch

    # Grid ingredients (ch23-25: plate_bit, cooked_bit, ing0_count)
    grid_plate_bit = ov2[:, :, 23]
    grid_cooked_bit = ov2[:, :, 24]
    grid_ing0_count = ov2[:, :, 25]

    # Pot timer (ch29)
    pot_timer = ov2[:, :, 29]

    # Self inventory (ch5-7: plate_bit, cooked_bit, ing0_count)
    self_inv_plate = ov2[:, :, 5]
    self_inv_cooked = ov2[:, :, 6]
    self_inv_ing0 = ov2[:, :, 7]

    # Other inventory (ch13-15)
    other_inv_plate = ov2[:, :, 13]
    other_inv_cooked = ov2[:, :, 14]
    other_inv_ing0 = ov2[:, :, 15]

    # --- Pot state 변환 ---
    # CEC v1은 COOKED bit(dyn & 0x2) 기반으로 cooking/done 판단.
    # OV2 자동요리(start_cooking_interaction=False)에서는 COOKED bit 없이
    # pot_timer가 카운트다운되므로, obs의 grid_cooked_bit(ch24)를 함께 체크.
    #
    # OV2 pot 상태:
    #   cooked_bit=1 & extra>0 → cooking (v1 COOKED bit 기반)
    #   cooked_bit=1 & extra=0 → ready/done
    #   cooked_bit=0 & ing0>0  → filling (onions in pot, not cooking yet)
    #   cooked_bit=0 & ing0=0  → empty
    # OV2 자동요리 모드에서 cooked_bit=0 & pot_timer>0 → filling으로 처리 (CEC v1 호환)
    is_pot = ov2_pot.astype(jnp.bool_)

    # pot에 있는 onion 수 (grid ingredient count at pot positions)
    pot_onions = jnp.where(is_pot, grid_ing0_count, 0.0)
    # pot 위치의 cooked bit
    pot_cooked = jnp.where(is_pot, grid_cooked_bit, 0.0)

    # pot 상태 분류 (CEC v1 COOKED bit 기반)
    pot_is_cooking = is_pot & (pot_cooked > 0) & (pot_timer > 0)
    pot_is_ready = is_pot & (pot_cooked > 0) & (pot_timer == 0)
    pot_is_filling = is_pot & (pot_cooked == 0) & (pot_onions > 0)
    pot_is_empty = is_pot & (pot_cooked == 0) & (pot_onions == 0)

    # CEC ch16: onions_in_pot (0-3) — filling pots only
    cec_onions_in_pot = jnp.where(pot_is_filling | pot_is_empty, pot_onions, 0.0)

    # CEC ch18: onions_in_soup (0 or 3) — cooking or done
    cec_onions_in_soup = jnp.where(pot_is_cooking | pot_is_ready, 3.0, 0.0)

    # CEC ch20: cooking time remaining
    # CEC v1: countdown 19→1 (max 19). OV2: extra는 POT_COOK_TIME(20)→1.
    cec_cook_time = jnp.where(pot_is_cooking, jnp.minimum(pot_timer, 19.0), 0.0)

    # CEC ch21: soup ready
    cec_soup_ready = pot_is_ready.astype(jnp.float32)

    # --- Loose items on grid ---
    # plate on grid: grid_plate_bit at non-agent, non-pot locations
    # onion on grid: grid_ing0_count > 0 at non-pot, non-pile locations
    # 간단하게: grid ingredients에서 pot/pile이 아닌 위치의 아이템
    not_pot = ~is_pot
    not_pile = (ov2_onion_pile == 0)
    not_plate_pile = (ov2_plate_pile == 0)

    cec_plate_on_grid = jnp.where(not_pot & not_plate_pile, grid_plate_bit, 0.0)
    cec_onion_on_grid = jnp.where(not_pot & not_pile, grid_ing0_count, 0.0)
    # Agent가 들고 있는 아이템도 해당 위치에 표시 (CEC 방식)
    # self가 plate 들고 있으면 self 위치에 plate 표시
    cec_plate_on_grid = cec_plate_on_grid + self_inv_plate + other_inv_plate
    cec_onion_on_grid = cec_onion_on_grid + self_inv_ing0 + other_inv_ing0

    # Held soup: cooked_bit이면 dish (soup ready)
    # CEC에서 agent가 soup을 들고 있으면 ch21(soup_ready)에도 반영
    held_soup_self = (self_inv_cooked > 0).astype(jnp.float32)
    held_soup_other = (other_inv_cooked > 0).astype(jnp.float32)
    cec_soup_ready = cec_soup_ready + held_soup_self + held_soup_other

    # --- Wall 처리 ---
    # CEC v1 로직: maze_map에 loose item이나 agent held item이 overwrite된 후
    # maze_map == WALL을 체크하므로, 카운터 위에 아이템이 놓여있거나
    # agent가 아이템을 들고 서 있는 카운터는 wall=0이 됨.
    # loose item이 있는 셀: grid_plate_bit > 0 or grid_ing0_count > 0 (pot/pile 제외)
    has_loose_item = ((grid_plate_bit > 0) | (grid_cooked_bit > 0) | (grid_ing0_count > 0)) & not_pot
    # agent가 item을 들고 있는 위치
    has_held_self = ((self_inv_plate > 0) | (self_inv_cooked > 0) | (self_inv_ing0 > 0)) & (self_pos > 0)
    has_held_other = ((other_inv_plate > 0) | (other_inv_cooked > 0) | (other_inv_ing0 > 0)) & (other_pos > 0)
    # wall에서 이런 셀 제외
    adjusted_wall = ov2_wall * ~has_loose_item * ~has_held_self * ~has_held_other

    # 전체 9×9를 wall로 초기화한 뒤, 게임 영역의 wall 채널만 교체
    cec = cec.at[y_off:y_off+h, x_off:x_off+w, 11].set(adjusted_wall)
    # 게임 영역 바깥은 이미 wall=1로 설정됨

    # --- 채널 매핑: 게임 영역에 배치 ---
    def _place(ch, data):
        """CEC ch에 OV2 영역의 data를 패딩 오프셋에 맞게 배치."""
        return cec.at[y_off:y_off+h, x_off:x_off+w, ch].set(data)

    cec = _place(0, self_pos)
    cec = _place(1, other_pos)
    cec = _place(2, self_dir[:, :, 0])   # RIGHT/EAST
    cec = _place(3, self_dir[:, :, 1])   # DOWN/SOUTH
    cec = _place(4, self_dir[:, :, 2])   # LEFT/WEST
    cec = _place(5, self_dir[:, :, 3])   # UP/NORTH
    cec = _place(6, other_dir[:, :, 0])  # RIGHT/EAST
    cec = _place(7, other_dir[:, :, 1])  # DOWN/SOUTH
    cec = _place(8, other_dir[:, :, 2])  # LEFT/WEST
    cec = _place(9, other_dir[:, :, 3])  # UP/NORTH
    cec = _place(10, ov2_pot)           # pot
    # ch11 (wall) already set above
    cec = _place(12, ov2_onion_pile)    # onion pile
    # ch13 (tomato pile) = 0
    # --- CEC `_9x9` 레이아웃과의 차이 보정 ---
    # OV2 canonical 과 CEC 훈련 분포의 plate/goal/pot 위치가 다른 경우, 누락 객체를 주입.
    # swap 은 더 이상 사용하지 않지만 legacy 호환을 위해 분기는 유지.
    fix = LAYOUT_OV1_FIX.get(layout_name, {})
    if fix.get("swap", False):
        cec_plate_pile_ch = ov2_goal
        cec_goal_ch = ov2_plate_pile
    else:
        cec_plate_pile_ch = ov2_plate_pile
        cec_goal_ch = ov2_goal
    cec = _place(14, cec_plate_pile_ch)  # plate pile
    cec = _place(15, cec_goal_ch)        # goal
    # CEC `_9x9` 에만 있는 추가 객체 (pot/plate/goal 각 채널에 1.0 세팅 + wall 제거)
    extra_pot_cells = fix.get("extra_pot", [])
    extra_plate_cells = fix.get("extra_plate", [])
    extra_goal_cells = fix.get("extra_goal", [])
    for (ey, ex) in extra_pot_cells:
        # ch10 = pot location. extra pot 은 OV2 에 대응 객체가 없으므로 항상 "빈 pot" 상태.
        # → onions_in_pot(ch16)/onions_in_soup(ch18)/cook_time(ch20)/soup_ready(ch21)
        #   은 모두 0 으로 두면 된다 (cec 는 0 으로 초기화됐고 _place 로 덮어쓰지 않음).
        cec = cec.at[y_off + ey, x_off + ex, 10].set(1.0)
    for (ey, ex) in extra_plate_cells:
        cec = cec.at[y_off + ey, x_off + ex, 14].set(1.0)
    for (ey, ex) in extra_goal_cells:
        cec = cec.at[y_off + ey, x_off + ex, 15].set(1.0)
    # 추가 객체 위치의 wall(ch11) 제거 — CEC 입장에서 pot/plate/goal 은 non-wall 셀.
    for (ey, ex) in extra_pot_cells + extra_plate_cells + extra_goal_cells:
        cec = cec.at[y_off + ey, x_off + ex, 11].set(0.0)
    cec = _place(16, cec_onions_in_pot)
    # ch17 (tomato in pot) = 0
    cec = _place(18, cec_onions_in_soup)
    # ch19 (tomato in soup) = 0
    cec = _place(20, cec_cook_time)
    cec = _place(21, cec_soup_ready)
    cec = _place(22, cec_plate_on_grid)
    cec = _place(23, cec_onion_on_grid)
    # ch24 (tomato on grid) = 0

    # --- Urgency (ch25) ---
    steps_remaining = max_steps - current_step
    urgency = jnp.where(steps_remaining < 40, 1.0, 0.0)
    cec = cec.at[:, :, 25].set(urgency)

    return cec


def ov2_obs_batch_to_cec(
    ov2_obs_dict: Dict[str, jnp.ndarray],  # {"agent_0": (H,W,30), "agent_1": (H,W,30)}
    layout_name: str,
    current_step: int = 0,
    max_steps: int = 400,
) -> Dict[str, jnp.ndarray]:
    """OV2 obs dict → CEC obs dict 변환."""
    return {
        agent: ov2_obs_to_cec(obs, layout_name, current_step, max_steps)
        for agent, obs in ov2_obs_dict.items()
    }
