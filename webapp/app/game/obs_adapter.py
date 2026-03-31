"""
Observation Adapter — overcooked-ai state → JaxMARL obs (H, W, C).

JaxMARL의 get_obs_default() (overcooked.py:592-736)을 numpy로 재구현.
모든 인코딩은 JaxMARL의 비트 연산 결과와 100% 동일해야 한다.

채널 구조 (cramped_room, 1 ingredient, 30 channels):
  [0]     self agent position (binary grid)
  [1-4]   self agent direction (one-hot: UP=0, DOWN=1, RIGHT=2, LEFT=3)
  [5-6]   self agent inventory (plate bit, cooked bit)
  [7]     self agent inventory ingredient 0 (개수: 0~3)
  [8]     other agent position
  [9-12]  other agent direction (same order)
  [13-14] other agent inventory (plate, cooked)
  [15]    other agent inventory ingredient 0 (개수: 0~3)
  [16-21] static objects (wall, goal, pot, recipe_ind, button_recipe, plate_pile)
  [22]    ingredient pile 0 (onion pile)
  [23-24] ingredients on grid (plate bit, cooked bit)
  [25]    ingredients on grid ingredient 0 (개수: 0~3)
  [26-28] recipe indicators (plate, cooked, ingredient 0)
  [29]    pot timer (JaxMARL: 카운트다운, POT_COOK_TIME에서 0으로)

JaxMARL DynamicObject 비트 인코딩:
  bit 0: PLATE (1)
  bit 1: COOKED (2)
  bits 2-3: ingredient 0 개수 (0~3)
  bits 4-5: ingredient 1 개수 (0~3)  (num_ingredients > 1일 때)
"""
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


# overcooked-ai terrain 문자 → JaxMARL StaticObject 매핑
TERRAIN_TO_STATIC = {
    "X": 1,   # WALL
    "S": 4,   # GOAL (serving/delivery)
    "D": 4,   # GOAL (delivery — overcooked-ai v2 uses D)
    "P": 5,   # POT
    "B": 9,   # PLATE_PILE
    "O": 10,  # ONION_PILE (INGREDIENT_PILE_BASE + 0)
    " ": 0,   # EMPTY
}

# JaxMARL Direction enum: UP=0, DOWN=1, RIGHT=2, LEFT=3
# overcooked-ai orientation: (0,-1)=NORTH/UP, (0,1)=SOUTH/DOWN, (1,0)=EAST/RIGHT, (-1,0)=WEST/LEFT
ORIENTATION_TO_DIR_IDX = {
    (0, -1): 0,   # NORTH → UP=0
    (0, 1): 1,    # SOUTH → DOWN=1
    (1, 0): 2,    # EAST  → RIGHT=2
    (-1, 0): 3,   # WEST  → LEFT=3
}

# JaxMARL POT_COOK_TIME (settings.py)
POT_COOK_TIME = 20


def _encode_dynamic_object(obj, is_cooking=False, is_ready=False, num_ingredients=1):
    """
    overcooked-ai 오브젝트 → JaxMARL bitpacked 정수값.

    JaxMARL DynamicObject 인코딩:
      PLATE = 1 << 0 = 1
      COOKED = 1 << 1 = 2
      ingredient(idx) = (1 << 2) << (2 * idx)
        ingredient(0) = 4, ingredient(1) = 16, ...
      재료 개수 누적: 양파 3개 = 4*3 = 12 → bits 2-3 = 3
    """
    val = 0
    if obj.name == "dish":
        val = 1  # PLATE
    elif obj.name == "onion":
        val = 1 << 2  # ingredient(0) = 4
    elif obj.name == "tomato" and num_ingredients > 1:
        val = 1 << 4  # ingredient(1) = 16
    elif obj.name == "soup":
        # 팟 안의 수프 상태에 따라 비트 설정
        # JaxMARL: 요리 완성(COOKED 비트) = is_ready 또는 is_cooking 완료
        if is_ready:
            val |= 1 | 2  # PLATE | COOKED
        elif is_cooking:
            val |= 2  # COOKED만 (요리 진행 중)
        # else: 부분 수프 (재료만 넣음, 아직 요리 시작 안 함) → plate=0, cooked=0

        # 재료 개수 누적 (각 재료당 4씩 더함)
        for ing in obj.ingredients:
            if ing == "onion":
                val += (1 << 2)  # 4 per onion
            elif ing == "tomato" and num_ingredients > 1:
                val += (1 << 4)  # 16 per tomato
    return val


def _bitpacked_to_channels(val, num_ingredients=1):
    """
    JaxMARL _ingridient_layers() 재현.
    bitpacked 정수 → [plate, cooked, ing0_count, ing1_count, ...] 리스트.

    shift = [0, 1, 2, 4, ...]  →  mask = [0x1, 0x1, 0x3, 0x3, ...]
    """
    shifts = [0, 1] + [2 * (i + 1) for i in range(num_ingredients)]
    masks = [0x1, 0x1] + [0x3] * num_ingredients
    return [(int(val) >> s) & m for s, m in zip(shifts, masks)]


def overcooked_state_to_jaxmarl_obs(
    state: OvercookedState,
    mdp: OvercookedGridworld,
    agent_idx: int,
    num_ingredients: int = 1,
) -> np.ndarray:
    """
    overcooked-ai OvercookedState → JaxMARL-compatible obs tensor.

    Args:
        state: overcooked-ai game state
        mdp: OvercookedGridworld instance (for terrain info)
        agent_idx: which agent's perspective (0 or 1). self=agent_idx, other=1-agent_idx.
        num_ingredients: number of ingredient types (default 1 for onion-only layouts)

    Returns:
        obs: np.ndarray shape (H, W, C) dtype int32 (JaxMARL과 동일)
    """
    terrain = mdp.terrain_mtx
    height = len(terrain)
    width = len(terrain[0])

    # 채널 수 계산: 18 + 4*(num_ingredients + 2)
    num_channels = 18 + 4 * (num_ingredients + 2)
    obs = np.zeros((height, width, num_channels), dtype=np.int32)

    other_idx = 1 - agent_idx
    players = state.players

    self_player = players[agent_idx]
    other_player = players[other_idx]

    # === Agent layers ===
    def _encode_agent(player, start_ch):
        """agent → position(1) + direction(4) + inventory(2 + num_ingredients)"""
        y, x = player.position[1], player.position[0]
        # position
        obs[y, x, start_ch] = 1
        # direction (one-hot, JaxMARL 순서: UP=0, DOWN=1, RIGHT=2, LEFT=3)
        dir_idx = ORIENTATION_TO_DIR_IDX.get(player.orientation, 0)
        obs[y, x, start_ch + 1 + dir_idx] = 1
        # inventory (bitpacked → channel decomposition)
        inv_ch = start_ch + 5
        held = player.held_object
        if held is not None:
            inv_val = _encode_dynamic_object(held, is_cooking=False, is_ready=False,
                                              num_ingredients=num_ingredients)
            channels = _bitpacked_to_channels(inv_val, num_ingredients)
            for i, v in enumerate(channels):
                obs[y, x, inv_ch + i] = v

    ch = 0
    # Self agent: channels 0 ~ (6 + num_ingredients)
    _encode_agent(self_player, ch)
    agent_ch_size = 5 + 2 + num_ingredients  # pos(1) + dir(4) + plate(1) + cooked(1) + ingredients
    ch += agent_ch_size

    # Other agent
    _encode_agent(other_player, ch)
    ch += agent_ch_size

    # === Static object layers (6 channels) ===
    # JaxMARL 순서: WALL, GOAL, POT, RECIPE_INDICATOR, BUTTON_RECIPE_INDICATOR, PLATE_PILE
    for r in range(height):
        for c in range(width):
            terrain_char = terrain[r][c]
            static_val = TERRAIN_TO_STATIC.get(terrain_char, 0)
            if static_val == 1:   # WALL
                obs[r, c, ch] = 1
            elif static_val == 4:  # GOAL
                obs[r, c, ch + 1] = 1
            elif static_val == 5:  # POT
                obs[r, c, ch + 2] = 1
            elif static_val == 9:  # PLATE_PILE
                obs[r, c, ch + 5] = 1
    ch += 6

    # === Ingredient pile layers (num_ingredients channels) ===
    for r in range(height):
        for c in range(width):
            terrain_char = terrain[r][c]
            if terrain_char == "O":
                obs[r, c, ch] = 1  # onion pile
            elif terrain_char == "T" and num_ingredients > 1:
                obs[r, c, ch + 1] = 1  # tomato pile
    ch += num_ingredients

    # === Ingredients on grid (2 + num_ingredients channels) ===
    # JaxMARL: state.grid[:,:,1]의 bitpacked 값을 _ingridient_layers()로 디코딩
    for obj_pos, obj in state.objects.items():
        y, x = obj_pos[1], obj_pos[0]
        is_cooking = getattr(obj, "is_cooking", False)
        is_ready = getattr(obj, "is_ready", False)
        obj_val = _encode_dynamic_object(obj, is_cooking, is_ready, num_ingredients)
        channels = _bitpacked_to_channels(obj_val, num_ingredients)
        for i, v in enumerate(channels):
            obs[y, x, ch + i] = v
    ch += 2 + num_ingredients

    # === Recipe indicators (2 + num_ingredients channels) ===
    # 표준 레이아웃(cramped_room 등)에는 RECIPE_INDICATOR 셀이 없음 → 0으로 둠
    # v2 레이아웃(cramped_room_v2 등)에는 "R" 셀이 있을 수 있으나 현재 webapp에서 미사용
    ch += 2 + num_ingredients

    # === Extra layers ===
    # Pot timer layer — JaxMARL: 카운트다운 (POT_COOK_TIME에서 시작, 매 스텝 -1, 0이면 완성)
    # overcooked-ai: cooking_tick은 카운트업 (0에서 시작, 매 스텝 +1)
    # 변환: jaxmarl_timer = cook_time - cooking_tick
    for obj_pos, obj in state.objects.items():
        if hasattr(obj, "name") and obj.name == "soup":
            y, x = obj_pos[1], obj_pos[0]
            # 팟 위치인지 확인
            if terrain[y][x] == "P":
                is_cooking = getattr(obj, "is_cooking", False)
                is_ready = getattr(obj, "is_ready", False)
                if is_cooking and not is_ready:
                    # overcooked-ai: _cooking_tick은 1부터 시작, 매 스텝 +1
                    cooking_tick = getattr(obj, "_cooking_tick", 0) or 0
                    try:
                        cook_time = obj.cook_time
                    except (ValueError, AttributeError):
                        cook_time = POT_COOK_TIME
                    # JaxMARL 카운트다운: cook_time - cooking_tick
                    obs[y, x, ch] = max(0, cook_time - cooking_tick)
                # is_ready → timer = 0 (기본값), 부분 수프(아직 요리 안 시작) → timer = 0

    return obs


def get_obs_shape(layout_name: str = None, mdp: OvercookedGridworld = None, num_ingredients: int = 1):
    """레이아웃별 obs shape 반환."""
    if mdp is None:
        mdp = OvercookedGridworld.from_layout_name(layout_name)
    h = len(mdp.terrain_mtx)
    w = len(mdp.terrain_mtx[0])
    c = 18 + 4 * (num_ingredients + 2)
    return (h, w, c)
