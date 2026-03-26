"""
Phase 3: Observation Adapter — overcooked-ai state → JaxMARL obs (H, W, C).

⚠️ 위험지점 #1: 채널 매핑이 정확해야 모델이 올바르게 동작.
JaxMARL의 get_obs_default() (overcooked.py:587-731)을 numpy로 재구현.

채널 구조 (cramped_room, 1 ingredient, 30 channels):
  [0]     self agent position (binary grid)
  [1-4]   self agent direction (one-hot: E, S, W, N)
  [5-6]   self agent inventory (plate bit, cooked bit)
  [7]     self agent inventory ingredient 0
  [8]     other agent position
  [9-12]  other agent direction
  [13-14] other agent inventory (plate, cooked)
  [15]    other agent inventory ingredient 0
  [16-21] static objects (wall, goal, pot, recipe_ind, button_recipe, plate_pile)
  [22]    ingredient pile 0 (onion pile)
  [23-24] ingredients on grid (plate bit, cooked bit)
  [25]    ingredients on grid ingredient 0
  [26-27] recipe indicators (plate bit, cooked bit)
  [28]    recipe indicators ingredient 0
  [29]    pot timer
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

# Direction index mapping (overcooked-ai orientation to JaxMARL direction index)
# overcooked-ai: (0,-1)=NORTH, (0,1)=SOUTH, (-1,0)=WEST, (1,0)=EAST
ORIENTATION_TO_DIR_IDX = {
    (1, 0): 0,    # EAST  → right
    (0, 1): 1,    # SOUTH → down
    (-1, 0): 2,   # WEST  → left
    (0, -1): 3,   # NORTH → up
}


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
        obs: np.ndarray shape (H, W, C) dtype uint8
    """
    terrain = mdp.terrain_mtx
    height = len(terrain)
    width = len(terrain[0])

    # 채널 수 계산: 18 + 4*(num_ingredients + 2)
    num_channels = 18 + 4 * (num_ingredients + 2)
    obs = np.zeros((height, width, num_channels), dtype=np.uint8)

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
        # direction (one-hot)
        dir_idx = ORIENTATION_TO_DIR_IDX.get(player.orientation, 0)
        obs[y, x, start_ch + 1 + dir_idx] = 1
        # inventory
        inv_ch = start_ch + 5  # plate bit, cooked bit, then ingredients
        held = player.held_object
        if held is not None:
            if held.name == "dish":
                obs[y, x, inv_ch] = 1  # plate bit
            elif held.name == "soup":
                obs[y, x, inv_ch] = 1      # plate bit
                obs[y, x, inv_ch + 1] = 1  # cooked bit
                # ingredients in soup
                for ing in held.ingredients:
                    if ing == "onion":
                        obs[y, x, inv_ch + 2] = 1
                    elif ing == "tomato" and num_ingredients > 1:
                        obs[y, x, inv_ch + 3] = 1
            elif held.name == "onion":
                obs[y, x, inv_ch + 2] = 1
            elif held.name == "tomato" and num_ingredients > 1:
                obs[y, x, inv_ch + 3] = 1

    ch = 0
    # Self agent: channels 0 ~ (6 + num_ingredients)
    _encode_agent(self_player, ch)
    agent_ch_size = 5 + 2 + num_ingredients  # pos(1) + dir(4) + plate(1) + cooked(1) + ingredients
    ch += agent_ch_size

    # Other agent
    _encode_agent(other_player, ch)
    ch += agent_ch_size

    # === Static object layers (6 channels) ===
    static_encoding = [1, 4, 5, 6, 7, 9]  # WALL, GOAL, POT, RECIPE_IND, BUTTON_RECIPE, PLATE_PILE
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
    # Dynamic objects on the grid (not held by players, not in pots)
    for obj_pos, obj in state.objects.items():
        y, x = obj_pos[1], obj_pos[0]
        if obj.name == "dish":
            obs[y, x, ch] = 1
        elif obj.name == "onion":
            obs[y, x, ch + 2] = 1
        elif obj.name == "tomato" and num_ingredients > 1:
            obs[y, x, ch + 3] = 1
        elif obj.name == "soup":
            obs[y, x, ch] = 1      # plate
            obs[y, x, ch + 1] = 1  # cooked
            for ing in obj.ingredients:
                if ing == "onion":
                    obs[y, x, ch + 2] = 1
                elif ing == "tomato" and num_ingredients > 1:
                    obs[y, x, ch + 3] = 1
    ch += 2 + num_ingredients

    # === Recipe indicators (2 + num_ingredients channels) ===
    # JaxMARL에서는 recipe indicator가 있는 셀에 레시피 정보를 인코딩
    # overcooked-ai에서는 항상 onion soup이 기본 레시피 (all_orders)
    # 여기서는 간단히 모든 pot 위치에 레시피 = onion으로 인코딩
    ch += 2 + num_ingredients

    # === Extra layers ===
    # Pot timer layer
    for obj_pos, obj in state.objects.items():
        if hasattr(obj, "name") and obj.name == "soup":
            y, x = obj_pos[1], obj_pos[0]
            # pot에 있는 soup의 cooking tick
            if hasattr(obj, "cooking_tick"):
                obs[y, x, ch] = min(obj.cooking_tick, 255)
            elif hasattr(obj, "_cooking_tick"):
                obs[y, x, ch] = min(obj._cooking_tick, 255)

    return obs


def get_obs_shape(layout_name: str = None, mdp: OvercookedGridworld = None, num_ingredients: int = 1):
    """레이아웃별 obs shape 반환."""
    if mdp is None:
        mdp = OvercookedGridworld.from_layout_name(layout_name)
    h = len(mdp.terrain_mtx)
    w = len(mdp.terrain_mtx[0])
    c = 18 + 4 * (num_ingredients + 2)
    return (h, w, c)
