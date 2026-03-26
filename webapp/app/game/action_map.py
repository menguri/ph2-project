"""
Action 매핑: JaxMARL ↔ overcooked-ai ↔ 키보드 입력.
"""
from overcooked_ai_py.mdp.actions import Action, Direction

# JaxMARL action indices (overcooked_v2/common.py Actions enum)
# right=0, down=1, left=2, up=3, stay=4, interact=5
JAXMARL_ACTIONS = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
    4: "stay",
    5: "interact",
}

# overcooked-ai action tuples
OVERCOOKED_AI_ACTIONS = [
    Direction.EAST,       # 0: right  → (1, 0)
    Direction.SOUTH,      # 1: down   → (0, 1)
    Direction.WEST,       # 2: left   → (-1, 0)
    Direction.NORTH,      # 3: up     → (0, -1)
    Action.STAY,          # 4: stay   → (0, 0)
    Action.INTERACT,      # 5: interact
]

# overcooked-ai → JaxMARL index
OVERCOOKED_TO_JAXMARL = {
    Direction.EAST: 0,
    Direction.SOUTH: 1,
    Direction.WEST: 2,
    Direction.NORTH: 3,
    Action.STAY: 4,
    Action.INTERACT: 5,
}

# 키보드 키 → JaxMARL action index
KEYBOARD_TO_ACTION = {
    "ArrowRight": 0,
    "ArrowDown": 1,
    "ArrowLeft": 2,
    "ArrowUp": 3,
    "d": 0,
    "s": 1,
    "a": 2,
    "w": 3,
    " ": 5,       # space = interact
    "e": 5,       # e = interact
    "Enter": 5,   # enter = interact
}


def jaxmarl_to_overcooked(action_idx: int):
    """JaxMARL action index → overcooked-ai action tuple."""
    return OVERCOOKED_AI_ACTIONS[action_idx]


def overcooked_to_jaxmarl(action_tuple) -> int:
    """overcooked-ai action tuple → JaxMARL action index."""
    return OVERCOOKED_TO_JAXMARL.get(action_tuple, 4)  # default: stay


def keyboard_to_action(key: str) -> int:
    """키보드 키 이름 → JaxMARL action index."""
    return KEYBOARD_TO_ACTION.get(key, 4)  # default: stay
