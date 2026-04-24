"""webapp 에서 CEC 전용 V1 engine 세션 지원을 위한 helper 도구.

사용자 요구: "webapp 에서 CEC 로드할 때는 CEC 가 겪던 V1 engine 들고 와서 사용자한테는
똑같은 레이아웃과 모양이지만, CEC 한테는 걔가 학습하던 거 그대로 주기".

아키텍처:
  primary engine = V1 Overcooked (CEC_LAYOUTS[f"{layout}_9"])   ← ground truth
  CEC 입력:   V1 env.get_obs(state)                             ← native (9,9,26)
  BC 궤적:   v1_state_to_ov2_obs_adapter.get_ov2_obs(state)     ← OV2 호환 (H,W,30) uint8
  UI 렌더:   v1_state_to_ai_render_state(state)                 ← overcooked-ai OvercookedState

webapp engine.py 는 CEC 세션에서 이 모듈의 V1EngineSession 을 사용.
CEC 이외 모델 (PH2/SP/E3T/FCP/MEP) 은 기존 overcooked-ai 경로 유지.

Note: 이 모듈은 webapp 에 import 만 하면 되도록 설계됐고, cec_integration 외부 파일을
수정하지 않음. webapp 측 wire-up 은 별도 작업 (engine.py 에 `is_cec_model → use V1EngineSession` 분기 추가).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

from .cec_layouts import CEC_LAYOUTS
from .obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter

# UI / BC pos (OV2 convention: agent_0 / agent_1) → V1 Overcooked engine slot 매핑.
# V1 과 OV2 의 agent 초기 위치가 달라서 사용자 UI 에서 "agent_0" 으로 보이는 chef 가
# V1 engine 내부에서는 다른 slot 일 수 있음. BC 도 OV2 trajectory pos 로 학습됐으므로
# 같은 매핑 사용.
#
# V1 CEC_LAYOUTS[_9] agent_idx (x, y) vs OV2 native agent_positions:
#   cramped_room:    V1=[(1,1),(3,1)]  OV2=[(3,1),(1,1)]  → 완전 swap
#   coord_ring:      V1=[(2,1),(2,3)]  OV2=[(2,1),(1,2)]  → slot 0 일치, slot 1 불일치
#   forced_coord:    V1=[(1,2),(3,2)]  OV2=[(3,1),(1,2)]  → OV2_pos_1 → V1 slot 0
#   counter_circuit: V1=[(1,2),(6,2)]  OV2=[(6,3),(2,1)]  → 완전 불일치 (어쩔 수 없이 pos 그대로)
BC_POS_TO_V1_SLOT = {
    "cramped_room":    {0: 1, 1: 0},   # swap: OV2 agent_0 → V1 slot 1 at (3,1)
    "coord_ring":      {0: 0, 1: 1},   # slot 0 일치, slot 1 best-effort
    "forced_coord":    {0: 1, 1: 0},   # OV2 agent_1 → V1 slot 0 at (1,2) 일치
    "counter_circuit": {0: 0, 1: 1},   # 어느 매핑도 완전 매칭 안 됨 → identity
    "asymm_advantages":{0: 0, 1: 1},
}

# webapp UI 에서 사용하는 이름. 동일 테이블 — UI agent_i (OV2 convention) → V1 slot.
UI_PLAYER_TO_V1_SLOT = BC_POS_TO_V1_SLOT


# V1 Actions enum 은 {right=0, down=1, left=2, up=3} 으로 라벨되어 있으나
# V1 DIR_TO_VEC 은 [NORTH, SOUTH, EAST, WEST] 순서라 action index → 실제 이동방향이
# 라벨과 뒤바뀌어 있음 (V1 action 0 = NORTH/UP, action 2 = EAST/RIGHT 등).
# OV2 는 action index 와 direction 이 정상 매칭 (action 0 = RIGHT 등).
#
# BC (OV2 학습) 가 V1 engine 에서 돌아야 할 때 같은 physical direction 이 되도록
# action index 를 remap:
#   BC wants OV2 action a → apply V1 action ACTION_REMAP_OV2_TO_V1[a]
#
# 반대 방향 (V1 학습 정책을 OV2 engine 에 보낼 때) 는 동일한 테이블 사용 (involution).
ACTION_REMAP_OV2_TO_V1 = np.array([2, 1, 3, 0, 4, 5], dtype=np.int32)
ACTION_REMAP_V1_TO_OV2 = np.array([2, 1, 3, 0, 4, 5], dtype=np.int32)  # 실제 동일 테이블


V1_EMPTY = OBJECT_TO_INDEX["empty"]
V1_WALL = OBJECT_TO_INDEX["wall"]
V1_ONION = OBJECT_TO_INDEX["onion"]
V1_ONION_PILE = OBJECT_TO_INDEX["onion_pile"]
V1_PLATE = OBJECT_TO_INDEX["plate"]
V1_PLATE_PILE = OBJECT_TO_INDEX["plate_pile"]
V1_GOAL = OBJECT_TO_INDEX["goal"]
V1_POT = OBJECT_TO_INDEX["pot"]
V1_DISH = OBJECT_TO_INDEX["dish"]

# V1 maze_map padding (agent_view_size=5 → padding=4)
_V1_MAZE_PAD = 4

# V1 dir_idx → overcooked-ai orientation vector (물리 semantic 기준)
# V1 DIR_TO_VEC: 0=NORTH/(0,-1), 1=SOUTH/(0,1), 2=EAST/(1,0), 3=WEST/(-1,0)
# V1 action enum 라벨 (right=0 등) 과 DIR_TO_VEC 인덱스가 뒤바뀌어 있는데,
# state 에 저장되는 agent_dir_idx 는 DIR_TO_VEC 기준 물리 방향. 따라서 orient 도
# 같은 물리 벡터로 그려야 UI 에서 chef 가 가리키는 방향이 실제 이동 방향과 일치.
_V1_DIR_TO_ORIENT = {
    0: (0, -1),   # NORTH / UP
    1: (0, 1),    # SOUTH / DOWN
    2: (1, 0),    # EAST / RIGHT
    3: (-1, 0),   # WEST / LEFT
}


# ---------------------------------------------------------------------------
# 1. V1 state → overcooked-ai style state dict (for UI rendering)
# ---------------------------------------------------------------------------

def v1_state_to_ai_render_state(
    v1_state,
    layout: str,
    extras_as_wall: bool = True,
) -> Dict[str, Any]:
    """V1 Overcooked State → overcooked-ai 호환 dict (webapp frontend 렌더링용).

    extras_as_wall: True 면 CEC 학습 레이아웃의 여분 pot/plate/goal 을 UI 에서 wall 로
      변환 → OV2 canonical layout 의 비주얼 그대로 보여줌. False 면 extras 도 표시.

    반환 dict 의 shape (frontend 호환):
      {
        "players": [{"pos": (x, y), "orient": (dx, dy), "held": "onion"|"plate"|"dish"|None}, ...],
        "pots":    [{"pos": (x, y), "onions": int, "is_cooking": bool, "is_ready": bool, "cook_time_remaining": int}, ...],
        "objects": [{"pos": (x, y), "name": "onion"|"plate"|"dish"}, ...],
        "grid":    [[str, ...], ...]  # 각 셀 "W"/"P"/"B"/"O"/"G"/" "
      }
    """
    pad = _V1_MAZE_PAD
    maze = np.asarray(v1_state.maze_map)
    agent_pos = np.asarray(v1_state.agent_pos)
    agent_dir_idx = np.asarray(v1_state.agent_dir_idx)
    agent_inv = np.asarray(v1_state.agent_inv)

    h, w = 9, 9

    # grid 문자열 행렬 (UI 렌더용)
    grid = [[" " for _ in range(w)] for _ in range(h)]
    pots_info = []
    objects = []

    # 레이아웃 별 extras (wall 로 표시할 좌표 - OV2 에 없는 CEC extras)
    from .obs_adapter_v2 import LAYOUT_OV1_FIX
    extras_coords = set()
    fix = LAYOUT_OV1_FIX.get(layout, {})
    if extras_as_wall:
        for key in ("extra_pot", "extra_plate", "extra_goal"):
            for (y, x) in fix.get(key, []):
                extras_coords.add((x, y))

    for y in range(h):
        for x in range(w):
            obj = int(maze[y + pad, x + pad, 0])
            status = int(maze[y + pad, x + pad, 2])
            cell_xy = (x, y)

            if cell_xy in extras_coords:
                grid[y][x] = "W"
                continue

            if obj == V1_WALL:
                grid[y][x] = "W"
            elif obj == V1_POT:
                grid[y][x] = "P"
                onions = 0
                is_cooking = False
                is_ready = False
                cook_time_remaining = 0
                # V1 pot_status: 23=empty, 22..20=filling, 19..1=cooking, 0=ready
                if status >= 23:
                    onions = 0
                elif status >= 20:
                    onions = 23 - status        # 22→1, 21→2, 20→3 (about to cook)
                    is_cooking = status < 23    # filling = not cooking
                    is_cooking = False          # strictly, only 1..19 is cooking
                    if status == 20:
                        onions = 3
                elif status > 0:
                    onions = 3
                    is_cooking = True
                    cook_time_remaining = status
                else:  # status == 0
                    onions = 3
                    is_ready = True
                pots_info.append({
                    "pos": cell_xy,
                    "onions": onions,
                    "is_cooking": is_cooking,
                    "is_ready": is_ready,
                    "cook_time_remaining": cook_time_remaining,
                })
            elif obj == V1_PLATE_PILE:
                grid[y][x] = "B"
            elif obj == V1_ONION_PILE:
                grid[y][x] = "O"
            elif obj == V1_GOAL:
                grid[y][x] = "G"
            elif obj == V1_ONION:
                grid[y][x] = " "
                objects.append({"pos": cell_xy, "name": "onion"})
            elif obj == V1_PLATE:
                grid[y][x] = " "
                objects.append({"pos": cell_xy, "name": "plate"})
            elif obj == V1_DISH:
                grid[y][x] = " "
                objects.append({"pos": cell_xy, "name": "dish"})
            else:
                grid[y][x] = " "

    # Agents (maze_map 에 agent 도 표기되지만 agent_pos 로 직접 추출)
    players = []
    for i in range(agent_pos.shape[0]):
        px, py = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        orient = _V1_DIR_TO_ORIENT.get(int(agent_dir_idx[i]), (0, -1))
        inv_val = int(agent_inv[i])
        held = None
        if inv_val == V1_ONION:
            held = "onion"
        elif inv_val == V1_PLATE:
            held = "plate"
        elif inv_val == V1_DISH:
            held = "dish"
        players.append({"pos": (px, py), "orient": orient, "held": held})

    return {
        "players": players,
        "pots": pots_info,
        "objects": objects,
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# 2. V1 engine session wrapper
# ---------------------------------------------------------------------------

@dataclass
class V1EngineSession:
    """webapp CEC 세션용 V1 engine wrapper.

    세션 수명: 한 episode (reset → step 반복 → done).
    """
    layout: str
    max_steps: int = 400
    seed: int = 0

    def __post_init__(self):
        cec_name = f"{self.layout}_9"
        if cec_name not in CEC_LAYOUTS:
            raise KeyError(f"CEC_LAYOUTS 에 {cec_name} 없음")
        self.env = V1Overcooked(
            layout=CEC_LAYOUTS[cec_name], random_reset=False, max_steps=self.max_steps,
        )
        self.ov2_adapter = V1StateToOV2ObsAdapter(
            target_layout=self.layout, max_steps=self.max_steps,
        )
        self._key = jax.random.PRNGKey(self.seed)
        self.state = None
        self.done = False
        self.t = 0
        self.total_reward = 0.0

    def reset(self) -> Dict[str, Any]:
        self._key, k = jax.random.split(self._key)
        _, self.state = self.env.reset(k)
        self.t = 0
        self.done = False
        self.total_reward = 0.0
        return self.get_render_state()

    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict[str, Any], float, bool]:
        """action_dict = {"agent_0": int, "agent_1": int}
        Returns (render_state, reward, done).
        """
        self._key, k = jax.random.split(self._key)
        env_act = {a: jnp.int32(int(action_dict[a])) for a in self.env.agents}
        _, self.state, reward, done_dict, _ = self.env.step(k, self.state, env_act)
        r = float(reward["agent_0"])
        self.total_reward += r
        self.t += 1
        self.done = bool(done_dict["__all__"])
        return self.get_render_state(), r, self.done

    # -- obs producers --
    def get_cec_obs_v1(self, agent_idx: int = 0) -> jnp.ndarray:
        """CEC 입력용 V1 native obs (9, 9, 26)."""
        obs_dict = self.env.get_obs(self.state)
        return obs_dict[f"agent_{agent_idx}"]

    def get_human_obs_ov2(self, agent_idx: int = 0) -> np.ndarray:
        """BC 궤적 수집용 OV2 호환 obs (H, W, 30) uint8."""
        ov2_dict = self.ov2_adapter.get_ov2_obs(self.state, current_step=self.t)
        return np.asarray(ov2_dict[f"agent_{agent_idx}"]).astype(np.uint8)

    def get_render_state(self) -> Dict[str, Any]:
        """UI 렌더링용 overcooked-ai 호환 dict."""
        return v1_state_to_ai_render_state(self.state, self.layout, extras_as_wall=True)

    def build_terrain_mtx(self, extras_as_wall: bool = True):
        """V1 layout 기반 UI terrain_mtx (overcooked-ai 포맷: 2D list of chars).

        4 지원 layout 은 V1 core == OV2 native (extras 숨김 후), asymm_advantages 는
        CEC template 기반의 독자 terrain. CEC V1 engine 세션에서는 항상 V1 기반 terrain 을
        사용해야 agent 위치와 grid 가 모순되지 않음.

        chars: 'X'=wall/counter, 'O'=onion pile, 'D'=dish (plate) pile,
               'S'=serving counter (goal), 'P'=pot, ' '=empty.
        """
        from .obs_adapter_v2 import LAYOUT_OV1_FIX
        import numpy as np
        h, w = 9, 9
        grid = [['X'] * w for _ in range(h)]  # default wall

        d = self.env.layout if hasattr(self.env, "layout") else None
        # V1 Overcooked stores layout as FrozenDict
        layout_dict = self._v1_env.layout if hasattr(self, "_v1_env") else None
        # self.env 는 V1Overcooked 인스턴스 — layout 속성 접근
        try:
            layout_dict = self.env.layout
        except Exception:
            layout_dict = None
        if layout_dict is None:
            # fallback: CEC_LAYOUTS 에서 가져옴
            from .cec_layouts import CEC_LAYOUTS
            layout_dict = CEC_LAYOUTS[f"{self.layout}_9"]

        def flat_to_xy(flat_list):
            arr = np.asarray(flat_list)
            return [(int(f) % w, int(f) // w) for f in arr.flatten()]

        walls = set(flat_to_xy(layout_dict["wall_idx"]))
        pots = set(flat_to_xy(layout_dict["pot_idx"]))
        goals = set(flat_to_xy(layout_dict["goal_idx"]))
        plates = set(flat_to_xy(layout_dict["plate_pile_idx"]))
        onions = set(flat_to_xy(layout_dict["onion_pile_idx"]))

        # 먼저 wall/static object 로 채움 (wall_idx 는 모든 static object 포함)
        for (x, y) in walls:
            grid[y][x] = 'X'
        for (x, y) in pots:
            grid[y][x] = 'P'
        for (x, y) in goals:
            grid[y][x] = 'S'
        for (x, y) in plates:
            grid[y][x] = 'D'
        for (x, y) in onions:
            grid[y][x] = 'O'

        # 빈 셀은 empty ' '. wall_idx 에 없으면서 객체 좌표에도 없는 곳은 empty.
        # 추가로 agent 초기 위치는 empty 로 보정 (V1 layout agent_idx).
        agents = flat_to_xy(layout_dict["agent_idx"])
        for (x, y) in agents:
            if grid[y][x] not in ('P', 'S', 'D', 'O'):
                grid[y][x] = ' '
        # wall_idx 에 없고 static obj 도 아닌 core 셀 = empty
        for y in range(h):
            for x in range(w):
                if ((x, y) not in walls and (x, y) not in pots and (x, y) not in goals
                        and (x, y) not in plates and (x, y) not in onions):
                    grid[y][x] = ' '

        # extras 숨김 (CEC template 에 있지만 OV2 canonical 에 없는 obj 를 wall 로).
        if extras_as_wall:
            fix = LAYOUT_OV1_FIX.get(self.layout, {})
            for key in ("extra_pot", "extra_plate", "extra_goal"):
                for (y, x) in fix.get(key, []):
                    grid[y][x] = 'X'

        return grid

    def get_webapp_state_json(self) -> Dict[str, Any]:
        """webapp engine.py::_serialize_state 의 JSON 포맷과 동일한 dict 반환.

        구조:
          {
            "players": [{"position": [x,y], "orientation": [dx,dy], "held_object": {...}}, ...],
            "objects": {"x,y": {"name": ..., "position": [x,y], ...}, ...}
          }

        extras (CEC 학습 전용 추가 pot/plate/goal) 는 unreachable 이라 UI 렌더에서 숨김 →
        objects dict 에서 제외 (무슨 아이템이 있어도 extras 셀에서는 발생하지 않으므로 실제
        영향 없음).
        """
        pad = _V1_MAZE_PAD
        maze = np.asarray(self.state.maze_map)
        agent_pos = np.asarray(self.state.agent_pos)
        agent_dir_idx = np.asarray(self.state.agent_dir_idx)
        agent_inv = np.asarray(self.state.agent_inv)

        # extras 좌표 (obs_adapter_v2 의 LAYOUT_OV1_FIX 기준)
        from .obs_adapter_v2 import LAYOUT_OV1_FIX
        fix = LAYOUT_OV1_FIX.get(self.layout, {})
        extras_xy = set()
        for key in ("extra_pot", "extra_plate", "extra_goal"):
            for (y, x) in fix.get(key, []):
                extras_xy.add((x, y))

        # Players
        players = []
        _inv_name = {V1_ONION: "onion", V1_PLATE: "dish", V1_DISH: "soup"}
        for i in range(agent_pos.shape[0]):
            px, py = int(agent_pos[i, 0]), int(agent_pos[i, 1])
            orient = _V1_DIR_TO_ORIENT.get(int(agent_dir_idx[i]), (0, -1))
            inv_val = int(agent_inv[i])
            held = None
            if inv_val in _inv_name:
                name = _inv_name[inv_val]
                held = {"name": name, "position": [px, py]}
                if name == "soup":
                    # 완성된 수프 들고 있음: 3 onion, is_ready
                    held["ingredients"] = ["onion", "onion", "onion"]
                    held["cooking_tick"] = 20
                    held["is_cooking"] = False
                    held["is_ready"] = True
                    held["cook_time"] = 20
            players.append({
                "position": [px, py],
                "orientation": list(orient),
                "held_object": held,
            })

        # Objects: scan maze_map for pots with contents + loose items
        objects = {}
        v1_h = maze.shape[0] - 2 * pad
        v1_w = maze.shape[1] - 2 * pad
        for y in range(v1_h):
            for x in range(v1_w):
                if (x, y) in extras_xy:
                    continue
                obj_idx = int(maze[y + pad, x + pad, 0])
                status = int(maze[y + pad, x + pad, 2])

                if obj_idx == V1_POT and status < 23:
                    # Pot with contents (status 22..0)
                    if status >= 20:
                        # Filling (1..3 onions), not cooking
                        onion_count = 23 - status
                        is_cooking = False
                        is_ready = False
                        cooking_tick = -1
                    elif status > 0:
                        # Cooking 1..19 ticks remaining
                        onion_count = 3
                        is_cooking = True
                        is_ready = False
                        cooking_tick = 20 - status  # overcooked-ai convention
                    else:
                        # Ready
                        onion_count = 3
                        is_cooking = False
                        is_ready = True
                        cooking_tick = 20
                    obj_data = {
                        "name": "soup",
                        "position": [x, y],
                        "ingredients": ["onion"] * onion_count,
                        "cooking_tick": cooking_tick,
                        "is_cooking": is_cooking,
                        "is_ready": is_ready,
                        "cook_time": 20,
                    }
                    objects[f"{x},{y}"] = obj_data

                elif obj_idx == V1_ONION:
                    objects[f"{x},{y}"] = {
                        "name": "onion", "position": [x, y],
                    }
                elif obj_idx == V1_PLATE:
                    objects[f"{x},{y}"] = {
                        "name": "dish", "position": [x, y],
                    }
                elif obj_idx == V1_DISH:
                    objects[f"{x},{y}"] = {
                        "name": "soup",
                        "position": [x, y],
                        "ingredients": ["onion", "onion", "onion"],
                        "cooking_tick": 20,
                        "is_cooking": False,
                        "is_ready": True,
                        "cook_time": 20,
                    }

        # UI 관점으로 player 순서 재정렬 (OV2 convention 와 동일한 시작 위치/순서 유지).
        #   ui_map: UI slot → V1 slot
        # players 는 V1 slot 순서로 채워졌으므로, UI slot i 에 보여줄 player 는 players[ui_map[i]].
        ui_map = UI_PLAYER_TO_V1_SLOT.get(self.layout, {0: 0, 1: 1})
        ordered_players = [players[ui_map[0]], players[ui_map[1]]]

        return {
            "players": ordered_players,
            "objects": objects,
        }
