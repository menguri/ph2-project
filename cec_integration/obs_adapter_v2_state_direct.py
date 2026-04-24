"""OV2 state (JaxMARL overcooked_v2 State) → V1 State → V1 env.get_obs → CEC 26ch.

human-proxy 에서 CEC 가 OV2 env 에 붙을 때 byte-exact V1 obs 를 받기 위해 사용.
`obs_adapter_from_ai.py` 와 유사한 철학 — V1 의 native get_obs 를 그대로 쓰면 CEC 훈련
분포와 byte-exact 일치. 단 입력이 overcooked-ai state 가 아니라 OV2 State 라는 차이.

기존 `cec_integration/obs_adapter.py::CECObsAdapter` 는 extras 주입이 없어 CEC `_9x9`
레이아웃의 2개 plate/goal/pot 중 1개가 누락된 obs 를 만듦. 이 신규 adapter 는
`obs_adapter_v2.py::LAYOUT_OV1_FIX` 와 동일한 extras 를 V1 state 구성 시 주입한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX, DIR_TO_VEC, make_overcooked_map,
)
from jaxmarl.environments.overcooked.overcooked import (
    State as V1State,
    Overcooked as V1Overcooked,
    POT_EMPTY_STATUS, POT_FULL_STATUS, POT_READY_STATUS, MAX_ONIONS_IN_POT,
)
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject

from .cec_layouts import CEC_LAYOUTS


# V1 상수
V1_EMPTY = OBJECT_TO_INDEX["empty"]
V1_WALL = OBJECT_TO_INDEX["wall"]
V1_ONION = OBJECT_TO_INDEX["onion"]
V1_PLATE = OBJECT_TO_INDEX["plate"]
V1_DISH = OBJECT_TO_INDEX["dish"]

# OV2 Direction → V1 agent_dir_idx
# 물리 semantic 은 두 engine 에서 동일 (V1 DIR_TO_VEC 인덱스 0..3 = NORTH/SOUTH/EAST/WEST,
# OV2 Direction 인덱스 0..3 = UP/DOWN/RIGHT/LEFT). identity 매핑이 맞음.
# V1 Actions enum 라벨이 뒤바뀌어 있지만 state 저장 semantic 은 DIR_TO_VEC 기준.
_V2_DIR_TO_V1_DIR = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

POT_COOK_TIME = 20  # OV2 기본값

# CEC `_9x9` 레이아웃에 있고 OV2 canonical 레이아웃에 없는 extras (obs_adapter_v2.py 와 동일 테이블)
_LAYOUT_EXTRAS = {
    "cramped_room":     {"extra_pot": [(0, 0)], "extra_plate": [(3, 0)], "extra_goal": [(3, 4)]},
    "coord_ring":       {"extra_plate": [(0, 0)], "extra_goal": [(4, 4)]},
    "forced_coord":     {"extra_plate": [(4, 0)], "extra_goal": [(4, 4)]},
    "counter_circuit":  {"extra_plate": [(0, 0)], "extra_goal": [(0, 7)]},
    "asymm_advantages": {},
}


def _v2_inv_to_v1(dyn_val: int) -> int:
    """OV2 bitpacked inventory → V1 OBJECT_TO_INDEX enum."""
    dyn_val = int(dyn_val)
    has_plate = (dyn_val & 0x1) != 0
    has_cooked = (dyn_val & 0x2) != 0
    onion_count = (dyn_val >> 2) & 0x3
    if has_plate and has_cooked and onion_count > 0:
        return V1_DISH
    if has_plate and not has_cooked and onion_count == 0:
        return V1_PLATE
    if not has_plate and not has_cooked and onion_count > 0:
        return V1_ONION
    return V1_EMPTY


def _v2_pot_to_v1_status(dyn_val: int, extra: int) -> int:
    """OV2 pot dyn/extra → V1 pot_status.

    V1 convention:
      23=empty, 22/21/20=filling (1/2/3 onions idle), 19..1=cooking, 0=ready.

    OV2 semantics (step_cooking_interaction=False / auto_cook 모드):
      - 1~3 onion filling, not cooking : dyn=count<<2, extra=0, has_cooked=0
      - Cooking in progress           : dyn=3<<2(=12), extra=1..19, has_cooked=0
          · 3번째 onion 드롭되는 step 에 auto_cook 이 extra=POT_COOK_TIME(20) 으로 세팅하고,
            같은 step 의 _cook 이 extra 를 19 로 감소시킴. 따라서 end-of-step snapshot
            에서 cooking pot 은 항상 extra ∈ [1, 19], has_cooked=0, dyn=12.
      - Ready                         : dyn=(3<<2)|0x2=14, extra=0, has_cooked=1
    """
    dyn_val = int(dyn_val)
    extra = int(extra)
    has_cooked = (dyn_val & 0x2) != 0
    onion_count = (dyn_val >> 2) & 0x3

    # Ready 상태: cooked bit set + timer 만료
    if has_cooked and extra == 0:
        return POT_READY_STATUS  # 0

    # Cooking 진행 중: extra > 0 (OV2 에선 cooked bit 없이 cooking; 안전하게 has_cooked 도 허용)
    if extra > 0:
        return min(extra, POT_FULL_STATUS - 1)  # 1..19

    # Idle (cooking 아님): onion 개수에 따라 filling 상태
    # onion_count=0 → 23 (empty), 1 → 22, 2 → 21, 3 → 20 (about to cook in V1)
    return POT_EMPTY_STATUS - onion_count


def _v2_dyn_loose_to_v1(dyn_val: int) -> int:
    """OV2 grid cell 에 놓인 loose item 의 OV2 dyn → V1 item index.

    plate pile / onion pile / pot 이 아닌 셀에서 실제 loose 아이템:
      - PLATE bit only → V1 PLATE
      - COOKED + PLATE + 3 onion → V1 DISH (완성 수프 놓음)
      - onion_count > 0 only → V1 ONION
    """
    return _v2_inv_to_v1(dyn_val)  # 동일한 비트 의미


@dataclass
class OV2StateToCECDirectAdapter:
    """OV2 state → V1 State → V1 env.get_obs → CEC obs (byte-exact V1)."""

    target_layout: str
    max_steps: int = 400

    def __post_init__(self):
        cec_name = f"{self.target_layout}_9"
        if cec_name not in CEC_LAYOUTS:
            raise KeyError(f"CEC_LAYOUTS 에 {cec_name} 없음")
        self._layout_dict = CEC_LAYOUTS[cec_name]
        self._v1_env = V1Overcooked(layout=self._layout_dict, random_reset=False,
                                    max_steps=self.max_steps)
        self._height = self._v1_env.height  # 9
        self._width = self._v1_env.width
        self._num_agents = self._v1_env.num_agents

        # 정적 레이아웃 필드 (고정) — CEC 9x9 기준 이미 extras 포함
        self._wall_map = self._build_wall_map()
        self._goal_pos = self._flat_to_xy(self._layout_dict["goal_idx"])
        self._plate_pile_pos = self._flat_to_xy(self._layout_dict["plate_pile_idx"])
        self._onion_pile_pos = self._flat_to_xy(self._layout_dict["onion_pile_idx"])
        self._pot_pos = self._flat_to_xy(self._layout_dict["pot_idx"])
        self._pot_pos_set = {(int(p[0]), int(p[1]))
                              for p in np.asarray(self._pot_pos)}

    def _flat_to_xy(self, flat_idx):
        arr = np.asarray(flat_idx)
        x = arr % self._width
        y = arr // self._width
        return jnp.stack([x, y], axis=-1).astype(jnp.uint32)

    def _build_wall_map(self):
        flat = np.asarray(self._layout_dict["wall_idx"])
        wall = np.zeros((self._height, self._width), dtype=bool)
        for idx in flat:
            wall[int(idx) // self._width, int(idx) % self._width] = True
        return jnp.array(wall, dtype=jnp.bool_)

    def build_v1_state(self, v2_state, current_step: int = 0) -> V1State:
        """OV2 State → V1 State."""
        # Agents (OV2 agents 구조체는 .pos, .dir, .inventory 포함)
        num_agents = self._num_agents
        agents = v2_state.agents
        # OV2 pos 는 struct 형태: pos.x, pos.y
        agent_pos_list = []
        agent_dir_idx_list = []
        agent_inv_list = []
        for i in range(num_agents):
            px = int(agents.pos.x[i])
            py = int(agents.pos.y[i])
            agent_pos_list.append([px, py])
            v2_dir = int(agents.dir[i])
            agent_dir_idx_list.append(int(_V2_DIR_TO_V1_DIR[v2_dir]))
            agent_inv_list.append(_v2_inv_to_v1(int(agents.inventory[i])))
        agent_pos = jnp.array(agent_pos_list, dtype=jnp.uint32)
        agent_dir_idx = jnp.array(agent_dir_idx_list, dtype=jnp.int32)
        agent_dir = DIR_TO_VEC[agent_dir_idx]
        agent_inv = jnp.array(agent_inv_list, dtype=jnp.uint32)

        # OV2 grid: (ov2_h, ov2_w, 3) [static, dyn, extra]
        grid = np.asarray(v2_state.grid)
        ov2_h, ov2_w = grid.shape[0], grid.shape[1]
        static_ch = grid[:, :, 0]
        dyn_ch = grid[:, :, 1]
        extra_ch = grid[:, :, 2]

        # Pot status — CEC pot_pos 순서대로 채움. OV2 에 없는 extra pot 은 EMPTY(23).
        pot_status_list = []
        pot_pos_np = np.asarray(self._pot_pos)
        for i in range(len(pot_pos_np)):
            x, y = int(pot_pos_np[i, 0]), int(pot_pos_np[i, 1])
            if y < ov2_h and x < ov2_w and static_ch[y, x] == StaticObject.POT:
                pot_status_list.append(_v2_pot_to_v1_status(dyn_ch[y, x], extra_ch[y, x]))
            else:
                pot_status_list.append(POT_EMPTY_STATUS)
        pot_status = jnp.array(pot_status_list, dtype=jnp.uint32)

        # Loose items on OV2 grid (non-pot, non-pile, non-plate_pile cells with dyn values)
        onion_pos, plate_pos, dish_pos = [], [], []
        for y in range(ov2_h):
            for x in range(ov2_w):
                s = int(static_ch[y, x])
                if s == StaticObject.POT:
                    continue
                # ingredient pile 셀에도 loose item 은 의미 없음 (pile 자체는 static)
                d = int(dyn_ch[y, x])
                if d == 0:
                    continue
                v1_item = _v2_dyn_loose_to_v1(d)
                if v1_item == V1_ONION:
                    onion_pos.append([x, y])
                elif v1_item == V1_PLATE:
                    plate_pos.append([x, y])
                elif v1_item == V1_DISH:
                    dish_pos.append([x, y])

        onion_pos_arr = jnp.array(onion_pos, dtype=jnp.uint32) if onion_pos else jnp.zeros((0, 2), dtype=jnp.uint32)
        plate_pos_arr = jnp.array(plate_pos, dtype=jnp.uint32) if plate_pos else jnp.zeros((0, 2), dtype=jnp.uint32)
        dish_pos_arr = jnp.array(dish_pos, dtype=jnp.uint32) if dish_pos else jnp.zeros((0, 2), dtype=jnp.uint32)

        # make_overcooked_map: CEC 9x9 레이아웃의 static 좌표는 이미 extras 포함되어 있음
        maze_map = make_overcooked_map(
            self._wall_map, self._goal_pos, agent_pos, agent_dir_idx,
            self._plate_pile_pos, self._onion_pile_pos, self._pot_pos, pot_status,
            onion_pos_arr, plate_pos_arr, dish_pos_arr,
            pad_obs=True, num_agents=num_agents,
            agent_view_size=self._v1_env.agent_view_size,
        )

        return V1State(
            agent_pos=agent_pos, agent_dir=agent_dir, agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv, goal_pos=self._goal_pos, pot_pos=self._pot_pos,
            wall_map=self._wall_map, maze_map=maze_map,
            time=int(current_step), terminal=False,
        )

    def get_cec_obs(self, v2_state, agent_idx: int = 0, current_step: int = 0) -> jnp.ndarray:
        v1_state = self.build_v1_state(v2_state, current_step=current_step)
        obs_dict = self._v1_env.get_obs(v1_state)
        return obs_dict[f"agent_{agent_idx}"]

    def get_cec_obs_both(self, v2_state, current_step: int = 0):
        v1_state = self.build_v1_state(v2_state, current_step=current_step)
        return self._v1_env.get_obs(v1_state)
