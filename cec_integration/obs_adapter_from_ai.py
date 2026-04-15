"""overcooked-ai state → V1 JaxMARL State → V1 env.get_obs → CEC (9, 9, 26) obs.

CEC 는 V1 JaxMARL Overcooked 엔진에서 학습됐으므로, overcooked-ai state 를 V1 State 로
바로 재구성하고 V1 env.get_obs 를 그대로 호출하면 CEC 훈련 분포와 byte-exact obs 가 나온다.
OV2 포맷을 경유하는 `obs_adapter.py` → `obs_adapter_v2.py` 체인의 bitpacked↔enum, POT lifecycle
변환 손실을 우회한다.

사용처:
  webapp/app/agent/inference.py::ModelManager._get_action_cec → 이 adapter 로 교체

Run (unit test):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp \
        ./overcooked_v2/bin/python cec_integration/scripts/test_ai_to_v1_adapter.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    DIR_TO_VEC,
    make_overcooked_map,
)
from jaxmarl.environments.overcooked.overcooked import (
    State as V1State,
    Overcooked as V1Overcooked,
    POT_EMPTY_STATUS,  # 23 = empty
    POT_FULL_STATUS,   # 20 = 3 onions, cooking starts
    POT_READY_STATUS,  # 0 = ready
    MAX_ONIONS_IN_POT, # 3
)

from .cec_layouts import CEC_LAYOUTS


# V1 OBJECT_TO_INDEX 값 (상수 별칭)
V1_EMPTY = OBJECT_TO_INDEX["empty"]       # 1
V1_WALL = OBJECT_TO_INDEX["wall"]         # 2
V1_ONION = OBJECT_TO_INDEX["onion"]       # 3
V1_ONION_PILE = OBJECT_TO_INDEX["onion_pile"]  # 4
V1_PLATE = OBJECT_TO_INDEX["plate"]       # 5
V1_PLATE_PILE = OBJECT_TO_INDEX["plate_pile"]  # 6
V1_GOAL = OBJECT_TO_INDEX["goal"]         # 7
V1_POT = OBJECT_TO_INDEX["pot"]           # 8
V1_DISH = OBJECT_TO_INDEX["dish"]         # 9

# overcooked-ai Direction tuple → V1 agent_dir_idx (V1 Actions: right=0, down=1, left=2, up=3)
#   overcooked-ai Direction: EAST=(1,0), SOUTH=(0,1), WEST=(-1,0), NORTH=(0,-1)
_ORIENTATION_TO_V1_DIR = {
    (1, 0): 0,   # EAST  → right
    (0, 1): 1,   # SOUTH → down
    (-1, 0): 2,  # WEST  → left
    (0, -1): 3,  # NORTH → up
}

# V1 convention 에서 cooking timer. V1 의 pot_status 는 요리 중이면 19→1 (1 step 지나면 19, 완성 시 0)
POT_COOK_TIME = 20  # overcooked-ai 기본값 (soup.cook_time)


def _held_to_v1_inv(held) -> int:
    """overcooked-ai held_object → V1 agent_inv enum 값.

    overcooked-ai:
      - None            → 빈 손
      - Onion           → name="onion"
      - Tomato          → name="tomato" (CEC 는 onion 전용 레이아웃만 씀)
      - Dish            → name="dish"  (빈 접시 = V1 의 plate)
      - SoupState (배달 가능한 완성 수프 = V1 의 dish)
          · held 된 soup 은 항상 is_ready=True 여야 함 (pot 밖으로 나온 수프는 plate 에 담긴 상태)
    """
    if held is None:
        return V1_EMPTY
    name = getattr(held, "name", None)
    if name == "onion":
        return V1_ONION
    if name == "dish":
        # overcooked-ai dish = 빈 접시 → V1 의 plate (index 5)
        return V1_PLATE
    if name == "soup":
        # held 된 soup 은 완성된 것 → V1 의 dish (index 9)
        return V1_DISH
    # 나머지 (tomato 등) 는 CEC 레이아웃에 없음
    return V1_EMPTY


def _soup_to_v1_pot_status(soup) -> int:
    """overcooked-ai SoupState → V1 pot_status 정수.

    V1 pot_status 인코딩:
      23 = 빈 pot
      22 = 1 onion
      21 = 2 onions
      20 = 3 onions (꽉 참, 요리 아직 시작 안 함) — 다음 step 에 cooking 시작
      19..1 = 요리 진행 중 (19→0 카운트다운)
      0 = 요리 완료 (수프 ready)

    overcooked-ai SoupState:
      - ingredients (list), cooking_tick (int), is_cooking (bool), is_ready (bool)
      - cooking_tick: -1 = idle, 0+ = cooking 진행
    """
    if soup is None:
        return POT_EMPTY_STATUS  # 23
    if soup.is_ready:
        return POT_READY_STATUS  # 0
    if soup.is_cooking:
        # cooking_tick 0 (방금 시작) → V1 pot_status 19. cooking_tick=19 → pot_status 0.
        # V1 은 cooking 시작 후 다음 step 에 19 로 decrement. 즉 pot_status = POT_FULL_STATUS - 1 - cooking_tick.
        # overcooked-ai cooking_tick 범위: 0 ~ cook_time(20). 실제 요리 끝은 cook_time = POT_COOK_TIME 이면
        # is_ready 가 True 가 됨. 그러므로 cooking_tick 은 0..19 사이.
        tick = int(getattr(soup, "_cooking_tick", -1))
        if tick < 0:
            # cook 안 시작한 전이 단계 — is_cooking 은 True 지만 tick = -1 또는 0 직전
            return POT_FULL_STATUS - 1  # 19
        # V1: cooking_tick=0 이 시작 → pot_status=19. tick 1 → status 18. ...
        status = POT_FULL_STATUS - 1 - tick
        return max(POT_READY_STATUS, min(POT_FULL_STATUS - 1, status))
    # 요리 시작 안 함, 재료만 있음 — 23-n (n=0,1,2,3)
    n = len(soup.ingredients)
    if n >= MAX_ONIONS_IN_POT:
        # 3개 다 차있는 상태 — 이론상 overcooked-ai 는 webapp 패치로 자동 요리 시작이지만
        # obs 생성 시점엔 방금 3번째 넣고 아직 step 안 된 상태일 수 있음
        return POT_FULL_STATUS  # 20
    return POT_EMPTY_STATUS - n  # 23 - n


def _categorize_loose_item(obj) -> Optional[int]:
    """카운터 위 loose 아이템 → V1 OBJECT_TO_INDEX (onion/plate/dish) 또는 None."""
    name = getattr(obj, "name", None)
    if name == "onion":
        return V1_ONION
    if name == "dish":
        # 빈 접시 — V1 plate
        return V1_PLATE
    if name == "soup":
        # 카운터 위에 놓인 완성 수프 = V1 dish
        return V1_DISH
    return None


@dataclass
class OvercookedAIToCECAdapter:
    """overcooked-ai 의 OvercookedState 를 V1 State 로 재구성해서 V1 get_obs 로 CEC 26ch obs 생성.

    target_layout 은 webapp 쓰는 이름 (예: "cramped_room") — 내부에서 `_9` suffix 붙여
    CEC_LAYOUTS 의 9×9 레이아웃을 로드한다.
    """

    target_layout: str
    max_steps: int = 400

    def __post_init__(self):
        cec_name = f"{self.target_layout}_9"
        if cec_name not in CEC_LAYOUTS:
            raise KeyError(f"CEC_LAYOUTS 에 {cec_name} 없음. 지원: {list(CEC_LAYOUTS)}")
        layout_dict = CEC_LAYOUTS[cec_name]
        self._v1_env = V1Overcooked(layout=layout_dict, random_reset=False,
                                    max_steps=self.max_steps)
        self._height = self._v1_env.height  # 9
        self._width = self._v1_env.width    # 9
        self._num_agents = self._v1_env.num_agents  # 2

        # 정적 레이아웃 필드 (CEC 9×9 그대로) — V1 make_overcooked_map 에 넣을 좌표들
        self._wall_map = self._build_wall_map(layout_dict)
        self._goal_pos = self._flat_to_xy(layout_dict["goal_idx"])
        self._plate_pile_pos = self._flat_to_xy(layout_dict["plate_pile_idx"])
        self._onion_pile_pos = self._flat_to_xy(layout_dict["onion_pile_idx"])
        self._pot_pos = self._flat_to_xy(layout_dict["pot_idx"])
        # pot_pos in (x, y) tuples for loose-item lookup
        self._pot_pos_set = {(int(xy[0]), int(xy[1])) for xy in np.asarray(self._pot_pos)}

    def _flat_to_xy(self, flat_idx: jnp.ndarray) -> jnp.ndarray:
        """9x9 flat index → (n, 2) (x, y) array."""
        arr = np.asarray(flat_idx)
        x = arr % self._width
        y = arr // self._width
        return jnp.stack([x, y], axis=-1).astype(jnp.uint32)

    def _build_wall_map(self, layout_dict) -> jnp.ndarray:
        """9x9 wall_map (bool) — True at wall cells."""
        flat = np.asarray(layout_dict["wall_idx"])
        wall = np.zeros((self._height, self._width), dtype=bool)
        for idx in flat:
            y = int(idx) // self._width
            x = int(idx) % self._width
            wall[y, x] = True
        return jnp.array(wall, dtype=jnp.bool_)

    def build_v1_state(self, ai_state, mdp, current_step: int = 0) -> V1State:
        """overcooked-ai state → V1 State."""
        # 1) Agents
        players = ai_state.players
        agent_pos_list = []
        agent_dir_idx_list = []
        agent_inv_list = []
        for p in players[: self._num_agents]:
            # overcooked-ai position (x, y). V1 agent_pos 도 (x, y).
            agent_pos_list.append([int(p.position[0]), int(p.position[1])])
            orientation = tuple(p.orientation)
            dir_idx = _ORIENTATION_TO_V1_DIR.get(orientation, 0)
            agent_dir_idx_list.append(dir_idx)
            agent_inv_list.append(_held_to_v1_inv(p.held_object))
        agent_pos = jnp.array(agent_pos_list, dtype=jnp.uint32)
        agent_dir_idx = jnp.array(agent_dir_idx_list, dtype=jnp.int32)
        agent_dir = DIR_TO_VEC[agent_dir_idx]
        agent_inv = jnp.array(agent_inv_list, dtype=jnp.uint32)

        # 2) Pot status — CEC 9×9 의 pot_pos 순서대로 값 채우기
        #    CEC 의 cramped_room_9 는 2 개 pot 가 있고, 그 중 하나만 ai 에서 실제로 존재.
        #    ai 에 없는 extra pot 은 23 (빈 상태) 로 둠.
        pot_status_list = []
        ai_pots = {tuple(pos): obj for pos, obj in ai_state.objects.items()
                   if getattr(obj, "name", None) == "soup"}
        pot_pos_np = np.asarray(self._pot_pos)
        for i in range(len(pot_pos_np)):
            x, y = int(pot_pos_np[i, 0]), int(pot_pos_np[i, 1])
            soup = ai_pots.get((x, y))
            pot_status_list.append(_soup_to_v1_pot_status(soup))
        pot_status = jnp.array(pot_status_list, dtype=jnp.uint32)

        # 3) Loose items on counters (non-pot 위치의 onion/plate/dish)
        #    ai_state.objects 에서 pot 이 아닌 위치의 오브젝트만 추출
        onion_pos_list = []
        plate_pos_list = []
        dish_pos_list = []
        for pos, obj in ai_state.objects.items():
            pos_xy = (int(pos[0]), int(pos[1]))
            if pos_xy in self._pot_pos_set:
                continue  # pot 위치는 pot_status 로 처리
            cat = _categorize_loose_item(obj)
            if cat == V1_ONION:
                onion_pos_list.append([pos_xy[0], pos_xy[1]])
            elif cat == V1_PLATE:
                plate_pos_list.append([pos_xy[0], pos_xy[1]])
            elif cat == V1_DISH:
                dish_pos_list.append([pos_xy[0], pos_xy[1]])

        onion_pos = jnp.array(onion_pos_list, dtype=jnp.uint32) if onion_pos_list else jnp.zeros((0, 2), dtype=jnp.uint32)
        plate_pos = jnp.array(plate_pos_list, dtype=jnp.uint32) if plate_pos_list else jnp.zeros((0, 2), dtype=jnp.uint32)
        dish_pos = jnp.array(dish_pos_list, dtype=jnp.uint32) if dish_pos_list else jnp.zeros((0, 2), dtype=jnp.uint32)

        # 4) maze_map 재구성 (V1 의 make_overcooked_map 사용)
        maze_map = make_overcooked_map(
            self._wall_map,
            self._goal_pos,
            agent_pos,
            agent_dir_idx,
            self._plate_pile_pos,
            self._onion_pile_pos,
            self._pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self._num_agents,
            agent_view_size=self._v1_env.agent_view_size,
        )

        return V1State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=self._goal_pos,
            pot_pos=self._pot_pos,
            wall_map=self._wall_map,
            maze_map=maze_map,
            time=int(current_step),
            terminal=False,
        )

    def get_cec_obs(self, ai_state, mdp, agent_idx: int = 0,
                    current_step: int = 0) -> jnp.ndarray:
        """overcooked-ai state → CEC (9, 9, 26) obs for given agent.

        Args:
            ai_state: overcooked-ai OvercookedState
            mdp: overcooked-ai OvercookedGridworld (terrain 정보 — 현재는 layout 에서 가져오므로
                 호환용으로만 받음)
            agent_idx: 0 or 1 (obs 관점)
            current_step: urgency layer 계산용

        Returns:
            CEC 26ch obs, shape (9, 9, 26), dtype uint8 (V1 env.get_obs 출력 그대로)
        """
        v1_state = self.build_v1_state(ai_state, mdp, current_step=current_step)
        obs_dict = self._v1_env.get_obs(v1_state)
        return obs_dict[f"agent_{agent_idx}"]

    def get_cec_obs_both(self, ai_state, mdp, current_step: int = 0):
        """양 agent 의 CEC obs 를 한 번에 (V1 state 재구성 1 회로 efficient)."""
        v1_state = self.build_v1_state(ai_state, mdp, current_step=current_step)
        return self._v1_env.get_obs(v1_state)
