"""V1 state → OV2 obs adapter (역변환).

용도: CEC 평가 경로에서 V1 Overcooked engine 을 primary 로 쓰고, 파트너 정책(BC 등)
     이나 사용자 UI 에게는 OV2-style obs/render 를 공급해야 할 때.

철학: `obs_adapter_v2_state_direct.py` 는 OV2 State → V1 State → V1 get_obs 경로.
     본 모듈은 정확히 그 역 — V1 State → synthetic OV2 State → OvercookedV2.get_obs.
     OV2 env 의 get_obs 는 grid(static/dyn/extra) 와 agents 만 쓰므로 BC 가 학습 때
     본 obs 분포와 byte-exact 일치.

extras 처리: CEC `_9x9` 는 OV2 canonical 에 없는 pot/plate/goal 을 여분으로 가짐.
  - 모두 unreachable cell 에 있으므로 V1 engine 에서도 실제 interact 대상이 아님.
  - V1 state 에서 그 cell 의 pot_status 를 OV2 에 옮기지 않고 그냥 wall 로 둠.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
from jaxmarl.environments.overcooked.overcooked import (
    POT_FULL_STATUS, POT_READY_STATUS, POT_EMPTY_STATUS,
)
from jaxmarl.environments.overcooked_v2.overcooked import (
    OvercookedV2, State as OV2State,
)
from jaxmarl.environments.overcooked_v2.common import (
    Agent, Position, DynamicObject, StaticObject,
)


# V1 OBJECT_TO_INDEX 상수 (cec-zero-shot/ph2 JaxMARL 동일)
V1_EMPTY = OBJECT_TO_INDEX["empty"]        # 1
V1_WALL = OBJECT_TO_INDEX["wall"]          # 2
V1_ONION = OBJECT_TO_INDEX["onion"]        # 3
V1_ONION_PILE = OBJECT_TO_INDEX["onion_pile"]  # 4
V1_PLATE = OBJECT_TO_INDEX["plate"]        # 5
V1_PLATE_PILE = OBJECT_TO_INDEX["plate_pile"]  # 6
V1_GOAL = OBJECT_TO_INDEX["goal"]          # 7
V1_POT = OBJECT_TO_INDEX["pot"]            # 8
V1_DISH = OBJECT_TO_INDEX["dish"]          # 9

# V1 agent_dir_idx 의 물리적 의미 (DIR_TO_VEC 기준):
#   0 = NORTH/UP (0,-1), 1 = SOUTH/DOWN (0,1), 2 = EAST/RIGHT (1,0), 3 = WEST/LEFT (-1,0)
# V1 Actions enum 라벨 (right=0, down=1, left=2, up=3) 은 DIR_TO_VEC 순서와 뒤바뀌어
# 있지만, 실제로 state 에 저장되는 agent_dir_idx 의 물리 semantic 은 DIR_TO_VEC 기준.
# OV2 Direction enum (UP=0, DOWN=1, RIGHT=2, LEFT=3) 도 동일한 물리 semantic.
# 따라서 dir index 는 **identity** 매핑.
_V1_DIR_TO_OV2 = np.array([0, 1, 2, 3], dtype=np.int32)

# V1 Overcooked 의 기본 agent_view_size=5 → maze_map padding=4
_V1_MAZE_PAD = 4

# 레이아웃별 V1(9×9 CEC_LAYOUTS[_9]) coord → OV2 native coord 변환 오프셋
# OV2_y = V1_y + y_shift (OV2 grid 상단에 추가 wall row 가 있는 레이아웃 보정)
_LAYOUT_Y_SHIFT = {
    "asymm_advantages": 1,  # OV2 asymm_advantages 는 상단 wall row 1개 추가됨
}


def _v1_inv_to_ov2(v1_val: int) -> int:
    """V1 OBJECT_TO_INDEX 값 → OV2 bitpacked 인벤토리 값.

    OV2 DynamicObject bitpack: bit0=PLATE, bit1=COOKED, bits2+=ingredient count (2 bits per ingredient).
    """
    v = int(v1_val)
    if v == V1_ONION:
        return 1 << 2                               # 1 onion, no plate/cooked
    if v == V1_PLATE:
        return 0x1                                  # plate bit only
    if v == V1_DISH:
        return 0x1 | 0x2 | (3 << 2)                 # plate + cooked + 3 onions = 완성 수프
    return 0                                        # empty


def _v1_pot_status_to_ov2(v1_status: int):
    """V1 pot_status (23=empty .. 0=ready) → (ov2_dyn, ov2_extra) at pot cell.

    OV2 실제 snapshot 규약 (관측된 state):
      - 1~3 onion filling (cooking 아님): dyn=count<<2, extra=0, has_cooked=0
      - Cooking 중 (1..19 tick): dyn=3<<2(=12), extra=1..19, has_cooked=0
          · COOKED bit 는 cooking 완료 step 에만 set.
      - Ready: dyn=(3<<2)|0x2=14, extra=0, has_cooked=1
    """
    s = int(v1_status)
    ONION_3 = 3 << 2
    COOKED = 0x2
    if s >= POT_EMPTY_STATUS:                       # 23 = empty
        return (0, 0)
    if s > POT_FULL_STATUS:                         # 22, 21 (1, 2 onions filling)
        count = POT_EMPTY_STATUS - s                # 23-s = 1 or 2
        return (count << 2, 0)
    if s == POT_FULL_STATUS:                        # 20 = V1 "filled 3 onions, transition"
        # V1 의 transient 상태; OV2 end-of-step snapshot 에는 등장 안 함.
        # 가장 가까운 OV2 상태: 3 onions idle (dyn=12, extra=0, has_cooked=0).
        return (ONION_3, 0)
    if s > POT_READY_STATUS:                        # 1..19 cooking
        return (ONION_3, s)                         # dyn=12 (no cooked), extra=s
    # s == 0: ready
    return (ONION_3 | COOKED, 0)                    # dyn=14 (cooked), extra=0


@dataclass
class V1StateToOV2ObsAdapter:
    """V1 Overcooked State → OV2 (H, W, 30) obs dict.

    사용 예:
        adapter = V1StateToOV2ObsAdapter("cramped_room", max_steps=400)
        ov2_obs_dict = adapter.get_ov2_obs(v1_state, current_step=t)
        # ov2_obs_dict["agent_0"], ["agent_1"] — shape (H, W, 30)
    """

    target_layout: str
    max_steps: int = 400

    def __post_init__(self):
        self.ov2_env = OvercookedV2(
            layout=self.target_layout,
            max_steps=self.max_steps,
            random_reset=False,
            random_agent_positions=False,
        )
        _, init_state = self.ov2_env.reset(jax.random.PRNGKey(0))
        self._static_ch = np.asarray(init_state.grid[:, :, 0])     # (H, W)
        self._ov2_h = int(self._static_ch.shape[0])
        self._ov2_w = int(self._static_ch.shape[1])
        self._y_shift = int(_LAYOUT_Y_SHIFT.get(self.target_layout, 0))
        # 기본 recipe (onion 3개)
        self._recipe = DynamicObject.get_recipe_encoding(jnp.array([0, 0, 0]))

        # ── JIT 경로용 static 캐시 ────────────────────────────────
        # static grid (numpy → jnp) : is_pot mask 등을 runtime 에 jnp 로 씀
        self._static_ch_jnp = jnp.asarray(self._static_ch, dtype=jnp.int32)
        self._is_ov2_pot = (self._static_ch_jnp == jnp.int32(StaticObject.POT))

        # V1 inventory value → OV2 bitpack lookup (size 10 — V1 obj indices 1..9)
        inv_map = np.zeros(10, dtype=np.int32)
        inv_map[V1_ONION] = 1 << 2                      # 1 onion in hand
        inv_map[V1_PLATE] = 0x1                         # empty plate
        inv_map[V1_DISH] = 0x1 | 0x2 | (3 << 2)         # 완성 수프
        self._inv_map_jnp = jnp.asarray(inv_map, dtype=jnp.int32)

        # V1 agent_dir identity remap (constant)
        self._dir_remap_jnp = jnp.asarray(_V1_DIR_TO_OV2, dtype=jnp.int32)

    def _build_grid(self, v1_state) -> np.ndarray:
        """V1 maze_map 에서 pot_status + loose item 을 추출해 OV2 (H, W, 3) grid 구성."""
        v1_maze = np.asarray(v1_state.maze_map)
        pad = _V1_MAZE_PAD
        # inner cell (x, y) ↔ maze_map[y+pad, x+pad]
        H = self._ov2_h
        W = self._ov2_w
        dyn = np.zeros((H, W), dtype=np.int32)
        extra = np.zeros((H, W), dtype=np.int32)

        # (1) Pot state — V1 pot_pos 순회
        v1_pot_pos = np.asarray(v1_state.pot_pos)
        for p in v1_pot_pos:
            px, py = int(p[0]), int(p[1])
            if py + pad < 0 or py + pad >= v1_maze.shape[0]:
                continue
            if px + pad < 0 or px + pad >= v1_maze.shape[1]:
                continue
            status = int(v1_maze[py + pad, px + pad, 2])
            ox = px
            oy = py + self._y_shift
            if 0 <= ox < W and 0 <= oy < H and int(self._static_ch[oy, ox]) == StaticObject.POT:
                d, e = _v1_pot_status_to_ov2(status)
                dyn[oy, ox] = d
                extra[oy, ox] = e
            # else: V1 extras — OV2 에는 없음. skip.

        # (2) Loose items — V1 maze_map obj_idx 가 onion/plate/dish 인 cell
        # V1 inner grid 크기 (layout 기반). CEC_LAYOUTS[_9] 는 9x9. agent_view_size 로 pad 된 maze_map.
        v1_h = v1_maze.shape[0] - 2 * pad
        v1_w = v1_maze.shape[1] - 2 * pad
        for y in range(v1_h):
            for x in range(v1_w):
                obj = int(v1_maze[y + pad, x + pad, 0])
                if obj not in (V1_ONION, V1_PLATE, V1_DISH):
                    continue
                ox = x
                oy = y + self._y_shift
                if not (0 <= ox < W and 0 <= oy < H):
                    continue
                # OV2 static 이 pot/pile 이면 loose item 인코딩 충돌 가능 → skip.
                s = int(self._static_ch[oy, ox])
                if s in (StaticObject.POT,):
                    continue
                if obj == V1_ONION:
                    dyn[oy, ox] = 1 << 2
                elif obj == V1_PLATE:
                    dyn[oy, ox] = 0x1
                elif obj == V1_DISH:
                    dyn[oy, ox] = 0x1 | 0x2 | (3 << 2)

        grid = np.stack([self._static_ch.astype(np.int32), dyn, extra], axis=-1)
        return grid

    def _build_agents(self, v1_state) -> Agent:
        v1_pos = np.asarray(v1_state.agent_pos).astype(np.int32)  # (N, 2) [x, y]
        v1_dir = np.asarray(v1_state.agent_dir_idx).astype(np.int32)
        v1_inv = np.asarray(v1_state.agent_inv).astype(np.int32)

        xs = jnp.array(v1_pos[:, 0], dtype=jnp.int32)
        ys = jnp.array(v1_pos[:, 1] + self._y_shift, dtype=jnp.int32)
        dirs = jnp.array(_V1_DIR_TO_OV2[v1_dir], dtype=jnp.int32)
        inv = jnp.array([_v1_inv_to_ov2(v) for v in v1_inv], dtype=jnp.int32)

        return Agent(pos=Position(x=xs, y=ys), dir=dirs, inventory=inv)

    def build_ov2_state(self, v1_state, current_step: int = 0) -> OV2State:
        grid_np = self._build_grid(v1_state)
        grid = jnp.array(grid_np, dtype=jnp.int32)
        agents = self._build_agents(v1_state)
        return OV2State(
            agents=agents,
            grid=grid,
            time=jnp.array(int(current_step), dtype=jnp.int32),
            terminal=jnp.array(False),
            recipe=self._recipe,
            new_correct_delivery=jnp.array(False),
            ingredient_permutations=None,
        )

    def get_ov2_obs(self, v1_state, current_step: int = 0) -> Dict[str, jnp.ndarray]:
        """V1 state → OV2 obs dict ({"agent_0": (H,W,30), "agent_1": (H,W,30)})."""
        ov2_state = self.build_ov2_state(v1_state, current_step)
        return self.ov2_env.get_obs(ov2_state)

    # ═══════════════════════════════════════════════════════════════════
    #   JIT-able 경로 (numpy/Python for-loop 제거)
    #
    #   동일 V1 state 입력에 대해 기존 numpy 경로와 byte-exact 결과를 내야 함.
    #   verify 스크립트 : cec_integration/scripts/verify_v1_to_ov2_jit.py
    # ═══════════════════════════════════════════════════════════════════

    def _build_grid_jit(self, v1_state) -> jnp.ndarray:
        """`_build_grid` 의 JIT 버전. v1_state.maze_map 을 jnp 로 그대로 사용."""
        pad = _V1_MAZE_PAD
        maze = v1_state.maze_map                                   # (H_pad, W_pad, 3)
        # inner grid: v1 native (9, 9, 3) — 축 순서는 (y, x, ch)
        inner = jax.lax.dynamic_slice(
            maze, (pad, pad, 0),
            (maze.shape[0] - 2 * pad, maze.shape[1] - 2 * pad, maze.shape[2]),
        )
        obj_v1 = inner[:, :, 0]                                     # (v1_h, v1_w)
        status_v1 = inner[:, :, 2]

        H = self._ov2_h
        W = self._ov2_w
        y_shift = self._y_shift

        # (1) Loose items (onion / plate / dish) on V1 inner grid
        onion_mask = (obj_v1 == V1_ONION)
        plate_mask = (obj_v1 == V1_PLATE)
        dish_mask = (obj_v1 == V1_DISH)
        dyn_v1 = (
            onion_mask.astype(jnp.int32) * jnp.int32(1 << 2)
            + plate_mask.astype(jnp.int32) * jnp.int32(0x1)
            + dish_mask.astype(jnp.int32) * jnp.int32(0x1 | 0x2 | (3 << 2))
        )

        # V1 inner → OV2 grid 위치로 배치 (y_shift 만큼 아래로 밀림).
        # V1 inner shape = (v1_h, v1_w). y_shift > 0 이면 상단 y_shift 행은 0.
        # OV2 shape = (H, W). 보통 H = v1_h + y_shift, W = v1_w.
        # (실제 수치: cramped_room v1_h=v1_w=9, H=9, W=9, y_shift=0.
        #  asymm_advantages y_shift=1, OV2 H=6 but v1_h=9 → 하단 잘림. dynamic_update_slice 사용)
        dyn_loose_ov2 = jnp.zeros((H, W), dtype=jnp.int32)
        # dyn_v1 을 (y_shift, 0) 위치에 배치 후 OV2 grid 에 얹기.
        # 잘리는 경우 (H < v1_h + y_shift) — 그냥 슬라이스:
        v1_h_eff = min(dyn_v1.shape[0], H - y_shift) if H - y_shift > 0 else 0
        v1_w_eff = min(dyn_v1.shape[1], W)
        if v1_h_eff > 0 and v1_w_eff > 0:
            dyn_loose_ov2 = dyn_loose_ov2.at[
                y_shift:y_shift + v1_h_eff, 0:v1_w_eff
            ].set(dyn_v1[:v1_h_eff, :v1_w_eff])

        # POT cell 에 loose item 인코딩이 쓰이지 않도록 mask
        dyn_loose_ov2 = dyn_loose_ov2 * (~self._is_ov2_pot).astype(jnp.int32)

        # (2) Pots — V1 pot_pos 는 runtime value (shape 고정). scatter 로 처리.
        pot_pos = jnp.asarray(v1_state.pot_pos, dtype=jnp.int32)    # (N_pot, 2) [x, y]
        pot_xs_v1 = pot_pos[:, 0]
        pot_ys_v1 = pot_pos[:, 1]

        # V1 inner 가 9×9 이므로 원본도 0..8 범위. (음수 or 9+ 는 invalid — valid mask 적용)
        v1_h = maze.shape[0] - 2 * pad
        v1_w = maze.shape[1] - 2 * pad
        valid_v1 = (
            (pot_xs_v1 >= 0) & (pot_xs_v1 < v1_w)
            & (pot_ys_v1 >= 0) & (pot_ys_v1 < v1_h)
        )
        # clamp for safe indexing (invalid 은 결과에서 valid_v1 로 0 처리)
        safe_xs = jnp.clip(pot_xs_v1, 0, v1_w - 1)
        safe_ys = jnp.clip(pot_ys_v1, 0, v1_h - 1)
        pot_statuses = status_v1[safe_ys, safe_xs]                  # (N_pot,)

        dyn_pot_per, extra_pot_per = jax.vmap(_v1_pot_status_to_ov2_jit)(pot_statuses)

        # OV2 좌표
        pot_oxs = pot_xs_v1
        pot_oys = pot_ys_v1 + y_shift
        in_ov2 = (
            (pot_oxs >= 0) & (pot_oxs < W)
            & (pot_oys >= 0) & (pot_oys < H)
        )
        safe_oxs = jnp.clip(pot_oxs, 0, W - 1)
        safe_oys = jnp.clip(pot_oys, 0, H - 1)
        ov2_is_pot_here = self._is_ov2_pot[safe_oys, safe_oxs]
        pot_valid = valid_v1 & in_ov2 & ov2_is_pot_here
        pot_dyn_eff = dyn_pot_per * pot_valid.astype(jnp.int32)
        pot_extra_eff = extra_pot_per * pot_valid.astype(jnp.int32)

        dyn_pot_ov2 = jnp.zeros((H, W), dtype=jnp.int32)
        extra_pot_ov2 = jnp.zeros((H, W), dtype=jnp.int32)
        dyn_pot_ov2 = dyn_pot_ov2.at[safe_oys, safe_oxs].set(pot_dyn_eff)
        extra_pot_ov2 = extra_pot_ov2.at[safe_oys, safe_oxs].set(pot_extra_eff)

        # (3) 최종 dyn/extra:  pot 셀은 pot 값, 그 외는 loose 값
        is_pot_f = self._is_ov2_pot.astype(jnp.int32)
        dyn = jnp.where(is_pot_f.astype(bool), dyn_pot_ov2, dyn_loose_ov2)
        extra = jnp.where(is_pot_f.astype(bool), extra_pot_ov2, jnp.zeros_like(extra_pot_ov2))

        grid = jnp.stack([self._static_ch_jnp, dyn, extra], axis=-1)
        return grid

    def _build_agents_jit(self, v1_state) -> Agent:
        v1_pos = jnp.asarray(v1_state.agent_pos, dtype=jnp.int32)   # (N, 2) [x, y]
        v1_dir = jnp.asarray(v1_state.agent_dir_idx, dtype=jnp.int32)
        v1_inv = jnp.asarray(v1_state.agent_inv, dtype=jnp.int32)

        xs = v1_pos[:, 0]
        ys = v1_pos[:, 1] + jnp.int32(self._y_shift)
        dirs = self._dir_remap_jnp[v1_dir]
        inv = self._inv_map_jnp[v1_inv]

        return Agent(pos=Position(x=xs, y=ys), dir=dirs, inventory=inv)

    def build_ov2_state_jit(self, v1_state, current_step=0) -> OV2State:
        grid = self._build_grid_jit(v1_state)
        agents = self._build_agents_jit(v1_state)
        # current_step 은 jnp 로 받을 수도 있고 Python int 로 받을 수도 있음
        t = jnp.asarray(current_step, dtype=jnp.int32)
        return OV2State(
            agents=agents,
            grid=grid,
            time=t,
            terminal=jnp.array(False),
            recipe=self._recipe,
            new_correct_delivery=jnp.array(False),
            ingredient_permutations=None,
        )

    def get_ov2_obs_jit(self, v1_state, current_step=0) -> Dict[str, jnp.ndarray]:
        """JIT-able 버전. 동일 state 입력에 대해 `get_ov2_obs` 와 동일 obs 반환."""
        ov2_state = self.build_ov2_state_jit(v1_state, current_step)
        return self.ov2_env.get_obs(ov2_state)


# ───────────────────────────────────────────────────────────────────
#   JIT-able helper fns
# ───────────────────────────────────────────────────────────────────

def _v1_pot_status_to_ov2_jit(s: jnp.ndarray):
    """JIT 버전. s (jnp int32) 하나를 받아 (dyn, extra) 반환.
    기존 `_v1_pot_status_to_ov2` 와 동일한 매핑:
      s >= 23            → (0, 0)
      s == 22/21         → (count<<2, 0)
      s == 20            → (ONION_3, 0)
      1 <= s <= 19       → (ONION_3, s)
      s == 0             → (ONION_3 | COOKED, 0)
    """
    ONION_3 = jnp.int32(3 << 2)          # 12
    COOKED = jnp.int32(0x2)
    POT_EMPTY = jnp.int32(POT_EMPTY_STATUS)   # 23
    POT_FULL = jnp.int32(POT_FULL_STATUS)     # 20
    POT_READY = jnp.int32(POT_READY_STATUS)   # 0

    s = jnp.int32(s)

    # dyn
    dyn = jnp.where(
        s >= POT_EMPTY, jnp.int32(0),
        jnp.where(
            s == jnp.int32(22), jnp.int32(1 << 2),
            jnp.where(
                s == jnp.int32(21), jnp.int32(2 << 2),
                jnp.where(
                    s == POT_READY, ONION_3 | COOKED,
                    ONION_3,  # 20, 19..1 모두 dyn = ONION_3
                )
            )
        )
    )
    # extra = s if 1..19 (cooking), else 0
    cooking = (s > POT_READY) & (s < POT_FULL)
    extra = jnp.where(cooking, s, jnp.int32(0))
    return dyn, extra
