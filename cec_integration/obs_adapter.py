"""ph2 overcooked_v2 state → CEC v1 (9,9,26) observation adapter.

CEC was trained against the legacy `jaxmarl.environments.overcooked` env
(v1) which produces a 26-channel grid observation. ph2-project drives
its rollouts on `overcooked_v2`, whose State / grid encoding is entirely
different. This module bridges the two by reconstructing a v1 `State`
from a v2 `State`, then calling the v1 env's `get_obs` to produce the
exact tensor shape CEC expects.

Strategy: keep one *prebuilt* v1 `Overcooked` env around (used purely as
a get_obs function), and on every step rebuild only the fields that
get_obs reads:

    - maze_map[..., 0]  : v1 OBJECT_TO_INDEX static cells (padded)
    - maze_map[..., 2]  : v1 pot status (23..0) at pot cells (padded)
    - agent_pos         : (n_agents, 2)  [x, y] in the padded-grid coords
    - agent_dir_idx     : (n_agents,)    v1 action index 0..3
    - agent_inv         : (n_agents,)    v1 OBJECT_TO_INDEX
    - time              : scalar (urgency layer)

The v2 grid is always placed at the top-left of a 9x9 frame, with the
remaining cells set to WALL — this matches CEC's `make_9x9_layout` which
embeds the smaller training grids the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
from jaxmarl.environments.overcooked.overcooked import (
    State as V1State,
    Overcooked as V1Overcooked,
)
from jaxmarl.environments.overcooked_v2.common import StaticObject

from .cec_layouts import CEC_LAYOUTS

# ---------------------------------------------------------------------------
# Constants

V1_EMPTY = OBJECT_TO_INDEX["empty"]      # 1
V1_WALL = OBJECT_TO_INDEX["wall"]        # 2
V1_ONION = OBJECT_TO_INDEX["onion"]      # 3
V1_ONION_PILE = OBJECT_TO_INDEX["onion_pile"]  # 4
V1_PLATE = OBJECT_TO_INDEX["plate"]      # 5
V1_PLATE_PILE = OBJECT_TO_INDEX["plate_pile"]  # 6
V1_GOAL = OBJECT_TO_INDEX["goal"]        # 7
V1_POT = OBJECT_TO_INDEX["pot"]          # 8
V1_DISH = OBJECT_TO_INDEX["dish"]        # 9
V1_AGENT = OBJECT_TO_INDEX["agent"]      # 10

# v1: action index for direction (right=0, down=1, left=2, up=3)
# v2: Direction.UP=0, DOWN=1, RIGHT=2, LEFT=3
# So v2 dir → v1 dir_idx is [3, 1, 0, 2]
_V2_DIR_TO_V1_DIR_IDX = jnp.array([3, 1, 0, 2], dtype=jnp.int32)

# v2 inventory bitflag → v1 OBJECT_TO_INDEX
#   bit0 (1) : PLATE
#   bit1 (2) : COOKED
#   bit2+ (4..) : ingredients (count of base ingredient 0)
# We only support num_ingredients == 1 (onions only) which matches all
# CEC training layouts.
def _v2_inv_to_v1(inv: jnp.ndarray) -> jnp.ndarray:
    has_plate = (inv & 0x1) != 0
    has_cooked = (inv & 0x2) != 0
    onion_count = (inv >> 2) & 0x3
    is_dish = has_plate & has_cooked & (onion_count > 0)
    is_plate = has_plate & ~has_cooked & (onion_count == 0)
    is_onion = ~has_plate & ~has_cooked & (onion_count > 0)
    return jnp.where(
        is_dish, V1_DISH,
        jnp.where(
            is_plate, V1_PLATE,
            jnp.where(is_onion, V1_ONION, V1_EMPTY),
        ),
    )


def _v2_dyn_to_v1_static(static_v2, dyn) -> jnp.ndarray:
    """Loose-item encoding for v1 maze_map at non-pot cells.

    On counters (WALL) we may have a plate / onion / dish sitting on top.
    """
    has_plate = (dyn & 0x1) != 0
    has_cooked = (dyn & 0x2) != 0
    onion_count = (dyn >> 2) & 0x3
    is_dish = has_plate & has_cooked & (onion_count > 0)
    is_plate = has_plate & ~has_cooked & (onion_count == 0)
    is_onion = ~has_plate & ~has_cooked & (onion_count > 0)
    # Counters / empty cells with a loose item show that item.
    return jnp.where(
        is_dish, V1_DISH,
        jnp.where(
            is_plate, V1_PLATE,
            jnp.where(is_onion, V1_ONION, jnp.uint32(0)),
        ),
    )


# v2 StaticObject → v1 OBJECT_TO_INDEX (for non-pot cells, used as the base
# static layer; loose items are overlaid on top).
_V2_TO_V1_STATIC_LUT = jnp.zeros(20, dtype=jnp.uint32)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.EMPTY].set(V1_EMPTY)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.WALL].set(V1_WALL)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.GOAL].set(V1_GOAL)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.POT].set(V1_POT)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.PLATE_PILE].set(V1_PLATE_PILE)
_V2_TO_V1_STATIC_LUT = _V2_TO_V1_STATIC_LUT.at[StaticObject.INGREDIENT_PILE_BASE].set(V1_ONION_PILE)


def _v2_pot_status_to_v1(dyn, extra) -> jnp.ndarray:
    """v1 pot status at a single pot cell.

    v1 encoding (only used at pot cells in maze_map[..., 2]):
        23 = empty
        22 = 1 onion (not cooking)
        21 = 2 onions (not cooking)
        20 = 3 onions (full, not cooking)
         1..19 = cooking countdown
         0 = ready / done
    """
    has_cooked = (dyn & 0x2) != 0
    onion_count = (dyn >> 2) & 0x3

    # Done if cooked bit set and timer hit zero
    done = has_cooked & (extra == 0)
    # Cooking if cooked bit set and timer still ticking
    cooking = has_cooked & (extra > 0)
    # Otherwise: filling — pot has 0..3 onions, no cooked bit
    return jnp.where(
        done, jnp.uint32(0),
        jnp.where(
            cooking, jnp.minimum(extra, 19).astype(jnp.uint32),
            (23 - onion_count).astype(jnp.uint32),
        ),
    )


# ---------------------------------------------------------------------------
# Adapter

@dataclass
class CECObsAdapter:
    """Stateless adapter — only holds the prebuilt v1 env we use for get_obs."""

    target_layout: str = "forced_coord_9"
    max_steps: int = 400

    def __post_init__(self):
        layout = CEC_LAYOUTS[self.target_layout]
        # The v1 env's max_steps drives the urgency layer; align it to the
        # rollout horizon used at training (CEC ippo_final.yaml uses 100,
        # but our integration callers can override).
        self._v1_env = V1Overcooked(layout=layout, random_reset=False, max_steps=self.max_steps)
        self._target_h = self._v1_env.height
        self._target_w = self._v1_env.width
        self._padding = self._v1_env.agent_view_size - 1  # 4
        ph = self._target_h + 2 * self._padding
        pw = self._target_w + 2 * self._padding
        self._padded_shape = (ph, pw, 3)

    # ------------------------------------------------------------------
    def _build_inner_static(self, v2_state):
        """Build the (target_h, target_w) v1 static-channel for the inner grid."""
        grid = v2_state.grid  # (h2, w2, 3)
        h2, w2 = grid.shape[0], grid.shape[1]
        static_v2 = grid[:, :, 0].astype(jnp.uint32)
        dyn = grid[:, :, 1].astype(jnp.uint32)

        # Base static via lookup table
        base = _V2_TO_V1_STATIC_LUT[static_v2]
        # Loose items only show up on non-pot cells
        loose = _v2_dyn_to_v1_static(static_v2, dyn)
        non_pot = static_v2 != int(StaticObject.POT)
        with_items = jnp.where((loose != 0) & non_pot, loose, base)

        # Embed into 9x9 padded with WALL
        inner = jnp.full(
            (self._target_h, self._target_w), V1_WALL, dtype=jnp.uint32
        )
        inner = inner.at[:h2, :w2].set(with_items)
        return inner

    def _build_inner_pot_status(self, v2_state):
        grid = v2_state.grid
        h2, w2 = grid.shape[0], grid.shape[1]
        static_v2 = grid[:, :, 0].astype(jnp.uint32)
        dyn = grid[:, :, 1].astype(jnp.uint32)
        extra = grid[:, :, 2].astype(jnp.uint32)
        is_pot = static_v2 == int(StaticObject.POT)
        pot_status_small = jnp.where(
            is_pot, _v2_pot_status_to_v1(dyn, extra), jnp.uint32(0)
        )
        inner = jnp.zeros(
            (self._target_h, self._target_w), dtype=jnp.uint32
        )
        inner = inner.at[:h2, :w2].set(pot_status_small)
        return inner

    def build_v1_state(self, v2_state) -> V1State:
        inner_static = self._build_inner_static(v2_state)
        inner_pot = self._build_inner_pot_status(v2_state)

        ph, pw, _ = self._padded_shape
        maze_map = jnp.zeros((ph, pw, 3), dtype=jnp.uint32)
        # Channel 0: static
        maze_map = maze_map.at[
            self._padding:self._padding + self._target_h,
            self._padding:self._padding + self._target_w,
            0,
        ].set(inner_static)
        # Channel 2: pot status
        maze_map = maze_map.at[
            self._padding:self._padding + self._target_h,
            self._padding:self._padding + self._target_w,
            2,
        ].set(inner_pot)

        # Agents — v2 grid sits at top-left of the 9x9 inner area, so the
        # v2 (x, y) coordinates carry over directly.
        agents = v2_state.agents
        agent_pos = jnp.stack(
            [agents.pos.x.astype(jnp.uint32), agents.pos.y.astype(jnp.uint32)],
            axis=-1,
        )
        agent_dir_idx = _V2_DIR_TO_V1_DIR_IDX[agents.dir.astype(jnp.int32)]
        agent_inv = _v2_inv_to_v1(agents.inventory.astype(jnp.uint32))

        # wall_map / goal_pos / pot_pos are not consumed by get_obs but the
        # v1 State dataclass requires them. Provide cheap stubs derived from
        # the inner static channel.
        wall_map = (inner_static == V1_WALL)
        # goal_pos and pot_pos: take their cell coordinates from the inner
        # static. We shape them as (k, 2). Stub with zeros if absent — get_obs
        # never reads these fields.
        goal_pos = jnp.zeros((1, 2), dtype=jnp.uint32)
        pot_pos = jnp.zeros((1, 2), dtype=jnp.uint32)

        # v1 State expects agent_dir as DIR_TO_VEC values too. Compute from idx.
        from jaxmarl.environments.overcooked.common import DIR_TO_VEC
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        return V1State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=jnp.asarray(v2_state.time, dtype=jnp.int32),
            terminal=v2_state.terminal,
        )

    def get_cec_obs(self, v2_state):
        """Returns the (9,9,26) per-agent obs dict CEC consumes."""
        v1_state = self.build_v1_state(v2_state)
        return self._v1_env.get_obs(v1_state)
