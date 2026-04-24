"""V1 Overcooked vs OV2 engine dynamics 비교.

동일 초기 state + 동일 action 시퀀스를 양 engine 에 돌려 매 step state diff 를 찍는다.
Layout: cramped_room.

목적: BC 가 V1 engine 에서 실패하는 근본 원인 (dynamics 어느 부분이 다른가) 파악.
"""
import os
import sys

sys.path.insert(0, "/home/mlic/mingukang/ph2-project")

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
from jaxmarl.environments.overcooked.common import DIR_TO_VEC, OBJECT_TO_INDEX
from jaxmarl.environments.overcooked_v2.common import StaticObject

from cec_integration.cec_layouts import CEC_LAYOUTS


LAYOUT = "cramped_room"
V1_TO_OV2_DIR = np.array([2, 1, 3, 0])   # V1 (right,down,left,up) → OV2 (right,down,left,up) — OV2 order (UP=0,DOWN=1,RIGHT=2,LEFT=3)

_V1_INV_NAME = {0: "empty", OBJECT_TO_INDEX["onion"]: "onion",
                OBJECT_TO_INDEX["plate"]: "plate", OBJECT_TO_INDEX["dish"]: "dish",
                OBJECT_TO_INDEX["empty"]: "empty"}


def v1_inv_to_str(v):
    return _V1_INV_NAME.get(int(v), f"other({int(v)})")


def ov2_inv_to_str(v):
    v = int(v)
    has_plate = v & 0x1
    has_cooked = v & 0x2
    count = (v >> 2) & 0x3
    if v == 0:
        return "empty"
    if has_plate and has_cooked and count == 3:
        return "dish"
    if has_plate and not has_cooked and count == 0:
        return "plate"
    if not has_plate and not has_cooked and count > 0:
        return f"onion×{count}"
    return f"raw({v})"


def extract_ov2_snapshot(state):
    grid = np.asarray(state.grid)
    agents = state.agents
    pos = [(int(agents.pos.x[i]), int(agents.pos.y[i])) for i in range(2)]
    dirs = [int(agents.dir[i]) for i in range(2)]
    inv = [ov2_inv_to_str(agents.inventory[i]) for i in range(2)]
    pots = {}
    loose = {}
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            static = int(grid[y, x, 0])
            dyn = int(grid[y, x, 1])
            extra = int(grid[y, x, 2])
            if static == StaticObject.POT and (dyn != 0 or extra != 0):
                pots[(x, y)] = f"dyn={dyn}({ov2_inv_to_str(dyn)}) extra={extra}"
            elif static != StaticObject.POT and dyn != 0:
                loose[(x, y)] = f"dyn={dyn}({ov2_inv_to_str(dyn)})"
    return {"pos": pos, "dir": dirs, "inv": inv, "pots": pots, "loose": loose}


def extract_v1_snapshot(state):
    pad = 4
    maze = np.asarray(state.maze_map)
    pos = np.asarray(state.agent_pos)
    dirs = np.asarray(state.agent_dir_idx)
    inv = np.asarray(state.agent_inv)
    pot_pos = np.asarray(state.pot_pos)
    pot_out = {}
    for p in pot_pos:
        x, y = int(p[0]), int(p[1])
        status = int(maze[y + pad, x + pad, 2])
        if status < 23:
            if status == 0:
                desc = f"READY"
            elif status < 20:
                desc = f"cooking (status={status}, ≈cook_time_remaining={status})"
            else:
                n = 23 - status
                desc = f"filling (status={status}, onions={n})"
            pot_out[(x, y)] = f"v1_status={status} ({desc})"
    loose_out = {}
    v1_h = maze.shape[0] - 2 * pad
    v1_w = maze.shape[1] - 2 * pad
    for y in range(v1_h):
        for x in range(v1_w):
            obj = int(maze[y + pad, x + pad, 0])
            if obj == OBJECT_TO_INDEX["onion"]:
                loose_out[(x, y)] = "onion"
            elif obj == OBJECT_TO_INDEX["plate"]:
                loose_out[(x, y)] = "plate"
            elif obj == OBJECT_TO_INDEX["dish"]:
                loose_out[(x, y)] = "dish"
    return {
        "pos": [(int(pos[i, 0]), int(pos[i, 1])) for i in range(2)],
        "dir": [int(V1_TO_OV2_DIR[int(dirs[i])]) for i in range(2)],
        "inv": [v1_inv_to_str(inv[i]) for i in range(2)],
        "pots": pot_out,
        "loose": loose_out,
    }


def print_side_by_side(v1s, ov2s, header=""):
    if header:
        print(f"  {header}")
    print(f"    V1  pos={v1s['pos']}  dir={v1s['dir']}  inv={v1s['inv']}")
    print(f"    OV2 pos={ov2s['pos']}  dir={ov2s['dir']}  inv={ov2s['inv']}")
    if v1s["pots"] or ov2s["pots"]:
        print(f"    V1  pots={v1s['pots']}")
        print(f"    OV2 pots={ov2s['pots']}")
    if v1s["loose"] or ov2s["loose"]:
        print(f"    V1  loose={v1s['loose']}")
        print(f"    OV2 loose={ov2s['loose']}")


def main():
    v1_env = V1Overcooked(layout=CEC_LAYOUTS[f"{LAYOUT}_9"], random_reset=False, max_steps=100)
    ov2_env = jaxmarl.make("overcooked_v2", layout=LAYOUT, max_steps=100, random_reset=False)

    key = jax.random.PRNGKey(0)
    _, v1_state = v1_env.reset(key)
    _, ov2_state = ov2_env.reset(key)

    # V1 initial dir 과 agent_pos 를 OV2 와 동일하게 정렬 (fair comparison)
    # OV2 cramped_room: agents at [(3,1), (1,1)], dir UP
    v1_state = v1_state.replace(
        agent_dir_idx=jnp.array([3, 3], dtype=jnp.int32),
        agent_dir=DIR_TO_VEC[jnp.array([3, 3])],
        agent_pos=jnp.array([[3, 1], [1, 1]], dtype=jnp.uint32),
    )

    print("=" * 80)
    print(f"Engine dynamics comparison on {LAYOUT}")
    print("=" * 80)

    # Initial snapshot
    v1_snap = extract_v1_snapshot(v1_state)
    ov2_snap = extract_ov2_snapshot(ov2_state)
    print_side_by_side(v1_snap, ov2_snap, header="[Initial state]")

    # Scripted action sequence (양쪽 agent 모두 delivery pipeline 시도)
    # OV2 cramped_room 기준:
    #   agent_0 at (3,1) UP: r(0)→dir=RIGHT, facing (4,1)=onion_pile
    #   agent_1 at (1,1) UP: l(2)→dir=LEFT, facing (0,1)=onion_pile
    #
    # Sequence: 3 ×  [pickup onion → move to pot → drop onion] → cook → plate → delivery
    actions = [
        # 1. agent_0 right (face onion pile), agent_1 left (face onion pile)
        (0, 2),
        # 2. both interact — pickup onion
        (5, 5),
        # 3. agent_0 left (move to (2,1)), agent_1 right (move to (2,1)?  collision — stay)
        #    실제로 agent_0 이 먼저 이동하든 agent_1 이 먼저 이동하든 engine dependent
        (2, 0),
        # 4. both up — face pot at (2,0)
        (3, 3),
        # 5. both interact — drop onion (but maybe only one can drop at a time — engine dependent)
        (5, 5),
    ]

    # OV2 → V1 action remap (V1 DIR_TO_VEC 과 Actions enum label 이 뒤바뀌어 있음)
    V1_REMAP = np.array([2, 1, 3, 0, 4, 5], dtype=np.int32)

    for i, (a0, a1) in enumerate(actions):
        # V1 step — action remap 적용 (OV2 semantic 을 V1 semantic 으로)
        va0, va1 = int(V1_REMAP[a0]), int(V1_REMAP[a1])
        k1, ke1 = jax.random.split(jax.random.PRNGKey(100 + i), 2)
        _, v1_state, v1_reward, v1_done, _ = v1_env.step(
            ke1, v1_state, {"agent_0": jnp.int32(va0), "agent_1": jnp.int32(va1)}
        )
        # OV2 step — 원본 action 그대로
        k2, ke2 = jax.random.split(jax.random.PRNGKey(100 + i), 2)
        _, ov2_state, ov2_reward, ov2_done, _ = ov2_env.step(
            ke2, ov2_state, {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        )

        v1_snap = extract_v1_snapshot(v1_state)
        ov2_snap = extract_ov2_snapshot(ov2_state)
        print(f"\n[step {i+1}] action=(a0={a0}, a1={a1})   V1_r={float(v1_reward['agent_0']):.0f}  OV2_r={float(ov2_reward['agent_0']):.0f}")
        print_side_by_side(v1_snap, ov2_snap)


if __name__ == "__main__":
    sys.exit(main() or 0)
