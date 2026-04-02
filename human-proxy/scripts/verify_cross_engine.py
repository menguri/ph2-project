#!/usr/bin/env python3
"""
Step 3: Berkeley human action을 overcooked-ai + JaxMARL 양쪽에서 replay하여 비교.

실제 human 협력 플레이(요리, 픽업, 배달)에서 두 엔진의 state가 일치하는지 검증.

사용법:
    cd human-proxy && python scripts/verify_cross_engine.py
"""
import json
import os
import pickle
import sys
import types
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import pandas as pd
import pandas.core.indexes
if not hasattr(pandas.core.indexes, "numeric"):
    pandas.core.indexes.numeric = types.ModuleType("pandas.core.indexes.numeric")
    pandas.core.indexes.numeric.Int64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = pandas.core.indexes.numeric

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))
sys.path.insert(0, str(PROJECT_ROOT / "JaxMARL"))

import jax
import jax.numpy as jnp

# JaxMARL
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2 as JaxOvercooked
from jaxmarl.environments.overcooked_v2.common import DynamicObject, StaticObject

# overcooked-ai
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.actions import Action

from app.game.engine import _load_custom_layout


BERKELEY_DATA = Path("/home/mlic/mingukang/zsc-basecamp/GAMMA/mapbt/envs/overcooked/"
                     "overcooked_berkeley/src/human_aware_rl/static/human_data/cleaned")

# Berkeley layout → JaxMARL layout
LAYOUT_MAP = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymm_advantages",
    "coordination_ring": "coord_ring",
}

# Berkeley action → JaxMARL int
OVERCOOKED_TO_JAXMARL = {
    (1, 0): 0,    # EAST → right
    (0, 1): 1,    # SOUTH → down
    (-1, 0): 2,   # WEST → left
    (0, -1): 3,   # NORTH → up
    (0, 0): 4,    # STAY
}

# JaxMARL int → overcooked-ai action
JAXMARL_TO_OVERCOOKED = {
    0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0), 5: "interact",
}

JAXMARL_DIR_TO_ORIENT = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}


def parse_berkeley_action_to_jaxmarl(a):
    if isinstance(a, str) and a.upper() == "INTERACT":
        return 5
    if isinstance(a, (list, tuple)):
        return OVERCOOKED_TO_JAXMARL.get(tuple(a), 4)
    return 4


def extract_jax_state(env, state):
    """JaxMARL state → 비교용 dict."""
    info = {"agents": [], "pots": []}
    for i in range(env.num_agents):
        agent = jax.tree.map(lambda x: x[i], state.agents)
        pos = (int(agent.pos.x), int(agent.pos.y))
        direction = int(agent.dir)
        inventory = int(agent.inventory)
        info["agents"].append({
            "pos": pos, "orient": JAXMARL_DIR_TO_ORIENT.get(direction),
            "inventory": inventory,
        })
    grid = np.array(state.grid)
    for r in range(env.height):
        for c in range(env.width):
            if int(grid[r, c, 0]) == StaticObject.POT:
                ings = int(grid[r, c, 1])
                timer = int(grid[r, c, 2])
                info["pots"].append({
                    "pos": (c, r),
                    "ingredients": ings,
                    "timer": timer,
                    "is_cooking": timer > 0 and not bool(ings & DynamicObject.COOKED),
                    "is_ready": bool(ings & DynamicObject.COOKED),
                    "ing_count": (ings >> 2) & 0x3,
                })
    return info


def extract_oc_state(mdp, state):
    """overcooked-ai state → 비교용 dict."""
    info = {"agents": [], "pots": []}
    for p in state.players:
        held = p.held_object
        inv = 0
        if held:
            if held.name == "dish":
                inv = 1
            elif held.name == "onion":
                inv = 4
            elif held.name == "soup":
                inv = 1 | 2  # plate + cooked
                inv += sum(4 for i in held.ingredients if i == "onion")
        info["agents"].append({
            "pos": p.position, "orient": p.orientation, "inventory": inv,
        })
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if obj.name == "soup":
                n = len(obj.ingredients)
                is_cooking = getattr(obj, "is_cooking", False)
                is_ready = getattr(obj, "is_ready", False)
                tick = getattr(obj, "_cooking_tick", -1)
                if is_cooking and not is_ready:
                    try:
                        ct = obj.cook_time
                    except:
                        ct = 20
                    timer = ct - tick
                else:
                    timer = 0

                ings_raw = 0
                if is_ready:
                    ings_raw |= 1 | 2
                elif is_cooking:
                    ings_raw |= 2
                ings_raw += n * 4

                info["pots"].append({
                    "pos": pos, "ingredients": ings_raw, "timer": timer,
                    "is_cooking": is_cooking, "is_ready": is_ready, "ing_count": n,
                })
        else:
            info["pots"].append({
                "pos": pos, "ingredients": 0, "timer": 0,
                "is_cooking": False, "is_ready": False, "ing_count": 0,
            })
    return info


def compare(jax_info, oc_info):
    diffs = []
    for i in range(len(jax_info["agents"])):
        ja, oa = jax_info["agents"][i], oc_info["agents"][i]
        if ja["pos"] != oa["pos"]:
            diffs.append(f"agent{i}_pos: jax={ja['pos']} oc={oa['pos']}")
        if ja["orient"] != oa["orient"]:
            diffs.append(f"agent{i}_dir: jax={ja['orient']} oc={oa['orient']}")
        if ja["inventory"] != oa["inventory"]:
            diffs.append(f"agent{i}_inv: jax={ja['inventory']} oc={oa['inventory']}")

    j_pots = sorted(jax_info["pots"], key=lambda p: p["pos"])
    o_pots = sorted(oc_info["pots"], key=lambda p: p["pos"])
    for jp, op in zip(j_pots, o_pots):
        if jp["ing_count"] != op["ing_count"]:
            diffs.append(f"pot{jp['pos']}_ings: jax={jp['ing_count']} oc={op['ing_count']}")
        if jp["timer"] != op["timer"]:
            diffs.append(f"pot{jp['pos']}_timer: jax={jp['timer']} oc={op['timer']}")
        if jp["is_cooking"] != op["is_cooking"]:
            diffs.append(f"pot{jp['pos']}_cooking: jax={jp['is_cooking']} oc={op['is_cooking']}")
        if jp["is_ready"] != op["is_ready"]:
            diffs.append(f"pot{jp['pos']}_ready: jax={jp['is_ready']} oc={op['is_ready']}")
    return diffs


def run_trial(berkeley_layout, trial_df, max_steps=400):
    """한 trial을 양쪽 엔진에서 replay."""
    jaxmarl_layout = LAYOUT_MAP[berkeley_layout]
    rows = list(trial_df.sort_values("cur_gameloop").iterrows())

    # 초기 state 파싱
    first_state = json.loads(rows[0][1]["state"])

    # JaxMARL 환경 초기화
    jax_env = JaxOvercooked(layout=jaxmarl_layout, max_steps=max_steps + 10)
    jax_key = jax.random.PRNGKey(0)
    jax_key, reset_key = jax.random.split(jax_key)
    _, jax_state = jax_env.reset(reset_key)

    # overcooked-ai 환경 초기화
    oc_mdp = _load_custom_layout(jaxmarl_layout)
    oc_state = oc_mdp.get_standard_start_state()

    # 양쪽 초기 위치를 Berkeley와 동기화
    for i in range(2):
        pos = tuple(first_state["players"][i]["position"])
        orient = tuple(first_state["players"][i]["orientation"])
        oc_state.players[i].position = pos
        oc_state.players[i].orientation = orient

        # JaxMARL은 reset state를 직접 수정해야 함
        orient_to_dir = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}
        new_agents = jax_state.agents.replace(
            pos=jax_state.agents.pos.replace(
                x=jax_state.agents.pos.x.at[i].set(pos[0]),
                y=jax_state.agents.pos.y.at[i].set(pos[1]),
            ),
            dir=jax_state.agents.dir.at[i].set(orient_to_dir.get(orient, 0)),
        )
        jax_state = jax_state.replace(agents=new_agents)

    total = 0
    mismatches = 0
    first_mismatch = None
    mismatch_types = {}

    steps_to_run = min(max_steps, len(rows) - 1)

    for step_i in range(steps_to_run):
        _, row = rows[step_i]

        # Berkeley action 파싱
        try:
            ja = json.loads(row["joint_action"])
        except:
            import ast
            ja = ast.literal_eval(row["joint_action"])

        a0_jax = parse_berkeley_action_to_jaxmarl(ja[0])
        a1_jax = parse_berkeley_action_to_jaxmarl(ja[1])
        a0_oc = JAXMARL_TO_OVERCOOKED[a0_jax]
        a1_oc = JAXMARL_TO_OVERCOOKED[a1_jax]

        # JaxMARL step
        jax_key, step_key = jax.random.split(jax_key)
        jax_actions = {"agent_0": jnp.int32(a0_jax), "agent_1": jnp.int32(a1_jax)}
        _, jax_state, jax_rewards, jax_dones, _ = jax_env.step(step_key, jax_state, jax_actions)

        # overcooked-ai step + auto-cook patch
        oc_next, _ = oc_mdp.get_state_transition(oc_state, (a0_oc, a1_oc))
        for pot_pos in oc_mdp.get_pot_locations():
            if oc_next.has_object(pot_pos):
                obj = oc_next.get_object(pot_pos)
                if (obj.name == "soup" and not obj.is_cooking
                        and not obj.is_ready and len(obj.ingredients) >= 3):
                    obj.begin_cooking()
                    obj.cook()
        oc_state = oc_next

        # 비교
        jax_info = extract_jax_state(jax_env, jax_state)
        oc_info = extract_oc_state(oc_mdp, oc_state)
        diffs = compare(jax_info, oc_info)

        total += 1
        if diffs:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (step_i + 1, diffs[:3])
            for d in diffs:
                cat = d.split(":")[0]
                mismatch_types[cat] = mismatch_types.get(cat, 0) + 1

        if bool(jax_dones.get("__all__", False)):
            break

    return total, mismatches, first_mismatch, mismatch_types


def main():
    print("=" * 70)
    print("Step 3: Berkeley human action cross-engine replay 검증")
    print("=" * 70)

    with open(BERKELEY_DATA / "2019_hh_trials_all.pickle", "rb") as f:
        df = pickle.load(f)

    all_pass = True

    for bk_layout, jax_layout in LAYOUT_MAP.items():
        print(f"\n[{bk_layout} → {jax_layout}]")
        sub = df[df["layout_name"] == bk_layout]
        scored = sub.groupby("trial_id").agg({"score": "max"}).query("score > 0")
        trial_ids = scored.index[:3] if len(scored) >= 3 else sub["trial_id"].unique()[:3]

        layout_total = 0
        layout_mismatch = 0

        for tid in trial_ids:
            trial = sub[sub["trial_id"] == tid]
            total, mismatches, first_mm, mm_types = run_trial(bk_layout, trial, max_steps=400)
            layout_total += total
            layout_mismatch += mismatches

            score = trial.sort_values("cur_gameloop").iloc[-1]["score"]
            pct = 100.0 * (total - mismatches) / total if total > 0 else 0
            status = "✓" if mismatches == 0 else "✗"
            print(f"  Trial {tid}: {total} steps, {mismatches} mismatches ({pct:.1f}% match), "
                  f"score={score} {status}")
            if first_mm:
                step_n, details = first_mm
                print(f"    첫 불일치 step {step_n}: {details}")
            if mm_types:
                for cat, cnt in sorted(mm_types.items(), key=lambda x: -x[1])[:3]:
                    print(f"    {cat}: {cnt}건")

        pct_total = 100.0 * (layout_total - layout_mismatch) / layout_total if layout_total > 0 else 0
        if layout_mismatch > 0:
            print(f"  ✗ {bk_layout}: {pct_total:.1f}% match ({layout_total - layout_mismatch}/{layout_total})")
            all_pass = False
        else:
            print(f"  ✓ {bk_layout}: 100% match ({layout_total}/{layout_total})")

    print(f"\n{'=' * 70}")
    print("PASS — 모든 레이아웃 cross-engine 일치" if all_pass else "FAIL — 불일치 발견")


if __name__ == "__main__":
    main()
