#!/usr/bin/env python3
"""
Step 1: Berkeley 데이터 무결성 검증.

Berkeley trajectory의 action sequence를 overcooked-ai에서 replay하여
기록된 state와 일치하는지 step-by-step 확인.

사용법:
    cd human-proxy && python scripts/verify_berkeley_replay.py
"""
import json
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pandas.core.indexes
if not hasattr(pandas.core.indexes, "numeric"):
    pandas.core.indexes.numeric = types.ModuleType("pandas.core.indexes.numeric")
    pandas.core.indexes.numeric.Int64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = pandas.core.indexes.numeric

# webapp의 obs_adapter 사용을 위해 path 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.actions import Action, Direction


BERKELEY_DATA = Path("/home/mlic/mingukang/zsc-basecamp/GAMMA/mapbt/envs/overcooked/overcooked_berkeley/src/human_aware_rl/static/human_data/cleaned")

TARGET_LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]

# Berkeley action → overcooked-ai action
def parse_berkeley_action(a):
    if isinstance(a, str) and a.upper() == "INTERACT":
        return Action.INTERACT
    if isinstance(a, (list, tuple)):
        return tuple(a)
    return Action.STAY


def load_berkeley_data(year="2019"):
    f = BERKELEY_DATA / f"{year}_hh_trials_all.pickle"
    with open(f, "rb") as fp:
        return pickle.load(fp)


def compare_states(recorded_state_dict, actual_state):
    """기록된 state와 실제 state 비교. 불일치 항목 리스트 반환."""
    mismatches = []

    rec_players = recorded_state_dict["players"]
    act_players = actual_state.players

    for i in range(2):
        # position
        rec_pos = tuple(rec_players[i]["position"])
        act_pos = tuple(act_players[i].position)
        if rec_pos != act_pos:
            mismatches.append(f"player{i}_pos: rec={rec_pos} act={act_pos}")

        # orientation
        rec_ori = tuple(rec_players[i]["orientation"])
        act_ori = tuple(act_players[i].orientation)
        if rec_ori != act_ori:
            mismatches.append(f"player{i}_ori: rec={rec_ori} act={act_ori}")

        # held object
        rec_held = rec_players[i].get("held_object")
        act_held = act_players[i].held_object
        rec_name = rec_held["name"] if rec_held else None
        act_name = act_held.name if act_held else None
        if rec_name != act_name:
            mismatches.append(f"player{i}_held: rec={rec_name} act={act_name}")

    return mismatches


def verify_layout(df, layout_name, max_trials=5):
    """한 레이아웃의 Berkeley trial들을 replay 검증."""
    sub = df[df["layout_name"] == layout_name]
    scored = sub.groupby("trial_id").agg({"score": "max"}).query("score > 0")

    if len(scored) == 0:
        print(f"  score > 0 trial 없음, 전체에서 선택")
        trial_ids = sub["trial_id"].unique()[:max_trials]
    else:
        trial_ids = scored.index[:max_trials]

    mdp = OvercookedGridworld.from_layout_name(layout_name)

    total_steps = 0
    total_match = 0
    total_mismatch = 0

    for tid in trial_ids:
        trial = sub[sub["trial_id"] == tid].sort_values("cur_gameloop")
        rows = list(trial.iterrows())

        # 첫 state로 초기화
        first_state_dict = json.loads(rows[0][1]["state"])
        state = OvercookedState.from_dict(first_state_dict)

        step_matches = 0
        step_mismatches = 0
        first_mismatch = None

        for i in range(len(rows) - 1):
            _, row = rows[i]
            _, next_row = rows[i + 1]

            # action 파싱
            try:
                ja = json.loads(row["joint_action"])
            except (json.JSONDecodeError, TypeError):
                import ast
                ja = ast.literal_eval(row["joint_action"])

            a0 = parse_berkeley_action(ja[0])
            a1 = parse_berkeley_action(ja[1])
            joint_action = (a0, a1)

            # overcooked-ai step
            next_state, _ = mdp.get_state_transition(state, joint_action)

            # 기록된 다음 state와 비교
            next_recorded = json.loads(next_row["state"])
            mismatches = compare_states(next_recorded, next_state)

            if mismatches:
                step_mismatches += 1
                if first_mismatch is None:
                    first_mismatch = (i + 1, mismatches[:3])
            else:
                step_matches += 1

            # 다음 step을 위해 기록된 state로 동기화 (drift 방지)
            state = OvercookedState.from_dict(next_recorded)

        total = step_matches + step_mismatches
        total_steps += total
        total_match += step_matches
        total_mismatch += step_mismatches

        pct = 100.0 * step_matches / total if total > 0 else 0
        score = rows[-1][1]["score"]
        status = "✓" if step_mismatches == 0 else "✗"
        print(f"  Trial {tid}: {total} steps, {step_matches}/{total} match ({pct:.1f}%), "
              f"score={score} {status}")
        if first_mismatch:
            step_num, details = first_mismatch
            print(f"    첫 불일치 step {step_num}: {details}")

    pct_total = 100.0 * total_match / total_steps if total_steps > 0 else 0
    return total_match, total_steps, pct_total


def main():
    print("=" * 70)
    print("Step 1: Berkeley 데이터 내부 replay 검증")
    print("=" * 70)

    df = load_berkeley_data("2019")
    all_pass = True

    for layout in TARGET_LAYOUTS:
        print(f"\n[{layout}]")
        match, total, pct = verify_layout(df, layout, max_trials=5)
        if pct < 99.0:
            print(f"  ✗ FAIL: {pct:.1f}% match ({match}/{total})")
            all_pass = False
        else:
            print(f"  ✓ PASS: {pct:.1f}% match ({match}/{total})")

    print(f"\n{'=' * 70}")
    print("PASS" if all_pass else "FAIL — 불일치 발견")


if __name__ == "__main__":
    main()
