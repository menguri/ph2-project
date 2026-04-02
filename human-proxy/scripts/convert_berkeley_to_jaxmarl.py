#!/usr/bin/env python3
"""
Berkeley human-human trajectory → JaxMARL 포맷 변환.

검증 완료 레이아웃: cramped_room, asymmetric_advantages, coordination_ring
(counter_circuit은 cook_time/early_cook 불일치로 폐기)

출력: human-proxy/data/berkeley/{layout}/pos_{0,1}/obs.npy + actions.npy + metadata.json

사용법:
    cd human-proxy && python scripts/convert_berkeley_to_jaxmarl.py
"""
import json
import os
import pickle
import sys
import types
from collections import defaultdict
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

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs


BERKELEY_DATA = Path("/home/mlic/mingukang/zsc-basecamp/GAMMA/mapbt/envs/overcooked/"
                     "overcooked_berkeley/src/human_aware_rl/static/human_data/cleaned")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "berkeley"

TARGET_LAYOUTS = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
}

OVERCOOKED_TO_JAXMARL_ACTION = {
    (1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3, (0, 0): 4,
}


def parse_action(a):
    if isinstance(a, str) and a.upper() == "INTERACT":
        return 5
    if isinstance(a, (list, tuple)):
        return OVERCOOKED_TO_JAXMARL_ACTION.get(tuple(a), 4)
    return 4


def convert_all():
    print("Berkeley → JaxMARL 변환 시작\n")

    for year in ["2019"]:
        pkl_path = BERKELEY_DATA / f"{year}_hh_trials_all.pickle"
        print(f"Loading {pkl_path.name}...")
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)

        for bk_layout, oc_layout in TARGET_LAYOUTS.items():
            sub = df[df["layout_name"] == bk_layout]
            if len(sub) == 0:
                continue

            mdp = OvercookedGridworld.from_layout_name(oc_layout)
            trials = sorted(sub["trial_id"].unique())

            # pos별 데이터 수집
            data = {0: {"obs": [], "actions": [], "meta": []},
                    1: {"obs": [], "actions": [], "meta": []}}

            for tid in trials:
                trial = sub[sub["trial_id"] == tid].sort_values("cur_gameloop")
                score = trial.iloc[-1]["score"]
                n_steps = 0

                for _, row in trial.iterrows():
                    try:
                        state_dict = json.loads(row["state"])
                        state = OvercookedState.from_dict(state_dict)
                    except Exception:
                        continue

                    try:
                        ja = json.loads(row["joint_action"])
                    except:
                        import ast
                        ja = ast.literal_eval(row["joint_action"])

                    for pos in [0, 1]:
                        obs = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=pos)
                        action = parse_action(ja[pos])
                        data[pos]["obs"].append(obs)
                        data[pos]["actions"].append(action)

                    n_steps += 1

                for pos in [0, 1]:
                    data[pos]["meta"].append({
                        "trial_id": int(tid),
                        "score": float(score),
                        "steps": n_steps,
                        "year": year,
                    })

            # 저장
            for pos in [0, 1]:
                if not data[pos]["obs"]:
                    continue
                obs_arr = np.stack(data[pos]["obs"]).astype(np.int32)
                act_arr = np.array(data[pos]["actions"], dtype=np.int32)

                out_dir = OUTPUT_DIR / bk_layout / f"pos_{pos}"
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(out_dir / "obs.npy", obs_arr)
                np.save(out_dir / "actions.npy", act_arr)

                metadata = {
                    "num_samples": len(data[pos]["obs"]),
                    "obs_shape": list(obs_arr.shape),
                    "num_episodes": len(data[pos]["meta"]),
                    "layout": bk_layout,
                    "position": pos,
                    "source": f"berkeley_{year}",
                    "episodes": data[pos]["meta"],
                }
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"  {bk_layout}/pos_{pos}: {len(data[pos]['obs']):,} samples "
                      f"from {len(data[pos]['meta'])} trials → {out_dir}")

    print("\n변환 완료!")


if __name__ == "__main__":
    convert_all()
