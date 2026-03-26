#!/usr/bin/env python3
"""
Phase 10: Trajectory pickle → BC 학습용 numpy 변환.

사용법:
    python scripts/export_bc_dataset.py [--traj-dir data/trajectories] [--out-dir data/bc_dataset]

출력:
    data/bc_dataset/{layout}/
        obs.npy       # (N, H, W, C) uint8 — human agent의 JaxMARL obs
        actions.npy   # (N,) int32 — human agent의 action indices
        metadata.json # 에피소드 정보, 통계
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_trajectories(traj_dir: str):
    """trajectory 디렉토리의 모든 pickle 파일 로드."""
    traj_path = Path(traj_dir)
    episodes = []
    for pkl_file in sorted(traj_path.rglob("*.pkl")):
        with open(pkl_file, "rb") as f:
            ep = pickle.load(f)
        episodes.append(ep)
    return episodes


def export_bc_dataset(traj_dir: str, out_dir: str, layout_filter: str = None):
    """trajectory pickle → BC numpy arrays."""
    episodes = load_trajectories(traj_dir)
    print(f"Loaded {len(episodes)} episodes")

    # 레이아웃별 그룹핑
    by_layout = {}
    for ep in episodes:
        layout = ep.get("layout", "unknown")
        if layout_filter and layout != layout_filter:
            continue
        by_layout.setdefault(layout, []).append(ep)

    for layout, eps in by_layout.items():
        all_obs = []
        all_actions = []
        ep_metadata = []

        for ep in eps:
            transitions = ep.get("transitions", [])
            for t in transitions:
                obs = t.get("obs_human")
                action = t.get("action_human")
                if obs is not None and action is not None:
                    all_obs.append(obs)
                    all_actions.append(action)

            ep_metadata.append({
                "episode_id": ep.get("episode_id"),
                "participant_id": ep.get("participant_id"),
                "algo_name": ep.get("algo_name"),
                "seed_id": ep.get("seed_id"),
                "final_score": ep.get("final_score"),
                "episode_length": ep.get("episode_length"),
                "human_player_index": ep.get("human_player_index"),
            })

        if not all_obs:
            print(f"  {layout}: no obs data, skipping")
            continue

        obs_array = np.stack(all_obs).astype(np.uint8)
        actions_array = np.array(all_actions, dtype=np.int32)

        out_path = Path(out_dir) / layout
        out_path.mkdir(parents=True, exist_ok=True)

        np.save(out_path / "obs.npy", obs_array)
        np.save(out_path / "actions.npy", actions_array)
        with open(out_path / "metadata.json", "w") as f:
            json.dump({
                "num_samples": len(all_obs),
                "obs_shape": list(obs_array.shape),
                "num_episodes": len(eps),
                "episodes": ep_metadata,
            }, f, indent=2)

        print(f"  {layout}: {len(all_obs)} samples from {len(eps)} episodes → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", default="data/trajectories")
    parser.add_argument("--out-dir", default="data/bc_dataset")
    parser.add_argument("--layout", default=None, help="Filter by layout name")
    args = parser.parse_args()
    export_bc_dataset(args.traj_dir, args.out_dir, args.layout)
