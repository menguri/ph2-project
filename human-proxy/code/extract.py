#!/usr/bin/env python3
"""
Webapp trajectory pickle → BC 학습용 numpy 변환 (포지션별 분리).

사용법:
    python code/extract.py --traj-dir ../webapp/data/trajectories --out-dir data/bc

출력:
    data/bc/{layout}/pos_{i}/
        obs.npy       # (N, H, W, C) uint8
        actions.npy   # (N,) int32
        metadata.json # 에피소드 정보
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_all_trajectories(traj_dir: str):
    """trajectory 디렉토리의 모든 pickle 파일 로드."""
    traj_path = Path(traj_dir)
    episodes = []
    for pkl_file in sorted(traj_path.rglob("*.pkl")):
        with open(pkl_file, "rb") as f:
            ep = pickle.load(f)
        episodes.append(ep)
    return episodes


def export_by_position(traj_dir: str, out_dir: str, layout_filter: str = None):
    """trajectory pickle → 레이아웃/포지션별 numpy arrays."""
    episodes = load_all_trajectories(traj_dir)
    print(f"총 {len(episodes)}개 에피소드 로드")

    # 레이아웃 + 포지션별 그룹핑
    # key: (layout, human_player_index)
    grouped = defaultdict(lambda: {"obs": [], "actions": [], "metadata": []})

    for ep in episodes:
        layout = ep.get("layout", "unknown")
        if layout_filter and layout != layout_filter:
            continue

        pos = ep.get("human_player_index", 0)
        key = (layout, pos)

        transitions = ep.get("transitions", [])
        for t in transitions:
            obs = t.get("obs_human")
            action = t.get("action_human")
            if obs is not None and action is not None:
                grouped[key]["obs"].append(obs)
                grouped[key]["actions"].append(action)

        grouped[key]["metadata"].append({
            "episode_id": ep.get("episode_id"),
            "participant_id": ep.get("participant_id"),
            "algo_name": ep.get("algo_name"),
            "seed_id": ep.get("seed_id"),
            "final_score": ep.get("final_score"),
            "episode_length": ep.get("episode_length"),
            "human_player_index": pos,
        })

    if not grouped:
        print("데이터 없음")
        return

    for (layout, pos), data in sorted(grouped.items()):
        if not data["obs"]:
            print(f"  {layout}/pos_{pos}: obs 없음, 스킵")
            continue

        obs_array = np.stack(data["obs"]).astype(np.uint8)
        actions_array = np.array(data["actions"], dtype=np.int32)

        out_path = Path(out_dir) / layout / f"pos_{pos}"
        out_path.mkdir(parents=True, exist_ok=True)

        np.save(out_path / "obs.npy", obs_array)
        np.save(out_path / "actions.npy", actions_array)

        meta = {
            "num_samples": len(data["obs"]),
            "obs_shape": list(obs_array.shape),
            "num_episodes": len(data["metadata"]),
            "layout": layout,
            "position": pos,
            "episodes": data["metadata"],
        }
        with open(out_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"  {layout}/pos_{pos}: {len(data['obs'])} samples "
              f"from {len(data['metadata'])} episodes → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webapp trajectory → BC 학습 데이터 변환")
    parser.add_argument("--traj-dir", default="../webapp/data/trajectories",
                        help="Webapp trajectory 디렉토리")
    parser.add_argument("--out-dir", default="data/bc",
                        help="출력 디렉토리")
    parser.add_argument("--layout", default=None,
                        help="특정 레이아웃만 필터링")
    args = parser.parse_args()
    export_by_position(args.traj_dir, args.out_dir, args.layout)
