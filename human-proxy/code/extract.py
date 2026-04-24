#!/usr/bin/env python3
"""
Webapp trajectory pickle → BC 학습용 numpy 변환 (포지션별 분리).

사용법:
    python code/extract.py --traj-dir ../webapp/data/trajectories --out-dir data/bc

    # obs_adapter 버그 수정 후 obs 재계산:
    python code/extract.py --traj-dir ../webapp/data/trajectories --out-dir data/bc --recompute-obs

출력:
    data/bc/{layout}/pos_{i}/
        obs.npy       # (N, H, W, C) int32
        actions.npy   # (N,) int32
        metadata.json # 에피소드 정보
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# webapp layout 이름 → overcooked-ai layout 이름 매핑
LAYOUT_NAME_MAP = {
    "cramped_room": "cramped_room",
    "asymm_advantages": "asymmetric_advantages",
    "coord_ring": "coordination_ring",
    "forced_coord": "forced_coordination",
    "counter_circuit": "counter_circuit",
}


def load_all_trajectories(traj_dir: str):
    """trajectory 디렉토리의 모든 pickle 파일 로드."""
    traj_path = Path(traj_dir)
    episodes = []
    for pkl_file in sorted(traj_path.rglob("*.pkl")):
        with open(pkl_file, "rb") as f:
            ep = pickle.load(f)
        episodes.append(ep)
    return episodes


def _setup_recompute():
    """obs 재계산에 필요한 모듈 임포트 및 초기화."""
    # webapp/app/game/obs_adapter.py를 import하기 위해 path 추가
    webapp_root = Path(__file__).resolve().parent.parent.parent / "webapp"
    sys.path.insert(0, str(webapp_root))

    from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

    return overcooked_state_to_jaxmarl_obs, OvercookedGridworld, OvercookedState


def _load_mdp_for_layout(layout: str, OvercookedGridworld):
    """JaxMARL OV2 호환 layout으로 OvercookedGridworld 생성.

    webapp의 custom layout 파일을 우선 사용하고,
    없으면 overcooked-ai 기본 layout으로 fallback.
    이렇게 해야 RL 학습/eval (OV2 기준)과 obs shape이 일치함.
    """
    # webapp custom layout 경로 (JaxMARL OV2 grid와 일치하도록 만든 layout 파일)
    webapp_layouts_dir = Path(__file__).resolve().parent.parent.parent / "webapp" / "app" / "game" / "layouts"

    # webapp에서 사용하는 layout 이름 매핑
    WEBAPP_LAYOUT_MAP = {
        "counter_circuit": "jaxmarl_counter_circuit",
        "forced_coord": "jaxmarl_forced_coordination",
        "cramped_room": "jaxmarl_cramped_room",
        "asymm_advantages": "jaxmarl_asymmetric_advantages",
        "coord_ring": "jaxmarl_coordination_ring",
    }

    webapp_name = WEBAPP_LAYOUT_MAP.get(layout)
    if webapp_name:
        layout_file = webapp_layouts_dir / f"{webapp_name}.layout"
        if layout_file.exists():
            import json as _json
            with open(layout_file) as f:
                layout_dict = _json.loads(f.read())
            grid = layout_dict.get("grid", layout_dict.get("layout"))
            if isinstance(grid, str):
                grid = [list(row) for row in grid.strip().split("\n")]
            mdp = OvercookedGridworld.from_grid(
                grid,
                base_layout_params={"start_order_list": None},
            )
            print(f"    [layout] webapp custom layout 사용: {layout_file.name} ({len(grid)}×{len(grid[0])})")
            return mdp

    # fallback: overcooked-ai 기본 layout
    overcooked_layout = LAYOUT_NAME_MAP.get(layout, layout)
    mdp = OvercookedGridworld.from_layout_name(overcooked_layout)
    h, w = mdp.shape
    print(f"    [layout] OV1 기본 layout 사용: {overcooked_layout} ({h}×{w})")
    return mdp


def export_by_position(traj_dir: str, out_dir: str, layout_filter: str = None,
                       recompute_obs: bool = False,
                       algo_exclude=None):
    """trajectory pickle → 레이아웃/포지션별 numpy arrays.

    algo_exclude: 제외할 algo_name 리스트 (예: ["cec"]).
    """
    episodes = load_all_trajectories(traj_dir)
    print(f"총 {len(episodes)}개 에피소드 로드")
    if algo_exclude:
        before = len(episodes)
        episodes = [ep for ep in episodes if ep.get("algo_name") not in set(algo_exclude)]
        print(f"  [algo-exclude {algo_exclude}] {before - len(episodes)}개 제외 → {len(episodes)}개")

    # recompute 모드 설정
    if recompute_obs:
        overcooked_state_to_jaxmarl_obs, OvercookedGridworld, OvercookedState = _setup_recompute()
        mdp_cache = {}  # layout별 mdp 캐싱
        print("  [recompute-obs] state dict에서 obs를 재계산합니다.")

    # 레이아웃 + 포지션별 그룹핑
    # key: (layout, human_player_index)
    grouped = defaultdict(lambda: {"obs": [], "actions": [], "metadata": []})
    skipped_no_state = 0

    for ep in episodes:
        layout = ep.get("layout", "unknown")
        if layout_filter and layout != layout_filter:
            continue

        pos = ep.get("human_player_index", 0)
        key = (layout, pos)

        # recompute 모드: layout별 mdp 준비 (JaxMARL OV2 호환 layout 우선 사용)
        if recompute_obs:
            if layout not in mdp_cache:
                try:
                    mdp_cache[layout] = _load_mdp_for_layout(layout, OvercookedGridworld)
                except Exception as e:
                    print(f"  [경고] layout '{layout}' mdp 생성 실패: {e}")
                    mdp_cache[layout] = None
            mdp = mdp_cache[layout]

        transitions = ep.get("transitions", [])
        for t in transitions:
            action = t.get("action_human")
            if action is None:
                continue

            if recompute_obs:
                # state dict에서 obs 재생성
                state_dict = t.get("state")
                if state_dict is None or mdp is None:
                    skipped_no_state += 1
                    continue
                try:
                    state = OvercookedState.from_dict(state_dict)
                    obs = overcooked_state_to_jaxmarl_obs(state, mdp, pos)
                except Exception as e:
                    skipped_no_state += 1
                    continue
            else:
                # 기존 방식: pickle에 저장된 obs_human 사용
                obs = t.get("obs_human")
                if obs is None:
                    continue

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

    if recompute_obs and skipped_no_state > 0:
        print(f"  [경고] state dict 없거나 복원 실패로 {skipped_no_state}건 스킵")

    if not grouped:
        print("데이터 없음")
        return

    for (layout, pos), data in sorted(grouped.items()):
        if not data["obs"]:
            print(f"  {layout}/pos_{pos}: obs 없음, 스킵")
            continue

        obs_array = np.stack(data["obs"]).astype(np.int32)
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
            "recomputed_obs": recompute_obs,
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
    parser.add_argument("--recompute-obs", action="store_true",
                        help="state dict에서 obs를 재계산 (obs_adapter 버그 수정 후 사용)")
    parser.add_argument("--algo-exclude", nargs="*", default=[],
                        help="제외할 algo_name 리스트 (예: --algo-exclude cec)")
    args = parser.parse_args()
    export_by_position(args.traj_dir, args.out_dir, args.layout,
                       args.recompute_obs, algo_exclude=args.algo_exclude)
