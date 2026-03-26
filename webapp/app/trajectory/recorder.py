"""
Phase 4: Trajectory Recorder — 에피소드 동안 매 timestep 데이터 수집.
"""
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


class TrajectoryRecorder:
    """에피소드 trajectory를 메모리에 쌓다가 종료 시 pickle 저장."""

    def __init__(self, save_dir: str, save_obs: bool = True, save_state: bool = True):
        self.save_dir = Path(save_dir)
        self.save_obs = save_obs
        self.save_state = save_state
        self._episode_data = None

    def start_episode(
        self,
        episode_id: str,
        participant_id: str,
        algo_name: str,
        seed_id: str,
        layout: str,
        human_player_index: int,
    ):
        self._episode_data = {
            "episode_id": episode_id,
            "participant_id": participant_id,
            "algo_name": algo_name,
            "seed_id": seed_id,
            "layout": layout,
            "human_player_index": human_player_index,
            "timestamp": datetime.utcnow().isoformat(),
            "final_score": 0,
            "episode_length": 0,
            "transitions": [],
        }

    def record_step(
        self,
        timestep: int,
        state_dict: Optional[dict],
        joint_action: list,
        reward: float,
        cumulative_score: int,
        obs_human: Optional[np.ndarray],
        action_human: int,
    ):
        if self._episode_data is None:
            return

        transition = {
            "timestep": timestep,
            "joint_action": joint_action,
            "reward": reward,
            "cumulative_score": cumulative_score,
            "action_human": action_human,
        }
        if self.save_state and state_dict is not None:
            transition["state"] = state_dict
        if self.save_obs and obs_human is not None:
            transition["obs_human"] = obs_human

        self._episode_data["transitions"].append(transition)

    def end_episode(self, final_score: int) -> Optional[str]:
        """에피소드 종료. pickle 저장 후 파일 경로 반환."""
        if self._episode_data is None:
            return None

        self._episode_data["final_score"] = final_score
        self._episode_data["episode_length"] = len(self._episode_data["transitions"])

        # 저장 경로: save_dir/{participant_id}/{episode_id}.pkl
        participant_dir = self.save_dir / self._episode_data["participant_id"]
        participant_dir.mkdir(parents=True, exist_ok=True)
        filepath = participant_dir / f"{self._episode_data['episode_id']}.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(self._episode_data, f)

        saved_path = str(filepath)
        self._episode_data = None
        return saved_path
