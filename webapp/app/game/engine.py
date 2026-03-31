"""
Phase 4: Game Engine — overcooked-ai 환경 래핑 + AI 추론 + trajectory 수집.
"""
import random
import uuid

import json
from pathlib import Path

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from .obs_adapter import overcooked_state_to_jaxmarl_obs
from .action_map import jaxmarl_to_overcooked, OVERCOOKED_AI_ACTIONS
from ..agent.inference import ModelManager
from ..trajectory.recorder import TrajectoryRecorder

# JaxMARL 레이아웃을 overcooked-ai 커스텀 레이아웃으로 등록
LAYOUTS_DIR = Path(__file__).parent / "layouts"

# webapp models 디렉토리의 layout 이름 → overcooked-ai에서 사용할 layout 이름
LAYOUT_NAME_MAP = {
    "cramped_room": "jaxmarl_cramped_room",
    "asymm_advantages": "jaxmarl_asymm_advantages",
    "coord_ring": "jaxmarl_coord_ring",
    "forced_coord": "jaxmarl_forced_coord",
    "counter_circuit": "jaxmarl_counter_circuit",
}


def _load_custom_layout(layout_name: str) -> OvercookedGridworld:
    """JaxMARL layout 파일에서 overcooked-ai OvercookedGridworld 생성."""
    jaxmarl_name = LAYOUT_NAME_MAP.get(layout_name, layout_name)
    layout_file = LAYOUTS_DIR / f"{jaxmarl_name}.layout"
    if layout_file.exists():
        with open(layout_file) as f:
            layout_dict = json.loads(f.read())
        # grid는 "\n"으로 split하여 list of strings로 전달
        grid_str = layout_dict["grid"]
        grid_lines = [line.strip() for line in grid_str.strip().split("\n") if line.strip()]
        mdp = OvercookedGridworld.from_grid(
            grid_lines,
            base_layout_params={
                "start_all_orders": layout_dict.get("start_all_orders", []),
                "rew_shaping_params": layout_dict.get("rew_shaping_params"),
            },
        )
        return _patch_mdp_no_early_cook(mdp)
    # fallback: overcooked-ai 기본 layout
    return _patch_mdp_no_early_cook(OvercookedGridworld.from_layout_name(layout_name))


def _patch_mdp_no_early_cook(mdp: OvercookedGridworld):
    """JaxMARL 호환: interact로 3개 미만 재료 요리 시작 차단.

    overcooked-ai는 재료 1개라도 interact하면 요리 시작.
    JaxMARL은 start_cooking_interaction=False → 3개 찰 때만 자동 시작.
    이 패치로 overcooked-ai에서도 동일하게 동작.
    """
    def patched_soup_to_be_cooked(state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return (
            obj.name == "soup"
            and not obj.is_cooking
            and not obj.is_ready
            and len(obj.ingredients) >= 3  # 원본은 > 0, JaxMARL은 >= 3
        )
    mdp.soup_to_be_cooked_at_location = patched_soup_to_be_cooked
    return mdp


class GameSession:
    """한 에피소드의 게임 세션."""

    def __init__(
        self,
        layout: str,
        model: ModelManager,
        recorder: TrajectoryRecorder,
        participant_id: str,
        episode_length: int = 400,
        human_player_index: int = None,
    ):
        self.layout = layout
        self.model = model
        self.recorder = recorder
        self.participant_id = participant_id
        self.episode_length = episode_length
        self.episode_id = str(uuid.uuid4())

        # human player index 랜덤 배정
        if human_player_index is None:
            self.human_idx = random.randint(0, 1)
        else:
            self.human_idx = human_player_index
        self.ai_idx = 1 - self.human_idx

        # overcooked-ai 환경 (JaxMARL 레이아웃 기준)
        self.mdp = _load_custom_layout(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=episode_length)
        self.state = self.env.state
        self.timestep = 0
        self.score = 0
        self.done = False
        self.collisions = 0
        self.deliveries = 0

        # 모델 hidden state 리셋
        self.model.reset_hidden()

        # trajectory 시작
        self.recorder.start_episode(
            episode_id=self.episode_id,
            participant_id=self.participant_id,
            algo_name=self.model.algo_name,
            seed_id=self.model.seed_id,
            layout=layout,
            human_player_index=self.human_idx,
        )

    def get_init_info(self) -> dict:
        """에피소드 시작 시 클라이언트에 보낼 정보."""
        return {
            "episode_id": self.episode_id,
            "layout": self.layout,
            "human_player_index": self.human_idx,
            "episode_length": self.episode_length,
            "terrain": self.mdp.terrain_mtx,
            "state": self._serialize_state(),
            "score": 0,
            "timestep": 0,
            "done": False,
        }

    def step(self, human_action_idx: int) -> dict:
        """한 timestep 진행."""
        if self.done:
            return {"done": True, "score": self.score, "timestep": self.timestep}

        # AI obs → AI action
        ai_obs = overcooked_state_to_jaxmarl_obs(
            self.state, self.mdp, agent_idx=self.ai_idx
        )
        ai_action_idx = self.model.get_action(ai_obs)

        # human obs (BC 데이터용)
        human_obs = overcooked_state_to_jaxmarl_obs(
            self.state, self.mdp, agent_idx=self.human_idx
        )

        # joint action 구성 (index 순서: [agent_0_action, agent_1_action])
        actions = [None, None]
        actions[self.human_idx] = jaxmarl_to_overcooked(human_action_idx)
        actions[self.ai_idx] = jaxmarl_to_overcooked(ai_action_idx)
        joint_action = tuple(actions)

        # 이전 위치 기록 (충돌 감지용)
        prev_positions = [p.position for p in self.state.players]

        # 환경 step — overcooked-ai 2.0.0: (state, reward, done, info)
        next_state, reward, done, info = self.env.step(joint_action)

        # JaxMARL 호환: 재료 3개 차면 자동 요리 시작 (interact 불필요)
        self._auto_cook_full_pots(next_state)

        # 충돌 감지: 둘 다 이동 시도했는데 둘 다 안 움직인 경우
        new_positions = [p.position for p in next_state.players]
        human_tried_move = (human_action_idx < 4)  # 0~3 = 방향, 4=stay, 5=interact
        ai_tried_move = (ai_action_idx < 4)
        human_stuck = (prev_positions[self.human_idx] == new_positions[self.human_idx])
        ai_stuck = (prev_positions[self.ai_idx] == new_positions[self.ai_idx])
        if human_tried_move and human_stuck and ai_tried_move and ai_stuck:
            self.collisions += 1
        elif human_tried_move and human_stuck and new_positions[self.ai_idx] == prev_positions[self.human_idx]:
            # AI가 human 자리로 들어오려는 시도로 human이 막힘
            self.collisions += 1

        # 배달 감지: reward > 0 이면 배달 성공
        if reward > 0:
            self.deliveries += int(reward / 20)  # overcooked-ai 기본 배달 보상 = 20

        self.score += int(reward)
        self.state = next_state
        self.timestep += 1
        self.done = done or self.timestep >= self.episode_length

        # trajectory 기록
        try:
            state_dict = self.state.to_dict() if hasattr(self.state, "to_dict") else None
        except Exception:
            state_dict = None

        self.recorder.record_step(
            timestep=self.timestep,
            state_dict=state_dict,
            joint_action=[human_action_idx, ai_action_idx],
            reward=reward,
            cumulative_score=self.score,
            obs_human=human_obs,
            action_human=human_action_idx,
        )

        result = {
            "state": self._serialize_state(),
            "score": self.score,
            "timestep": self.timestep,
            "done": self.done,
            "reward": reward,
        }

        if self.done:
            self.recorder.end_episode(self.score)

        return result

    def _auto_cook_full_pots(self, state):
        """JaxMARL 호환: pot에 재료 3개가 차면 interact 없이 자동 요리 시작.

        JaxMARL에서는 auto_cook + grid_update(timer-1)가 같은 스텝에서 일어남.
        overcooked-ai에서는 step_environment_effects가 이미 실행된 후 이 함수가 호출되므로,
        begin_cooking() 후 cook()을 한 번 호출하여 1스텝 오프셋을 보정한다.
        """
        for pos in self.mdp.get_pot_locations():
            if state.has_object(pos):
                obj = state.get_object(pos)
                if (obj.name == "soup"
                    and not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                    obj.begin_cooking()
                    obj.cook()  # 1스텝 보정: JaxMARL과 타이밍 동기화

    def force_end(self):
        """비정상 종료 시 trajectory 저장."""
        if not self.done:
            self.done = True
            self.recorder.end_episode(self.score)

    def _serialize_state(self) -> dict:
        """overcooked-ai state를 클라이언트 렌더링용 JSON으로 변환."""
        players = []
        for p in self.state.players:
            held = None
            if p.held_object is not None:
                held = {"name": p.held_object.name}
                if hasattr(p.held_object, "ingredients"):
                    held["ingredients"] = list(p.held_object.ingredients)
                if hasattr(p.held_object, "cooking_tick"):
                    held["cooking_tick"] = p.held_object.cooking_tick
            players.append({
                "position": list(p.position),
                "orientation": list(p.orientation),
                "held_object": held,
            })

        objects = {}
        for pos, obj in self.state.objects.items():
            obj_data = {"name": obj.name, "position": list(pos)}
            try:
                obj_data["ingredients"] = list(obj.ingredients)
            except (AttributeError, ValueError):
                pass
            try:
                obj_data["cooking_tick"] = obj.cooking_tick
            except (AttributeError, ValueError):
                pass
            try:
                obj_data["is_cooking"] = obj.is_cooking
            except (AttributeError, ValueError):
                pass
            try:
                obj_data["is_ready"] = obj.is_ready
            except (AttributeError, ValueError):
                pass
            try:
                obj_data["cook_time"] = obj.cook_time
            except (AttributeError, ValueError):
                pass
            objects[f"{pos[0]},{pos[1]}"] = obj_data

        return {
            "players": players,
            "objects": objects,
        }
