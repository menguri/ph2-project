"""
Phase 4: Game Engine — overcooked-ai 환경 래핑 + AI 추론 + trajectory 수집.

CEC 모델은 V1 Overcooked engine (CEC 가 학습한 env) 을 primary 로 사용.
  - UI 렌더링: V1 state → overcooked-ai 호환 JSON (extras 는 wall 로 숨김 → 사용자 시점에는 OV2 cramped_room 과 동일한 모양).
  - CEC 입력: V1 env.get_obs 네이티브 (9, 9, 26).
  - Human BC 궤적: V1 state → OV2 obs (H, W, 30) via V1StateToOV2ObsAdapter.
비-CEC 모델은 기존 overcooked-ai 경로 유지.
"""
import os
import sys
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

# cec_integration 경로 확보 (webapp → project root → cec_integration)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# CEC 전용: V1 engine 세션 지원 레이아웃. 모두 CEC_LAYOUTS[_9] 기반으로 돌고
# terrain 도 V1 layout 기반으로 UI 에 전달 (4 지원 layout 은 OV2 native 와 동일,
# asymm_advantages 는 CEC template 기반 5x9 배치로 표시).
_CEC_V1_SUPPORTED_LAYOUTS = {"cramped_room", "coord_ring", "forced_coord", "counter_circuit", "asymm_advantages"}

# OV2 (frontend/human) action semantic → V1 action semantic 변환.
# V1 Actions enum 라벨이 DIR_TO_VEC 인덱스와 뒤바뀌어 있어서 physical direction 을 맞추려면
# remap 필요. V1 DIR_TO_VEC = [NORTH/UP, SOUTH/DOWN, EAST/RIGHT, WEST/LEFT].
# OV2 action: 0=right, 1=down, 2=left, 3=up, 4=stay, 5=interact.
# 같은 physical direction 의 V1 action index:
#   OV2 0 (RIGHT/EAST) → V1 2
#   OV2 1 (DOWN/SOUTH) → V1 1
#   OV2 2 (LEFT/WEST)  → V1 3
#   OV2 3 (UP/NORTH)   → V1 0
#   OV2 4 (stay)       → V1 4
#   OV2 5 (interact)   → V1 5
_OV2_TO_V1_ACTION = (2, 1, 3, 0, 4, 5)

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

        # overcooked-ai 환경 (JaxMARL 레이아웃 기준) — terrain 렌더링 & 비-CEC 경로에 사용
        self.mdp = _load_custom_layout(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=episode_length)
        self.state = self.env.state
        self.timestep = 0
        self.score = 0
        self.done = False
        self.collisions = 0
        self.deliveries = 0
        # 객관 지표 추적 (H1/H3/H4)
        self.ai_idle_steps = 0                    # AI stay + stuck 합산
        # per-task, per-agent 카운트 — [human_count, ai_count]
        # task 종류: onion_pickup, plate_pickup, pot_deposit, soup_pickup, delivery
        self._task_events = {
            "onion_pickup": [0, 0],
            "plate_pickup": [0, 0],
            "pot_deposit": [0, 0],
            "soup_pickup": [0, 0],
            "delivery": [0, 0],
        }

        # CEC 모델 + 지원 레이아웃이면 V1 engine session 을 primary 로 사용.
        # CEC 가 학습한 V1 Overcooked + CEC_LAYOUTS[_9] 에서 돌되, UI 에는 OV2 외관 그대로.
        self.cec_v1_session = None
        if (getattr(self.model, "source", None) == "cec"
                and layout in _CEC_V1_SUPPORTED_LAYOUTS):
            from cec_integration.webapp_v1_engine_helpers import V1EngineSession
            self.cec_v1_session = V1EngineSession(
                layout=layout, max_steps=episode_length, seed=random.randint(0, 2**31 - 1),
            )
            self.cec_v1_session.reset()

        # 모델 hidden state 리셋
        self.model.reset_hidden()

        # trajectory 시작 — CEC V1 engine 세션에서는 사용자 trajectory 저장하지 않음
        if self.cec_v1_session is None:
            self.recorder.start_episode(
                episode_id=self.episode_id,
                participant_id=self.participant_id,
                algo_name=self.model.algo_name,
                seed_id=self.model.seed_id,
                layout=layout,
                human_player_index=self.human_idx,
            )

    def warmup(self) -> None:
        """JAX JIT compile을 미리 트리거 (첫 step lag 방지).

        game loop 시작 전에 호출. env state 는 변경되지 않고,
        forward 로 인해 변동된 model hidden state 만 다시 초기화한다.
        실패해도 게임 진행에는 영향 없도록 예외는 삼킨다.
        """
        try:
            if self.cec_v1_session is not None:
                # CEC V1 경로 — _step_cec_v1 에서 실제로 호출하는 함수와 **동일 경로** 로 warmup
                # (다른 함수로 warmup 하면 JIT trace 가 달라져 첫 step 에서 재컴파일 발생.
                #  이전 버전에서 CEC 만 3-2-1 끝나고도 멈췄다 파바박 움직인 원인.)
                from cec_integration.webapp_v1_engine_helpers import UI_PLAYER_TO_V1_SLOT
                sess = self.cec_v1_session
                ui_map = UI_PLAYER_TO_V1_SLOT.get(self.layout, {0: 0, 1: 1})
                ai_v1_slot = ui_map[self.ai_idx]
                v1_obs = sess.get_cec_obs_v1(agent_idx=ai_v1_slot)
                self.model.get_action_cec_v1_obs(v1_obs, cec_slot=ai_v1_slot)
            else:
                ai_obs = overcooked_state_to_jaxmarl_obs(
                    self.state, self.mdp, agent_idx=self.ai_idx
                )
                self.model.get_action(ai_obs, layout_name=self.layout)
        except Exception as e:
            print(f"[warmup] skipped due to: {e}")
        finally:
            # warmup 에서 변경된 hidden state 를 다시 리셋
            self.model.reset_hidden()

    def get_init_info(self) -> dict:
        """에피소드 시작 시 클라이언트에 보낼 정보.

        CEC 세션이어도 **4개 aligned layout (cramped/coord/forced/counter)** 는
        V1 reachable obj 가 OV2 terrain 과 1:1 일치하므로 OV2 terrain 을 그대로 보여준다
        (=비-CEC 알고리즘과 같은 UI). V1 extras 는 OV2 terrain 기준으로도 wall cell 이라
        사용자 화면에 뜨지 않고 CEC engine 내부에서 물리적으로도 unreachable 이라 영향 없음.
        asymm_advantages 만 V1 과 OV2 레이아웃이 구조적으로 달라 V1 terrain 을 사용.
        """
        if self.cec_v1_session is not None and self.layout == "asymm_advantages":
            terrain = self.cec_v1_session.build_terrain_mtx(extras_as_wall=True)
        else:
            terrain = self.mdp.terrain_mtx
        return {
            "episode_id": self.episode_id,
            "layout": self.layout,
            "human_player_index": self.human_idx,
            "episode_length": self.episode_length,
            "terrain": terrain,
            "state": self._serialize_state(),
            "score": 0,
            "timestep": 0,
            "done": False,
        }

    def step(self, human_action_idx: int) -> dict:
        """한 timestep 진행."""
        if self.done:
            return {"done": True, "score": self.score, "timestep": self.timestep}

        # CEC + V1 engine 경로 — 완전히 분리된 스텝 로직
        if self.cec_v1_session is not None:
            return self._step_cec_v1(human_action_idx)

        # AI obs → AI action
        # 비-CEC 경로: PH2/SP/E3T/FCP/MEP. 기존 OV2-style JaxMARL obs 경로.
        ai_obs = overcooked_state_to_jaxmarl_obs(
            self.state, self.mdp, agent_idx=self.ai_idx
        )
        ai_action_idx = self.model.get_action(ai_obs, layout_name=self.layout)

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
        # 이전 보유 물품 (task 이벤트 감지용)
        prev_held = [self._held_name(p) for p in self.state.players]
        # 이전 pot 재료 수 (pot_deposit 감지용)
        prev_pot_counts = self._pot_ingredient_counts(self.state)

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

        # AI idle 추적: stay 이거나 이동 시도했는데 못 움직인 경우
        if (not ai_tried_move) or ai_stuck:
            # interact(5) 는 생산적 행동일 수 있으므로 제외
            if ai_action_idx != 5:
                self.ai_idle_steps += 1

        # Task event 감지 (per-agent count)
        self._record_task_events(
            prev_held=prev_held,
            next_state=next_state,
            prev_pot_counts=prev_pot_counts,
            actions=[human_action_idx, ai_action_idx],
            reward=reward,
        )

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

    def _step_cec_v1(self, human_action_idx: int) -> dict:
        """CEC + V1 engine primary step.

        UI slot (OV2 convention) ↔ V1 engine slot 매핑 을 `UI_PLAYER_TO_V1_SLOT` 로 적용해
        사용자에게는 이전 UX (시작 위치 / chef 순서) 가 그대로 유지되도록 한다.

        흐름:
          1. UI human_idx/ai_idx → V1 slot 변환.
          2. V1 state → CEC 네이티브 obs (AI 의 V1 slot 관점).
          3. model.get_action_cec_v1_obs → ai_action_idx.
          4. V1 state → OV2 obs (human 의 V1 slot 관점) for BC trajectory.
          5. V1EngineSession.step(action_dict) — V1 slot 기준 joint action 적용.
          6. 점수/충돌/배달 집계, trajectory 기록, JSON 응답.
        """
        sess = self.cec_v1_session

        # UI slot → V1 slot 매핑 (cramped_room 등 layout 에서 swap 필요)
        from cec_integration.webapp_v1_engine_helpers import UI_PLAYER_TO_V1_SLOT
        ui_map = UI_PLAYER_TO_V1_SLOT.get(self.layout, {0: 0, 1: 1})
        human_v1_slot = ui_map[self.human_idx]
        ai_v1_slot = ui_map[self.ai_idx]

        # (1) CEC 네이티브 obs — AI 의 V1 slot 관점
        v1_obs = sess.get_cec_obs_v1(agent_idx=ai_v1_slot)

        # (2) AI action — CEC runtime slot = V1 slot
        ai_action_idx = self.model.get_action_cec_v1_obs(v1_obs, cec_slot=ai_v1_slot)

        # (3) CEC V1 세션에서는 사용자 trajectory 저장하지 않으므로 human_obs 합성 생략.

        # 이전 위치 기록 (충돌 감지용) — V1 agent_pos 에서
        prev_positions = [tuple(np.asarray(sess.state.agent_pos)[i]) for i in range(2)]

        # (4) V1 engine step
        # human_action_idx: frontend 에서 OV2 semantic → V1 semantic remap
        # ai_action_idx: CEC 는 V1 학습 분포 그대로 출력 → remap 없이 V1 slot 에 전달
        human_v1 = _OV2_TO_V1_ACTION[int(human_action_idx)]
        action_dict = {
            f"agent_{human_v1_slot}": int(human_v1),
            f"agent_{ai_v1_slot}": int(ai_action_idx),
        }
        _, reward, done = sess.step(action_dict)

        # 충돌 감지
        new_positions = [tuple(np.asarray(sess.state.agent_pos)[i]) for i in range(2)]
        human_tried_move = (human_action_idx < 4)
        ai_tried_move = (ai_action_idx < 4)
        # V1 slot 기준 인덱싱 (human_idx/ai_idx 는 UI convention)
        human_stuck = (prev_positions[human_v1_slot] == new_positions[human_v1_slot])
        ai_stuck = (prev_positions[ai_v1_slot] == new_positions[ai_v1_slot])
        if human_tried_move and human_stuck and ai_tried_move and ai_stuck:
            self.collisions += 1
        elif human_tried_move and human_stuck and new_positions[ai_v1_slot] == prev_positions[human_v1_slot]:
            self.collisions += 1

        # 배달 감지 (V1 delivery reward = 20)
        if reward > 0:
            self.deliveries += int(reward / 20)

        self.score += int(reward)
        self.timestep += 1
        self.done = done or self.timestep >= self.episode_length

        # CEC V1 engine 세션에서는 사용자 trajectory 저장하지 않음.
        # (human_obs 는 inference 용으로 이미 합성했지만 파일로 쓰지는 않음)

        result = {
            "state": self._serialize_state_cec_v1(),
            "score": self.score,
            "timestep": self.timestep,
            "done": self.done,
            "reward": reward,
        }
        return result

    def _serialize_state_cec_v1(self) -> dict:
        """CEC V1 engine session 의 state → webapp 클라이언트 JSON.

        V1EngineSession.get_webapp_state_json() 이 overcooked-ai _serialize_state 와
        동일한 스키마로 변환해준다. extras 는 내부에서 wall 로 숨겨져 objects 에 안 들어감.
        """
        return self.cec_v1_session.get_webapp_state_json()

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
                    # cook() 호출 제거 — V1 타이밍 맞추기 위한 +1 tick 보정이었으나
                    # 실제로 over-correction 하던 코드. 제거 후 begin_cooking() 만 호출하면
                    # cooking_tick=0 으로 시작하고 overcooked-ai step_environment_effects 가
                    # 다음 step 에 자연스럽게 cook() 를 호출한다.

    # ===== 객관 지표 헬퍼 =====
    @staticmethod
    def _held_name(player):
        """플레이어가 들고 있는 물품명 (없으면 None)."""
        if player.held_object is None:
            return None
        return getattr(player.held_object, "name", None)

    def _pot_ingredient_counts(self, state) -> dict:
        """각 pot 위치별 재료 개수 (cooking/ready 이후는 제외하여 deposit 과 혼동 방지)."""
        counts = {}
        for pos in self.mdp.get_pot_locations():
            if state.has_object(pos):
                obj = state.get_object(pos)
                if obj.name == "soup" and not obj.is_cooking and not obj.is_ready:
                    counts[pos] = len(obj.ingredients)
                else:
                    counts[pos] = None  # cooking/ready → deposit 불가 상태
            else:
                counts[pos] = 0
        return counts

    def _record_task_events(self, prev_held, next_state, prev_pot_counts, actions, reward):
        """Task 이벤트를 각 agent 의 per-task count 로 누적.

        task 정의:
          - onion_pickup: 빈 손 → onion
          - plate_pickup: 빈 손 → dish
          - soup_pickup:  dish → soup
          - pot_deposit:  pot 의 onion ingredient 수 증가 (agent 는 가장 가까이 있던 쪽으로 귀속)
          - delivery:     soup → None & reward>0
        """
        new_held = [self._held_name(p) for p in next_state.players]
        for idx in range(2):
            role = 0 if idx == self.human_idx else 1  # 0=human, 1=AI
            before, after = prev_held[idx], new_held[idx]
            if before is None and after == "onion":
                self._task_events["onion_pickup"][role] += 1
            elif before is None and after == "dish":
                self._task_events["plate_pickup"][role] += 1
            elif before == "dish" and after == "soup":
                self._task_events["soup_pickup"][role] += 1
            elif before == "soup" and after is None and reward > 0:
                self._task_events["delivery"][role] += 1

        # pot_deposit: 재료 수 증가분 → interact 하며 인접한 agent 에게 귀속
        new_pot_counts = self._pot_ingredient_counts(next_state)
        for pos, prev_c in prev_pot_counts.items():
            new_c = new_pot_counts.get(pos)
            if prev_c is None or new_c is None:
                continue
            if new_c > prev_c:
                # interact(5) 를 쏜 agent 중 pot 과 인접한 쪽으로 귀속 (대부분 1명)
                diff = new_c - prev_c
                candidates = []
                for idx in range(2):
                    if actions[idx] != 5:
                        continue
                    p = self.state.players[idx]
                    # interact 로 놓는 대상은 바라보는 방향의 타일
                    fx = p.position[0] + p.orientation[0]
                    fy = p.position[1] + p.orientation[1]
                    if (fx, fy) == pos:
                        candidates.append(idx)
                # 후보가 없으면 (timing 예외) 가장 가까운 agent 선택
                if not candidates:
                    for idx in range(2):
                        p = next_state.players[idx]
                        dx = abs(p.position[0] - pos[0])
                        dy = abs(p.position[1] - pos[1])
                        if dx + dy <= 1:
                            candidates.append(idx)
                if candidates:
                    idx = candidates[0]
                    role = 0 if idx == self.human_idx else 1
                    self._task_events["pot_deposit"][role] += diff

    @property
    def role_specialization(self) -> float:
        """역할 분화 지수 (H4 핵심).

        각 task 유형별로: overlap = 2 * min(h, a), total = h + a.
        RSI = 1 - Σoverlap / Σtotal  → 0 = 완전 중복, 1 = 완전 분리.
        총 task 가 0 이면 정의 불가 → 1.0 반환.
        """
        total = 0
        overlap = 0
        for counts in self._task_events.values():
            h, a = counts
            total += h + a
            overlap += 2 * min(h, a)
        if total == 0:
            return 1.0
        return 1.0 - overlap / total

    @property
    def idle_time_ratio(self) -> float:
        """AI 비생산 timestep 비율 (H1 sanity)."""
        if self.timestep == 0:
            return 0.0
        return self.ai_idle_steps / self.timestep

    def force_end(self):
        """비정상 종료 시 trajectory 저장 (CEC V1 세션에서는 저장 안 함)."""
        if not self.done:
            self.done = True
            if self.cec_v1_session is None:
                self.recorder.end_episode(self.score)

    def _serialize_state(self) -> dict:
        """State → 클라이언트 렌더링용 JSON.

        CEC V1 engine 세션이면 V1 state 기반, 아니면 overcooked-ai state 기반.
        둘 다 동일한 schema (players + objects) 를 반환해서 frontend 는 구분 불필요.
        """
        if self.cec_v1_session is not None:
            return self._serialize_state_cec_v1()
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
