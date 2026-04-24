"""webapp GameSession CEC V1-engine 통합 smoke test.

1. ModelManager 에 CEC 체크포인트 로드
2. GameSession 생성 — CEC 이라 V1 engine 자동 사용
3. 10 step self-play (human=random, ai=CEC) → reward/state JSON 정상 확인

Run:
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/test_webapp_cec_v1_session.py
"""
import os
import sys

PROJECT_ROOT = "/home/mlic/mingukang/ph2-project"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import numpy as np

from app.agent.inference import ModelManager
from app.game.engine import GameSession
from app.trajectory.recorder import TrajectoryRecorder


LAYOUT = "cramped_room"
NUM_STEPS = 30


class _NoopRecorder:
    """TrajectoryRecorder 대체. 파일 I/O 없음."""
    def start_episode(self, **kwargs):
        pass
    def record_step(self, **kwargs):
        pass
    def end_episode(self, *args, **kwargs):
        pass


def main():
    mm = ModelManager()
    ckpt = os.path.join(PROJECT_ROOT, "webapp", "models", LAYOUT, "cec", "run0", "ckpt_final")
    mm.load(ckpt_path=ckpt, algo_name="CEC", seed_id="run0")
    print(f"Model loaded: source={mm.source}", flush=True)

    session = GameSession(
        layout=LAYOUT, model=mm, recorder=_NoopRecorder(),
        participant_id="smoke", episode_length=NUM_STEPS, human_player_index=0,
    )
    print(f"Session: cec_v1_session={session.cec_v1_session is not None}, "
          f"human={session.human_idx}, ai={session.ai_idx}", flush=True)

    info = session.get_init_info()
    print(f"\nInit info keys: {list(info.keys())}", flush=True)
    print(f"terrain rows={len(info['terrain'])} cols={len(info['terrain'][0])}", flush=True)
    print(f"Initial state.players: {info['state']['players']}", flush=True)
    print(f"Initial state.objects: {info['state']['objects']}", flush=True)

    rng = np.random.RandomState(0)
    for t in range(NUM_STEPS):
        human_act = int(rng.randint(0, 6))
        result = session.step(human_act)
        if t < 3 or result['reward'] > 0 or result['done']:
            print(f"  t={t+1:2d} human_act={human_act} r={result['reward']} "
                  f"score={result['score']} done={result['done']} "
                  f"p0={result['state']['players'][0]['position']} "
                  f"p1={result['state']['players'][1]['position']} "
                  f"objs={len(result['state']['objects'])}",
                  flush=True)
        if result['done']:
            break

    print(f"\nFinal score: {session.score} deliveries={session.deliveries} collisions={session.collisions}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
