#!/usr/bin/env python3
"""
모든 (layout × algo) 조합에서 모델 로드 + GameSession step 동작 검증.

확인:
  1. 모델 로드 성공
  2. 20 스텝 step() 정상 실행
  3. 새 객관 지표 (role_specialization, idle_time_ratio, task_events) 접근 가능

사용법:
    cd webapp && JAX_PLATFORMS=cpu python scripts/test_all_algos_step.py
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

WEBAPP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBAPP_ROOT))

import yaml
import random

from app.agent.inference import ModelManager
from app.agent.loader import scan_models_dir
from app.game.engine import GameSession
from app.trajectory.recorder import TrajectoryRecorder


def main():
    with open(WEBAPP_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_dir = WEBAPP_ROOT / cfg["agent"]["model_dir"]
    algo_filter = cfg["agent"].get("algo_filter")

    models_by_layout = scan_models_dir(str(model_dir))
    if algo_filter:
        models_by_layout = {
            L: [m for m in mods if m["algo_name"] in algo_filter]
            for L, mods in models_by_layout.items()
        }

    # Each (layout, algo) pair: pick 1 run
    test_targets = []
    for layout in sorted(models_by_layout.keys()):
        by_algo = {}
        for m in models_by_layout[layout]:
            by_algo.setdefault(m["algo_name"], []).append(m)
        for algo in sorted(by_algo.keys()):
            test_targets.append((layout, algo, by_algo[algo][0]))

    print(f"Testing {len(test_targets)} (layout, algo) pairs\n")

    results = []
    for layout, algo, minfo in test_targets:
        tag = f"{layout}/{algo}/{minfo['seed_id']}"
        try:
            mm = ModelManager()
            mm.load(
                ckpt_path=minfo["ckpt_path"],
                policy_source=cfg["agent"].get("policy_source", "params"),
                stochastic=cfg["agent"].get("stochastic", True),
                algo_name=algo,
                seed_id=minfo["seed_id"],
            )

            recorder = TrajectoryRecorder(
                save_dir=str(WEBAPP_ROOT / "data/_test_trajectories"),
                save_obs=False,
                save_state=False,
            )
            session = GameSession(
                layout=layout,
                model=mm,
                recorder=recorder,
                participant_id="test_probe",
                episode_length=40,
                human_player_index=0,
            )
            _ = session.get_init_info()

            rng = random.Random(hash(tag) & 0xFFFFFFFF)
            for _ in range(20):
                a = rng.randint(0, 5)
                r = session.step(a)
                if r["done"]:
                    break

            rsi = session.role_specialization
            idle = session.idle_time_ratio
            te_sum = sum(h + a for h, a in session._task_events.values())

            assert 0.0 <= rsi <= 1.0
            assert 0.0 <= idle <= 1.0

            print(f"  OK  {tag:60s}  score={session.score:3d}  "
                  f"col={session.collisions:2d}  del={session.deliveries:2d}  "
                  f"RSI={rsi:.2f}  idle={idle:.2f}  tasks={te_sum}")
            results.append((tag, True, None))
        except Exception as e:
            tb = traceback.format_exc()
            short = f"{type(e).__name__}: {e}"
            print(f"  FAIL {tag:60s}  {short}")
            print(tb)
            results.append((tag, False, short))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== Summary: {n_ok}/{len(results)} passed ===")
    fails = [r for r in results if not r[1]]
    if fails:
        print("\nFailures:")
        for tag, _, err in fails:
            print(f"  - {tag}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
