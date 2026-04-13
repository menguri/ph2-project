"""Webapp CEC 모델 로드 + 추론 검증.

webapp의 ModelManager를 사용하여 CEC 체크포인트 로드 → get_action() 호출.
"""
import sys
import os

# webapp 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))
sys.path.insert(0, PROJECT_ROOT)

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np


def main() -> int:
    print("=" * 60)
    print("Webapp CEC 모델 로드 검증")
    print("=" * 60)

    # 1. ModelManager 로드 테스트
    from app.agent.inference import ModelManager

    # webapp/models/forced_coord/cec/run0/ckpt_final/ 경로
    ckpt_path = os.path.join(PROJECT_ROOT, "webapp", "models", "forced_coord", "cec", "run0", "ckpt_final")

    if not os.path.exists(ckpt_path):
        print(f"[FAIL] 체크포인트 경로 없음: {ckpt_path}")
        return 1

    print(f"\n[1] ModelManager.load(algo_name='CEC')")
    mm = ModelManager()
    try:
        mm.load(ckpt_path=ckpt_path, algo_name="CEC", seed_id="run0")
        print(f"  [PASS] source={mm.source}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        return 1

    # 2. get_action 테스트 (forced_coord obs: 5x5x30)
    print(f"\n[2] get_action(obs, layout_name='forced_coord')")
    dummy_obs = np.zeros((5, 5, 30), dtype=np.float32)
    # 최소한 agent pos 설정 (ch0 = self pos)
    dummy_obs[1, 3, 0] = 1.0  # self at (1,3)
    dummy_obs[2, 1, 8] = 1.0  # other at (2,1)
    # direction (UP)
    dummy_obs[1, 3, 1] = 1.0  # self dir UP
    dummy_obs[2, 1, 9] = 1.0  # other dir UP
    # wall 설정
    for y in range(5):
        for x in range(5):
            if (y == 0) or (y == 4) or (x == 0) or (x == 4):
                dummy_obs[y, x, 16] = 1.0  # wall

    try:
        action = mm.get_action(dummy_obs, layout_name="forced_coord")
        print(f"  [PASS] action={action} (0-5 범위)")
        assert 0 <= action < 6
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 3. 모든 레이아웃 CEC 모델 로드 테스트
    print(f"\n[3] 5개 레이아웃 CEC 모델 로드")
    layouts = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
    layout_obs_shapes = {
        "cramped_room": (4, 5, 30),
        "forced_coord": (5, 5, 30),
        "counter_circuit": (5, 8, 30),
        "coord_ring": (5, 5, 30),
        "asymm_advantages": (5, 9, 30),
    }
    all_pass = True
    for layout in layouts:
        ckpt = os.path.join(PROJECT_ROOT, "webapp", "models", layout, "cec", "run0", "ckpt_final")
        if not os.path.exists(ckpt):
            print(f"  [{layout:20s}] SKIP (경로 없음)")
            continue
        try:
            m = ModelManager()
            m.load(ckpt_path=ckpt, algo_name="CEC", seed_id="run0")
            obs_shape = layout_obs_shapes[layout]
            obs = np.zeros(obs_shape, dtype=np.float32)
            action = m.get_action(obs, layout_name=layout)
            print(f"  [{layout:20s}] PASS (action={action})")
        except Exception as e:
            print(f"  [{layout:20s}] FAIL: {e}")
            all_pass = False

    # 4. 모델 스캔 테스트
    print(f"\n[4] scan_models_dir에서 CEC 감지")
    from app.agent.loader import scan_models_dir
    model_dir = os.path.join(PROJECT_ROOT, "webapp", "models")
    models_by_layout = scan_models_dir(model_dir)
    cec_count = 0
    for layout, models in models_by_layout.items():
        for m in models:
            if m["algo_name"] == "cec":
                cec_count += 1
                print(f"  {layout}/{m['algo_name']}/{m['seed_id']}: {m['ckpt_path'][-40:]}")
    print(f"  CEC 모델 총 {cec_count}개 감지")

    print(f"\n{'=' * 60}")
    print(f"결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
