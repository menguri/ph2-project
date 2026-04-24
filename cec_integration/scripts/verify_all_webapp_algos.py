"""webapp 의 모든 algo × layout 조합 로딩 + 첫 inference smoke 테스트.

algo: sp, e3t, fcp, ph2, mep, gamma, cec
layout: cramped_room, asymm_advantages, coord_ring, counter_circuit, forced_coord

각 조합에서:
  1. ModelManager 로드 성공 여부
  2. 더미 obs 로 get_action 호출 → action 0-5 범위 확인 (connection break 안 나는지)

결과: 로드/추론 실패한 조합 리스트.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring",
           "counter_circuit", "forced_coord"]
ALGOS = ["sp", "e3t", "fcp", "ph2", "mep", "gamma", "cec"]


def _build_dummy_obs(layout_name: str, num_ingredients: int = 1):
    """layout 별 dummy OV2-format obs (H, W, 30) 생성."""
    from app.game.engine import _load_custom_layout
    mdp = _load_custom_layout(layout_name)
    h = len(mdp.terrain_mtx)
    w = len(mdp.terrain_mtx[0])
    c = 18 + 4 * (num_ingredients + 2)
    return np.zeros((h, w, c), dtype=np.float32)


def main():
    from app.agent.inference import ModelManager

    results = {}
    models_root = PROJECT_ROOT / "webapp" / "models"

    for layout in LAYOUTS:
        for algo in ALGOS:
            pair = f"{layout:18s} / {algo:6s}"
            algo_dir = models_root / layout / algo
            if not algo_dir.exists():
                results[pair] = "SKIP (no dir)"
                continue
            # run0 우선
            ckpt_candidates = []
            for run_dir in sorted(algo_dir.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith("run"):
                    final = run_dir / "ckpt_final"
                    if final.is_dir():
                        ckpt_candidates.append(final)
                        break
            if not ckpt_candidates:
                results[pair] = "SKIP (no ckpt_final)"
                continue
            ckpt = ckpt_candidates[0]

            # 로드 + 추론
            try:
                mm = ModelManager()
                mm.load(ckpt_path=str(ckpt), algo_name=algo.upper(), seed_id="run0")
                obs = _build_dummy_obs(layout)
                action = mm.get_action(obs, layout_name=layout)
                if isinstance(action, int) and 0 <= action < 6:
                    results[pair] = f"OK  action={action}"
                else:
                    results[pair] = f"FAIL invalid action={action}"
            except Exception as e:
                msg = str(e).splitlines()[0][:120]
                results[pair] = f"FAIL: {msg}"

    # 출력
    print("=" * 70)
    print(f"{'layout/algo':26s}  result")
    print("=" * 70)
    n_ok = n_fail = n_skip = 0
    fail_list = []
    for pair, r in results.items():
        print(f"  {pair}  {r}")
        if r.startswith("OK"):
            n_ok += 1
        elif r.startswith("FAIL"):
            n_fail += 1
            fail_list.append((pair, r))
        else:
            n_skip += 1
    print("=" * 70)
    print(f"OK={n_ok}  FAIL={n_fail}  SKIP={n_skip}")
    if fail_list:
        print("\n실패 항목:")
        for p, r in fail_list:
            print(f"  {p}: {r}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
