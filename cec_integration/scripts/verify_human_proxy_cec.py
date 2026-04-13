"""Human-proxy CEC 14개 체크포인트 로드 + 추론 검증.

forced_coord_9의 모든 *_improved 체크포인트를 load_cec_policies()로 로드.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["JAX_PLATFORMS"] = "cpu"

import importlib.util
import jax
import jax.numpy as jnp
import numpy as np


def _import_policy():
    """human-proxy/code/policy.py를 직접 import (code 모듈 충돌 회피)."""
    # human-proxy/code/ 내 상대 import 해결 위해 path 추가
    hp_code_dir = os.path.join(PROJECT_ROOT, "human-proxy", "code")
    if hp_code_dir not in sys.path:
        sys.path.insert(0, hp_code_dir)
    spec = importlib.util.spec_from_file_location(
        "hp_policy",
        os.path.join(hp_code_dir, "policy.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 60)
    print("Human-proxy CEC 전체 체크포인트 로드 검증")
    print("=" * 60)

    policy_mod = _import_policy()
    load_cec_policies = policy_mod.load_cec_policies
    CECPolicy = policy_mod.CECPolicy

    ckpt_dir = os.path.join(PROJECT_ROOT, "cec_integration", "ckpts", "forced_coord_9")
    layout = "forced_coord"

    print(f"\nckpt_dir: {ckpt_dir}")
    print(f"layout: {layout}")

    # 1. 전체 로드
    print(f"\n[1] load_cec_policies() 실행")
    policies = load_cec_policies(ckpt_dir, layout, stochastic=True)
    print(f"  로드 성공: {len(policies)}개")

    if len(policies) == 0:
        print("  [FAIL] 로드된 policy 없음")
        return 1

    # 2. 각 policy로 추론 테스트
    print(f"\n[2] 각 policy compute_action() 테스트")
    dummy_obs = np.zeros((5, 5, 30), dtype=np.float32)
    dummy_obs[1, 3, 0] = 1.0   # self pos
    dummy_obs[2, 1, 8] = 1.0   # other pos
    dummy_obs[1, 3, 1] = 1.0   # self dir UP
    dummy_obs[2, 1, 9] = 1.0   # other dir UP
    # pot 위치 (forced_coord)
    dummy_obs[0, 3, 18] = 1.0  # pot
    dummy_obs[1, 3, 18] = 1.0  # pot (대략적)
    # wall
    for y in range(5):
        for x in range(5):
            if (y == 0) or (y == 4) or (x == 0) or (x == 4):
                dummy_obs[y, x, 16] = 1.0

    all_pass = True
    rng = jax.random.PRNGKey(0)
    results = []

    for i, policy in enumerate(policies):
        try:
            hstate = policy.init_hstate()
            rng, sub = jax.random.split(rng)
            action, new_hstate, extras = policy.compute_action(
                dummy_obs, done=False, hstate=hstate, key=sub
            )
            assert 0 <= action < 6, f"invalid action: {action}"
            results.append((i, "PASS", action))
        except Exception as e:
            results.append((i, "FAIL", str(e)))
            all_pass = False

    # 결과 테이블
    print(f"\n{'idx':>4s} | {'status':>6s} | {'action':>6s}")
    print("-" * 25)
    for idx, status, info in results:
        print(f"{idx:4d} | {status:>6s} | {info!s:>6s}")

    print(f"\n총 {len(policies)}개 중 {sum(1 for _,s,_ in results if s=='PASS')}개 PASS")

    # 3. 체크포인트 이름 로그
    print(f"\n[3] 체크포인트 상세 로그")
    from pathlib import Path
    ckpt_path = Path(ckpt_dir)
    improved_dirs = sorted([d.name for d in ckpt_path.iterdir() if d.is_dir() and "improved" in d.name])
    print(f"  *_improved 디렉토리 총 {len(improved_dirs)}개:")
    for d in improved_dirs:
        print(f"    {d}")

    print(f"\n{'=' * 60}")
    print(f"결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
