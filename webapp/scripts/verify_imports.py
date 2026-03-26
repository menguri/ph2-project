#!/usr/bin/env python3
"""Phase 1 검증: 모든 핵심 import 및 환경 동작 확인."""
import os
import sys
from pathlib import Path

# webapp 루트 기준으로 경로 설정
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent

# baseline 모델 코드 접근
BASELINE_PATH = PROJECT_ROOT / "baseline"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}")
        if result:
            print(f"        {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


results = []

print("=" * 50)
print("Phase 1 Verification")
print("=" * 50)

# 1. overcooked-ai
print("\n[1] overcooked-ai (overcooked_ai_py)")
results.append(check(
    "import OvercookedGridworld",
    lambda: __import__("overcooked_ai_py.mdp.overcooked_mdp", fromlist=["OvercookedGridworld"]) and None
))
results.append(check(
    "import OvercookedEnv",
    lambda: __import__("overcooked_ai_py.mdp.overcooked_env", fromlist=["OvercookedEnv"]) and None
))
results.append(check(
    "import Action, Direction",
    lambda: __import__("overcooked_ai_py.mdp.actions", fromlist=["Action", "Direction"]) and None
))


def _test_cramped_room():
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    state = mdp.get_standard_start_state()
    h, w = len(mdp.terrain_mtx), len(mdp.terrain_mtx[0])
    n_players = len(state.players)
    return f"Grid: {h}x{w}, Players: {n_players}"


results.append(check("cramped_room env creation + start state", _test_cramped_room))

# 2. JAX / Flax / Orbax
print("\n[2] JAX / Flax / Orbax")


def _test_jax():
    import jax
    devices = jax.devices()
    return f"JAX {jax.__version__}, devices: {[str(d) for d in devices]}"


results.append(check("JAX (CPU)", _test_jax))
results.append(check(
    "flax.linen",
    lambda: __import__("flax.linen", fromlist=["nn"]) and None
))
results.append(check(
    "orbax.checkpoint",
    lambda: __import__("orbax.checkpoint", fromlist=["PyTreeCheckpointer"]) and None
))
results.append(check(
    "distrax",
    lambda: __import__("distrax") and None
))

# 3. JaxMARL
print("\n[3] JaxMARL")
results.append(check(
    "jaxmarl overcooked_v2 Actions",
    lambda: __import__("jaxmarl.environments.overcooked_v2.common", fromlist=["Actions"]) and None
))

# 4. ph2 overcooked_v2_experiments (editable install)
print("\n[4] ph2 overcooked_v2_experiments")


def _test_ph2_models():
    from overcooked_v2_experiments.ppo.models.model import get_actor_critic
    return f"get_actor_critic loaded from {get_actor_critic.__module__}"


results.append(check("ph2 get_actor_critic", _test_ph2_models))

# 5. baseline ActorCriticRNN (via importlib)
print("\n[5] baseline model code (importlib)")


def _test_baseline_models():
    # baseline의 rnn.py가 relative import를 사용하므로
    # 단순 importlib로는 로드 불가. sys.path + sys.modules swap으로 처리.
    # Phase 2에서 정식 구현. 여기서는 파일 존재 + 클래스 정의 확인만.
    baseline_rnn_path = PROJECT_ROOT / "baseline" / "overcooked_v2_experiments" / "ppo" / "models" / "rnn.py"
    if not baseline_rnn_path.exists():
        raise FileNotFoundError(f"{baseline_rnn_path} not found")
    content = baseline_rnn_path.read_text()
    assert "class ActorCriticRNN" in content, "ActorCriticRNN class not found"
    assert "class ScannedRNN" in content, "ScannedRNN class not found"

    # baseline store.py도 확인
    baseline_store = PROJECT_ROOT / "baseline" / "overcooked_v2_experiments" / "ppo" / "utils" / "store.py"
    assert baseline_store.exists(), f"{baseline_store} not found"
    return f"baseline model files verified at {baseline_rnn_path.parent}"


results.append(check("baseline ActorCriticRNN", _test_baseline_models))

# 6. Web framework
print("\n[6] Web framework")
results.append(check(
    "fastapi",
    lambda: __import__("fastapi") and None
))
results.append(check(
    "uvicorn",
    lambda: __import__("uvicorn") and None
))

# 7. Database
print("\n[7] Database")
results.append(check(
    "sqlalchemy",
    lambda: __import__("sqlalchemy") and None
))
results.append(check(
    "aiosqlite",
    lambda: __import__("aiosqlite") and None
))

# 8. Config
print("\n[8] Configuration")
results.append(check(
    "yaml",
    lambda: __import__("yaml") and None
))


def _test_config_load():
    import yaml
    config_path = WEBAPP_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return f"Loaded config with {len(config)} top-level keys"


results.append(check("config.yaml load", _test_config_load))

# Summary
print("\n" + "=" * 50)
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} passed")
if all(results):
    print("Phase 1 verification PASSED")
else:
    print("Phase 1 verification FAILED")
sys.exit(0 if all(results) else 1)
