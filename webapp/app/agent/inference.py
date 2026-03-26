"""
Phase 2: 모델 추론 — ActorCriticRNN forward pass on CPU.
baseline/ph2 모델 모두 지원.
"""
import importlib.util
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .loader import load_checkpoint_cpu, detect_model_source, select_params

# ph2 모델 (venv에 editable install)
from overcooked_v2_experiments.ppo.models.model import (
    get_actor_critic as ph2_get_actor_critic,
    initialize_carry as ph2_initialize_carry,
)

# baseline 모델 — importlib로 로드 (relative import 문제 방지를 위해 sys.path 활용)
# main.py에서 baseline/ 경로를 sys.path에 추가했으므로
# baseline의 overcooked_v2_experiments를 직접 import하면 ph2와 충돌.
# → baseline 모델은 파일 경로 기반으로 로드.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASELINE_MODELS_DIR = PROJECT_ROOT / "baseline" / "overcooked_v2_experiments" / "ppo" / "models"

_baseline_modules = {}


def _load_baseline_module(name: str):
    """baseline 모델 모듈을 importlib로 로드 (캐시)."""
    if name in _baseline_modules:
        return _baseline_modules[name]

    file_path = BASELINE_MODELS_DIR / f"{name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Baseline model file not found: {file_path}")

    # 먼저 의존 모듈들을 로드 (abstract, common, e3t 순서)
    deps = {
        "rnn": ["abstract", "common", "e3t"],
        "model": ["abstract", "common", "e3t", "rnn"],
    }
    for dep in deps.get(name, []):
        if dep not in _baseline_modules:
            _load_baseline_module(dep)

    spec = importlib.util.spec_from_file_location(
        f"baseline_models.{name}",
        str(file_path),
        submodule_search_locations=[str(BASELINE_MODELS_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)

    # relative import를 위해 parent package 시뮬레이션
    import sys
    pkg_name = f"baseline_models"
    if pkg_name not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                pkg_name,
                str(BASELINE_MODELS_DIR / "__init__.py"),
                submodule_search_locations=[str(BASELINE_MODELS_DIR)],
            )
        )
        sys.modules[pkg_name] = pkg

    mod.__package__ = pkg_name
    sys.modules[f"baseline_models.{name}"] = mod
    spec.loader.exec_module(mod)
    _baseline_modules[name] = mod
    return mod


class ModelManager:
    """AI agent 추론 관리자. checkpoint 로드 → get_action() API."""

    def __init__(self):
        self.network = None
        self.params = None
        self.hidden = None
        self.source = None  # "baseline" or "ph2"
        self.config = None
        self.algo_name = None
        self.seed_id = None
        self.stochastic = True
        self._rng = jax.random.PRNGKey(0)
        self._apply_fn = None

    def load(
        self,
        ckpt_path: str,
        policy_source: str = "params",
        stochastic: bool = True,
        algo_name: str = "",
        seed_id: str = "",
    ):
        """checkpoint 로드 + 모델 초기화."""
        self.stochastic = stochastic
        self.algo_name = algo_name
        self.seed_id = seed_id

        ckpt = load_checkpoint_cpu(ckpt_path)
        self.config = ckpt["config"]

        # PH2 모델은 ind policy 사용 (cross-play 기준)
        source = detect_model_source(self.config)
        if source == "ph2":
            raw_params = select_params(ckpt, "ind")
        else:
            raw_params = select_params(ckpt, policy_source)

        # params 구조: {"params": {...actual_params...}} 형태일 수 있음
        if isinstance(raw_params, dict) and "params" in raw_params and len(raw_params) == 1:
            self.params = raw_params["params"]
        else:
            self.params = raw_params

        self.source = source
        self._build_network()
        self._make_forward_fn()
        self.reset_hidden()

    def _build_network(self):
        """config에 맞는 네트워크 구축."""
        if self.source == "ph2":
            # ind policy용: blocked_encoder 비활성화
            cfg = dict(self.config)
            has_blocked = "blocked_encoder" in self.params
            if not has_blocked:
                cfg["PH1_ENABLED"] = False
                cfg["PH1_STATE_MODE"] = False
            self.network = ph2_get_actor_critic(cfg)
        else:
            # baseline 네트워크: ph2 코드로도 로드 가능하지만 param 키 다름
            # → baseline 모델 코드를 사용해야 함
            # 하지만 baseline의 relative import 문제로 직접 로드가 복잡
            # → 대안: ph2의 ActorCriticRNN을 사용하되, param 키를 리매핑
            self.network = self._build_baseline_network_via_ph2()

    def _build_baseline_network_via_ph2(self):
        """baseline param 키를 ph2 키로 리매핑 후 ph2 네트워크 사용."""
        # baseline params: CNN_0, LayerNorm_0, ScannedRNN_0, Dense_0~3, (predictor)
        # ph2 params: shared_encoder, shared_encoder_ln, ScannedRNN_0, Dense_0~3, (predictor)
        remap = {
            "CNN_0": "shared_encoder",
            "LayerNorm_0": "shared_encoder_ln",
        }
        remapped = {}
        for k, v in self.params.items():
            new_key = remap.get(k, k)
            remapped[new_key] = v
        self.params = remapped

        # baseline config를 ph2 형식으로 변환
        model_config = dict(self.config.get("model", {}))
        # baseline은 USE_PREDICTION, ph2는 ACTION_PREDICTION
        if self.config.get("USE_PREDICTION", False):
            model_config["ACTION_PREDICTION"] = True
        else:
            model_config["ACTION_PREDICTION"] = False

        ph2_config = dict(self.config)
        ph2_config["model"] = model_config
        ph2_config["ACTION_PREDICTION"] = model_config["ACTION_PREDICTION"]
        ph2_config["TRANSFORMER_ACTION"] = False
        self.config = ph2_config

        return ph2_get_actor_critic(ph2_config)

    def reset_hidden(self):
        """에피소드 시작 시 GRU hidden state 리셋."""
        if self.config is None:
            return
        self.hidden = ph2_initialize_carry(self.config, batch_size=1)
        self._rng = jax.random.PRNGKey(np.random.randint(0, 2**31))

    def _make_forward_fn(self):
        """네트워크별 JIT forward 함수 생성."""
        network = self.network
        # ind policy는 blocked_encoder가 없으므로 blocked_states 불필요
        # blocked_encoder가 params에 있는 경우에만 dummy blocked_states 전달
        has_blocked = "blocked_encoder" in self.params

        if has_blocked:
            ph1_k = int(self.config.get("PH1_MAX_PENALTY_COUNT", 1))
            @jax.jit
            def _forward(params, hidden, obs, done):
                T, B = obs.shape[:2]
                H, W, C = obs.shape[2:]
                blocked = jnp.zeros((T, B, ph1_k, H, W, C), dtype=obs.dtype)
                x = (obs, done)
                new_hidden, pi, value, extras = network.apply(
                    {"params": params}, hidden, x,
                    blocked_states=blocked,
                )
                return new_hidden, pi.logits
        else:
            @jax.jit
            def _forward(params, hidden, obs, done):
                x = (obs, done)
                new_hidden, pi, value, extras = network.apply(
                    {"params": params}, hidden, x
                )
                return new_hidden, pi.logits

        self._apply_fn = _forward

    def get_action(self, obs: np.ndarray) -> int:
        """
        obs: (H, W, C) numpy array → action: int (0-5)
        """
        # (H,W,C) → (1, 1, H, W, C) for (T=1, B=1)
        obs_jax = jnp.array(obs, dtype=jnp.float32)[None, None, ...]
        done_jax = jnp.zeros((1, 1), dtype=jnp.bool_)

        new_hidden, logits = self._apply_fn(
            self.params, self.hidden, obs_jax, done_jax
        )
        self.hidden = new_hidden

        # logits: (1, 1, 6)
        logits_2d = logits[0, 0]  # (6,)

        if self.stochastic:
            self._rng, subkey = jax.random.split(self._rng)
            action = jax.random.categorical(subkey, logits_2d).item()
        else:
            action = jnp.argmax(logits_2d).item()

        return int(action)
