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

# CEC 통합: obs 변환 + 추론 (lazy import — CEC 모델이 없으면 에러 안 남)
_cec_runtime_cls = None
_cec_obs_adapter = None

def _ensure_cec_imports():
    global _cec_runtime_cls, _cec_obs_adapter
    if _cec_runtime_cls is None:
        from cec_integration.cec_runtime import CECRuntime
        from cec_integration.obs_adapter_v2 import ov2_obs_to_cec
        _cec_runtime_cls = CECRuntime
        _cec_obs_adapter = ov2_obs_to_cec

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


def _patch_e3t_predictor():
    """E3T 체크포인트의 3층 PartnerPredictor에 맞게 baseline 모듈을 패치.

    학습 코드(baseline)의 PartnerPredictor는 5층이지만, E3T 체크포인트는
    3층(Dense 128→64, 64→64, 64→6)으로 학습됨.
    baseline rnn.py가 참조하는 PartnerPredictor 클래스를 교체하여
    use_prediction=True 호출 시 체크포인트 params와 일치시킨다.
    """
    import flax.linen as nn
    from flax.linen.initializers import orthogonal, constant

    class PartnerPredictor3L(nn.Module):
        """체크포인트 호환 3층 PartnerPredictor."""
        action_dim: int = 6
        num_partners: int = 1

        @nn.compact
        def __call__(self, embedding):
            output_dim = self.action_dim * self.num_partners
            x = embedding
            x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.leaky_relu(x)
            x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.leaky_relu(x)
            x = nn.Dense(output_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-10)
            x = x / norm
            return x

    # baseline rnn.py가 참조하는 PartnerPredictor를 교체
    rnn_mod = _load_baseline_module("rnn")
    rnn_mod.PartnerPredictor = PartnerPredictor3L

    # e3t 모듈도 교체 (rnn.py가 from .e3t import PartnerPredictor 했을 수 있으므로)
    e3t_mod = _load_baseline_module("e3t")
    e3t_mod.PartnerPredictor = PartnerPredictor3L


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
        self._use_prediction = False  # E3T용: use_prediction 플래그

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

        # CEC 모델: 별도 경로 (Orbax가 아닌 CEC 자체 체크포인트 형식)
        if algo_name.upper() == "CEC":
            self._load_cec(ckpt_path)
            return

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
        elif self.source == "baseline_native":
            # MEP 등 baseline 네트워크를 직접 사용 (리매핑 없이 원본 params 그대로)
            self.network = self._build_baseline_network_native()
        else:
            # SP, E3T, FCP 등: ph2 네트워크에 param 키 리매핑
            self.network = self._build_baseline_network_via_ph2()

    def _build_baseline_network_native(self):
        """baseline ActorCriticRNN을 직접 로드하여 원본 params 그대로 사용."""
        # E3T: 체크포인트 predictor(3층)와 baseline 코드(5층) 불일치 패치
        if "predictor" in self.params:
            _patch_e3t_predictor()
            self._use_prediction = True
        baseline_model_mod = _load_baseline_module("model")
        network = baseline_model_mod.get_actor_critic(self.config)
        # baseline initialize_carry도 baseline 모듈에서 가져옴
        self._baseline_initialize_carry = baseline_model_mod.initialize_carry
        return network

    def _build_baseline_network_via_ph2(self):
        """baseline param 키를 ph2 키로 리매핑 후 ph2 네트워크 사용."""
        # baseline params: CNN_0, LayerNorm_0, ScannedRNN_0, Dense_0~3, (predictor)
        # ph2 params: shared_encoder, shared_encoder_ln, ScannedRNN_0, Dense_0~3, (predictor)
        remap = {
            "CNN_0": "shared_encoder",
            "CNNGamma_0": "shared_encoder",
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
        """에피소드 시작 시 hidden state 리셋."""
        if self.source == "cec":
            self.reset_cec_hidden()
            self._rng = jax.random.PRNGKey(np.random.randint(0, 2**31))
            return
        if self.config is None:
            return
        if self.source == "baseline_native":
            self.hidden = self._baseline_initialize_carry(self.config, batch_size=1)
        else:
            self.hidden = ph2_initialize_carry(self.config, batch_size=1)
        self._rng = jax.random.PRNGKey(np.random.randint(0, 2**31))

    def _make_forward_fn(self):
        """네트워크별 JIT forward 함수 생성."""
        network = self.network

        if self.source == "baseline_native":
            # baseline ActorCriticRNN: actor_only=True로 value head 스킵
            use_pred = self._use_prediction  # E3T: True, 나머지: False
            @jax.jit
            def _forward(params, hidden, obs, done):
                x = (obs, done)
                new_hidden, pi, _value, _pred = network.apply(
                    {"params": params}, hidden, x,
                    actor_only=True,
                    use_prediction=use_pred,
                )
                return new_hidden, pi.logits
            self._apply_fn = _forward
            return

        # PH2 / baseline(리매핑) 경로
        is_ind = (self.source == "ph2")  # PH2는 항상 ind policy 로드
        has_blocked = ("blocked_encoder" in self.params) and (not is_ind)

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

    # ------------------------------------------------------------------
    # CEC 전용 로드/추론
    # ------------------------------------------------------------------
    def _load_cec(self, ckpt_path: str):
        """CEC checkpoint 로드. CECRuntime으로 추론."""
        _ensure_cec_imports()
        self._cec_runtime = _cec_runtime_cls(ckpt_path)
        self._cec_hidden = self._cec_runtime.init_hidden(2)
        self._cec_done = jnp.zeros((2,), dtype=jnp.bool_)
        self._cec_step = 0
        self.source = "cec"
        self.config = {}
        print(f"[CEC] Loaded from {ckpt_path}")

    def reset_cec_hidden(self):
        """CEC 에피소드 리셋."""
        if self.source == "cec":
            self._cec_hidden = self._cec_runtime.init_hidden(2)
            self._cec_done = jnp.zeros((2,), dtype=jnp.bool_)
            self._cec_step = 0

    # ------------------------------------------------------------------
    # get_action: 기존 모델 + CEC 통합
    # ------------------------------------------------------------------
    def get_action(self, obs: np.ndarray, layout_name: str = "") -> int:
        """
        obs: (H, W, C) numpy array → action: int (0-5)
        layout_name: CEC 모델일 때 obs 변환에 필요 (기존 모델은 무시)
        """
        if self.source == "cec":
            return self._get_action_cec(obs, layout_name)

        # 기존 경로 (ph2 / baseline)
        obs_jax = jnp.array(obs, dtype=jnp.float32)[None, None, ...]
        done_jax = jnp.zeros((1, 1), dtype=jnp.bool_)

        new_hidden, logits = self._apply_fn(
            self.params, self.hidden, obs_jax, done_jax
        )
        self.hidden = new_hidden

        logits_2d = logits[0, 0]

        if self.stochastic:
            self._rng, subkey = jax.random.split(self._rng)
            action = jax.random.categorical(subkey, logits_2d).item()
        else:
            action = jnp.argmax(logits_2d).item()

        return int(action)

    def _get_action_cec(self, obs: np.ndarray, layout_name: str) -> int:
        """CEC: OV2 obs (H,W,C) → (9,9,26) 변환 → CECRuntime 추론."""
        _ensure_cec_imports()
        # obs adapter: OV2 (H,W,30) → CEC (9,9,26)
        cec_obs = _cec_obs_adapter(
            jnp.array(obs, dtype=jnp.float32),
            layout_name,
            self._cec_step,
            400,
        )
        # CECRuntime은 (num_agents, 9, 9, 26)을 받지만
        # webapp에서는 AI agent 1명만 추론하므로 dummy agent 1명 추가
        obs_arr = jnp.stack([cec_obs, jnp.zeros_like(cec_obs)])

        self._rng, subkey = jax.random.split(self._rng)
        actions, self._cec_hidden, probs = self._cec_runtime.step(
            obs_arr, self._cec_hidden, self._cec_done, subkey
        )
        self._cec_step += 1

        return int(actions[0])
