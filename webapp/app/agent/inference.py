"""
Phase 2: 모델 추론 — ActorCriticRNN forward pass on CPU.
baseline/ph2 모델 모두 지원.
"""
import importlib.util
import os
import sys
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


def _detect_predictor_depth(predictor_params: dict | None) -> int:
    """체크포인트 params 에서 PartnerPredictor 의 hidden layer 수를 추정.

    E3T PartnerPredictor 는 N 개의 hidden Dense + 1 개의 output Dense 로 구성.
    → params 의 Dense_* 개수 - 1 = num_hidden_layers.

    실측 depth (webapp/models/*/e3t):
      cramped_room / coord_ring / asymm_advantages: 5 Dense (4 hidden)  ← 다수
      forced_coord / counter_circuit:               3 Dense (2 hidden)  ← 일부
    """
    if predictor_params is None:
        return 4  # baseline modern default
    n_dense = sum(1 for k in predictor_params if k.startswith("Dense_"))
    return max(1, n_dense - 1)


def _make_ph2_predictor_class(num_hidden_layers: int):
    """ph2 의 PartnerPredictor (depth hardcoded=2) 를 체크포인트 depth 에 맞춰 재생성.

    baseline 쪽은 `num_hidden_layers` 를 config/kwarg 로 지원하므로 패치 불필요.
    여기서는 오로지 ph2 네트워크 경로 (ph2 / baseline-via-ph2 remap) 에서 사용할
    동적 Predictor 클래스만 만든다.

    주의: __init__ 시그니처는 baseline 쪽과 맞춰 `num_hidden_layers` 도 optional 로
    받도록 한다. (baseline rnn 쪽이 실수로 ph2 경로에서도 해당 kwarg 를 넘기더라도
    TypeError 를 피하기 위함.) 값이 주어지면 그 값을 우선 사용한다.
    """
    import flax.linen as nn
    from flax.linen.initializers import orthogonal, constant

    default_n = num_hidden_layers

    class _Predictor(nn.Module):
        action_dim: int = 6
        num_partners: int = 1
        # baseline 호환성: config-driven depth override. None 이면 closure default 사용.
        num_hidden_layers: int | None = None

        @nn.compact
        def __call__(self, embedding):
            output_dim = self.action_dim * self.num_partners
            n = default_n if self.num_hidden_layers is None else int(self.num_hidden_layers)
            x = embedding
            for i in range(n):
                x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                             bias_init=constant(0.0))(x)
                # 마지막 hidden 만 tanh (baseline convention 과 일치 — activation 차이는
                # normalize 후 softmax 거치므로 inference 영향 미미하지만 가능한 한 맞춤)
                if i == n - 1:
                    x = nn.tanh(x)
                else:
                    x = nn.leaky_relu(x)
            x = nn.Dense(output_dim, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
            norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-10)
            return x / norm

    return _Predictor


def _patch_e3t_predictor(predictor_params: dict | None = None):
    """E3T 체크포인트의 predictor depth 에 맞춰 **ph2 쪽** PartnerPredictor 만 교체.

    baseline 쪽은 `num_hidden_layers` 를 config 로 제어 가능 (baseline/rnn.py 가
    `self.config.get("PREDICTOR_HIDDEN_LAYERS", 4)` 로 읽어 PartnerPredictor 에
    kwarg 로 넘김) → 여기서 패치하지 말고 config 수정으로 처리.

    ph2 쪽은 depth 가 2 로 하드코딩 되어 있어 checkpoint 가 다른 depth 면
    param 키 구조가 안 맞음 → 동적 재생성이 필요.
    """
    n = _detect_predictor_depth(predictor_params)
    cls = _make_ph2_predictor_class(n)

    try:
        import importlib
        ph2_e3t = importlib.import_module("overcooked_v2_experiments.ppo.models.e3t")
        ph2_e3t.PartnerPredictor = cls
        ph2_rnn = importlib.import_module("overcooked_v2_experiments.ppo.models.rnn")
        ph2_rnn.PartnerPredictor = cls
    except Exception:
        pass


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
        # predictor 가 params 에 있으면 (E3T, 또는 action_prediction 쓰는 PH2 등)
        # checkpoint 의 predictor depth 에 맞춰 baseline + ph2 의 PartnerPredictor 를 패치.
        # 매 load 마다 다시 해야 이전 load 의 patched 클래스가 남는 문제 방지.
        if "predictor" in self.params:
            _patch_e3t_predictor(self.params["predictor"])
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
        # E3T: checkpoint predictor depth 에 맞춰 config 수정 + ph2 module 패치.
        # baseline 의 ActorCriticRNN 은 `config["model"]` 을 self.config 로 받아서
        # `self.config.get("PREDICTOR_HIDDEN_LAYERS", 4)` 로 depth 를 읽는다
        # (baseline/model.py → get_actor_critic 에서 model_config 전달).
        # → 반드시 `config["model"]["PREDICTOR_HIDDEN_LAYERS"]` 에 넣어야 함.
        if "predictor" in self.params:
            n_hidden = _detect_predictor_depth(self.params["predictor"])
            self.config = dict(self.config)
            self.config["model"] = dict(self.config.get("model", {}))
            self.config["model"]["PREDICTOR_HIDDEN_LAYERS"] = n_hidden
            _patch_e3t_predictor(self.params["predictor"])
            self._use_prediction = True
        baseline_model_mod = _load_baseline_module("model")
        network = baseline_model_mod.get_actor_critic(self.config)
        # baseline initialize_carry도 baseline 모듈에서 가져옴
        self._baseline_initialize_carry = baseline_model_mod.initialize_carry
        return network

    def _build_baseline_network_via_ph2(self):
        """baseline param 키를 ph2 키로 리매핑 후 ph2 네트워크 사용."""
        # E3T: ph2 코드의 PartnerPredictor (3층) 와 체크포인트 (5층) 가 다를 수 있음 → 패치
        if "predictor" in self.params:
            _patch_e3t_predictor(self.params["predictor"])
            self._use_prediction = True
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
        """CEC legacy 경로: OV2 obs (H,W,C) → (9,9,26) 변환 → CECRuntime 추론.

        주: 이 경로는 overcooked-ai → OV2 → CEC 2단계 변환이라 dynamics 불일치 +
        pot lifecycle 손실이 있어 실제 webapp 에선 `get_action_cec_from_ai` 사용 권장.
        호환 목적으로 유지.
        """
        _ensure_cec_imports()
        cec_obs = _cec_obs_adapter(
            jnp.array(obs, dtype=jnp.float32),
            layout_name,
            self._cec_step,
            400,
        )
        obs_arr = jnp.stack([cec_obs, jnp.zeros_like(cec_obs)])

        self._rng, subkey = jax.random.split(self._rng)
        actions, self._cec_hidden, probs = self._cec_runtime.step(
            obs_arr, self._cec_hidden, self._cec_done, subkey
        )
        self._cec_step += 1

        return int(actions[0])

    def _ensure_cec_ai_adapter(self, layout_name: str):
        """레이아웃별 OvercookedAIToCECAdapter 를 lazy-load 후 캐시."""
        if not hasattr(self, "_cec_ai_adapters"):
            self._cec_ai_adapters = {}
        if layout_name not in self._cec_ai_adapters:
            # cec_integration 경로 확보
            cec_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if cec_root not in sys.path:
                sys.path.insert(0, cec_root)
            from cec_integration.obs_adapter_from_ai import OvercookedAIToCECAdapter
            self._cec_ai_adapters[layout_name] = OvercookedAIToCECAdapter(
                target_layout=layout_name, max_steps=400,
            )
        return self._cec_ai_adapters[layout_name]

    def get_action_cec_from_ai(self, ai_state, mdp, layout_name: str,
                               agent_idx: int = 0) -> int:
        """CEC 직접 경로: overcooked-ai state → V1 State → V1 get_obs → CEC 추론.

        slot alignment:
          V1 훈련 시 CEC 의 agent_0 은 `CEC_LAYOUTS[layout_9]["agent_idx"][0]` 좌표
          (예: cramped_room_9 → flat 10 = (x=1, y=1)) 에 고정.
          webapp overcooked-ai 는 `swap_agents=True` 레이아웃 (cramped_room,
          counter_circuit) 에서 agent_0 을 다른 좌표에 배치.
          → AI 의 실제 위치가 V1 의 agent_0 좌표와 일치하면 CEC slot 0 을,
            V1 의 agent_1 좌표와 일치하면 CEC slot 1 을 사용해서 훈련 convention 과 맞춤.

        Args:
            ai_state: overcooked-ai OvercookedState
            mdp: overcooked-ai OvercookedGridworld
            layout_name: e.g. "cramped_room"
            agent_idx: 0 or 1 (AI agent 가 webapp 의 slot 0 인지 1 인지)

        Returns:
            action index (0-5)
        """
        _ensure_cec_imports()
        adapter = self._ensure_cec_ai_adapter(layout_name)

        # V1 훈련 convention 의 agent_0 / agent_1 좌표 (x, y)
        from cec_integration.cec_layouts import CEC_LAYOUTS
        v1_layout = CEC_LAYOUTS[f"{layout_name}_9"]
        import numpy as np
        v1_agent_xy = []
        for flat in np.asarray(v1_layout["agent_idx"]):
            v1_agent_xy.append((int(flat) % 9, int(flat) // 9))  # (x, y)

        # AI 의 실제 (x, y)
        ai_pos = tuple(ai_state.players[agent_idx].position)
        # V1 convention 에서 이 위치가 agent_0 이면 slot 0, agent_1 이면 slot 1
        if ai_pos == v1_agent_xy[0]:
            cec_slot = 0
        elif ai_pos == v1_agent_xy[1]:
            cec_slot = 1
        else:
            # 이동 후라 V1 static agent_idx 와 매칭 안 됨 → 이전 slot 유지
            cec_slot = getattr(self, "_cec_assigned_slot", 0)
        self._cec_assigned_slot = cec_slot

        cec_obs = adapter.get_cec_obs(ai_state, mdp,
                                       agent_idx=agent_idx,
                                       current_step=self._cec_step)
        # CECRuntime 은 (num_agents=2, 9, 9, 26) 입력. cec_slot 에 배치, 나머지 dummy.
        dummy = jnp.zeros_like(cec_obs)
        if cec_slot == 0:
            obs_arr = jnp.stack([cec_obs, dummy])
        else:
            obs_arr = jnp.stack([dummy, cec_obs])

        self._rng, subkey = jax.random.split(self._rng)
        actions, self._cec_hidden, probs = self._cec_runtime.step(
            obs_arr, self._cec_hidden, self._cec_done, subkey
        )
        self._cec_step += 1
        return int(actions[cec_slot])

    def get_action_cec_v1_obs(self, v1_obs, cec_slot: int = 0) -> int:
        """CEC V1-engine 경로: V1 Overcooked.get_obs 결과를 직접 받아 CEC runtime 호출.

        webapp 에서 primary engine 을 V1 Overcooked 로 운영할 때 사용.
        V1EngineSession.get_cec_obs_v1(agent_idx) 의 출력을 그대로 넘기면 됨.

        Args:
            v1_obs: (9, 9, 26) — V1 native obs for the AI agent
            cec_slot: 0 or 1 — CEC runtime 내 AI 의 slot 위치 (V1 훈련 convention;
                      보통 AI 가 webapp ai_idx 로 놓이는 V1 slot 과 동일)

        Returns:
            action index (0-5)
        """
        _ensure_cec_imports()
        v1_obs = jnp.asarray(v1_obs)
        dummy = jnp.zeros_like(v1_obs)
        obs_arr = jnp.stack([v1_obs, dummy]) if cec_slot == 0 else jnp.stack([dummy, v1_obs])

        self._rng, subkey = jax.random.split(self._rng)
        actions, self._cec_hidden, probs = self._cec_runtime.step(
            obs_arr, self._cec_hidden, self._cec_done, subkey
        )
        self._cec_step += 1
        return int(actions[cec_slot])
