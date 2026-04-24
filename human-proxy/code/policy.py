#!/usr/bin/env python3
"""
BCPolicy + PPOPolicy — eval용 policy 클래스.

BCPolicy: CNN BC 모델로 행동 샘플링.
PPOPolicy: ph2 또는 baseline 체크포인트 로딩 + 추론.

PYTHONPATH 설정:
    - evaluate.py에서 --source ph2|baseline 인자로 결정
    - ph2: PH2, PH2-E3T 등 PH2 계열 알고리즘
    - baseline: SP, E3T, FCP, MEP, HSP, GAMMA 등
    - setup_pythonpath()를 반드시 import 전에 호출
"""
import os
import sys
from pathlib import Path
from typing import Any

import pickle

import chex
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import core
from jax.sharding import SingleDeviceSharding

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_pythonpath(source: str = "ph2"):
    """overcooked_v2_experiments 패키지 경로 설정. import 전에 호출 필수."""
    if source == "ph2":
        code_root = _PROJECT_ROOT / "ph2"
    elif source == "baseline":
        code_root = _PROJECT_ROOT / "baseline"
    else:
        raise ValueError(f"source must be 'ph2' or 'baseline', got {source}")

    jaxmarl_root = _PROJECT_ROOT / "JaxMARL"

    for p in [str(code_root), str(jaxmarl_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def detect_source_from_checkpoint(ckpt_path: Path) -> str:
    """체크포인트의 config에서 ph2인지 baseline인지 감지."""
    config, _ = load_ppo_checkpoint(ckpt_path)
    alg_name = str(config.get("ALG_NAME", ""))
    if "PH2" in alg_name.upper():
        return "ph2"
    return "baseline"


def detect_source_from_run_dir(run_dir: Path) -> str:
    """run 디렉토리 첫 번째 체크포인트에서 source 감지."""
    run_dir = Path(run_dir).resolve()
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            ckpt_final = d / "ckpt_final"
            if ckpt_final.exists():
                return detect_source_from_checkpoint(ckpt_final)
    return "baseline"


# ──────────────────────────────────────────────
# PPO 체크포인트 로드 (GPU→CPU 호환)
# ──────────────────────────────────────────────

_CPU_DEVICE = jax.devices("cpu")[0]
_CPU_SHARDING = SingleDeviceSharding(_CPU_DEVICE)


def _build_cpu_restore_args(tree):
    """checkpoint metadata 트리를 순회하며 CPU sharding restore args 생성."""
    if isinstance(tree, dict):
        return {k: _build_cpu_restore_args(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_build_cpu_restore_args(v) for v in tree)
    type_name = type(tree).__name__
    if "ArrayMetadata" in type_name:
        return ocp.ArrayRestoreArgs(sharding=_CPU_SHARDING)
    return ocp.RestoreArgs()


def load_ppo_checkpoint(ckpt_path: Path):
    """Orbax PPO 체크포인트를 CPU에서 로드. (config, params) 반환.
    PH2 체크포인트: params_ind 사용 (independent policy).
    Baseline 체크포인트: params 사용.

    E3T PartnerPredictor 구조 자동 감지:
      체크포인트의 /predictor/ 하위 Dense 개수로 `PREDICTOR_HIDDEN_LAYERS` 를
      config 에 주입 (modern=4, legacy=2). 이후 모델 생성 시 이 값을 씀.
    """
    ckpt_path = Path(ckpt_path).resolve()
    handler = ocp.PyTreeCheckpointHandler()
    meta = handler.metadata(ckpt_path)
    restore_args = _build_cpu_restore_args(meta.tree)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(ckpt_path), restore_args=restore_args)

    # PH2 체크포인트는 params_ind (independent policy) 사용
    params = ckpt["params_ind"] if "params_ind" in ckpt else ckpt["params"]
    config = dict(ckpt["config"]) if "config" in ckpt else {}

    # E3T PartnerPredictor: 체크포인트에서 Dense 개수 감지 → config["model"] 에 주입.
    # (ActorCriticRNN 은 config["model"] 만 받으므로 반드시 model 하위에 넣어야 반영됨.)
    # /predictor/Dense_K 최대 K + 1 == 총 Dense 수. num_hidden_layers = 총 - 1.
    try:
        params_root = params["params"] if "params" in params else params
        if "predictor" in params_root:
            dense_keys = [k for k in params_root["predictor"].keys()
                          if str(k).startswith("Dense_")]
            if dense_keys:
                num_dense = max(int(str(k).split("_")[1]) for k in dense_keys) + 1
                num_hidden = max(0, num_dense - 1)
                model_cfg = dict(config.get("model", {}))
                model_cfg["PREDICTOR_HIDDEN_LAYERS"] = num_hidden
                config["model"] = model_cfg
    except Exception:
        pass

    return config, params


# ──────────────────────────────────────────────
# BC 체크포인트 로드
# ──────────────────────────────────────────────

def load_bc_checkpoint(model_dir: Path):
    """pickle BC 체크포인트 로드. (config, params) 반환."""
    ckpt_path = model_dir / "checkpoint.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, ckpt["params"])
    return ckpt["config"], params


# ──────────────────────────────────────────────
# Stuck Detection
# ──────────────────────────────────────────────

STATE_HISTORY_LEN = 3


@chex.dataclass
class BCHState:
    actions: chex.Array
    pos: chex.Array
    direction: chex.Array

    @staticmethod
    def init_empty():
        return BCHState(
            actions=jnp.zeros((STATE_HISTORY_LEN,), dtype=jnp.int32),
            pos=jnp.zeros((STATE_HISTORY_LEN, 2), dtype=jnp.int32),
            direction=jnp.zeros((STATE_HISTORY_LEN, 4), dtype=jnp.int32),
        )

    def to_flat(self):
        return jnp.concatenate([
            self.actions.reshape(-1),
            self.pos.reshape(-1),
            self.direction.reshape(-1),
        ])

    @staticmethod
    def from_flat(arr):
        idx = 0
        actions = arr[idx:idx + STATE_HISTORY_LEN].reshape((STATE_HISTORY_LEN,))
        idx += STATE_HISTORY_LEN
        pos = arr[idx:idx + STATE_HISTORY_LEN * 2].reshape((STATE_HISTORY_LEN, 2))
        idx += STATE_HISTORY_LEN * 2
        direction = arr[idx:idx + STATE_HISTORY_LEN * 4].reshape((STATE_HISTORY_LEN, 4))
        return BCHState(actions=actions, pos=pos, direction=direction)

    def append(self, action, obs_flat_start):
        new_actions = jnp.concatenate([self.actions[1:], jnp.array([action])])
        return BCHState(actions=new_actions, pos=self.pos, direction=self.direction)

    def is_stuck(self):
        return jnp.all(self.actions == self.actions[0])


def _remove_and_renormalize(probs, indices):
    probs = probs.at[indices].set(0.0)
    total = probs.sum()
    fallback = jnp.ones_like(probs).at[indices].set(0.0)
    probs = jax.lax.select(total > 0, probs, fallback)
    return probs / probs.sum()


# ──────────────────────────────────────────────
# BCPolicy (train.py의 BCModel 사용)
# ──────────────────────────────────────────────

# train.py는 human-proxy/code/ 안에 있으므로 직접 import
from train import BCModel


class BCPolicy:
    """CNN 기반 BC policy. JaxMARL obs (H,W,C) → action.
    AbstractPolicy를 직접 상속하지 않고 동일 인터페이스 구현.
    (AbstractPolicy import는 setup_pythonpath 이후에만 가능)
    """

    def __init__(self, params, config, stochastic=True, unblock_if_stuck=True):
        self.params = params
        self.config = config
        self.stochastic = stochastic
        self.unblock_if_stuck = unblock_if_stuck
        self.network = BCModel(
            action_dim=config.get("action_dim", 6),
            cnn_features=config.get("cnn_features", 32),
            fc_dim=config.get("fc_dim", 64),
        )

    @classmethod
    def from_pretrained(cls, model_dir, stochastic=True, unblock_if_stuck=True):
        config, params = load_bc_checkpoint(Path(model_dir))
        return cls(params, config, stochastic=stochastic, unblock_if_stuck=unblock_if_stuck)

    def compute_action(self, obs, done, hstate, key):
        obs_float = obs.astype(jnp.float32) / 255.0
        obs_batch = obs_float[jnp.newaxis, ...]
        logits = self.network.apply({"params": self.params}, obs_batch)
        action_probs = jax.nn.softmax(logits[0])

        if self.unblock_if_stuck and hstate is not None:
            h_flat = hstate[0] if hstate.ndim == 2 else hstate
            h_flat = jnp.where(done, BCHState.init_empty().to_flat(), h_flat)
            bc_h = BCHState.from_flat(h_flat)
            is_stuck = bc_h.is_stuck()
            unstuck_probs = _remove_and_renormalize(action_probs, bc_h.actions)
            action_probs = jnp.where(is_stuck, unstuck_probs, action_probs)

        if self.stochastic:
            action = jax.random.choice(key, self.network.action_dim, p=action_probs)
        else:
            action = jnp.argmax(action_probs)

        if self.unblock_if_stuck and hstate is not None:
            bc_h = BCHState.from_flat(h_flat)
            bc_h = bc_h.append(action, None)
            hstate = bc_h.to_flat()[jnp.newaxis, ...]

        # ph2 rollout이 extras["value"]를 기대하므로 dummy 반환
        extras = {"value": jnp.float32(0.0)}
        return action, hstate, extras

    def init_hstate(self, batch_size, key=None):
        if self.unblock_if_stuck:
            h = BCHState.init_empty().to_flat()
            return jnp.repeat(h[jnp.newaxis, ...], batch_size, axis=0)
        return None


# ──────────────────────────────────────────────
# PPOPolicy 로드 함수 (setup_pythonpath 후 호출)
# ──────────────────────────────────────────────

def make_ppo_policy(config, params, stochastic=True):
    """setup_pythonpath 호출 후 PPOPolicy 생성."""
    from overcooked_v2_experiments.ppo.policy import PPOPolicy
    return PPOPolicy(params=params, config=config, stochastic=stochastic)


def load_ppo_policies_from_run_dir(run_dir, stochastic=True):
    """run 디렉토리에서 모든 seed의 ckpt_final 로드. list[PPOPolicy] 반환."""
    run_dir = Path(run_dir).resolve()
    policies = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            ckpt_final = d / "ckpt_final"
            if ckpt_final.exists():
                try:
                    config, params = load_ppo_checkpoint(ckpt_final)
                    policy = make_ppo_policy(config, params, stochastic)
                    policies.append(policy)
                except Exception as e:
                    print(f"  경고: {ckpt_final} 로드 실패: {e}")
    return policies


# ──────────────────────────────────────────────
# CEC Policy (cec_integration 사용)
# ──────────────────────────────────────────────

class CECPolicy:
    """CEC checkpoint를 AbstractPolicy 인터페이스로 래핑.

    compute_action(obs, done, hstate, key) → (action, new_hstate, extras)
    obs는 OV2 (H,W,C) 형식이며, 내부에서 CEC (9,9,26)으로 변환.
    """

    def __init__(self, ckpt_path: str, layout: str, stochastic: bool = True):
        # `from cec_integration.xxx import ...` 가 되려면 project root 가 sys.path 에 있어야 함
        # (cec_integration 디렉터리 자체가 아니라 그 부모)
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from cec_integration.cec_runtime import CECRuntime
        from cec_integration.obs_adapter_v2 import ov2_obs_to_cec
        from cec_integration.cec_layouts import CEC_LAYOUTS

        self._runtime = CECRuntime(ckpt_path)
        self._obs_adapter = ov2_obs_to_cec
        self._layout = layout
        self._stochastic = stochastic
        self._step = 0

        # V1 훈련 convention 의 agent_0 좌표 (x, y) — slot alignment 에 사용.
        # CEC 는 V1 에서 학습됐으므로 slot 0 의 LSTM 은 이 좌표에서 시작한 agent 의 경험으로 훈련됨.
        # OV2 의 swap_agents=True 레이아웃 (cramped_room, counter_circuit) 에서는 agent_0 좌표가 뒤바뀌어 있음.
        v1_layout = CEC_LAYOUTS[f"{layout}_9"]
        flat0 = int(np.asarray(v1_layout["agent_idx"])[0])
        self._v1_agent0_xy = (flat0 % 9, flat0 // 9)

    def compute_action(self, obs, done, hstate, key, **kwargs):
        """obs: (H, W, C) OV2 obs. hstate: (cec_lstm_hidden, step_counter) tuple.

        JIT-safe: Python int cast / self mutation 없음. step 은 hstate 에 flow.
        Slot alignment: AI 시작 좌표가 V1 agent_0 좌표면 slot 0, 아니면 slot 1.
        """
        cec_lstm, step_counter = hstate
        cec_obs = self._obs_adapter(obs, self._layout, step_counter, 400)

        # OV2 obs ch 0 = self_pos one-hot. argmax → (x,y) → V1 agent_0 좌표와 비교.
        H, W = obs.shape[0], obs.shape[1]
        self_flat = jnp.argmax(obs[:, :, 0].reshape(-1))
        self_x = self_flat % W
        self_y = self_flat // W
        is_slot_0 = (self_x == self._v1_agent0_xy[0]) & (self_y == self._v1_agent0_xy[1])

        dummy = jnp.zeros_like(cec_obs)
        # slot 0 경로: [cec_obs, dummy] / slot 1 경로: [dummy, cec_obs]
        obs_arr = jnp.where(
            is_slot_0,
            jnp.stack([cec_obs, dummy]),
            jnp.stack([dummy, cec_obs]),
        )
        done_arr = jnp.array([done, done])

        actions, new_cec_lstm, probs = self._runtime.step(obs_arr, cec_lstm, done_arr, key)

        # 배정된 slot 의 action 선택. JIT-safe scalar int32.
        slot_idx = jnp.where(is_slot_0, jnp.int32(0), jnp.int32(1))
        action = actions[slot_idx].astype(jnp.int32)
        new_step = jnp.where(done, jnp.int32(0), step_counter + jnp.int32(1))
        new_hstate = (new_cec_lstm, new_step)
        extras = {"value": jnp.float32(0.0)}
        return action, new_hstate, extras

    def init_hstate(self, batch_size=1, key=None):
        # (LSTM hidden, step_counter) tuple
        return (self._runtime.init_hidden(2), jnp.int32(0))


def load_cec_policies(ckpt_dir: str, layout: str, stochastic: bool = True):
    """CEC 체크포인트 디렉토리에서 policy 리스트 로드.

    ckpt_dir: cec_integration/ckpts/forced_coord_9/ 같은 경로.
    ckpt2_improved 서브디렉토리만 로드.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    policies = []
    for d in sorted(ckpt_dir.iterdir()):
        if d.is_dir() and d.name.endswith("ckpt2_improved"):
            try:
                policy = CECPolicy(str(d), layout, stochastic)
                policies.append(policy)
                print(f"  CEC 로드: {d.name}")
            except Exception as e:
                print(f"  경고: {d} 로드 실패: {e}")
    return policies
