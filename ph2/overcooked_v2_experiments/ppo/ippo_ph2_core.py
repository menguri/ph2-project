""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict, Union
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
import pickle
from models.rnn import ScannedRNN
import matplotlib.pyplot as plt
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from overcooked_v2_experiments.ppo.ph1_utils import (
    compute_ph1_probs,
    sample_target_idx,
    sample_multi_targets_from_pool,
)
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from flax import core
from overcooked_v2_experiments.ppo.utils.valuenorm import (
    ValueNormState, valuenorm_update, valuenorm_normalize, valuenorm_denormalize,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray
    partner_action: jnp.ndarray
    is_ego: jnp.ndarray
    blocked_states: jnp.ndarray
    prev_action: jnp.ndarray
    partner_prediction: jnp.ndarray
    hstate: any
    global_obs: jnp.ndarray   # (NUM_ACTORS, H, W, C_full) — CT recon 타겟용 full global state
    partner_gru_z: jnp.ndarray  # (NUM_ACTORS, D) — CT v3: partner GRU hidden state 복원 타겟
    avail_actions: jnp.ndarray  # (NUM_ACTORS, ACTION_DIM) — GridSpread masking용, 타 env는 placeholder zeros


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _prepare_env_spec(config: Dict[str, Any], env_config: Dict[str, Any]):
    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    return env_name, env_kwargs


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]
    # ref MAPPO 안정화 트릭 (GridSpread 한정 자동 활성화 — main.py gs_overrides).
    use_valuenorm = bool(model_config.get("USE_VALUENORM", False))
    use_huber_loss = bool(model_config.get("USE_HUBER_LOSS", False))
    huber_delta = float(model_config.get("HUBER_DELTA", 10.0))
    phase_log_prefix = str(config.get("PHASE_LOG_PREFIX", "")).strip()
    phase_log_fixed_seed = config.get("PHASE_LOG_FIXED_SEED", None)
    verbose_logs = bool(config.get("PH2_CORE_VERBOSE_LOGS", False))
    runtime_progress_debug = bool(config.get("PH2_PROGRESS_DEBUG", False))
    runtime_progress_every = max(1, int(config.get("PH2_PROGRESS_DEBUG_EVERY", 100)))

    def _vprint(*args, **kwargs):
        if verbose_logs:
            print(*args, **kwargs)

    ph1_enabled = bool(config.get("PH1_ENABLED", False))
    # Execution policy input is always the environment-provided observation.
    # PH1 only injects the blocked target (tilde{s}) via `blocked_states`.
    use_ph1_partner_pred = bool(config.get("PH1_USE_PARTNER_PRED", ph1_enabled))
    ph1_epsilon = float(config.get("PH1_EPSILON", 0.0))
    ph2_epsilon_cfg = float(config.get("PH2_EPSILON", -1.0))
    ph2_epsilon = (
        float(np.clip(ph2_epsilon_cfg, 0.0, 1.0))
        if ph2_epsilon_cfg >= 0.0
        else float(np.clip(ph1_epsilon, 0.0, 1.0))
    )
    ph1_pool_size = int(config.get("PH1_POOL_SIZE", 100))
    ph1_multi_penalty_enabled = bool(config.get("PH1_MULTI_PENALTY_ENABLED", False))
    ph1_pair_mode = bool(config.get("PH1_PAIR_MODE", False))
    ph1_max_penalty_count = int(config.get("PH1_MAX_PENALTY_COUNT", 1))
    ph1_max_penalty_count = max(1, ph1_max_penalty_count)
    ph1_penalty_slots = ph1_max_penalty_count if ph1_multi_penalty_enabled else 1
    # pair_mode: 각 penalty position이 (원본 + swap) 2개씩 → 슬롯 수 2배
    if ph1_pair_mode:
        ph1_penalty_slots = ph1_penalty_slots * 2
    ph1_raw_distance = bool(config.get("PH1_RAW_DISTANCE", False))
    ph1_multi_penalty_single_weight = float(
        config.get("PH1_MULTI_PENALTY_SINGLE_WEIGHT", 2.0)
    )
    ph1_multi_penalty_other_weight = float(
        config.get("PH1_MULTI_PENALTY_OTHER_WEIGHT", 1.0)
    )
    ph1_warmup_steps = int(config.get("PH1_WARMUP_STEPS", 0))
    ph1_beta_base = float(config.get("PH1_BETA", 1.0))
    ph1_beta_schedule_enabled = bool(config.get("PH1_BETA_SCHEDULE_ENABLED", False))
    ph1_beta_start = float(config.get("PH1_BETA_START", 0.0))
    ph1_beta_end = float(config.get("PH1_BETA_END", 2.0))
    ph1_beta_schedule_horizon_cfg = int(
        config.get("PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS", -1)
    )
    abort_on_nan = bool(config.get("ABORT_ON_NAN", True))

    # CycleTransformer settings
    transformer_action = bool(config.get("TRANSFORMER_ACTION", False))
    transformer_v2 = bool(config.get("TRANSFORMER_V2", False))
    transformer_window_size = int(config.get("TRANSFORMER_WINDOW_SIZE", 16))
    transformer_d_c = int(config.get("TRANSFORMER_D_C", 128))
    transformer_recon_coef = float(config.get("TRANSFORMER_RECON_COEF", 1.0))
    transformer_pred_coef = float(config.get("TRANSFORMER_PRED_COEF", 1.0))
    transformer_cycle_coef = float(config.get("TRANSFORMER_CYCLE_COEF", 0.5))
    transformer_v3 = bool(config.get("TRANSFORMER_V3", False))

    # Z Prediction / Cycle Loss settings (CT OFF 모드 전용)
    z_prediction_enabled = bool(config.get("Z_PREDICTION_ENABLED", False))
    z_pred_loss_coef = float(config.get("Z_PRED_LOSS_COEF", 1.0))
    cycle_loss_enabled = bool(config.get("CYCLE_LOSS_ENABLED", False))
    cycle_loss_coef = float(config.get("CYCLE_LOSS_COEF", 0.1))

    # OV1 vs OV2 감지: env kwargs에 agent_view_size 있으면 partial obs (OV2)
    # OV2 → CT recon target은 per-actor full obs (get_obs_default 활용)
    # OV1 → agent 0의 full obs를 모든 actor에 동일하게 사용 (PH1 pool과 동일)
    _env_kwargs = env_config.get("ENV_KWARGS", {})
    is_partial_obs = "agent_view_size" in _env_kwargs

    # E3T 설정 로드
    alg_name = config.get("ALG_NAME", "SP")
    alg_name_u = str(alg_name).upper()
    e3t_enabled = ("E3T" in alg_name_u)
    e3t_epsilon = config.get("E3T_EPSILON", 0.05)
    use_partner_modeling = config.get("USE_PARTNER_MODELING", True)
    action_prediction = bool(config.get("ACTION_PREDICTION", True))
    prediction_enabled = bool(action_prediction)
    learner_use_blocked_input = bool(
        config.get("LEARNER_USE_BLOCKED_INPUT", ph1_enabled)
    )
    population_use_blocked_input = bool(config.get("POPULATION_USE_BLOCKED_INPUT", False))
    population_use_partner_pred_input = bool(config.get("POPULATION_USE_PARTNER_PRED_INPUT", False))
    ph1_probs_use_population_model = bool(config.get("PH1_PROBS_USE_POPULATION_MODEL", False))
    env_name, env_kwargs = _prepare_env_spec(config, env_config)
    # PH2 joint-match controls
    ph2_match_schedule = bool(config.get("PH2_MATCH_SCHEDULE", False))
    ph2_role = str(config.get("PH2_ROLE", "")).strip().lower()
    ph2_ratio_stage1 = float(config.get("PH2_RATIO_STAGE1", 0.0))
    ph2_ratio_stage2 = float(config.get("PH2_RATIO_STAGE2", 0.0))
    ph2_ratio_stage3 = float(config.get("PH2_RATIO_STAGE3", 0.0))
    ph2_fixed_ind_prob = config.get("PH2_FIXED_IND_PROB", 0.5)
    if ph2_fixed_ind_prob is not None:
        ph2_fixed_ind_prob = float(np.clip(float(ph2_fixed_ind_prob), 0.0, 1.0))

    # Optional device selection for environment to mitigate GPU OOM.
    # Usage via Hydra override: +ENV_DEVICE=cpu  (default: gpu / auto)
    env_device = config.get("ENV_DEVICE", None)  # None -> default placement
    if env_device is not None and env_device not in ("cpu", "gpu"):
        raise ValueError(f"ENV_DEVICE must be one of 'cpu','gpu' (or unset). Got: {env_device}")

    # Create env under a device context so large static buffers (layout grids etc.) live on CPU when requested.
    if env_device == "cpu":
        with jax.default_device(jax.devices("cpu")[0]):
            env = jaxmarl.make(env_name, **env_kwargs)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)

    ACTION_DIM = env.action_space(env.agents[0]).n
    is_spread = (env_name == "GridSpread")
    num_partners = env.num_agents - 1  # 2-agent: 1, 3-agent: 2
    policy_pred_dim = ACTION_DIM * num_partners  # pred_logits 전체 차원

    model_config["ACTION_DIM"] = ACTION_DIM
    model_config["NUM_PARTNERS"] = num_partners
    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"]
        * model_config["NUM_STEPS"]
        // model_config["NUM_MINIBATCHES"]
    )

    steps_per_update = model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
    total_timesteps = int(model_config["TOTAL_TIMESTEPS"])
    if ph1_beta_schedule_horizon_cfg <= 0:
        ph1_beta_schedule_horizon_env_steps = max(1, total_timesteps - ph1_warmup_steps)
    else:
        ph1_beta_schedule_horizon_env_steps = max(1, ph1_beta_schedule_horizon_cfg)
    ph2_total_updates = max(1, int(model_config["NUM_UPDATES"]))

    def _ratio_to_prob(x):
        # If ratio is given as probability (<=1), use directly.
        # If ratio is >1, interpret as odds-like value and squash to (0,1).
        return jnp.where(x <= 1.0, x, x / (x + 1.0))

    def _current_ph1_beta(update_step):
        if not ph1_beta_schedule_enabled:
            return jnp.float32(ph1_beta_base), jnp.float32(0.0)

        env_step = (
            jnp.asarray(update_step, dtype=jnp.float32)
            * jnp.float32(steps_per_update)
        )
        progress = (
            (env_step - jnp.float32(ph1_warmup_steps))
            / jnp.float32(ph1_beta_schedule_horizon_env_steps)
        )
        progress = jnp.clip(progress, 0.0, 1.0)
        beta_t = (
            jnp.float32(ph1_beta_start)
            + progress * jnp.float32(ph1_beta_end - ph1_beta_start)
        )
        return beta_t, progress

    def _phase2_ind_match_prob(update_step):
        if ph2_fixed_ind_prob is not None:
            return jnp.float32(ph2_fixed_ind_prob)
        progress = jnp.asarray(update_step, dtype=jnp.float32) / jnp.float32(ph2_total_updates)
        ratio = jnp.where(
            progress < (1.0 / 3.0),
            jnp.float32(ph2_ratio_stage1),
            jnp.where(
                progress < (2.0 / 3.0),
                jnp.float32(ph2_ratio_stage2),
                jnp.float32(ph2_ratio_stage3),
            ),
        )
        return jnp.clip(_ratio_to_prob(ratio), 0.0, 1.0)

    num_checkpoints = int(config["NUM_CHECKPOINTS"])
    checkpoint_steps = jnp.linspace(
        0,
        model_config["NUM_UPDATES"],
        num_checkpoints,
        endpoint=True,
        dtype=jnp.int32,
    )
    if num_checkpoints > 0:
        # make sure the last checkpoint is the last update step
        checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    _vprint("Checkpoint steps: ", checkpoint_steps)

    def _update_checkpoint(checkpoint_states, params, i):
        return jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y),
            checkpoint_states,
            params,
        )

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env, replace_info=False)
    else:
        env = LogWrapper(env, replace_info=False)

    _is_mpe = env_name.startswith("MPE_")

    def _extract_pos_axes(log_env_state):
        # ToyCoop: state.agent_pos (2,2) → [x, y] 형태
        if env_name == "ToyCoop":
            pos = log_env_state.env_state.agent_pos  # (NUM_ENVS, 2, 2)
            return pos[:, :, 1], pos[:, :, 0]  # y, x
        # GridSpread: state.agent_pos (NUM_ENVS, n_agents, 2) — [x, y]
        if is_spread:
            pos = log_env_state.env_state.agent_pos  # (NUM_ENVS, n_agents, 2)
            return pos[:, :, 1], pos[:, :, 0]  # y, x
        # MPE: state.p_pos (NUM_ENVS, num_entities, 2) — 에이전트가 앞쪽
        if _is_mpe:
            pos = log_env_state.env_state.p_pos  # (NUM_ENVS, num_entities, 2)
            return pos[:, :env.num_agents, 1], pos[:, :env.num_agents, 0]  # y, x
        return log_env_state.env_state.agents.pos.y, log_env_state.env_state.agents.pos.x

    def _extract_global_full_obs(log_env_state):
        # MPE: state = concat(obs_agent0, obs_agent1) → 28차원 (spread 기준)
        # 양쪽 에이전트 정보를 합쳐서 global state로 사용 (egocentric 좌표계 문제 해소)
        if _is_mpe:
            obs_dict = jax.vmap(env.get_obs)(log_env_state.env_state)
            all_obs = jnp.concatenate(
                [obs_dict[a].astype(jnp.float32) for a in env.agents], axis=-1
            )  # (NUM_ENVS, obs_dim * num_agents) e.g. (NUM_ENVS, 28)
            return all_obs
        full = jax.vmap(env.get_obs_default)(log_env_state.env_state)
        return full[:, 0].astype(jnp.float32)

    def _extract_global_full_obs_per_actor(log_env_state):
        """각 actor 자신의 시점에서 full global obs 반환.

        반환 순서는 obs_batch와 동일: [agent0_env0..N, agent1_env0..N]

        Returns:
            (NUM_ACTORS, *obs_shape) — obs_batch와 동일한 actor 순서
        """
        if _is_mpe:
            # MPE: 모든 actor에 대해 동일한 global state = concat(obs_0, obs_1)
            obs_dict = jax.vmap(env.get_obs)(log_env_state.env_state)
            global_state = jnp.concatenate(
                [obs_dict[a].astype(jnp.float32) for a in env.agents], axis=-1
            )  # (NUM_ENVS, 28)
            # 모든 actor에 동일한 global state 제공 (agent-invariant)
            return jnp.tile(global_state, [env.num_agents, 1])  # (NUM_ACTORS, 28)
        full = jax.vmap(env.get_obs_default)(log_env_state.env_state)
        # full: (NUM_ENVS, num_agents, H, W, C_full)
        # N-agent 일반화: 각 agent의 full obs를 순서대로 concat
        per_agent = [full[:, i].astype(jnp.float32) for i in range(env.num_agents)]
        return jnp.concatenate(per_agent, axis=0)  # (NUM_ACTORS, H, W, C_full)

    # Wrap reset/step with backend-specific jits if ENV_DEVICE explicitly set.
    reset_fn = env.reset
    step_fn = env.step
    if env_device == "cpu":
        reset_fn = jax.jit(env.reset, backend="cpu")
        step_fn = jax.jit(env.step, backend="cpu")
    elif env_device == "gpu":
        reset_fn = jax.jit(env.reset, backend="gpu")
        step_fn = jax.jit(env.step, backend="gpu")

    # Optional mixed precision for stored observations to reduce memory of trajectory buffers.
    cast_obs_bf16 = bool(config.get("CAST_OBS_BF16", False))

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]

        lr_warmup = model_config["LR_WARMUP"]
        update_steps = model_config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        _vprint("Update steps: ", update_steps)
        _vprint("Warmup epochs: ", warmup_steps)
        _vprint("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    train_idxs = jnp.linspace(
        0,
        env.num_agents,
        model_config["NUM_ENVS"],
        dtype=jnp.int32,
        endpoint=False,
    )
    train_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    train_mask_flat = batchify(
        train_mask_dict, env.agents, model_config["NUM_ACTORS"]
    ).squeeze()

    _vprint("train_mask_flat", train_mask_flat.shape)
    _vprint("train_mask_flat sum", train_mask_flat.sum())

    # NUM_ACTORS = (env.num_agents * NUM_ENVS)이므로, 각 슬롯이 어떤 에이전트인지 구분하기 위해
    # 반복되는 인덱스 벡터를 만들어 ego/partner 마스크를 구성한다.
    # batchify 결과는 Agent Major 순서 ([Ag0_Env0, ..., Ag0_EnvN, Ag1_Env0, ...])이므로
    # repeat를 사용하여 [0, 0, ..., 1, 1, ...] 형태로 만들어야 한다.
    actor_indices = jnp.repeat(
        jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
    )
    ego_actor_mask = actor_indices == 0
    partner_actor_mask = actor_indices == (env.num_agents - 1)

    # 3+ agent action prediction용: 각 agent의 partner indices (obs 순서: 자기 제외, index 정렬)
    if num_partners > 1:
        _partner_map = jnp.array([
            sorted(set(range(env.num_agents)) - {i}) for i in range(env.num_agents)
        ])  # (num_agents, num_partners)

    use_population_annealing = False
    if "POPULATION_ANNEAL_HORIZON" in config:
        _vprint("Using population annealing")
        use_population_annealing = True
        transition_begin = 0
        if "POPULATION_ANNEAL_BEGIN" in config:
            transition_begin = config["POPULATION_ANNEAL_BEGIN"]

        anneal_horizon = config["POPULATION_ANNEAL_HORIZON"]
        if anneal_horizon == 0:
            population_annealing_schedule = optax.constant_schedule(1.0)
        else:
            population_annealing_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=1.0,
                transition_steps=config["POPULATION_ANNEAL_HORIZON"] - transition_begin,
                transition_begin=transition_begin,
            )

    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
        initial_runner_state=None,
        log_seed_override=None,
        update_step_offset_override=None,
        population_params_override=None,
        num_update_steps_override=None,
    ):
        original_seed = rng[0]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(reset_fn)(reset_rng)

        # Execution input shape: environment-provided observation
        # (NUM_ENVS, H, W, C_obs)
        state_shape = obsv[env.agents[0]].shape[1:]

        # PH1 blocked target shape: global full state with agent info.
        if ph1_enabled:
            global_full_env0 = _extract_global_full_obs(env_state)  # (E, H, W, C_full)
            ph1_block_shape = global_full_env0.shape[1:]
        else:
            ph1_block_shape = None

        # CT v2: pixel decoder가 full global state shape을 알아야 함 → config에 주입
        # ph1_block_shape가 없으면 직접 추출 (ph1_enabled=False 케이스)
        if transformer_action and transformer_v2:
            _v2_state_shape = (
                ph1_block_shape
                if ph1_block_shape is not None
                else _extract_global_full_obs(env_state).shape[1:]
            )
            config["TRANSFORMER_STATE_SHAPE"] = list(_v2_state_shape)

        # INIT NETWORK
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)

        init_x = (
            jnp.zeros(
                (1, model_config["NUM_ENVS"], *state_shape),
            ),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

        if init_hstate is not None:
            if verbose_logs:
                if isinstance(init_hstate, tuple):
                    print("init_hstate (tuple)", [x.shape for x in init_hstate])
                else:
                    print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        _vprint("init_x", init_x[0].shape, init_x[1].shape)

        # E3T: Prepare dummy inputs for Single Initialization
        # action_prediction: prediction is generated internally after GRU, so partner_prediction
        # is always None at call sites. Init must match.
        dummy_partner_prediction = None
        dummy_blocked_states = None
        # NOTE: agent_idx conditioning removed

        # Keep init-time blocked input rule consistent with runtime apply rule.
        # PH2 phase2(ind) learner path uses population partner and should not consume blocked input.
        use_blocked_input_learner = learner_use_blocked_input
        if use_blocked_input_learner:
            if config.get("PH1_ENABLED", False):
                # PH1 always uses global full state as blocked target
                if ph1_multi_penalty_enabled or ph1_pair_mode:
                    init_shape = (
                        1,
                        model_config["NUM_ENVS"],
                        ph1_penalty_slots,
                    ) + ph1_block_shape
                else:
                    init_shape = (1, model_config["NUM_ENVS"]) + ph1_block_shape
                dummy_blocked_states = jnp.zeros(init_shape, dtype=jnp.float32)

        # Single Init: Initialize all parameters at once to avoid collision
        # 단일 초기화: 파라미터 충돌 방지를 위해 모든 모듈을 한 번에 초기화
        network_params = network.init(
            _rng,
            init_hstate,
            init_x,
            partner_prediction=dummy_partner_prediction,
            blocked_states=dummy_blocked_states,
        )

        if transformer_action:
            # Main network.init() calls encode_only in the CT branch, which initializes:
            #   ct_obs_encoder*, encoder (CausalTransformerEncoder), state_fc/ln, action_*
            # Missing after main init (not called by encode_only):
            #   cycle_hidden, cycle_out  ← only called in __call__ (full forward)
            #   ct_state_encoder*        ← only called in ct_encode_state
            # Initialize both separately and merge into params["cycle_transformer"].
            from flax.core import freeze, unfreeze
            _W = int(config.get("TRANSFORMER_WINDOW_SIZE", 16))
            _D_obs = int(model_config["GRU_HIDDEN_DIM"])
            # 1) Full CT forward init (adds cycle_hidden, cycle_out)
            _dummy_ct_windows = jnp.zeros((1, _W, _D_obs), dtype=jnp.float32)
            _ct_full_vars = network.init(
                _rng, _dummy_ct_windows, method=network.cycle_transformer_forward
            )
            # 2) CT state encoder init (adds ct_state_encoder, ct_state_encoder_ln)
            # ct_state_encoder는 full global state shape을 입력받음 (OV2에서 state_shape ≠ full obs shape)
            # ph1_block_shape = get_obs_default() 출력 shape (full grid, C_full channels)
            # ph1_enabled=False이면 여기서 직접 추출
            if ph1_block_shape is not None:
                _ct_state_shape = ph1_block_shape
            else:
                _ct_state_shape = _extract_global_full_obs(env_state).shape[1:]
            _dummy_obs_ct = jnp.zeros(
                (1, model_config["NUM_ENVS"]) + _ct_state_shape, dtype=jnp.float32
            )
            _ct_state_vars = network.init(_rng, _dummy_obs_ct, method=network.ct_encode_state)
            # Merge both into params["cycle_transformer"] (only add missing keys)
            _np = unfreeze(network_params)
            _ct_existing = set(_np["params"]["cycle_transformer"].keys())
            for k, v in unfreeze(_ct_full_vars)["params"]["cycle_transformer"].items():
                if k not in _ct_existing:
                    _np["params"]["cycle_transformer"][k] = v
            for k, v in unfreeze(_ct_state_vars)["params"]["cycle_transformer"].items():
                if k not in _ct_existing:
                    _np["params"]["cycle_transformer"][k] = v
            network_params = freeze(_np)

        # CycleDecoder init (CT OFF cycle loss용) — cycle_decode method 호출 시 필요한 params
        if cycle_loss_enabled and not transformer_action:
            from flax.core import freeze, unfreeze
            # CycleDecoder 입력 차원: z_pred(128) + pred(6) = 134 또는 pred(6) 또는 z_pred(128)
            _cd_in_dim = 0
            if z_prediction_enabled:
                _cd_in_dim += model_config["GRU_HIDDEN_DIM"]
            if action_prediction:
                _cd_in_dim += ACTION_DIM
            if _cd_in_dim > 0:
                _dummy_cd_in = jnp.zeros((1, _cd_in_dim), dtype=jnp.float32)
                _cd_vars = network.init(_rng, _dummy_cd_in, method=network.cycle_decode)
                _np = unfreeze(network_params)
                for k, v in unfreeze(_cd_vars)["params"].items():
                    if k not in _np["params"]:
                        _np["params"][k] = v
                network_params = freeze(_np)

        # ------------------------------------------------------------------
        # Optimizer 구성 (GridSpread SPLIT_OPTIMIZER 매칭)
        #   - LR_SCHEDULE="linear" → linear LR decay (LR → 0)
        #   - SPLIT_OPTIMIZER=True → actor/critic params 를 multi_transform 으로 분리,
        #       각각 max_grad_norm clip + adam(eps=1e-5). onpolicy r_mappo 매칭.
        #   - critic 식별: param path 에 "critic" substring (rnn.py 에서 GS 시 명시 name).
        # 다른 환경 (Overcooked 등) 은 ANNEAL_LR + cosine 기존 경로 그대로.
        # ------------------------------------------------------------------
        _lr_schedule_kind = str(model_config.get("LR_SCHEDULE", "")).lower()
        if _lr_schedule_kind == "linear":
            _total_train_steps = (
                model_config["NUM_UPDATES"]
                * model_config["UPDATE_EPOCHS"]
                * model_config["NUM_MINIBATCHES"]
            )
            _lr_fn = optax.linear_schedule(
                init_value=model_config["LR"],
                end_value=0.0,
                transition_steps=_total_train_steps,
            )
        elif model_config["ANNEAL_LR"]:
            _lr_fn = create_learning_rate_fn()
        else:
            _lr_fn = model_config["LR"]

        _split_optimizer = bool(model_config.get("SPLIT_OPTIMIZER", False))

        def _make_single_tx(lr_):
            return optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(lr_, eps=1e-5),
            )

        if _split_optimizer:
            from flax.core import FrozenDict as _FD
            def _label_params(params):
                def _walk(node, path):
                    if isinstance(node, (dict, _FD)):
                        return {k: _walk(v, path + (str(k),)) for k, v in node.items()}
                    is_critic = any("critic" in p.lower() for p in path)
                    return "critic" if is_critic else "actor"
                return _walk(params, ())

            tx = optax.multi_transform(
                {
                    "actor": _make_single_tx(_lr_fn),
                    "critic": _make_single_tx(_lr_fn),
                },
                param_labels=_label_params,
            )
            _labels_tree = _label_params(network_params)
            _flat_labels = jax.tree_util.tree_leaves(_labels_tree)
            from collections import Counter as _Counter
            print(f"[SPLIT-OPT][ph2] param label dist = {_Counter(_flat_labels)}")
        else:
            tx = _make_single_tx(_lr_fn)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        if initial_train_state is not None:
            train_state = initial_train_state

        # INIT ENV state already created above
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        # jax.debug.print("check2 {x}", x=init_hstate.flatten()[0])

        init_population_hstate = None
        init_population_annealing_mask = None
        if population is not None:
            is_policy_population = False
            if isinstance(population, AbstractPolicy):
                is_policy_population = True
                rng, _rng = jax.random.split(rng)
                init_population_hstate = population.init_hstate(
                    model_config["NUM_ACTORS"], key=_rng
                )
            else:
                assert (
                    population_config is not None
                ), "population_config cannot be None if population is not a policy"
                # population 체크포인트에 ACTION_DIM이 없을 수 있으므로 현재 환경 값 주입
                if "ACTION_DIM" not in population_config.get("model", {}):
                    population_config.setdefault("model", {})["ACTION_DIM"] = ACTION_DIM
                population_network = get_actor_critic(population_config)
                init_population_hstate = initialize_carry(
                    population_config, model_config["NUM_ACTORS"]
                )

                fcp_population_size = jax.tree_util.tree_flatten(population)[0][0].shape[0]
                print("FCP population size", fcp_population_size)

                # print(f"normal hstate {init_hstate.shape}")
                # print(f"population hstate {init_population_hstate.shape}")

            if use_population_annealing:

                def _sample_population_annealing_mask(step, rng):
                    return jax.random.uniform(
                        rng, (model_config["NUM_ENVS"],)
                    ) < population_annealing_schedule(step)

                def _make_train_mask(annealing_mask):
                    full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
                    return jnp.where(full_anneal_mask, train_mask_flat, True)

                rng, _rng = jax.random.split(rng)
                init_population_annealing_mask = _sample_population_annealing_mask(
                    0, _rng
                )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                last_partner_action,
                last_action,
                ego_idxs,
                blocked_states_env,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                ph1_pool_states,
                ph1_probs,
                ph1_pool_ready,
                phase2_ind_match_env_state,
                ph2_spread_ind_ego_mask_state,  # (NUM_ACTORS,) bool — GridSpread spec-ind ego 배정
                vn_state,
                rng,
            ) = runner_state

            # jax.debug.print("check3 {x}", x=initial_hstate.flatten()[0])
            ph1_beta_current, ph1_beta_progress = _current_ph1_beta(update_step)

            # COLLECT TRAJECTORIES
            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    population_hstate,
                    population_annealing_mask,
                    fcp_pop_agent_idxs,
                    last_partner_action,
                    last_action,
                    ego_idxs,
                    blocked_states_env,
                    ph1_pool_states,
                    episode_returns_penalized,
                    returned_episode_returns_penalized,
                    episode_hit_counts,
                    returned_episode_hit_counts,
                    phase2_ind_match_env_state,
                    ph2_spread_ind_ego_mask_state,  # (NUM_ACTORS,) bool
                    rng,
                ) = env_step_state

                # [E3T] Dynamic Ego Assignment Logic
                # Check episode completion using last_done
                # last_done is (NUM_ACTORS,), reshaped to check env-wise done
                # OvercookedV2 agents terminate simultaneously
                last_done_reshaped = last_done.reshape(env.num_agents, model_config["NUM_ENVS"])
                episode_done = last_done_reshaped[0] # (NUM_ENVS,)

                rng, _rng = jax.random.split(rng)
                new_random_idxs = jax.random.randint(_rng, (model_config["NUM_ENVS"],), 0, env.num_agents)

                # Update ego_idxs only for environments that just finished
                ego_idxs = jnp.where(episode_done, new_random_idxs, ego_idxs)

                # Calculate actor indices and is_ego
                actor_indices = jnp.repeat(
                    jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
                )
                
                # Expand ego_idxs (NUM_ENVS,) to match actor_indices (NUM_ACTORS,)
                # actor_indices is [0...0, 1...1] (Agent-Major)
                # target_ego_idxs should be [e0...en, e0...en]
                target_ego_idxs = jnp.tile(ego_idxs, env.num_agents)
                
                is_ego = (actor_indices == target_ego_idxs)
                partner_actor_mask = ~is_ego

                # partner 위치 계산 (N-agent 일반화)
                pos_y, pos_x = _extract_pos_axes(env_state)
                env_range = jnp.arange(model_config["NUM_ENVS"])
                # 첫 번째 partner (backward compat용)
                partner_idxs = (ego_idxs + 1) % env.num_agents
                partner_y = pos_y[partner_idxs, env_range]
                partner_x = pos_x[partner_idxs, env_range]
                partner_pos = jnp.stack([partner_y, partner_x], axis=-1)  # (NUM_ENVS, 2)

                # 에피소드 종료 시 blocked target 재샘플링
                if config.get("PH1_ENABLED", False):
                    rng, _rng = jax.random.split(rng)
                    rng_envs = jax.random.split(_rng, model_config["NUM_ENVS"])

                    current_global_step = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                    is_ph1_warmup = current_global_step < ph1_warmup_steps

                    if env_name == "ToyCoop":
                        # ---------------------------------------------------------
                        # ToyCoop: pool 미사용, 현재 goal 기준으로 k개 synthetic
                        # penalty state 직접 생성 (agent_pos만 랜덤, goal 겹침 없음)
                        # ---------------------------------------------------------
                        _tc_inner = env_state.env_state  # ToyCoop State
                        _tc_all_pos = jnp.array(
                            [[x, y] for x in range(5) for y in range(5)]
                        )  # (25, 2)

                        def _gen_toycoop_targets_one_env(rng_e, gp, ogp):
                            # green goal과 겹치지 않는 위치 마스크
                            mask = jnp.ones(25, dtype=jnp.bool_)
                            match0 = jnp.all(_tc_all_pos == gp[0], axis=-1)
                            match1 = jnp.all(_tc_all_pos == gp[1], axis=-1)
                            mask = mask & ~match0 & ~match1
                            valid_idx = jnp.where(mask, size=23)[0]

                            def _gen_one(rng_k):
                                perm = jax.random.permutation(rng_k, valid_idx.shape[0])
                                a0 = _tc_all_pos[valid_idx[perm[0]]]
                                a1 = _tc_all_pos[valid_idx[perm[1]]]
                                agent_pos = jnp.stack([a0, a1])
                                obs_grid = jnp.zeros((5, 5, 4))
                                obs_grid = obs_grid.at[agent_pos[0, 1], agent_pos[0, 0], 0].set(1)
                                obs_grid = obs_grid.at[agent_pos[1, 1], agent_pos[1, 0], 1].set(1)
                                obs_grid = obs_grid.at[gp[:, 1], gp[:, 0], 2].set(1)
                                obs_grid = obs_grid.at[ogp[:, 1], ogp[:, 0], 3].set(1)
                                return obs_grid  # (5, 5, 4)

                            # 원래 position 개수만큼 생성 (pair는 후처리)
                            _base_slots = ph1_penalty_slots // 2 if ph1_pair_mode else ph1_penalty_slots
                            rngs_k = jax.random.split(rng_e, _base_slots)
                            base_targets = jax.vmap(_gen_one)(rngs_k)  # (_base_slots, 5, 5, 4)

                            if ph1_pair_mode:
                                # Ch0↔Ch1 swap 버전 생성 → 원본과 interleave
                                ch0 = base_targets[:, :, :, 0]
                                ch1 = base_targets[:, :, :, 1]
                                swapped = base_targets.at[:, :, :, 0].set(ch1).at[:, :, :, 1].set(ch0)
                                # interleave: [orig_0, swap_0, orig_1, swap_1, ...]
                                paired = jnp.stack([base_targets, swapped], axis=1)  # (_base, 2, 5, 5, 4)
                                return paired.reshape(-1, 5, 5, 4)  # (_base*2, 5, 5, 4)
                            return base_targets  # (_base_slots, 5, 5, 4)

                        new_targets_multi = jax.vmap(_gen_toycoop_targets_one_env)(
                            rng_envs,
                            _tc_inner.goal_pos,
                            _tc_inner.other_goal_pos,
                        )  # (E, k*M, 5, 5, 4) where M=2 if pair_mode else 1

                        if ph1_multi_penalty_enabled or ph1_pair_mode:
                            new_targets = new_targets_multi  # (E, ph1_penalty_slots, 5, 5, 4)
                        else:
                            new_targets = new_targets_multi[:, 0]  # (E, 5, 5, 4)

                        # warmup 기간에는 -1 (None target)
                        none_target = jnp.full_like(new_targets, -1.0)
                        new_targets = jnp.where(is_ph1_warmup, none_target, new_targets)

                    else:
                        # 기존 OV2: pool에서 v-gap sampling
                        pool_size = ph1_pool_states.shape[1]

                    if env_name != "ToyCoop" and ph1_multi_penalty_enabled:

                        def _sample_multi_target(r, pool_env, probs_env):
                            none_targets = jnp.full(
                                (ph1_penalty_slots,) + pool_env.shape[1:],
                                -1,
                                dtype=pool_env.dtype,
                            )

                            def _do_sample(_):
                                targets, _mask, _count = sample_multi_targets_from_pool(
                                    r,
                                    pool_env,
                                    probs_env,
                                    max_penalty_count=ph1_penalty_slots,
                                    single_weight=ph1_multi_penalty_single_weight,
                                    other_weight=ph1_multi_penalty_other_weight,
                                )
                                return targets

                            return jax.lax.cond(
                                is_ph1_warmup | (~ph1_pool_ready),
                                lambda _: none_targets,
                                _do_sample,
                                operand=None,
                            )

                        new_targets = jax.vmap(_sample_multi_target)(
                            rng_envs, ph1_pool_states, ph1_probs
                        )
                    elif env_name != "ToyCoop":
                        # OV2 single-target: pool에서 v-gap sampling

                        def _sample_target(r, probs_env):
                            # If warmup or pool not ready, force normal (index = pool_size)
                            return jax.lax.cond(
                                is_ph1_warmup | (~ph1_pool_ready),
                                lambda _: jnp.int32(pool_size),
                                lambda _: sample_target_idx(r, probs_env),
                                operand=None,
                            )

                        target_idxs = jax.vmap(_sample_target)(rng_envs, ph1_probs)

                        def _pick_candidate(idx, pool_env):
                            is_none = (idx == pool_size)
                            safe_idx = jnp.clip(idx, 0, pool_size - 1)
                            cand = pool_env[safe_idx]
                            none_val = jnp.full_like(cand, -1)
                            return jax.lax.select(is_none, none_val, cand)

                        new_targets = jax.vmap(_pick_candidate)(
                            target_idxs, ph1_pool_states
                        )

                    # done인 환경만 업데이트
                    # blocked_states_env shape: (Envs, ...)
                    # episode_done: (Envs,)
                    # Reshape done to (Envs, 1...)
                    done_expanded = episode_done.reshape(
                        (model_config["NUM_ENVS"],)
                        + (1,) * (blocked_states_env.ndim - 1)
                    )

                    blocked_states_env = jnp.where(
                        done_expanded,
                        new_targets,
                        blocked_states_env,
                    )
                if config.get("PH1_ENABLED", False):
                    # PH1 Expansion Logic
                    # blocked_states_env: (Envs, ...)  <-- Same target for both agents
                    # Agent order is [Ag0_E0...Ag0_En, Ag1_E0...Ag1_En] (Agent-Major)

                    # N-agent 일반화: env-level blocked state를 모든 agent에 복제
                    blocked_states_actor = jnp.tile(
                        blocked_states_env,
                        [env.num_agents] + [1] * (blocked_states_env.ndim - 1)
                    )
                    
                else:
                    blocked_states_actor = jnp.full(
                        (model_config["NUM_ACTORS"], 2), -1, dtype=jnp.int32
                    )

                # PH2 ind role:
                #   phase2_ind_match_env_state=True (OV2) / ==2 (GridSpread) -> ind-ind
                #   phase2_ind_match_env_state=False (OV2) / ==1 (GridSpread) -> spec-ind
                # For ind-ind, disable blocked target input by forcing normal target (-1).
                if ph2_match_schedule and (ph2_role == "ind"):
                    if is_spread:
                        # GridSpread: int32 match_type, ind-ind = 2
                        ind_ind_env = (phase2_ind_match_env_state == 2)
                    else:
                        # OV2: bool, True = ind-ind
                        ind_ind_env = phase2_ind_match_env_state
                    ind_ind_actor_mask = jnp.tile(
                        ind_ind_env, env.num_agents
                    )
                    ind_ind_actor_mask = ind_ind_actor_mask.reshape(
                        (model_config["NUM_ACTORS"],)
                        + (1,) * (blocked_states_actor.ndim - 1)
                    )
                    normal_block = jnp.full_like(blocked_states_actor, -1)
                    blocked_states_actor = jnp.where(
                        ind_ind_actor_mask, normal_block, blocked_states_actor
                    )

                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *state_shape
                )
                if cast_obs_bf16:
                    obs_batch = obs_batch.astype(jnp.bfloat16)

                # Partner prediction is handled internally by the network (action-prediction mode)
                partner_prediction = None
                combined_prediction = jnp.zeros((model_config["NUM_ACTORS"], policy_pred_dim))

                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                # [STA-PH1] Handle Extra Return (Latent Embeddings)
                # ToyCoop/MPE: pool 미충전 시 -1.0 blocked states → 0.0 대체 (LayerNorm NaN 방지)
                _blocked_input = blocked_states_actor if learner_use_blocked_input else None
                if _blocked_input is not None and (env_name == "ToyCoop" or _is_mpe):
                    _blocked_input = jnp.where(_blocked_input == -1.0, 0.0, _blocked_input)
                # MPE (1D obs): ac_in에 time dim(np.newaxis)을 추가하므로 blocked_states도 동일하게 맞춤.
                # OV2 (grid obs): 이미 CNN이 batch dim을 처리하므로 newaxis 불필요.
                if _blocked_input is not None and _is_mpe:
                    _blocked_input = _blocked_input[np.newaxis, :]

                # === GridSpread 전용 action masking ===
                # env._env는 LogWrapper 내부 base env. get_avail_actions는 base env 메서드.
                if is_spread:
                    _avail_dict = jax.vmap(env._env.get_avail_actions)(env_state.env_state)
                    _avail_batch = jnp.stack(
                        [_avail_dict[a] for a in env.agents]
                    ).reshape(-1, ACTION_DIM)  # (NUM_ACTORS, ACTION_DIM)
                    _avail_in = _avail_batch[np.newaxis, :]  # (1, NUM_ACTORS, ACTION_DIM)
                    net_out = network.apply(
                        train_state.params,
                        hstate,
                        ac_in,
                        partner_prediction=partner_prediction,
                        blocked_states=_blocked_input,
                        avail_actions=_avail_in,
                    )
                else:
                    _avail_batch = jnp.zeros(
                        (model_config["NUM_ACTORS"], ACTION_DIM), dtype=jnp.int32
                    )
                    net_out = network.apply(
                        train_state.params,
                        hstate,
                        ac_in,
                        partner_prediction=partner_prediction,
                        blocked_states=_blocked_input,
                    )

                ph1_extras = {}
                if len(net_out) == 4:
                    hstate, pi, value, ph1_extras = net_out
                    # For action prediction: extract pred_logits from extras as combined_prediction
                    if (
                        action_prediction
                        and (e3t_enabled or (ph1_enabled and use_ph1_partner_pred))
                        and use_partner_modeling
                        and prediction_enabled
                    ):
                        _pred_logits = ph1_extras.get("pred_logits", None)
                        if _pred_logits is not None:
                            combined_prediction = _pred_logits.squeeze(0)
                else:
                    hstate, pi, value = net_out

                # jax.debug.print("check5 {x}", x=hstate.flatten()[0])

                num_action_choices = pi.logits.shape[-1]
                # policy 정규화 진행해야 하나? => 이미 함수 안에서 되어 있음.
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                # policy_action(action)과 환경에 전달할 action을 분리해서 관리한다.
                #  - trajectory/log_prob 계산에는 action(정책 샘플) 그대로 사용
                #  - 환경 실행 시에는 이후 mask 로직을 거친 action_for_env를 사용한다.
                action_for_env = action
                raw_policy_entropy = pi.entropy().reshape(
                    (model_config["NUM_ACTORS"],)
                )
                log_action_dim = jnp.log(jnp.array(num_action_choices, dtype=jnp.float32))
                safe_log_action_dim = jnp.maximum(log_action_dim, 1e-6)
                policy_entropy = raw_policy_entropy / safe_log_action_dim

                action_pick_mask = jnp.ones(
                    (model_config["NUM_ACTORS"],), dtype=jnp.bool_
                )
                train_update_mask = action_pick_mask
                phase2_ind_match_env = phase2_ind_match_env_state
                phase2_ind_match_prob = jnp.float32(0.0)
                ph2_spread_ind_ego_mask = ph2_spread_ind_ego_mask_state  # 기본값 (population 유무 관계없이)
                if population is not None:
                    _vprint("Using population")

                    obs_population = obs_batch
                    if isinstance(population, AbstractPolicy):
                        # obs_featurized = jax.vmap(
                        #     env.get_obs_for_type, in_axes=(0, None)
                        # )(env_state.env_state, ObservationType.FEATURIZED)
                        # obs_population = batchify(
                        #     obs_featurized, env.agents, model_config["NUM_ACTORS"]
                        # )
                        pass  # NOTE: Use grid-based observations for FCP agents as well

                    if is_policy_population:
                        rng, _rng = jax.random.split(rng)
                        pop_blocked_states = None
                        if population_use_blocked_input:
                            pop_blocked_states = blocked_states_actor
                        pop_actions, population_hstate, _ = population.compute_action(
                            obs_population,
                            last_done,
                            population_hstate,
                            _rng,
                            params=population_params_override,
                            blocked_states=pop_blocked_states,
                        )
                    else:

                        def _compute_population_actions(
                            policy_idx, obs_pop, obs_ld, fcp_h_state
                        ):
                            current_p = jax.tree.map(
                                lambda x: x[policy_idx], population
                            )
                            current_ac_in = (
                                obs_pop[np.newaxis, np.newaxis, :],
                                jnp.array([obs_ld])[np.newaxis, :],
                            )
                            new_fcp_h_state, fcp_pi, _ = population_network.apply(
                                current_p,
                                jax.tree.map(lambda x: x[np.newaxis, :], fcp_h_state),
                                current_ac_in,
                            )
                            fcp_action = fcp_pi.sample(seed=_rng)
                            return fcp_action.squeeze(), jax.tree.map(
                                lambda x: x.squeeze(axis=0), new_fcp_h_state
                            )

                        pop_actions, population_hstate = jax.vmap(
                            _compute_population_actions
                        )(
                            fcp_pop_agent_idxs,
                            obs_population,
                            last_done,
                            population_hstate,
                        )

                    action_pick_mask = train_mask_flat
                    train_update_mask = action_pick_mask
                    if ph2_match_schedule and (ph2_role in ("spec", "ind")):
                        rng, rng_match = jax.random.split(rng)
                        if is_spread:
                            # ===== GridSpread 전용: 3-way categorical (0=spec-spec, 1=spec-ind, 2=ind-ind) =====
                            # 에피소드 시작 시만 재샘플; 그 외는 이전 match type 유지
                            logits = jnp.log(jnp.array([1/3, 1/3, 1/3]))
                            sampled_match_type = jax.random.categorical(
                                rng_match, logits, shape=(model_config["NUM_ENVS"],)
                            )
                            phase2_ind_match_env = jnp.where(
                                episode_done,
                                sampled_match_type.astype(jnp.int32),
                                phase2_ind_match_env_state.astype(jnp.int32),
                            )

                            # 3-way 액터별 bool 마스크
                            is_ss_actor = jnp.tile(phase2_ind_match_env == 0, env.num_agents)  # spec-spec
                            is_si_actor = jnp.tile(phase2_ind_match_env == 1, env.num_agents)  # spec-ind
                            is_ii_actor = jnp.tile(phase2_ind_match_env == 2, env.num_agents)  # ind-ind

                            # spec-ind ego 배정: episode 시작 시만 noise-rank 재샘플
                            # n_ego ~ Uniform[1, N-1]: 최소 1 ind ego, 최소 1 spec partner 보장
                            rng, rng_n, rng_noise = jax.random.split(rng, 3)
                            n_ego_per_env = jax.random.randint(
                                rng_n, shape=(model_config["NUM_ENVS"],),
                                minval=1, maxval=env.num_agents  # maxval exclusive → [1, N-1]
                            )
                            noise = jax.random.uniform(
                                rng_noise, shape=(env.num_agents, model_config["NUM_ENVS"])
                            )
                            ranks = jnp.argsort(jnp.argsort(noise, axis=0), axis=0)  # (N, E)
                            new_ind_ego = (ranks < n_ego_per_env[None, :]).reshape(-1)  # (NUM_ACTORS,) agent-major
                            episode_done_actor = jnp.tile(episode_done, env.num_agents)
                            ph2_spread_ind_ego_mask = jnp.where(
                                episode_done_actor, new_ind_ego, ph2_spread_ind_ego_mask_state
                            )

                            if ph2_role == "spec":
                                # spec-spec(0): 전원 spec 행동·학습
                                # spec-ind(1): non-ego(~ind_ego) spec 행동, 학습 X
                                # ind-ind(2): 전원 ind pop 행동 (action_pick_mask=0 → pop_actions 사용), 학습 X
                                action_pick_mask = jnp.where(
                                    is_ss_actor,
                                    jnp.ones_like(train_mask_flat),
                                    jnp.where(is_si_actor, ~ph2_spread_ind_ego_mask, jnp.zeros_like(train_mask_flat))
                                )
                                train_update_mask = jnp.where(
                                    is_ss_actor,
                                    jnp.ones_like(train_mask_flat),
                                    jnp.zeros_like(train_mask_flat),
                                )
                            else:  # ind
                                # spec-spec(0): 전원 spec pop, ind 학습 X
                                # spec-ind(1): ind ego만 행동·학습, 나머지 spec pop
                                # ind-ind(2): 전원 ind 행동·학습
                                action_pick_mask = jnp.where(
                                    is_ss_actor,
                                    jnp.zeros_like(train_mask_flat),
                                    jnp.where(is_si_actor, ph2_spread_ind_ego_mask, jnp.ones_like(train_mask_flat))
                                )
                                train_update_mask = action_pick_mask
                        else:
                            # ===== OV2: 기존 binary bernoulli 로직 완전히 그대로 =====
                            phase2_ind_match_prob = _phase2_ind_match_prob(update_step)
                            sampled_ind_match_env = jax.random.bernoulli(
                                rng_match,
                                p=phase2_ind_match_prob,
                                shape=(model_config["NUM_ENVS"],),
                            )
                            phase2_ind_match_env = jnp.where(
                                episode_done,
                                sampled_ind_match_env,
                                phase2_ind_match_env_state,
                            )
                            ind_match_actor = jnp.tile(phase2_ind_match_env, env.num_agents)
                            ph2_spread_ind_ego_mask = ph2_spread_ind_ego_mask_state  # OV2에서 미사용, 그대로 전달

                            if ph2_role == "spec":
                                # spec policy always acts in spec-match; in ind-match it acts as one side only.
                                action_pick_mask = jnp.where(
                                    ind_match_actor,
                                    train_mask_flat,
                                    jnp.ones_like(train_mask_flat),
                                )
                                # spec updates only on spec-match samples.
                                train_update_mask = jnp.where(
                                    ind_match_actor,
                                    jnp.zeros_like(train_mask_flat),
                                    jnp.ones_like(train_mask_flat),
                                )
                            else:
                                # ind role:
                                #   ind_match_env=True  -> ind-ind (both actors are ind)
                                #   ind_match_env=False -> spec-ind (ind only on train_mask slots)
                                action_pick_mask = jnp.where(
                                    ind_match_actor,
                                    jnp.ones_like(train_mask_flat),
                                    train_mask_flat,
                                )
                                # Update only samples where ind policy actually acted
                                # (all actors in ind-ind, train-mask slots in spec-ind).
                                train_update_mask = action_pick_mask
                    elif use_population_annealing:
                        action_pick_mask = _make_train_mask(population_annealing_mask)
                        train_update_mask = action_pick_mask

                    # use action_pick_mask to select the action from the population or the network
                    action_for_env = jnp.where(action_pick_mask, action_for_env, pop_actions)

                # E3T 알고리즘: 파트너 정책 혼합 (Mixture Partner Policy)
                # [E3T] 파트너 행동 변경 로직 (Mixture Policy)
                # 파트너는 (1-epsilon) 확률로 자신의 정책을 따르고, epsilon 확률로 무작위 행동을 수행함
                if e3t_enabled:
                    # 1. rng 분리 (혼합 정책용)
                    rng, rng_mix = jax.random.split(rng)
                    
                    # 2. 베르누이 마스크 생성 (p=E3T_EPSILON)
                    # True일 경우 무작위 행동을 선택
                    mix_mask = jax.random.bernoulli(rng_mix, p=e3t_epsilon, shape=(model_config["NUM_ACTORS"],))
                    
                    # [중요] 파트너 에이전트(partner_actor_mask)인 경우에만 무작위 행동 혼합을 적용
                    # Ego 에이전트는 자신의 정책을 그대로 따름
                    mix_mask = mix_mask & partner_actor_mask
                    
                    # 3. 완전 무작위 행동 샘플링
                    rng, rng_rand = jax.random.split(rng)
                    # Fix: Ensure shape is (NUM_ACTORS,) not (1, NUM_ACTORS)
                    rand_action = jax.random.randint(
                        rng_rand,
                        (model_config["NUM_ACTORS"],),
                        0,
                        num_action_choices,
                        dtype=action_for_env.dtype,
                    )
                    
                    # 4. 실제 파트너 행동 결정
                    # 마스크가 True이면 무작위 행동, False이면 기존 정책 행동(ego_policy_action) 사용
                    actual_partner_action = jax.lax.select(mix_mask, rand_action, action_for_env.squeeze())
                    
                    # 5. 환경에 전달할 행동 업데이트
                    # Fix: Force int32 type
                    action_for_env = actual_partner_action.astype(jnp.int32)

                # [PH2 epsilon]
                # - spec-spec: PH1과 동일(발동 env마다 에이전트 1명 무작위 치환)
                # - spec-ind: specialist actor에만 epsilon 랜덤행동 적용
                if ph1_enabled and ph2_epsilon > 0.0:
                    rng, rng_eps = jax.random.split(rng)
                    env_mask = jax.random.bernoulli(
                        rng_eps, p=ph2_epsilon, shape=(model_config["NUM_ENVS"],)
                    )
                    selected_actor = jnp.zeros(
                        (model_config["NUM_ACTORS"],), dtype=jnp.bool_
                    )

                    if (
                        ph2_match_schedule
                        and (population is not None)
                        and (ph2_role in ("spec", "ind"))
                    ):
                        action_pick_mask_bool = action_pick_mask.astype(jnp.bool_)
                        if ph2_role == "spec":
                            # spec role:
                            #   OV2:  phase2_ind_match_env=True  -> spec-ind, False -> spec-spec
                            #   Spread: ==1 -> spec-ind, ==0 -> spec-spec, ==2 -> ind-ind (spec 미참여)
                            if is_spread:
                                spec_ind_env = (phase2_ind_match_env == 1)
                                spec_spec_env = (phase2_ind_match_env == 0)
                            else:
                                spec_ind_env = phase2_ind_match_env.astype(jnp.bool_)
                                spec_spec_env = ~spec_ind_env

                            rng, rng_pick = jax.random.split(rng)
                            rand_agent = jax.random.randint(
                                rng_pick,
                                (model_config["NUM_ENVS"],),
                                0,
                                env.num_agents,
                            )
                            selected_actor_spec_spec = actor_indices == jnp.tile(
                                rand_agent, env.num_agents
                            )
                            selected_actor_spec_spec = selected_actor_spec_spec & jnp.tile(
                                env_mask & spec_spec_env, env.num_agents
                            )

                            selected_actor_spec_ind = action_pick_mask_bool & jnp.tile(
                                env_mask & spec_ind_env, env.num_agents
                            )
                            selected_actor = (
                                selected_actor_spec_spec | selected_actor_spec_ind
                            )
                        else:
                            # ind role:
                            #   OV2:  phase2_ind_match_env=True  -> ind-ind, False -> spec-ind
                            #   Spread: ==2 -> ind-ind, ==1 -> spec-ind, ==0 -> spec-spec (ind 미참여)
                            if is_spread:
                                spec_ind_env = (phase2_ind_match_env == 1)
                                ind_ind_env = (phase2_ind_match_env == 2)
                            else:
                                spec_ind_env = ~phase2_ind_match_env.astype(jnp.bool_)
                                ind_ind_env = phase2_ind_match_env.astype(jnp.bool_)
                            spec_actor_mask = ~action_pick_mask_bool
                            selected_actor_spec_ind = spec_actor_mask & jnp.tile(
                                env_mask & spec_ind_env, env.num_agents
                            )
                            rng, rng_pick = jax.random.split(rng)
                            rand_agent = jax.random.randint(
                                rng_pick,
                                (model_config["NUM_ENVS"],),
                                0,
                                env.num_agents,
                            )
                            selected_actor_ind_ind = actor_indices == jnp.tile(
                                rand_agent, env.num_agents
                            )
                            selected_actor_ind_ind = selected_actor_ind_ind & jnp.tile(
                                env_mask & ind_ind_env, env.num_agents
                            )
                            selected_actor = (
                                selected_actor_spec_ind | selected_actor_ind_ind
                            )
                    else:
                        # Fallback: PH1 기본 동작
                        rng, rng_pick = jax.random.split(rng)
                        rand_agent = jax.random.randint(
                            rng_pick, (model_config["NUM_ENVS"],), 0, env.num_agents
                        )
                        selected_actor = actor_indices == jnp.tile(
                            rand_agent, env.num_agents
                        )
                        selected_actor = selected_actor & jnp.tile(
                            env_mask, env.num_agents
                        )

                    rng, rng_rand = jax.random.split(rng)
                    rand_action = jax.random.randint(
                        rng_rand,
                        (model_config["NUM_ACTORS"],),
                        0,
                        num_action_choices,
                        dtype=action_for_env.dtype,
                    )

                    action_for_env = jnp.where(selected_actor, rand_action, action_for_env)

                # Ensure action_for_env is squeezed (NUM_ACTORS,) to match carry shape
                # This fixes the scan carry shape mismatch error (int32[512] vs int32[1,512])
                if len(action_for_env.shape) > 1:
                    action_for_env = action_for_env.squeeze()

                # Update last_partner_action for next step history
                last_partner_action = action_for_env

                env_act = unbatchify(
                    action_for_env,
                    env.agents,
                    model_config["NUM_ENVS"],
                    env.num_agents,
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV: 환경 스텝 실행
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    step_fn, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # original_reward: 환경에서 직접 반환한 원본 보상 (예: 배달 성공 시 +20, 실패 시 -10 등)
                original_reward = jnp.array([reward[a] for a in env.agents])
                reward_no_penalty = {a: reward[a] for a in env.agents}

                # [PH1] Build global full state (with agent info) for this next env_state.
                # This is used for:
                #  - env-specific target pool (recent visited full states)
                #  - latent distance penalty computation
                global_full_next_env0 = None
                if ph1_enabled:
                    global_full_next_env0 = _extract_global_full_obs(env_state)  # (E, H, W, C_full)

                    # Update env-specific ring buffer pool: (E, Pool, H, W, C_full)
                    # ToyCoop: pool 미사용 (synthetic penalty state 직접 생성)
                    if env_name != "ToyCoop":
                        new_pool = jnp.roll(ph1_pool_states, shift=-1, axis=1)
                        new_pool = new_pool.at[:, -1].set(global_full_next_env0)
                        # GridSpread: all_covered=True 상태를 pool에서 제외하려면 아래 블록 주석 해제.
                        # if is_spread:
                        #     # info["shaped_reward_events"]["agent_0"]: (NUM_ENVS, 2)
                        #     # index 0 = all_covered float, index 1 = coverage_ratio
                        #     all_covered_env = info["shaped_reward_events"]["agent_0"][:, 0].astype(jnp.bool_)
                        #     ph1_pool_states = jnp.where(
                        #         (~all_covered_env)[:, jnp.newaxis, jnp.newaxis],  # (E,1,1) broadcast
                        #         new_pool,
                        #         ph1_pool_states,
                        #     )
                        # else:
                        #     ph1_pool_states = new_pool
                        ph1_pool_states = new_pool
                
                # [STA-PH1] Apply Latent Distance Penalty
                # Calculate Distance: || z(next_obs) - z(blocked) ||
                # We need z(next_obs). We can run the network encoder on 'obsv' (next observation).
                
                dist_penalty = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.float32)
                if (
                    ph1_enabled
                    and (global_full_next_env0 is not None)
                    and ("blocked_emb" in ph1_extras)
                    and (ph1_extras["blocked_emb"] is not None)
                ):
                    # N-agent 일반화: env-level global obs를 모든 agent에 복제
                    global_full_next_actor = jnp.tile(
                        global_full_next_env0,
                        [env.num_agents] + [1] * (global_full_next_env0.ndim - 1)
                    )  # (Actors, H, W, C_full)

                    if ph1_raw_distance:
                        # raw vector distance (binary 0/1 obs → flatten → L2)
                        # PH1_RAW_DISTANCE=true 시 MLP encoder 경유 없이 직접 비교
                        z_next = global_full_next_actor.reshape(
                            global_full_next_actor.shape[0], -1
                        ).astype(jnp.float32)  # (Actors, 100)

                        # blocked_states_actor: (Actors, H, W, C) or (Actors, K, H, W, C)
                        if blocked_states_actor.ndim >= 5:
                            # multi-penalty: (Actors, K, H, W, C) → (Actors, K, 100)
                            z_tilde_slots = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0], blocked_states_actor.shape[1], -1
                            ).astype(jnp.float32)
                        else:
                            # single: (Actors, H, W, C) → (Actors, 1, 100)
                            z_tilde_slots = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0], -1
                            ).astype(jnp.float32)[:, None, :]
                        # -1.0 sentinel → 0.0 (미생성 state 보호)
                        z_tilde_slots = jnp.where(z_tilde_slots == -1.0, 0.0, z_tilde_slots)
                    else:
                        # blocked_encoder MLP latent distance (기본값: 모든 환경)
                        z_next = network.apply(
                            train_state.params,
                            global_full_next_actor,
                            method=network.encode_blocked,
                        ).squeeze()  # (Actors, D)
                        z_tilde_slots_src = ph1_extras.get("blocked_emb_slots", None)
                        if z_tilde_slots_src is not None:
                            z_tilde_slots = (
                                z_tilde_slots_src[0]
                                if z_tilde_slots_src.ndim >= 4
                                else z_tilde_slots_src
                            )
                            if z_tilde_slots.ndim == 2:
                                z_tilde_slots = z_tilde_slots[:, None, :]
                        else:
                            z_tilde_src = ph1_extras["blocked_emb"]
                            z_tilde = (
                                z_tilde_src[0] if z_tilde_src.ndim >= 3 else z_tilde_src
                            )
                            if z_tilde.ndim == 1:
                                z_tilde = z_tilde[None, :]
                            z_tilde_slots = z_tilde[:, None, :]

                    num_slots = z_tilde_slots.shape[1]
                    z_next_slots = jnp.expand_dims(z_next, axis=1)
                    lat_dist_slots = jnp.sqrt(
                        jnp.sum((z_next_slots - z_tilde_slots) ** 2, axis=-1)
                    )
                    # pair_mode: 2개씩 묶어서 min distance → agent 순서 무관 penalty
                    if ph1_pair_mode and lat_dist_slots.shape[1] >= 2:
                        n_pairs = lat_dist_slots.shape[1] // 2
                        dist_pairs = lat_dist_slots[:, :n_pairs * 2].reshape(
                            lat_dist_slots.shape[0], n_pairs, 2
                        )
                        lat_dist_slots = jnp.min(dist_pairs, axis=-1)  # (Actors, n_pairs)

                    if blocked_states_actor is not None:
                        if blocked_states_actor.ndim >= 5:
                            flat_blocks = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0],
                                blocked_states_actor.shape[1],
                                -1,
                            )
                            valid_slots = ~jnp.all(flat_blocks == -1, axis=-1)
                        else:
                            flat_blocks = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0], -1
                            )
                            valid_slots = ~jnp.all(flat_blocks == -1, axis=-1)
                            valid_slots = valid_slots[:, None]
                    else:
                        valid_slots = jnp.ones(
                            (model_config["NUM_ACTORS"], num_slots),
                            dtype=jnp.bool_,
                        )
                    # pair_mode: valid_slots도 2개씩 묶기 (둘 중 하나라도 valid이면 valid)
                    if ph1_pair_mode and valid_slots.shape[1] >= 2:
                        n_vp = valid_slots.shape[1] // 2
                        valid_pairs = valid_slots[:, :n_vp * 2].reshape(
                            valid_slots.shape[0], n_vp, 2
                        )
                        valid_slots = jnp.any(valid_pairs, axis=-1)  # (Actors, n_pairs)
                        num_slots = n_vp
                    if valid_slots.shape[1] == 1 and num_slots > 1:
                        valid_slots = jnp.tile(valid_slots, (1, num_slots))
                    valid_slots = valid_slots[:, :num_slots]
                    lat_dist_slots = jnp.where(valid_slots, lat_dist_slots, 0.0)

                    ph1_omega = config.get("PH1_OMEGA", 1.0)
                    ph1_sigma = config.get("PH1_SIGMA", 1.0)
                    penalty_slots = ph1_omega * jnp.exp(-ph1_sigma * lat_dist_slots)
                    penalty_slots = jnp.where(valid_slots, penalty_slots, 0.0)

                    # [Warmup] Disable penalty if warming up
                    current_global_step = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                    is_ph1_warmup = current_global_step < ph1_warmup_steps
                    penalty_slots = jnp.where(is_ph1_warmup, 0.0, penalty_slots)
                    dist_penalty = jnp.sum(penalty_slots, axis=-1)
                    lat_dist = jnp.sum(lat_dist_slots, axis=-1)

                    # Shared reward penalty (env-level mean)
                    dist_penalty_env = dist_penalty.reshape(
                        env.num_agents, model_config["NUM_ENVS"]
                    ).mean(axis=0)
                    lat_dist_env = lat_dist.reshape(
                        env.num_agents, model_config["NUM_ENVS"]
                    ).mean(axis=0)
                    dist_penalty_env_slots = penalty_slots.reshape(
                        env.num_agents, model_config["NUM_ENVS"], num_slots
                    ).mean(axis=0)
                    lat_dist_env_slots = lat_dist_slots.reshape(
                        env.num_agents, model_config["NUM_ENVS"], num_slots
                    ).mean(axis=0)

                    reward = {k: v - dist_penalty_env for k, v in reward.items()}
                    info["ph1_penalty_env"] = dist_penalty_env
                    info["ph1_dist_env"] = lat_dist_env
                    info["ph1_penalty_env_slots"] = dist_penalty_env_slots
                    info["ph1_dist_env_slots"] = lat_dist_env_slots
                
                # anneal_factor: 보상 쉐이핑 감쇠 계수 (학습 초반 1.0 → REW_SHAPING_HORIZON에 걸쳐 0.0으로 선형 감소)
                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                
                # combined_reward: original_reward + (shaped_reward * anneal_factor)
                # - shaped_reward는 환경이 제공하는 중간 단계 보상 (예: 재료 집기, 냄비에 넣기 등)
                # - 학습 초반에는 쉐이핑 보상이 크게 반영되어 학습 신호가 풍부하고,
                # - 학습 후반에는 anneal_factor가 0에 가까워져 원본 보상만 사용 (과최적화 방지)
                if "shaped_reward" in info:
                    reward = jax.tree_util.tree_map(
                        lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                    )
                reward_for_update = reward
                if ph2_match_schedule and (ph2_role == "ind"):
                    # ind learner update should use pure reward path (no blocking penalty).
                    if "shaped_reward" in info:
                        reward_for_update = jax.tree_util.tree_map(
                            lambda x, y: x + y * anneal_factor,
                            reward_no_penalty,
                            info["shaped_reward"],
                        )
                    else:
                        reward_for_update = reward_no_penalty

                # penalty 포함/미포함 reward 모두 로깅
                if "shaped_reward" in info:
                    shaped_reward = jnp.array(
                        [info["shaped_reward"][a] for a in env.agents]
                    )
                else:
                    shaped_reward = jnp.zeros((env.num_agents, model_config["NUM_ENVS"]))
                combined_reward = jnp.array([reward[a] for a in env.agents])
                combined_reward_no_penalty = (
                    jnp.array([reward_no_penalty[a] for a in env.agents])
                    + shaped_reward * anneal_factor
                )

                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward
                info["combined_reward_no_penalty"] = combined_reward_no_penalty

                # penalty 포함 episode return (combined_reward의 env-level mean 누적)
                ep_done = done["__all__"]
                combined_reward_env = combined_reward.mean(axis=0)
                new_episode_returns_penalized = (
                    episode_returns_penalized + combined_reward_env
                )
                episode_returns_penalized = new_episode_returns_penalized * (1 - ep_done)
                returned_episode_returns_penalized = jax.lax.select(
                    ep_done,
                    new_episode_returns_penalized,
                    returned_episode_returns_penalized,
                )
                returned_episode_returns_penalized_actor = jnp.tile(
                    returned_episode_returns_penalized, env.num_agents
                )
                info["returned_episode_returns_penalized_env"] = (
                    returned_episode_returns_penalized
                )
                # Deprecated alias: use returned_episode_returns_penalized_env instead.
                info["returned_episode_returns_penalized"] = (
                    returned_episode_returns_penalized_actor
                )

                # --------------------------------------------------------------
                # entropy 디버깅 정보를 info에 추가
                #   - policy_entropy: 각 슬롯별 정책 엔트로피 (항상 기록)
                # --------------------------------------------------------------
                info["policy_entropy"] = policy_entropy
                if ph2_match_schedule and (population is not None):
                    info["ph2_ind_match_env"] = phase2_ind_match_env.astype(jnp.float32)
                    info["ph2_ind_match_prob"] = jnp.full(
                        (model_config["NUM_ENVS"],),
                        phase2_ind_match_prob,
                        dtype=jnp.float32,
                    )

                def _reshape_info_leaf(x):
                    if not hasattr(x, "shape"):
                        return x
                    if (
                        x.ndim >= 2
                        and x.shape[0] == env.num_agents
                        and x.shape[1] == model_config["NUM_ENVS"]
                    ):
                        return x.reshape((model_config["NUM_ACTORS"],) + x.shape[2:])
                    if x.shape == (model_config["NUM_ENVS"],):
                        x = jnp.tile(x, env.num_agents)
                    elif x.ndim >= 2 and x.shape[0] == model_config["NUM_ENVS"]:
                        x = jnp.tile(
                            x,
                            (env.num_agents,) + (1,) * (x.ndim - 1),
                        )
                    if x.shape == ():
                        x = jnp.full((model_config["NUM_ACTORS"],), x)
                    if x.ndim >= 1 and x.shape[0] == model_config["NUM_ACTORS"]:
                        return x.reshape((model_config["NUM_ACTORS"],) + x.shape[1:])
                    return x

                info = {
                    k: (
                        batchify(v, env.agents, model_config["NUM_ACTORS"]).squeeze()
                        if isinstance(v, dict) and all(a in v for a in env.agents)
                        else jax.tree_util.tree_map(_reshape_info_leaf, v)
                    )
                    for k, v in info.items()
                }

                done_batch = batchify(
                    done, env.agents, model_config["NUM_ACTORS"]
                ).squeeze()

                if use_population_annealing:
                    env_steps = (
                        update_step
                        * model_config["NUM_STEPS"]
                        * model_config["NUM_ENVS"]
                    )
                    rng, _rng = jax.random.split(rng)
                    new_population_annealing_mask = jnp.where(
                        done["__all__"],
                        _sample_population_annealing_mask(env_steps, _rng),
                        population_annealing_mask,
                    )
                else:
                    new_population_annealing_mask = population_annealing_mask

                if population is not None and not is_policy_population:
                    new_fcp_pop_agent_idxs = jnp.where(
                        jnp.tile(done["__all__"], env.num_agents),
                        jax.random.randint(
                            _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
                        ),
                        fcp_pop_agent_idxs,
                    )
                else:
                    new_fcp_pop_agent_idxs = fcp_pop_agent_idxs

                # E3T: 현재 스텝의 파트너 행동 (Target Label)
                # action (원본 정책 샘플)을 기준으로 계산 (action_for_env는 epsilon 혼합 후 값)
                if env.num_agents == 2:
                    # 2-agent: jnp.roll로 agent0↔agent1 swap
                    current_partner_action = jnp.roll(action.squeeze(), shift=model_config["NUM_ENVS"], axis=0)
                else:
                    # 3+ agent: obs 순서대로 각 partner의 action을 gather
                    _acts = action.squeeze().reshape(env.num_agents, model_config["NUM_ENVS"])
                    _gathered = _acts[_partner_map]  # (num_agents, num_partners, NUM_ENVS)
                    current_partner_action = _gathered.transpose(0, 2, 1).reshape(-1, num_partners)

                # partner GRU z (v3 또는 Z_PREDICTION: partner의 GRU hidden state를 supervision으로 사용)
                # 주의: jnp.roll은 2-agent 전용. 3+ agent에서는 비활성화.
                if (transformer_v3 or z_prediction_enabled) and env.num_agents == 2:
                    _raw_gru = hstate[0] if transformer_action else hstate
                    current_partner_gru_z = jax.lax.stop_gradient(
                        jnp.roll(_raw_gru, shift=model_config["NUM_ENVS"], axis=0)
                    )
                else:
                    current_partner_gru_z = jnp.zeros(
                        (model_config["NUM_ACTORS"],), dtype=jnp.float32
                    )

                # CT recon target용 full global obs 추출 (transformer_action=True일 때만)
                # OV1: agent 0 full obs를 tile → OV2/ToyCoop: 각 actor 자신의 시점 full obs
                if transformer_action and not transformer_v3:
                    # v1/v2: collect global_obs for reconstruction target
                    if is_partial_obs or env_name == "ToyCoop" or _is_mpe:
                        # ToyCoop/MPE: per-actor obs 사용
                        _global_obs = _extract_global_full_obs_per_actor(env_state)
                    else:
                        _agent0_full = _extract_global_full_obs(env_state)   # (NUM_ENVS, H, W, C_full)
                        # N-agent 일반화: agent0 full obs를 모든 agent에 tile
                        _tile_reps = [env.num_agents] + [1] * (len(_agent0_full.shape) - 1)
                        _global_obs = jnp.tile(_agent0_full, _tile_reps)   # (NUM_ACTORS, H, W, C_full)
                else:
                    _global_obs = jnp.zeros_like(obs_batch)  # dummy — v3 또는 CT 비활성화

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward_for_update, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    train_update_mask,
                    current_partner_action,
                    is_ego,
                    blocked_states_actor,
                    last_action,
                    combined_prediction,
                    hstate,
                    _global_obs,
                    current_partner_gru_z,
                    _avail_batch,
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    population_hstate,
                    new_population_annealing_mask,
                    new_fcp_pop_agent_idxs,
                    last_partner_action,
                    action.squeeze(),
                    ego_idxs,
                    blocked_states_env,
                    ph1_pool_states,
                    episode_returns_penalized,
                    returned_episode_returns_penalized,
                    episode_hit_counts,
                    returned_episode_hit_counts,
                    phase2_ind_match_env,
                    ph2_spread_ind_ego_mask,  # (NUM_ACTORS,) bool — updated per-episode
                    rng,
                )
                return env_step_state, transition

            env_step_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                last_partner_action,
                last_action,
                ego_idxs,
                blocked_states_env,
                ph1_pool_states,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                phase2_ind_match_env_state,
                ph2_spread_ind_ego_mask_state,  # (NUM_ACTORS,) bool
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                last_partner_action,
                next_last_action,
                next_ego_idxs,
                blocked_states_env,
                ph1_pool_states,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                next_phase2_ind_match_env_state,
                next_ph2_spread_ind_ego_mask_state,  # (NUM_ACTORS,) bool
                rng,
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *state_shape
            )
            if cast_obs_bf16:
                last_obs_batch = last_obs_batch.astype(jnp.bfloat16)
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            # [E3T] For action prediction, prediction is computed internally by network.apply
            partner_prediction = None
            # [E3T] last_val 계산 시점의 ego 마스크
            target_ego_idxs = jnp.tile(next_ego_idxs, env.num_agents)
            is_ego_last = (actor_indices == target_ego_idxs)

            # [PH1] last_val 계산에도 blocked_states를 전달
            blocked_states_actor_last = None
            if config.get("PH1_ENABLED", False):
                # N-agent 일반화: env-level blocked state를 모든 agent에 복제
                blocked_states_actor_last = jnp.tile(
                    blocked_states_env,
                    [env.num_agents] + [1] * (blocked_states_env.ndim - 1)
                )

            _blocked_last = (
                blocked_states_actor_last
                if learner_use_blocked_input
                else None
            )
            # MPE (1D obs): ac_in에 time dim 추가와 동일하게 blocked_states에도 맞춤
            if _blocked_last is not None and _is_mpe:
                _blocked_last = _blocked_last[np.newaxis, :]

            net_out = network.apply(
                train_state.params,
                next_initial_hstate,
                ac_in,
                partner_prediction=partner_prediction,
                blocked_states=_blocked_last,
            )
            
            if len(net_out) == 4:
                _, _, last_val, _ = net_out
            else:
                _, _, last_val = net_out

            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + model_config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + model_config["GAMMA"]
                        * model_config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # ValueNorm: 네트워크 출력은 normalized space → GAE 전 denormalize, 후 normalize.
            if use_valuenorm:
                _denorm_traj = traj_batch._replace(
                    value=valuenorm_denormalize(vn_state, traj_batch.value)
                )
                _denorm_last_val = valuenorm_denormalize(vn_state, last_val)
                advantages, targets = _calculate_gae(_denorm_traj, _denorm_last_val)
                vn_state = valuenorm_update(vn_state, targets)
                targets = valuenorm_normalize(vn_state, targets)
            else:
                advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_and_rng, batch_info):
                    train_state, mb_rng = train_state_and_rng
                    mb_rng, loss_rng = jax.random.split(mb_rng)
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, loss_rng):
                        hstate = init_hstate
                        if hstate is not None:
                            if isinstance(hstate, tuple):
                                hstate = tuple(x.squeeze(axis=0) for x in hstate)
                            else:
                                hstate = hstate.squeeze(axis=0)
                            # hstate = jax.lax.stop_gradient(hstate)

                        train_mask = True
                        if population is not None:
                            train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                        def _masked_mean(x, mask):
                            x = jnp.asarray(x, dtype=jnp.float32)
                            if isinstance(mask, bool):
                                return x.mean()
                            mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
                            if mask_arr.ndim == 0:
                                return x.mean()
                            while mask_arr.ndim < x.ndim:
                                mask_arr = mask_arr[..., None]
                            mask_f = mask_arr.astype(jnp.float32)
                            denom = jnp.maximum(mask_f.sum(), 1.0)
                            return (x * mask_f).sum() / denom

                        # --------------------------------------------------------------
                        # E3T Partner Prediction Loss & Gradient Blocking
                        # --------------------------------------------------------------
                        pred_loss = 0.0
                        pred_accuracy = 0.0
                        state_pred_z_loss = 0.0
                        state_pred_action_loss = 0.0
                        partner_prediction = None
                        
                        use_ph1 = config.get("PH1_ENABLED", False)

                        target_labels_flat = traj_batch.partner_action.astype(jnp.int32)

                        # RERUN NETWORK (single pass)
                        # For action prediction: pred_logits returned in extras (computed after GRU)
                        # ToyCoop 전용: -1.0 blocked states → 0.0 대체 (LayerNorm NaN 방지)
                        _blocked_for_loss = traj_batch.blocked_states if learner_use_blocked_input else None
                        if _blocked_for_loss is not None and (env_name == "ToyCoop" or _is_mpe):
                            _blocked_for_loss = jnp.where(_blocked_for_loss == -1.0, 0.0, _blocked_for_loss)

                        # GridSpread: traj_batch.avail_actions를 전달해서 rollout과 동일한 masked pi 재생성.
                        if is_spread:
                            net_out = network.apply(
                                params,
                                hstate,
                                (traj_batch.obs, traj_batch.done),
                                partner_prediction=partner_prediction,
                                blocked_states=_blocked_for_loss,
                                avail_actions=traj_batch.avail_actions,
                            )
                        else:
                            net_out = network.apply(
                                params,
                                hstate,
                                (traj_batch.obs, traj_batch.done),
                                partner_prediction=partner_prediction,
                                blocked_states=_blocked_for_loss,
                            )

                        if len(net_out) == 4:
                            _, pi, value, rerun_extras = net_out
                        else:
                            _, pi, value = net_out
                            rerun_extras = {}

                        # For action prediction: compute CE loss from pred_logits in extras
                        if (
                            action_prediction
                            and (e3t_enabled or use_ph1)
                            and use_partner_modeling
                            and prediction_enabled
                        ):
                            pred_logits = rerun_extras.get("pred_logits", None)
                            if pred_logits is not None:
                                if num_partners == 1:
                                    # 2-agent: pred_logits (MB, A), target (MB,)
                                    pred_loss_vec = optax.softmax_cross_entropy_with_integer_labels(
                                        logits=pred_logits,
                                        labels=target_labels_flat,
                                    )
                                    pred_loss = _masked_mean(pred_loss_vec, train_mask)
                                    pred_labels = jnp.argmax(pred_logits, axis=-1)
                                    pred_accuracy = _masked_mean(
                                        pred_labels == target_labels_flat,
                                        train_mask,
                                    )
                                else:
                                    # 3+ agent: pred_logits (..., A*num_partners), target (..., num_partners)
                                    # 마지막 차원만 (num_partners, ACTION_DIM)으로 reshape
                                    _pred_split = pred_logits.reshape(
                                        *pred_logits.shape[:-1], num_partners, ACTION_DIM
                                    )
                                    _targets = target_labels_flat  # (MB, num_partners)
                                    _total_pred_loss = jnp.float32(0.0)
                                    _total_correct = jnp.float32(0.0)
                                    for p in range(num_partners):
                                        # _pred_split: (..., num_partners, A), _targets: (..., num_partners)
                                        _p_loss = optax.softmax_cross_entropy_with_integer_labels(
                                            logits=_pred_split[..., p, :], labels=_targets[..., p]
                                        )
                                        _total_pred_loss = _total_pred_loss + _masked_mean(_p_loss, train_mask)
                                        _p_pred = jnp.argmax(_pred_split[..., p, :], axis=-1)
                                        _total_correct = _total_correct + _masked_mean(
                                            _p_pred == _targets[..., p], train_mask
                                        )
                                    pred_loss = _total_pred_loss / num_partners
                                    pred_accuracy = _total_correct / num_partners
                                pred_loss = pred_loss * config.get("PRED_LOSS_COEF", 1.0)

                        log_prob = pi.log_prob(traj_batch.action)

                        # --------------------------------------------------
                        # Z Prediction + Cycle Loss (CT OFF 모드 전용)
                        # --------------------------------------------------
                        z_pred_loss = 0.0
                        off_cycle_loss = 0.0

                        if not transformer_action:
                            # Z Prediction: MSE(z_partner_hat, sg(partner_gru_z))
                            if z_prediction_enabled:
                                _z_partner_hat = rerun_extras.get("z_partner_hat")
                                if _z_partner_hat is not None:
                                    _target_z = jax.lax.stop_gradient(traj_batch.partner_gru_z)
                                    z_pred_diff = jnp.mean((_z_partner_hat - _target_z) ** 2, axis=-1)
                                    z_pred_loss = _masked_mean(z_pred_diff, train_mask) * z_pred_loss_coef

                            # Cycle Loss: sg(predictions) → CycleDecoder → z_hat ≈ sg(z_GRU)
                            if cycle_loss_enabled:
                                _cycle_parts = []
                                if z_prediction_enabled and rerun_extras.get("z_partner_hat") is not None:
                                    _cycle_parts.append(jax.lax.stop_gradient(rerun_extras["z_partner_hat"]))
                                if action_prediction and pred_logits is not None:
                                    _cycle_parts.append(jax.lax.stop_gradient(pred_logits))

                                if _cycle_parts:
                                    _cycle_in = jnp.concatenate(_cycle_parts, axis=-1)
                                    # Flatten T*B for method call, then reshape
                                    _T, _B = _cycle_in.shape[:2]
                                    _cycle_in_flat = _cycle_in.reshape(_T * _B, -1)
                                    _z_hat_flat = network.apply(
                                        params, _cycle_in_flat, method=network.cycle_decode,
                                    )
                                    _z_hat = _z_hat_flat.reshape(_T, _B, -1)
                                    _gru_target = jax.lax.stop_gradient(rerun_extras["gru_output"])
                                    _cycle_diff = jnp.mean((_z_hat - _gru_target) ** 2, axis=-1)
                                    off_cycle_loss = _masked_mean(_cycle_diff, train_mask) * cycle_loss_coef

                        # --------------------------------------------------
                        # CycleTransformer auxiliary losses
                        # (only when transformer_action=True; zero otherwise)
                        # --------------------------------------------------
                        ct_recon_loss = 0.0
                        ct_action_loss = 0.0
                        ct_cycle_loss = 0.0
                        ct_recon_per_channel = jnp.zeros(1)

                        if transformer_action:
                            W = transformer_window_size
                            D_obs = model_config["GRU_HIDDEN_DIM"]

                            # CT obs emb from rerun (T, B, D_obs) — CT's own encoder output
                            # stop_gradient 제거: CT loss가 ct_obs_encoder까지 gradient를 흘려 학습시킴
                            ct_obs_emb_seq = rerun_extras["ct_obs_emb"]
                            done_seq = traj_batch.done  # (T, B) bool
                            T_dim = ct_obs_emb_seq.shape[0]
                            B_dim = ct_obs_emb_seq.shape[1]

                            # Reconstruct obs_window sequence from zeros using CT obs emb
                            def _window_step_loss(carry, inputs):
                                obs_window, step_idx = carry
                                ct_obs_emb_t, done_t = inputs
                                obs_window = jnp.where(
                                    done_t[:, None, None],
                                    jnp.zeros_like(obs_window),
                                    obs_window,
                                )
                                step_idx = jnp.where(
                                    done_t, jnp.zeros_like(step_idx), step_idx
                                )
                                obs_window = jnp.roll(obs_window, shift=-1, axis=1)
                                obs_window = obs_window.at[:, -1, :].set(ct_obs_emb_t)
                                slots = jnp.arange(W)
                                valid_from = W - 1 - jnp.minimum(step_idx, W - 1)
                                padding_mask = slots[None, :] >= valid_from[:, None]
                                return (obs_window, step_idx + 1), (obs_window, padding_mask)

                            init_w = jnp.zeros((B_dim, W, D_obs))
                            init_si = jnp.zeros(B_dim, dtype=jnp.int32)
                            _, (obs_windows_seq, pad_masks_seq) = jax.lax.scan(
                                _window_step_loss,
                                (init_w, init_si),
                                (ct_obs_emb_seq, done_seq),
                            )
                            # obs_windows_seq: (T, B, W, D_obs)
                            # pad_masks_seq:   (T, B, W)

                            obs_windows_flat = obs_windows_seq.reshape(T_dim * B_dim, W, D_obs)
                            pad_masks_flat = pad_masks_seq.reshape(T_dim * B_dim, W)

                            # Full forward: (C, state_out, a_hat, C_prime)
                            # v1: state_out = z_hat (N, D_obs)
                            # v2: state_out = s_hat (N, H, W, C_full)
                            # v3: state_out = z_partner_hat (N, D_gru)
                            C_flat, state_out_flat, a_hat_flat, C_prime_flat = network.apply(
                                params,
                                obs_windows_flat,
                                pad_masks_flat,
                                method=network.cycle_transformer_forward,
                            )

                            valid_mask = ~traj_batch.done  # (T, B)

                            if transformer_v3:
                                # CT v3: partner GRU z 복원
                                z_partner_hat = state_out_flat.reshape(T_dim, B_dim, -1)
                                a_hat = a_hat_flat.reshape(T_dim, B_dim, -1)
                                C = C_flat.reshape(T_dim, B_dim, -1)
                                C_prime = C_prime_flat.reshape(T_dim, B_dim, -1)

                                # L_z_recon: MSE(z_partner_hat, sg(partner_gru_z))
                                target_z = jax.lax.stop_gradient(traj_batch.partner_gru_z)
                                z_recon_diff = jnp.mean((z_partner_hat - target_z) ** 2, axis=-1)
                                ct_recon_loss = _masked_mean(z_recon_diff, valid_mask)

                                # L_action: CE(a_hat, partner_action)
                                ct_action_loss_vec = optax.softmax_cross_entropy_with_integer_labels(
                                    logits=a_hat,
                                    labels=traj_batch.partner_action.astype(jnp.int32),
                                )
                                ct_action_loss = _masked_mean(ct_action_loss_vec, valid_mask)

                                # L_cycle: MSE(C_prime, sg(C))
                                cycle_diff = jnp.mean(
                                    (C_prime - jax.lax.stop_gradient(C)) ** 2, axis=-1
                                )
                                ct_cycle_loss = _masked_mean(cycle_diff, valid_mask)

                                # 계수 적용
                                ct_recon_loss = ct_recon_loss * transformer_recon_coef
                                ct_action_loss = ct_action_loss * transformer_pred_coef
                                ct_cycle_loss = ct_cycle_loss * transformer_cycle_coef
                                ct_recon_per_channel = jnp.zeros(1)  # v3: dummy
                            else:
                                # 기존 v1/v2 loss (변경 없음)
                                C = C_flat.reshape(T_dim, B_dim, -1)
                                a_hat = a_hat_flat.reshape(T_dim, B_dim, -1)
                                C_prime = C_prime_flat.reshape(T_dim, B_dim, -1)

                                if transformer_v2:
                                    # v2: pixel space reconstruction
                                    # state_out_flat = s_hat (N, H, W, C_full)
                                    _v2_shape = tuple(config.get("TRANSFORMER_STATE_SHAPE", []))
                                    H_v2, W_v2, C_v2 = _v2_shape
                                    s_hat = state_out_flat.reshape(T_dim, B_dim, H_v2, W_v2, C_v2)
                                    # recon target: global_obs 그대로 (no ct_state_encoder)
                                    recon_target_v2 = jax.lax.stop_gradient(traj_batch.global_obs)
                                    # (T, B, H, W, C_full)

                                    # L_recon: MSE per pixel, mean over spatial dims
                                    recon_diff_v2 = (s_hat - recon_target_v2) ** 2
                                    # (T, B, H, W, C_full)
                                    recon_diff_scalar = jnp.mean(recon_diff_v2, axis=(-3, -2, -1))
                                    # (T, B)
                                    ct_recon_loss = _masked_mean(recon_diff_scalar, valid_mask)

                                    # per-channel loss: mean over T, B, H, W → (C_full,)
                                    # valid step만 포함 (valid_mask 적용)
                                    valid_mask_v2 = valid_mask[:, :, None, None, None]
                                    recon_diff_masked = jnp.where(valid_mask_v2, recon_diff_v2, 0.0)
                                    n_valid = jnp.sum(valid_mask).clip(1)
                                    ct_recon_per_channel = (
                                        jnp.sum(recon_diff_masked, axis=(0, 1, 2, 3)) / n_valid
                                    )  # (C_full,)
                                else:
                                    # v1: latent space reconstruction
                                    z_hat = state_out_flat.reshape(T_dim, B_dim, D_obs)
                                    # recon target: ct_state_encoder(global_obs) → D_obs latent
                                    # ct_state_encoder params는 gradient 없음 (고정 random projection)
                                    recon_target = jax.lax.stop_gradient(
                                        network.apply(
                                            params,
                                            jax.lax.stop_gradient(traj_batch.global_obs),
                                            method=network.ct_encode_state,
                                        )
                                    )  # (T, B, D_obs)

                                    recon_diff = jnp.mean(
                                        (z_hat - recon_target) ** 2, axis=-1
                                    )  # (T, B)
                                    ct_recon_loss = _masked_mean(recon_diff, valid_mask)
                                    ct_recon_per_channel = jnp.zeros(1)  # v1: dummy

                                # L_action: CE(a_hat, partner_action)
                                ct_action_loss_vec = optax.softmax_cross_entropy_with_integer_labels(
                                    logits=a_hat,
                                    labels=traj_batch.partner_action.astype(jnp.int32),
                                )  # (T, B)
                                ct_action_loss = _masked_mean(ct_action_loss_vec, valid_mask)

                                # L_cycle: MSE(C_prime, sg(C))
                                cycle_diff = jnp.mean(
                                    (C_prime - jax.lax.stop_gradient(C)) ** 2, axis=-1
                                )  # (T, B)
                                ct_cycle_loss = _masked_mean(cycle_diff, valid_mask)

                                ct_recon_loss = ct_recon_loss * transformer_recon_coef
                                ct_action_loss = ct_action_loss * transformer_pred_coef
                                ct_cycle_loss = ct_cycle_loss * transformer_cycle_coef

                        # def safe_mean(x, mask):
                        #     x_safe = jnp.where(mask, x, 0.0)
                        #     total = jnp.sum(x_safe)
                        #     count = jnp.sum(mask)
                        #     return total / count

                        # def safe_std(x, mask, eps=1e-8):
                        #     m = safe_mean(x, mask)
                        #     diff_sq = (x - m) ** 2
                        #     variance = safe_mean(diff_sq, mask)
                        #     return jnp.sqrt(variance + eps)

                        # CALCULATE VALUE LOSS
                        # ValueNorm 활성 시 value/targets 는 normalized space.
                        # USE_HUBER_LOSS 시 MSE 대신 Huber.
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        if use_huber_loss:
                            def _huber_fn(x):
                                return jnp.where(
                                    jnp.abs(x) <= huber_delta,
                                    0.5 * x ** 2,
                                    huber_delta * (jnp.abs(x) - 0.5 * huber_delta),
                                )
                            value_losses = _huber_fn(value - targets)
                            value_losses_clipped = _huber_fn(value_pred_clipped - targets)
                            value_loss = jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean(where=train_mask)
                        else:
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean(where=train_mask)
                        # value_loss = safe_mean(value_loss, train_mask)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean(where=train_mask)) / (
                            gae.std(where=train_mask) + 1e-8
                        )
                        # gae_mean = safe_mean(gae, train_mask)
                        # gae_std = safe_std(gae, train_mask)
                        # gae = (gae - gae_mean) / (gae_std + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - model_config["CLIP_EPS"],
                                1.0 + model_config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=train_mask)
                        # loss_actor = safe_mean(loss_actor, train_mask)
                        entropy = pi.entropy().mean(where=train_mask)
                        # entropy = safe_mean(pi.entropy(), train_mask)
                        ratio = ratio.mean(where=train_mask)
                        # ratio = safe_mean(ratio, train_mask)

                        # ENT_COEF 선형 스케줄링 (optional):
                        #   ENT_COEF_ANNEAL_STEPS > 0이면 [0, ANNEAL_STEPS]에서 START→END 선형 감쇠.
                        _ent_anneal_steps = float(model_config.get("ENT_COEF_ANNEAL_STEPS", 0) or 0)
                        if _ent_anneal_steps > 0:
                            _ent_start = float(model_config.get("ENT_COEF_START", model_config["ENT_COEF"]))
                            _ent_end = float(model_config.get("ENT_COEF_END", model_config["ENT_COEF"]))
                            _env_steps_f = (
                                update_step
                                * model_config["NUM_STEPS"]
                                * model_config["NUM_ENVS"]
                            ).astype(jnp.float32)
                            _frac = jnp.clip(_env_steps_f / _ent_anneal_steps, 0.0, 1.0)
                            ent_coef_used = _ent_start + (_ent_end - _ent_start) * _frac
                        else:
                            ent_coef_used = model_config["ENT_COEF"]

                        total_loss = (
                            loss_actor
                            + model_config["VF_COEF"] * value_loss
                            - ent_coef_used * entropy
                            + pred_loss
                            + z_pred_loss
                            + off_cycle_loss
                            + ct_recon_loss
                            + ct_action_loss
                            + ct_cycle_loss
                        )

                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            pred_loss,
                            pred_accuracy,
                            state_pred_z_loss,
                            state_pred_action_loss,
                            ct_recon_loss,
                            ct_action_loss,
                            ct_cycle_loss,
                            ct_recon_per_channel,  # v2: (C_full,), v1: zeros(1)
                            z_pred_loss,
                            off_cycle_loss,
                        )

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            loss_rng,
                        )

                        # jax.debug.print(
                        #     "grads {x}, hstate {y}, mask {z}",
                        #     x=jax.tree_util.tree_flatten(grads)[0][0][0],
                        #     y=init_hstate.flatten()[0],
                        #     z=traj_batch.train_mask.sum(),
                        # )

                        new_train_state = train_state.apply_gradients(grads=grads)
                        return (new_train_state, mb_rng), total_loss

                    def _no_op():
                        # jax.debug.print("No update")
                        # ct_recon_per_channel: v2=(C_full,), v1=zeros(1) — shape 일관성 유지
                        _dummy_per_ch = jnp.zeros(
                            config.get("TRANSFORMER_STATE_SHAPE", [0, 0, 1])[-1]  # C_full (마지막 차원)
                        ) if transformer_v2 else jnp.zeros(1)
                        return (train_state, mb_rng), (
                            0.0,
                            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _dummy_per_ch, 0.0, 0.0),
                        )

                    # jax.debug.print(
                    #     "train_mask {x}, {y}",
                    #     x=traj_batch.train_mask.sum(),
                    #     y=traj_batch.train_mask.any(),
                    # )

                    train_state_and_rng, total_loss = jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )
                    return train_state_and_rng, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                num_actors = model_config["NUM_ACTORS"]

                hstate = init_hstate
                if hstate is not None:
                    if isinstance(hstate, tuple):
                        # print("hstate shape (tuple)", [x.shape for x in hstate])
                        hstate = tuple(x[jnp.newaxis, :] for x in hstate)
                        # print("hstate shape (tuple)", [x.shape for x in hstate])
                    else:
                        # print("hstate shape", hstate.shape)
                        hstate = hstate[jnp.newaxis, :]
                        # print("hstate shape", hstate.shape)

                batch = (
                    hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # print(
                #     "batch shapes:",
                #     batch[0].shape,
                #     batch[1].obs.shape,
                #     batch[1].done.shape,
                #     batch[2].shape,
                #     batch[3].shape,
                # )
                # print("hstate shape", hstate.shape)

                permutation = jax.random.permutation(_rng, num_actors)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                rng, mb_rng_init = jax.random.split(rng)
                (train_state, _), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, mb_rng_init), minibatches
                )

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
            )
            # UPDATE_EPOCHS 횟수만큼 미니배치 업데이트를 반복 수행
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # 환경 스텝 수집 중 기록된 정보를 메트릭 딕셔너리로 가져옴
            # (shaped_reward, original_reward, anneal_factor, combined_reward 등 포함)
            metric = traj_batch.info

            # [STA-PH1] Logging Logic
            if config.get("PH1_ENABLED", False):
                n_actors = model_config["NUM_ACTORS"]
                half = n_actors // 2

                def _get_agent_means(key):
                    if key not in metric:
                        return 0.0, 0.0
                    val = metric[key]
                    val_mean = val.mean(axis=0)
                    return val_mean[:half].mean(), val_mean[half:].mean()

                if "ph1_penalty_env" in metric:
                    metric["ph1_penalty_env_mean"] = metric["ph1_penalty_env"].mean()
                if "ph1_dist_env" in metric:
                    metric["ph1_dist_env_mean"] = metric["ph1_dist_env"].mean()
                if "ph1_penalty_env_slots" in metric:
                    for slot_idx in range(ph1_penalty_slots):
                        metric[f"ph1_penalty_env_slot{slot_idx + 1}_mean"] = (
                            metric["ph1_penalty_env_slots"][..., slot_idx].mean()
                        )
                if "ph1_dist_env_slots" in metric:
                    for slot_idx in range(ph1_penalty_slots):
                        metric[f"ph1_dist_env_slot{slot_idx + 1}_mean"] = (
                            metric["ph1_dist_env_slots"][..., slot_idx].mean()
                        )

                rew0, rew1 = _get_agent_means("combined_reward")
                metric["reward_agent0_penalized"] = rew0
                metric["reward_agent1_penalized"] = rew1

                pure0, pure1 = _get_agent_means("combined_reward_no_penalty")
                metric["reward_agent0_pure"] = pure0
                metric["reward_agent1_pure"] = pure1

            # --------------------------------------------------------------
            # entropy 통계 파생 메트릭
            #   - policy_entropy_mean: rollout 전체 평균 엔트로피
            # --------------------------------------------------------------
            if "policy_entropy" in metric:
                metric["policy_entropy_mean"] = metric["policy_entropy"].mean()

            # 손실 함수 관련 메트릭 추출
            total_loss, aux_data = loss_info
            (
                value_loss,
                loss_actor,
                entropy,
                ratio,
                pred_loss,
                pred_accuracy,
                state_pred_z_loss,
                state_pred_action_loss,
                ct_recon_loss,
                ct_action_loss,
                ct_cycle_loss,
                ct_recon_per_channel,  # v2: (C_full,), v1: zeros(1)
                z_pred_loss,
                off_cycle_loss,
            ) = aux_data

            # PPO 학습 손실 메트릭 추가
            metric["total_loss"] = total_loss      # 전체 손실 (actor + value + entropy + pred)
            metric["value_loss"] = value_loss      # 가치 함수(critic) MSE 손실
            metric["loss_actor"] = loss_actor      # 정책(actor) clipped surrogate 손실
            metric["entropy"] = entropy            # 정책 엔트로피 (탐험 정도)
            metric["ratio"] = ratio                # PPO ratio (new_prob / old_prob)
            metric["pred_loss"] = pred_loss        # 파트너 행동 예측 손실
            metric["pred_accuracy"] = pred_accuracy # 파트너 행동 예측 정확도
            metric["state_pred_z_loss"] = state_pred_z_loss
            metric["state_pred_action_loss"] = state_pred_action_loss
            metric["z_pred_loss"] = z_pred_loss          # partner GRU z 복원 손실
            metric["off_cycle_loss"] = off_cycle_loss    # CT OFF cycle consistency 손실
            metric["ct_recon_loss"] = ct_recon_loss
            metric["ct_action_loss"] = ct_action_loss
            metric["ct_cycle_loss"] = ct_cycle_loss
            # v2: per-channel recon loss 로깅 (각 채널별 재구성 품질 확인용)
            # ct_recon_per_channel shape after scan: (UPDATE_EPOCHS, NUM_MINIBATCHES, C_full)
            # → epoch/minibatch 차원 평균 후 채널별로 로깅
            if transformer_v2:
                _per_ch_mean = ct_recon_per_channel.mean(axis=(0, 1))  # (C_full,)
                for _ch_i in range(_per_ch_mean.shape[0]):
                    metric[f"ct_recon_ch{_ch_i}"] = _per_ch_mean[_ch_i]
            metric["ph1_beta_current"] = ph1_beta_current
            metric["ph1_beta_progress"] = ph1_beta_progress

            # Phase-specific masked logging for PH2:
            # - phase1: spec-spec 중심 reward
            # - phase2: spec-ind 중심 reward/penalty/distance
            train_mask_for_log = traj_batch.train_mask.astype(jnp.float32)
            metric["train_mask_ratio"] = train_mask_for_log.mean()

            def _masked_mean_with_train_mask(x):
                x_arr = jnp.asarray(x, dtype=jnp.float32)
                if x_arr.ndim == 0:
                    return x_arr
                mask_arr = train_mask_for_log
                while mask_arr.ndim < x_arr.ndim:
                    mask_arr = mask_arr[..., None]
                denom = jnp.maximum(jnp.sum(mask_arr), 1.0)
                return jnp.sum(x_arr * mask_arr) / denom

            def _masked_mean_with_mask(x, mask):
                x_arr = jnp.asarray(x, dtype=jnp.float32)
                if x_arr.ndim == 0:
                    return x_arr
                mask_arr = jnp.asarray(mask, dtype=jnp.float32)
                while mask_arr.ndim < x_arr.ndim:
                    mask_arr = mask_arr[..., None]
                denom = jnp.maximum(jnp.sum(mask_arr), 1.0)
                return jnp.sum(x_arr * mask_arr) / denom

            def _masked_mean_with_mask_slots(x, mask):
                x_arr = jnp.asarray(x, dtype=jnp.float32)
                if x_arr.ndim < 3:
                    return x_arr
                mask_arr = jnp.asarray(mask, dtype=jnp.float32)
                while mask_arr.ndim < x_arr.ndim:
                    mask_arr = mask_arr[..., None]
                num = jnp.sum(x_arr * mask_arr, axis=(0, 1))
                denom = jnp.maximum(jnp.sum(mask_arr, axis=(0, 1)), 1.0)
                return num / denom

            if "combined_reward" in metric:
                metric["reward"] = _masked_mean_with_train_mask(metric["combined_reward"])
            if phase_log_prefix == "phase2":
                if "combined_reward_no_penalty" in metric:
                    metric["reward"] = _masked_mean_with_train_mask(
                        metric["combined_reward_no_penalty"]
                    )
                    if "ph2_ind_match_env" in metric:
                        # phase2/ind-ind/reward:
                        # ph2_ind_match_env == 1.0 => ind-ind
                        ind_ind_mask = train_mask_for_log * jnp.asarray(
                            metric["ph2_ind_match_env"], dtype=jnp.float32
                        )
                        metric["ind-ind/reward"] = _masked_mean_with_mask(
                            metric["combined_reward_no_penalty"], ind_ind_mask
                        )
                # phase2(ind)에서 penalty/distance는 spec-ind 구간만 집계.
                # ph2_ind_match_env == 1.0 => ind-ind, 0.0 => spec-ind
                spec_ind_mask = train_mask_for_log
                if "ph2_ind_match_env" in metric:
                    spec_ind_mask = train_mask_for_log * (
                        1.0 - jnp.asarray(metric["ph2_ind_match_env"], dtype=jnp.float32)
                    )
                if "ph1_penalty_env" in metric:
                    metric["penalty"] = _masked_mean_with_mask(
                        metric["ph1_penalty_env"], spec_ind_mask
                    )
                if "ph1_dist_env" in metric:
                    metric["distance"] = _masked_mean_with_mask(
                        metric["ph1_dist_env"], spec_ind_mask
                    )
                if "ph1_penalty_env_slots" in metric:
                    slot_penalty = _masked_mean_with_mask_slots(
                        metric["ph1_penalty_env_slots"], spec_ind_mask
                    )
                    for slot_idx in range(ph1_penalty_slots):
                        metric[f"penalty_slot{slot_idx + 1}"] = slot_penalty[slot_idx]
                if "ph1_dist_env_slots" in metric:
                    slot_distance = _masked_mean_with_mask_slots(
                        metric["ph1_dist_env_slots"], spec_ind_mask
                    )
                    for slot_idx in range(ph1_penalty_slots):
                        metric[f"distance_slot{slot_idx + 1}"] = slot_distance[slot_idx]

            # 모든 메트릭 값을 배치/스텝 차원에 대해 평균 계산 (스칼라로 축약)
            # 이미 스칼라인 경우(PH1 메트릭 등)는 그대로 둔다.
            def _safe_mean(x):
                if hasattr(x, "mean"):
                    return x.mean()
                return x

            metric = jax.tree_util.tree_map(_safe_mean, metric)

            # GridSpread success_rate 파생: episode 종료 중 all_covered 비율.
            if is_spread and "success_at_done" in metric and "ep_done_flag" in metric:
                metric["success_rate"] = metric["success_at_done"] / (metric["ep_done_flag"] + 1e-8)

            # 업데이트 스텝 카운터 증가 및 메트릭에 기록
            update_step += 1
            metric["update_step"] = update_step
            # 환경과 상호작용한 총 스텝 수 계산 (update * rollout_length * num_envs)
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]                                                                 
            )

            if runtime_progress_debug:
                should_print = (update_step % runtime_progress_every) == 0

                def _host_progress_cb(us, es):
                    us_i = int(np.array(us))
                    es_i = int(np.array(es))
                    if phase_log_prefix:
                        print(
                            f"[PH2RT] {phase_log_prefix} update_step={us_i} env_step={es_i}",
                            flush=True,
                        )
                    else:
                        print(f"[PH2RT] update_step={us_i} env_step={es_i}", flush=True)

                jax.lax.cond(
                    should_print,
                    lambda _: jax.debug.callback(_host_progress_cb, update_step, metric["env_step"]),
                    lambda _: None,
                    operand=None,
                )

            # --------------------------------------------------------------
            # Hard-stop on NaN/Inf in training state/losses/metrics
            # --------------------------------------------------------------
            if abort_on_nan:
                def _all_finite_tree(tree):
                    leaves = jax.tree_util.tree_leaves(tree)
                    if len(leaves) == 0:
                        return jnp.array(True)
                    flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
                    return jnp.all(jnp.stack(flags))

                params_finite = _all_finite_tree(train_state.params)
                losses_finite = jnp.all(
                    jnp.stack(
                        [
                            jnp.all(jnp.isfinite(total_loss)),
                            jnp.all(jnp.isfinite(value_loss)),
                            jnp.all(jnp.isfinite(loss_actor)),
                            jnp.all(jnp.isfinite(entropy)),
                            jnp.all(jnp.isfinite(ratio)),
                            jnp.all(jnp.isfinite(pred_loss)),
                            jnp.all(jnp.isfinite(pred_accuracy)),
                            jnp.all(jnp.isfinite(state_pred_z_loss)),
                            jnp.all(jnp.isfinite(state_pred_action_loss)),
                        ]
                    )
                )
                metric_finite = _all_finite_tree(metric)
                all_finite = params_finite & losses_finite & metric_finite

                def _check_and_abort_nan(all_ok, us, es, p_ok, l_ok, m_ok,
                                        tl, vl, la, ent, rat, pl, pa, spz, spa):
                    """callback 내부에서 조건 체크 — jax.lax.cond + debug.callback 조합이
                    pmap/vmap 컨텍스트에서 false positive를 내는 문제 우회."""
                    if bool(np.asarray(all_ok).flat[0]):
                        return  # 정상 — abort 하지 않음
                    us_i = int(np.asarray(us).flat[0])
                    es_i = int(np.asarray(es).flat[0])
                    msg = (
                        f"[ABORT_ON_NAN] update_step={us_i}, env_step={es_i}\n"
                        f"  params_finite={np.asarray(p_ok)}, losses_finite={np.asarray(l_ok)}, metric_finite={np.asarray(m_ok)}\n"
                        f"  total_loss={np.asarray(tl)}, value_loss={np.asarray(vl)}, "
                        f"loss_actor={np.asarray(la)}, entropy={np.asarray(ent)}\n"
                        f"  ratio={np.asarray(rat)}, pred_loss={np.asarray(pl)}, "
                        f"pred_acc={np.asarray(pa)}\n"
                        f"  state_pred_z={np.asarray(spz)}, state_pred_action={np.asarray(spa)}"
                    )
                    raise RuntimeError(msg)

                jax.debug.callback(
                    _check_and_abort_nan,
                    all_finite,
                    update_step,
                    metric["env_step"],
                    params_finite, losses_finite, metric_finite,
                    total_loss, value_loss, loss_actor, entropy, ratio,
                    pred_loss, pred_accuracy, state_pred_z_loss, state_pred_action_loss,
                )

            # WandB 로깅 콜백: JAX 계산 그래프 밖에서 실행되도록 debug.callback 사용
            # 각 시드별로 구분된 네임스페이스(rng{seed}/)를 prefix로 추가하여 로깅
            def callback(metric, original_seed, log_seed_override_host):
                # vmap을 사용하면 metric과 original_seed가 배치(배열) 형태로 들어옵니다.
                # 따라서 배열인 경우 순회하며 각각 로깅해야 합니다.
                
                # numpy 변환 (호스트 측 실행이므로 안전)
                original_seed = np.array(original_seed)

                if log_seed_override_host is not None:
                    seed_np = np.array(log_seed_override_host)
                    seed = int(seed_np.reshape(-1)[0]) if seed_np.size > 0 else 0
                    if original_seed.ndim > 0:
                        single_metric = {
                            k: (v[0] if (hasattr(v, "__len__") and not np.isscalar(v)) else v)
                            for k, v in metric.items()
                        }
                    else:
                        single_metric = metric
                    if phase_log_prefix:
                        wandb_log = {f"rng{seed}/{phase_log_prefix}/{k}": v for k, v in single_metric.items()}
                    else:
                        wandb_log = {f"rng{seed}/{k}": v for k, v in single_metric.items()}
                    wandb.log(wandb_log)
                    return

                # PH2 unified logging mode:
                # always log into a fixed seed namespace (first seed) so phase1/phase2
                # curves are continuous under one run bucket.
                if phase_log_fixed_seed is not None:
                    seed = int(phase_log_fixed_seed)
                    if original_seed.ndim > 0:
                        single_metric = {
                            k: (v[0] if (hasattr(v, "__len__") and not np.isscalar(v)) else v)
                            for k, v in metric.items()
                        }
                    else:
                        single_metric = metric
                    if phase_log_prefix:
                        wandb_log = {f"rng{seed}/{phase_log_prefix}/{k}": v for k, v in single_metric.items()}
                    else:
                        wandb_log = {f"rng{seed}/{k}": v for k, v in single_metric.items()}
                    wandb.log(wandb_log)
                    return
                
                if original_seed.ndim > 0:
                    # 배치가 있는 경우 (vmap 사용 시)
                    for i in range(original_seed.shape[0]):
                        seed = original_seed[i]
                        # 해당 시드의 메트릭만 추출
                        single_metric = {k: v[i] for k, v in metric.items()}
                        
                        # print(f"[DEBUG] Logging for seed {seed}")
                        if phase_log_prefix:
                            wandb_log = {f"rng{int(seed)}/{phase_log_prefix}/{k}": v for k, v in single_metric.items()}
                        else:
                            wandb_log = {f"rng{int(seed)}/{k}": v for k, v in single_metric.items()}
                        wandb.log(wandb_log)
                else:
                    # 스칼라인 경우 (단일 시드)
                    # print(f"[DEBUG] Logging for seed {original_seed}")
                    if phase_log_prefix:
                        wandb_log = {f"rng{int(original_seed)}/{phase_log_prefix}/{k}": v for k, v in metric.items()}
                    else:
                        wandb_log = {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                    wandb.log(wandb_log)

            jax.debug.callback(callback, metric, original_seed, log_seed_override)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    train_state.params,
                    jnp.argmax(checkpoint_idx_selector),
                )

            # [STA-PH1] Update per-env sampling probabilities using the env-specific pool.
            next_ph1_pool_states = ph1_pool_states
            next_ph1_probs = ph1_probs

            if ph1_enabled and env_name != "ToyCoop":
                # ToyCoop은 pool/v-gap 미사용 (synthetic penalty state 직접 생성)
                use_partner_pred = (
                    ph1_enabled
                    and use_ph1_partner_pred
                    and use_partner_modeling
                    and prediction_enabled
                    and not action_prediction  # action_prediction: internal after GRU, not external input
                )
                ph1_prob_apply_fn = network.apply
                ph1_prob_params = train_state.params
                ph1_prob_use_blocked_input = learner_use_blocked_input
                if (
                    ph1_probs_use_population_model
                    and (population is not None)
                    and is_policy_population
                    and hasattr(population, "network")
                    and hasattr(population, "params")
                ):
                    ph1_prob_apply_fn = population.network.apply
                    ph1_prob_params = population.params
                    ph1_prob_use_blocked_input = population_use_blocked_input
                ph1_prob_blocked_slots = (
                    ph1_penalty_slots if ph1_prob_use_blocked_input else 1
                )

                # Use the most recent rollout step as the reference batch per environment.
                # Shape conventions:
                #  - traj_batch.obs: (T, Actors, H, W, C_obs)
                #  - traj_batch.done: (T, Actors)
                #  - traj_batch.hstate: pytree with leading (T, Actors, ...)
                # We reshape actors -> (Agents, Envs, ...), then take env-wise batches of size B=Agents.
                obs_last = traj_batch.obs[-1].reshape(env.num_agents, model_config["NUM_ENVS"], *state_shape)
                done_last = traj_batch.done[-1].reshape(env.num_agents, model_config["NUM_ENVS"])

                h_last = jax.tree_util.tree_map(
                    lambda x: x[-1].reshape(env.num_agents, model_config["NUM_ENVS"], *x.shape[2:]),
                    traj_batch.hstate,
                )

                if use_partner_pred:
                    pp_last = traj_batch.partner_prediction[-1].reshape(
                        env.num_agents, model_config["NUM_ENVS"], policy_pred_dim
                    )
                else:
                    pp_last = jnp.zeros(
                        (env.num_agents, model_config["NUM_ENVS"], policy_pred_dim),
                        dtype=jnp.float32,
                    )

                def _compute_env_probs(obs_env, done_env, h_env, pool_env, pp_env):
                    # obs_env: (Agents, H, W, C_obs)
                    # done_env: (Agents,)
                    # h_env: pytree with (Agents, ...)
                    # pool_env: (Pool, H, W, C_full)
                    probs_env, _v = compute_ph1_probs(
                        ph1_prob_apply_fn,
                        ph1_prob_params,
                        obs_env,
                        done_env,
                        h_env,
                        pool_env,
                        pp_env,
                        use_partner_pred=use_partner_pred,
                        use_blocked_input=ph1_prob_use_blocked_input,
                        blocked_input_slots=ph1_prob_blocked_slots,
                        beta=ph1_beta_current,
                        normal_prob=config.get("PH1_NORMAL_PROB", 0.5),
                    )
                    return probs_env

                computed_probs = jax.vmap(_compute_env_probs)(
                    jnp.swapaxes(obs_last, 0, 1),
                    jnp.swapaxes(done_last, 0, 1),
                    jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), h_last),
                    next_ph1_pool_states,
                    jnp.swapaxes(pp_last, 0, 1),
                )

                next_ph1_probs = jax.lax.select(ph1_pool_ready, computed_probs, ph1_probs)
                ph1_pool_ready = True

            runner_state = (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                last_partner_action,
                next_last_action,
                next_ego_idxs,
                blocked_states_env,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                next_ph1_pool_states,
                next_ph1_probs,
                ph1_pool_ready,
                next_phase2_ind_match_env_state,
                next_ph2_spread_ind_ego_mask_state,  # (NUM_ACTORS,) bool
                vn_state,
                rng,
            )
            return runner_state, metric

        initial_update_step = 0
        if update_step_offset is not None:
            initial_update_step = update_step_offset
        if (update_step_offset_override is not None) and (initial_runner_state is None):
            initial_update_step = update_step_offset_override

        if initial_runner_state is None:
            initial_checkpoints = jax.tree_util.tree_map(
                lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
                train_state.params,
            )

            if num_checkpoints > 0:
                initial_checkpoints = jax.lax.cond(
                    (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    initial_checkpoints,
                    train_state.params,
                    0,
                )

            init_fcp_pop_idxs = None
            if population is not None and not is_policy_population:
                init_fcp_pop_idxs = jax.random.randint(
                    _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
                )

            rng, _rng = jax.random.split(rng)

            initial_last_partner_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32)
            initial_last_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32)

            # [E3T] Initial Ego Indices (Random init)
            initial_ego_idxs = jax.random.randint(_rng, (model_config["NUM_ENVS"],), 0, env.num_agents)

            # 에피소드 시작 시 partner 위치 및 blocked target 초기화
            pos_y, pos_x = _extract_pos_axes(env_state)
            partner_idxs = (initial_ego_idxs + 1) % env.num_agents
            env_range = jnp.arange(model_config["NUM_ENVS"]) # 추가
            
            partner_y = pos_y[partner_idxs, env_range]
            partner_x = pos_x[partner_idxs, env_range]
            partner_pos = jnp.stack([partner_y, partner_x], axis=-1)
            if ph1_enabled:
                # PH1 blocked target is always the global full state.
                # ph1_penalty_slots는 이미 pair_mode 반영됨 (×2)
                if ph1_multi_penalty_enabled or ph1_pair_mode:
                    init_shape = (
                        model_config["NUM_ENVS"],
                        ph1_penalty_slots,
                    ) + ph1_block_shape
                else:
                    init_shape = (model_config["NUM_ENVS"],) + ph1_block_shape
                initial_blocked_states_env = jnp.full(init_shape, -1.0, dtype=jnp.float32)
            else:
                initial_blocked_states_env = jnp.full(
                    (model_config["NUM_ENVS"], 2), -1, dtype=jnp.int32
                )
            # Debug: print initial blocked state set (first few envs)
            # jax.debug.print("Initial blocked_states_env shape: {}", initial_blocked_states_env.shape)
            # jax.debug.print("Initial blocked_states_env sample (first 8): {}", initial_blocked_states_env[:8])
            
            # Debug: print possible blocked coords for first env
            # [PH1] Initialize Target Pool
            ph1_pool_size = config.get("PH1_POOL_SIZE", 100)
            
            # Per-env ring buffer pool: (Envs, Pool, H, W, C_full)
            if ph1_enabled:
                pool_shape = (model_config["NUM_ENVS"], ph1_pool_size) + ph1_block_shape
                initial_ph1_pool_states = jnp.full(pool_shape, -1.0, dtype=jnp.float32)
            else:
                # Placeholder to keep runner_state structure consistent
                initial_ph1_pool_states = jnp.zeros((model_config["NUM_ENVS"], ph1_pool_size, 1), dtype=jnp.float32)

            # Per-env probs: (Envs, Pool+1) (Last is None)
            initial_ph1_probs = jnp.zeros((model_config["NUM_ENVS"], ph1_pool_size + 1), dtype=jnp.float32)
            initial_ph1_probs = initial_ph1_probs.at[:, -1].set(1.0)

            initial_ph1_pool_ready = False
            if ph2_match_schedule:
                rng, rng_init_match = jax.random.split(rng)
                if is_spread:
                    # GridSpread: 3-way categorical 초기화
                    logits = jnp.log(jnp.array([1/3, 1/3, 1/3]))
                    initial_phase2_ind_match_env = jax.random.categorical(
                        rng_init_match, logits, shape=(model_config["NUM_ENVS"],)
                    ).astype(jnp.int32)
                else:
                    # OV2: 기존 bernoulli 그대로
                    initial_phase2_ind_match_env = jax.random.bernoulli(
                        rng_init_match,
                        p=_phase2_ind_match_prob(initial_update_step),
                        shape=(model_config["NUM_ENVS"],),
                    )
            else:
                initial_phase2_ind_match_env = jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.bool_)

            runner_state = (
                train_state,
                initial_checkpoints,
                env_state,
                obsv,
                jnp.zeros((model_config["NUM_ACTORS"]), dtype=bool),
                initial_update_step,
                init_hstate,
                init_population_hstate,
                init_population_annealing_mask,
                init_fcp_pop_idxs,
                initial_last_partner_action,
                initial_last_action,
                initial_ego_idxs,
                initial_blocked_states_env,
                jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.float32),
                jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.float32),
                jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.int32),
                jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.int32),
                initial_ph1_pool_states,
                initial_ph1_probs,
                initial_ph1_pool_ready,
                initial_phase2_ind_match_env,
                jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.bool_),  # ph2_spread_ind_ego_mask
                ValueNormState.create(),
                _rng,
            )
        else:
            runner_state = initial_runner_state
            # Backward compat: 이전 단계에서 vn_state 없이 저장된 runner_state (24 elements)면
            # 마지막 _rng 앞에 ValueNormState.create() 삽입 → 새 코드 25 elements 와 일치.
            if len(runner_state) == 24:
                _rs_list = list(runner_state)
                _trailing_rng = _rs_list.pop()
                _rs_list.append(ValueNormState.create())
                _rs_list.append(_trailing_rng)
                runner_state = tuple(_rs_list)
            if population is not None:
                runner_list = list(runner_state)
                if runner_list[7] is None:
                    if isinstance(population, AbstractPolicy):
                        runner_list[7] = population.init_hstate(model_config["NUM_ACTORS"])
                    else:
                        runner_list[7] = initialize_carry(
                            population_config, model_config["NUM_ACTORS"]
                        )
                        if runner_list[9] is None:
                            rng_rs = runner_list[-1]
                            rng_rs, rng_fcp = jax.random.split(rng_rs)
                            runner_list[-1] = rng_rs
                            runner_list[9] = jax.random.randint(
                                rng_fcp,
                                (model_config["NUM_ACTORS"],),
                                0,
                                fcp_population_size,
                            )
                if use_population_annealing and runner_list[8] is None:
                    rng_rs = runner_list[-1]
                    rng_rs, rng_pa = jax.random.split(rng_rs)
                    runner_list[-1] = rng_rs
                    runner_list[8] = _sample_population_annealing_mask(
                        runner_list[5], rng_pa
                    )
                runner_state = tuple(runner_list)
            if initial_train_state is not None:
                runner_state = (initial_train_state,) + tuple(runner_state[1:])

        max_num_update_steps = int(model_config["NUM_UPDATES"])
        if update_step_num_overwrite is not None:
            max_num_update_steps = int(update_step_num_overwrite)

        # PH2 compile-once mode:
        # run a fixed-shape compiled train function, and choose effective update
        # count at runtime via while_loop (no retrace for segment length changes).
        dynamic_update_mode = bool(config.get("PH2_DYNAMIC_UPDATES", False))
        if dynamic_update_mode:
            if max_num_update_steps <= 0:
                final_update_step = jnp.asarray(runner_state[5], dtype=jnp.int32)
                metric = {
                    "update_step": final_update_step,
                    "env_step": final_update_step * jnp.int32(steps_per_update),
                }
                return {"runner_state": runner_state, "metrics": metric}

            runtime_num_updates = (
                jnp.int32(max_num_update_steps)
                if num_update_steps_override is None
                else jnp.asarray(num_update_steps_override, dtype=jnp.int32)
            )
            runtime_num_updates = jnp.clip(
                runtime_num_updates,
                jnp.int32(1),
                jnp.int32(max_num_update_steps),
            )
            # First update to initialize metric tree; subsequent steps are masked by runtime_num_updates.
            runner_state, metric = _update_step(runner_state, None)

            if max_num_update_steps > 1:
                def _scan_body(carry, step_i):
                    curr_runner_state, curr_metric = carry

                    def _do_step(_):
                        return _update_step(curr_runner_state, None)

                    def _skip_step(_):
                        return curr_runner_state, curr_metric

                    next_runner_state, next_metric = jax.lax.cond(
                        step_i < runtime_num_updates,
                        _do_step,
                        _skip_step,
                        operand=None,
                    )
                    return (next_runner_state, next_metric), None

                (runner_state, metric), _ = jax.lax.scan(
                    _scan_body,
                    (runner_state, metric),
                    jnp.arange(1, max_num_update_steps, dtype=jnp.int32),
                )
        else:
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, max_num_update_steps
            )

        # jax.debug.print("Runner state {x}", x=runner_state)
        # jax.debug.print("neg5 {x}", x=runner_state[-5])
        return {"runner_state": runner_state, "metrics": metric}

    return train
