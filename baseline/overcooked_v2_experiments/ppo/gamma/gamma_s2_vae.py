"""
GAMMA S2 VAE: Population rollout 수집 → VAE 학습 → 순수 VAE 파트너 RL 학습.

Phase A: VAE 학습
  1. S1 population 멤버들의 self-play rollout 수집 (raw obs + action)
  2. VAE encoder/decoder 학습 (ELBO) — VAE 자체 obs encoder 사용

Phase B: 순수 VAE 파트너 RL (원본 GAMMA와 동일)
  1. 에피소드 시작 시 z ~ N(0, I) 샘플링
  2. VAE decoder(raw_obs, z) → partner action (VAE 자체 obs encoder 사용)
  3. Ego agent는 PPO로 학습

원본: mapbt/scripts/overcooked_population/coordinator/shared_runner.py
"""

import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from pathlib import Path
from typing import NamedTuple
from flax.training.train_state import TrainState
import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper, LogWrapper
import wandb

from overcooked_v2_experiments.ppo.models.rnn import ActorCriticRNN
from overcooked_v2_experiments.ppo.gamma.vae_model import GAMMAVAE
from overcooked_v2_experiments.ppo.utils.store import store_checkpoint
from overcooked_v2_experiments.ppo.utils.utils import get_num_devices
from overcooked_v2_experiments.ppo.utils.valuenorm import (
    ValueNormState, valuenorm_update, valuenorm_normalize, valuenorm_denormalize,
)


class S2Transition(NamedTuple):
    done: jnp.ndarray
    ego_obs: jnp.ndarray
    ego_action: jnp.ndarray
    ego_log_prob: jnp.ndarray
    critic_value: jnp.ndarray
    reward: jnp.ndarray
    info: dict
    global_obs: jnp.ndarray  # MAPPO centralized critic 용


# =====================================================================
# Phase A: Rollout 수집 + VAE 학습
# =====================================================================

def _collect_population_rollouts(pop_params, config, env, network, n_episodes, rng):
    """Population self-play rollout 수집 — JAX JIT + lax.scan + vmap 버전.

    Python for 루프 대신 jax.lax.scan(steps) + jax.vmap(episodes)로 전체 JIT 컴파일.
    done 이후 스텝은 invalid_mask로 마킹하고 이후 Python 단에서 필터링.
    """
    model_config = config["model"]
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]
    N = jax.tree_util.tree_leaves(pop_params)[0].shape[0]
    max_steps = 400
    eps_per_member = max(n_episodes // N, 1)

    num_env_agents = env.num_agents

    def _rollout_single_episode(member_params, rng):
        """단일 에피소드 롤아웃 (lax.scan, 고정 max_steps). N-agent 지원."""
        rng, k_reset = jax.random.split(rng)
        obs, state = env.reset(k_reset)
        # N개 에이전트 hstate 초기화
        hstates = jnp.stack([
            ActorCriticRNN.initialize_carry(1, GRU_HIDDEN_DIM)
            for _ in range(num_env_agents)
        ])  # (N, 1, GRU_HIDDEN_DIM)
        done = jnp.zeros((1,), dtype=jnp.bool_)

        def _step(carry, _):
            obs, state, hstates, done, rng = carry

            rng, k_env = jax.random.split(rng)
            rng, *agent_keys = jax.random.split(rng, num_env_agents + 1)

            # 각 에이전트 행동 생성
            new_hstates = []
            actions_list = []
            obs_list = []
            for i in range(num_env_agents):
                obs_i = obs[env.agents[i]][jnp.newaxis, jnp.newaxis]  # (1,1,*obs)
                h_i = hstates[i]
                h_new, pi_i, _, _ = network.apply(member_params, h_i, (obs_i, done[jnp.newaxis]), actor_only=True)
                a_i = pi_i.sample(seed=agent_keys[i]).squeeze(0)
                new_hstates.append(h_new)
                actions_list.append(a_i.squeeze())
                obs_list.append(obs[env.agents[i]])

            env_act = {env.agents[i]: actions_list[i] for i in range(num_env_agents)}
            obs_next, state_next, _, done_dict, _ = env.step(k_env, state, env_act)
            done_next = jnp.array([done_dict["__all__"]])

            rec_invalid = done.squeeze()
            rec_obs = jnp.stack(obs_list)        # (N, *obs_shape)
            rec_acts = jnp.stack(actions_list)    # (N,)

            return (obs_next, state_next, jnp.stack(new_hstates), done_next, rng), \
                   (rec_obs, rec_acts, rec_invalid)

        init_carry = (obs, state, hstates, done, rng)
        _, (obs_seq, act_seq, invalid_mask) = jax.lax.scan(
            _step, init_carry, None, max_steps,
        )
        # obs_seq: (max_steps, N, *obs_shape), act_seq: (max_steps, N), invalid_mask: (max_steps,)
        return obs_seq, act_seq, invalid_mask

    # member × episode 병렬 rollout (이중 vmap)
    # pop_params: leading axis = N (member). ep_rngs_all: (N, eps_per_member, 2)
    _rollout_all = jax.jit(
        jax.vmap(
            jax.vmap(_rollout_single_episode, in_axes=(None, 0)),
            in_axes=(0, 0),
        )
    )

    rng, rollout_rng = jax.random.split(rng)
    ep_rngs_all = jax.random.split(rollout_rng, N * eps_per_member)
    ep_rngs_all = ep_rngs_all.reshape((N, eps_per_member, 2))

    obs_all, act_all, invalid_all = _rollout_all(pop_params, ep_rngs_all)
    # shapes: (N, eps, max_steps, num_agents, *obs_shape) / (N, eps, max_steps, num_agents) / (N, eps, max_steps)
    obs_all = np.array(obs_all)
    act_all = np.array(act_all)
    invalid_all = np.array(invalid_all)

    all_obs, all_actions = [], []
    for m in range(N):
        for ep in range(eps_per_member):
            T = int(np.sum(~invalid_all[m, ep]))
            if T > 10:
                for ai in range(num_env_agents):
                    all_obs.append(obs_all[m, ep, :T, ai])
                    all_actions.append(act_all[m, ep, :T, ai])

    print(f"[VAE] Collected {len(all_obs)} trajectories from {N} members")
    return all_obs, all_actions


def _prepare_chunks(all_obs, all_actions, chunk_length):
    """Trajectory를 고정 길이 chunk로 분할."""
    chunks_obs, chunks_act = [], []
    for obs_traj, act_traj in zip(all_obs, all_actions):
        T = len(obs_traj)
        if T < chunk_length:
            continue
        for start in range(0, T - chunk_length + 1, chunk_length // 2):
            end = start + chunk_length
            if end > T:
                break
            chunks_obs.append(obs_traj[start:end])
            chunks_act.append(act_traj[start:end])
    return chunks_obs, chunks_act


def _train_vae(config, all_obs, all_actions, obs_shape, rng):
    """VAE를 ELBO loss로 학습. raw obs를 직접 처리 (자체 obs encoder)."""
    z_dim = config.get("GAMMA_VAE_Z_DIM", 16)
    hidden_dim = config.get("GAMMA_VAE_HIDDEN_DIM", 256)
    action_dim = config["model"]["ACTION_DIM"]
    vae_lr = config.get("GAMMA_VAE_LR", 5e-4)
    vae_epochs = config.get("GAMMA_VAE_EPOCHS", 500)
    batch_size = config.get("GAMMA_VAE_BATCH_SIZE", 64)
    chunk_length = config.get("GAMMA_VAE_CHUNK_LENGTH", 100)
    kl_penalty_final = config.get("GAMMA_VAE_KL_PENALTY", 1.0)
    kl_penalty_init = config.get("GAMMA_VAE_KL_INIT", 0.0)
    obs_encoder_type = config["model"].get("OBS_ENCODER", "CNN")

    vae = GAMMAVAE(
        hidden_dim=hidden_dim, z_dim=z_dim,
        action_dim=action_dim, obs_encoder_type=obs_encoder_type,
    )

    # Init
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((chunk_length, batch_size, *obs_shape))
    dummy_act = jnp.zeros((chunk_length, batch_size), dtype=jnp.int32)
    dummy_carry = jnp.zeros((batch_size, hidden_dim))
    vae_params = vae.init(init_rng, dummy_obs, dummy_act, dummy_carry, dummy_carry, init_rng)

    weight_decay = config.get("GAMMA_VAE_WEIGHT_DECAY", 1e-4)
    tx = optax.adamw(vae_lr, weight_decay=weight_decay)
    opt_state = tx.init(vae_params)

    chunks_obs, chunks_act = _prepare_chunks(all_obs, all_actions, chunk_length)
    n_chunks = len(chunks_obs)
    print(f"[VAE] {n_chunks} chunks, chunk_length={chunk_length}, obs_shape={obs_shape}")

    # === Pre-stack chunks once to device → per-batch host transfer 제거 ===
    chunks_obs_arr = jnp.stack([jnp.asarray(c) for c in chunks_obs], axis=0)  # (n_chunks, T, *obs_shape)
    chunks_act_arr = jnp.stack([jnp.asarray(c) for c in chunks_act], axis=0)  # (n_chunks, T)

    n_batches = n_chunks // batch_size  # drop remainder (jit 고정 shape)
    if n_batches < 1:
        raise ValueError(f"[VAE] n_chunks({n_chunks}) < batch_size({batch_size})")
    print(f"[VAE] n_batches per epoch = {n_batches}, dropping {n_chunks - n_batches * batch_size} chunks/epoch")

    def _loss_fn(params, obs_batch, act_batch, kl_w, rng):
        T, B = obs_batch.shape[:2]
        enc_carry = jnp.zeros((B, hidden_dim))
        dec_carry = jnp.zeros((B, hidden_dim))
        logits, z_mean, z_logvar, _, _, _ = vae.apply(
            params, obs_batch, act_batch, enc_carry, dec_carry, rng,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        act_onehot = jax.nn.one_hot(act_batch, action_dim)
        recon_loss = -(log_probs * act_onehot).sum(-1).sum(0).mean()
        kl = GAMMAVAE.kl_divergence(z_mean, z_logvar).mean()
        total = recon_loss + kl_w * kl
        pred = jnp.argmax(logits, axis=-1)
        acc = (pred == act_batch).mean()
        return total, (recon_loss, kl, acc)

    def _batch_step(carry, batch_idx):
        params, opt_state, rng, kl_w = carry
        obs_b = jnp.take(chunks_obs_arr, batch_idx, axis=0)  # (B, T, ...)
        obs_b = jnp.swapaxes(obs_b, 0, 1)                     # (T, B, ...)
        act_b = jnp.take(chunks_act_arr, batch_idx, axis=0)
        act_b = jnp.swapaxes(act_b, 0, 1)
        rng, step_rng = jax.random.split(rng)
        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, obs_b, act_b, kl_w, step_rng,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, rng, kl_w), (loss, aux[2])  # (loss, acc)

    def _epoch_step(carry, kl_w):
        params, opt_state, rng = carry
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_chunks)[: n_batches * batch_size]
        perm = perm.reshape(n_batches, batch_size)
        (params, opt_state, rng, _), (losses, accs) = jax.lax.scan(
            _batch_step, (params, opt_state, rng, kl_w), perm,
        )
        return (params, opt_state, rng), (losses.mean(), accs.mean())

    # epoch별 kl_w 스케줄
    kl_ws = kl_penalty_init + (kl_penalty_final - kl_penalty_init) * (
        jnp.arange(vae_epochs, dtype=jnp.float32) / max(vae_epochs - 1, 1)
    )

    # 전체 학습을 하나의 scan으로 jit → Python overhead 제거
    @jax.jit
    def _train_all(params, opt_state, rng):
        (params, opt_state, rng), (epoch_losses, epoch_accs) = jax.lax.scan(
            _epoch_step, (params, opt_state, rng), kl_ws,
        )
        return params, opt_state, epoch_losses, epoch_accs

    vae_params, opt_state, epoch_losses, epoch_accs = _train_all(
        vae_params, opt_state, rng,
    )
    epoch_losses = np.array(epoch_losses)
    epoch_accs = np.array(epoch_accs)

    for epoch in range(vae_epochs):
        if epoch % 10 == 0:
            print(f"[VAE] epoch {epoch}/{vae_epochs}: loss={epoch_losses[epoch]:.4f} acc={epoch_accs[epoch]:.3f}")
        wandb.log({
            "vae/loss": float(epoch_losses[epoch]),
            "vae/accuracy": float(epoch_accs[epoch]),
            "vae/epoch": epoch,
        })

    return vae, vae_params


# =====================================================================
# Phase B: 순수 VAE 파트너 RL (원본 GAMMA: train_coordinator_vs_vae)
# =====================================================================

def _build_s2_vae_train(config, vae, vae_params):
    """
    순수 VAE decoder 파트너 PPO 학습 — pure `train(rng)` 클로저 생성 후 반환.
    jit/vmap/pmap은 호출자가 적용 (Phase B 멀티시드 병렬 학습 지원).
    `vae`, `vae_params`는 Python closure로 캡처되어 vmap 축에 올라가지 않음(broadcast).
    """
    model_config = config["model"]
    env_config = config["env"]

    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    env_raw = jaxmarl.make(env_name, **env_kwargs)
    ACTION_DIM = env_raw.action_space(env_raw.agents[0]).n

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env_raw, replace_info=False)
    else:
        env = LogWrapper(env_raw, replace_info=False)

    NUM_ENVS = model_config["NUM_ENVS"]
    NUM_STEPS = model_config["NUM_STEPS"]
    NUM_UPDATES = int(model_config["TOTAL_TIMESTEPS"] // NUM_STEPS // NUM_ENVS)
    NUM_MINIBATCHES = model_config["NUM_MINIBATCHES"]
    UPDATE_EPOCHS = model_config["UPDATE_EPOCHS"]
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]
    num_checkpoints = config.get("NUM_CHECKPOINTS", 3)

    z_dim = config.get("GAMMA_VAE_Z_DIM", 16)
    vae_hidden_dim = config.get("GAMMA_VAE_HIDDEN_DIM", 64)

    mappo_mode = bool(config.get("MAPPO_MODE", False))
    use_valuenorm = bool(model_config.get("USE_VALUENORM", False))

    network = ActorCriticRNN(action_dim=ACTION_DIM, config=model_config)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    def _make_lr_fn():
        base_lr = model_config["LR"]
        if not model_config.get("ANNEAL_LR", True):
            return base_lr
        warmup_ratio = model_config.get("LR_WARMUP", 0.05)
        warmup_steps = int(warmup_ratio * NUM_UPDATES)
        steps_per_epoch = NUM_MINIBATCHES * UPDATE_EPOCHS
        warmup_fn = optax.linear_schedule(0.0, base_lr, max(warmup_steps * steps_per_epoch, 1))
        cosine_fn = optax.cosine_decay_schedule(base_lr, max((NUM_UPDATES - warmup_steps) * steps_per_epoch, 1))
        return optax.join_schedules([warmup_fn, cosine_fn], [warmup_steps * steps_per_epoch])

    lr_fn = _make_lr_fn()
    tx = optax.chain(optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]), optax.adam(lr_fn, eps=1e-5))

    # MAPPO: global obs shape 추출 (train() 밖에서 concrete 값으로 계산 — pmap/vmap 호환)
    _pre_rng = jax.random.PRNGKey(0)
    _pre_obs, _pre_state = jax.vmap(env.reset)(jax.random.split(_pre_rng, NUM_ENVS))
    obs_shape = _pre_obs[env.agents[0]].shape[1:]  # (H, W, C)
    if mappo_mode:
        _full = jax.vmap(env._env.get_obs_default)(_pre_state.env_state)
        _global_sample = _full[:, 0].astype(jnp.float32)
        global_obs_dim = 1
        for d in _global_sample.shape[1:]:
            global_obs_dim *= d
        print(f"[GAMMA-S2-VAE] MAPPO global_obs_dim={global_obs_dim}")
    else:
        global_obs_dim = 0

    def _extract_global_obs(env_state):
        _full = jax.vmap(env._env.get_obs_default)(env_state.env_state)
        _g = _full[:, 0].astype(jnp.float32)
        return _g.reshape(NUM_ENVS, -1)

    def train(rng):

        # Ego network init
        init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x = (jnp.zeros((1, NUM_ENVS, *obs_shape)), jnp.zeros((1, NUM_ENVS)))
        init_gobs = jnp.zeros((1, NUM_ENVS, global_obs_dim)) if mappo_mode else None
        rng, _rng = jax.random.split(rng)
        ego_params = network.init(_rng, init_h, init_x, global_obs=init_gobs)
        ego_state = TrainState.create(apply_fn=network.apply, params=ego_params, tx=tx)

        # Env init
        rng, _rng = jax.random.split(rng)
        last_obs, env_state = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        last_done = jnp.zeros((NUM_ENVS,), dtype=jnp.bool_)

        ego_hstate = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        dec_carry = jnp.zeros((NUM_ENVS, vae_hidden_dim))
        rng, z_rng = jax.random.split(rng)
        z_per_env = jax.random.normal(z_rng, (NUM_ENVS, z_dim))

        # Checkpoint buffer
        checkpoint_steps = jnp.linspace(0, NUM_UPDATES, max(num_checkpoints, 1), endpoint=True, dtype=jnp.int32)
        if num_checkpoints > 0:
            checkpoint_steps = checkpoint_steps.at[-1].set(NUM_UPDATES)
        ck_buf = jax.tree_util.tree_map(
            lambda p: jnp.zeros((max(num_checkpoints, 1),) + p.shape, p.dtype), ego_state.params,
        )
        ck_buf = jax.tree_util.tree_map(lambda b, p: b.at[0].set(p), ck_buf, ego_state.params)
        vn_state = ValueNormState.create()

        def _update_step(runner_state, unused):
            (ego_state, env_state, last_obs, last_done,
             ego_hstate, dec_carry, z_per_env,
             ck_buf, vn_state, update_step, rng) = runner_state

            # ---- ROLLOUT: 순수 VAE 파트너 ----
            # Bidirectional ego slot: 원본 GAMMA coordinator 는 env 인덱스 `e % 2` 로
            # ego 가 agent_0/agent_1 에 번갈아 배치됨. 짝수 env → ego=agent_0, 홀수 env → ego=agent_1.
            # 이렇게 하면 ego 정책이 양쪽 slot 역할을 모두 학습 (forced_coord 처럼 역할이 비대칭인
            # 레이아웃에서 특히 중요).
            ego_slot0 = (jnp.arange(NUM_ENVS) % 2) == 0  # (E,) bool, fixed across updates

            def _env_step(step_state, _):
                (env_state, last_obs, last_done,
                 ego_hs, dec_c, z_env, update_step, rng) = step_state

                obs_0 = last_obs[env.agents[0]]           # (E, ...)
                obs_1 = last_obs[env.agents[1]]           # (E, ...)
                # Broadcast mask to obs shape
                obs_mask = ego_slot0.reshape((NUM_ENVS,) + (1,) * (obs_0.ndim - 1))
                ego_obs = jnp.where(obs_mask, obs_0, obs_1)      # ego's own obs
                partner_obs = jnp.where(obs_mask, obs_1, obs_0)  # the other agent's obs
                done_t = last_done

                # z_change_prob: 매 step 확률적으로 z 재샘플 (원본 동작) + 에피 경계에선 항상 재샘플
                z_change_prob = float(config.get("GAMMA_VAE_Z_CHANGE_PROB", 0.0))
                rng, z_rng, z_flip_rng = jax.random.split(rng, 3)
                new_z = jax.random.normal(z_rng, z_env.shape)
                flip_prob = jax.random.uniform(z_flip_rng, (NUM_ENVS,))
                should_change = (flip_prob < z_change_prob) | done_t
                z_env = jnp.where(should_change[:, jnp.newaxis], new_z, z_env)
                # decoder carry: 에피 경계에서만 초기화
                dec_c = jnp.where(done_t[:, jnp.newaxis], jnp.zeros_like(dec_c), dec_c)

                # Global obs for MAPPO critic (agent_0 view, 항상 고정 — 대칭이라 ego slot 과 무관)
                if mappo_mode:
                    _gobs = _extract_global_obs(env_state)
                    _gobs_in = _gobs[jnp.newaxis]
                else:
                    _gobs = jnp.zeros((NUM_ENVS, 1), dtype=jnp.float32)
                    _gobs_in = None

                # Ego agent
                rng, k_ego = jax.random.split(rng)
                ego_hs, ego_pi, crit_val, _ = network.apply(
                    ego_state.params, ego_hs,
                    (ego_obs[jnp.newaxis], done_t[jnp.newaxis]),
                    global_obs=_gobs_in,
                )
                ego_action = ego_pi.sample(seed=k_ego).squeeze(0)
                ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)
                crit_val = crit_val.squeeze(0)

                # Partner: 순수 VAE decoder (raw obs → VAE 자체 encoder → action)
                vae_logits, dec_c = vae.apply(
                    vae_params,
                    partner_obs[jnp.newaxis],  # (1, E, *obs_shape) = (T=1, B=E, ...)
                    z_env,                      # (E, z_dim)
                    dec_c,                      # (E, hidden_dim)
                    method=GAMMAVAE.decode,
                )
                vae_logits = vae_logits.squeeze(0)  # (E, A)
                rng, k_vae = jax.random.split(rng)
                partner_action = jax.random.categorical(k_vae, vae_logits)

                # Step env — per-env 로 ego/partner action 을 각 slot 에 배치
                agent0_action = jnp.where(ego_slot0, ego_action, partner_action)
                agent1_action = jnp.where(ego_slot0, partner_action, ego_action)
                env_act = {env.agents[0]: agent0_action, env.agents[1]: agent1_action}
                rng, k_env = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(k_env, NUM_ENVS), env_state, env_act,
                )
                done_env = done["__all__"]

                anneal = rew_shaping_anneal(update_step * NUM_STEPS * NUM_ENVS)
                if "shaped_reward" in info:
                    reward = jax.tree_util.tree_map(lambda r, s: r + s * anneal, reward, info["shaped_reward"])
                # Ego reward: per-env 에서 ego 가 있는 slot 의 reward 를 꺼냄
                ego_reward = jnp.where(ego_slot0, reward[env.agents[0]], reward[env.agents[1]])
                info.pop("shaped_reward", None)
                info.pop("shaped_reward_events", None)
                info = jax.tree_util.tree_map(lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info)

                transition = S2Transition(
                    done=done_env, ego_obs=ego_obs, ego_action=ego_action,
                    ego_log_prob=ego_log_prob, critic_value=crit_val,
                    reward=ego_reward, info=info, global_obs=_gobs,
                )
                return (env_state, obsv, done_env, ego_hs, dec_c, z_env, update_step, rng), transition

            final, traj = jax.lax.scan(
                _env_step,
                (env_state, last_obs, last_done, ego_hstate, dec_carry, z_per_env, update_step, rng),
                None, NUM_STEPS,
            )
            (env_state, last_obs, last_done, ego_hstate, dec_carry, z_per_env, _, rng) = final

            # ---- GAE + PPO ----
            # Ego's last obs depends on its slot (bidirectional)
            _obs0_last = last_obs[env.agents[0]]
            _obs1_last = last_obs[env.agents[1]]
            _obs_mask_last = ego_slot0.reshape((NUM_ENVS,) + (1,) * (_obs0_last.ndim - 1))
            ego_obs_last = jnp.where(_obs_mask_last, _obs0_last, _obs1_last)
            init_h_bs = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            _last_gobs = traj.global_obs[-1][jnp.newaxis] if mappo_mode else None
            _, _, last_val, _ = network.apply(
                ego_state.params, init_h_bs,
                (ego_obs_last[jnp.newaxis], last_done[jnp.newaxis]),
                global_obs=_last_gobs,
            )
            last_val = last_val.squeeze(0)

            def _get_adv(carry, xs):
                gae, nv = carry
                done, value, reward = xs
                delta = reward + model_config["GAMMA"] * nv * (1 - done) - value
                gae = delta + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            # ValueNorm: denormalize for GAE
            if use_valuenorm:
                gae_values = valuenorm_denormalize(vn_state, traj.critic_value)
                gae_last = valuenorm_denormalize(vn_state, last_val)
            else:
                gae_values = traj.critic_value
                gae_last = last_val

            _, advantages = jax.lax.scan(
                _get_adv, (jnp.zeros_like(gae_last), gae_last),
                (traj.done, gae_values, traj.reward), reverse=True, unroll=16,
            )
            targets = advantages + gae_values

            # ValueNorm: update + normalize targets
            if use_valuenorm:
                vn_state = valuenorm_update(vn_state, targets)
                targets = valuenorm_normalize(vn_state, targets)

            mb_size = NUM_ENVS // NUM_MINIBATCHES
            init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

            def _split_mb(x):
                return jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:])), 1, 0,
                )

            def _update_minibatch(state, mb):
                (mb_ah, mb_obs, mb_done, mb_act, mb_lp, mb_val, mb_adv, mb_tgt, mb_gobs) = mb
                mb_ah = mb_ah.squeeze(0)
                def _loss(params):
                    _gobs_loss = mb_gobs if mappo_mode else None
                    _, pi, value, _ = network.apply(params, mb_ah, (mb_obs, mb_done), global_obs=_gobs_loss)
                    lp = pi.log_prob(mb_act)
                    ratio = jnp.exp(lp - mb_lp)
                    adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    l1 = ratio * adv_n
                    l2 = jnp.clip(ratio, 1 - model_config["CLIP_EPS"], 1 + model_config["CLIP_EPS"]) * adv_n
                    actor_loss = -jnp.minimum(l1, l2).mean()
                    v_cl = mb_val + (value - mb_val).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                    # 원본 GAMMA: use_huber_loss=True, huber_delta=10.0
                    _huber_delta = float(model_config.get("HUBER_DELTA", 10.0))
                    def _huber(err):
                        return jnp.where(
                            jnp.abs(err) <= _huber_delta,
                            0.5 * err ** 2,
                            _huber_delta * (jnp.abs(err) - 0.5 * _huber_delta),
                        )
                    critic_loss = jnp.maximum(
                        _huber(value - mb_tgt), _huber(v_cl - mb_tgt)
                    ).mean()
                    entropy = pi.entropy().mean()
                    total = actor_loss + model_config.get("VF_COEF", 0.5) * critic_loss - model_config["ENT_COEF"] * entropy
                    return total, (actor_loss, critic_loss, entropy)
                (total, aux), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
                return state.apply_gradients(grads=grads), (total, aux[0], aux[1], aux[2])

            def _update_epoch(es, _):
                state, rng_e = es
                rng_e, _r = jax.random.split(rng_e)
                perm = jax.random.permutation(_r, NUM_ENVS)
                batch = (
                    jnp.take(init_h_mb, perm, axis=0)[jnp.newaxis],
                    jnp.take(traj.ego_obs, perm, axis=1),
                    jnp.take(traj.done, perm, axis=1),
                    jnp.take(traj.ego_action, perm, axis=1),
                    jnp.take(traj.ego_log_prob, perm, axis=1),
                    jnp.take(traj.critic_value, perm, axis=1),
                    jnp.take(advantages, perm, axis=1),
                    jnp.take(targets, perm, axis=1),
                    jnp.take(traj.global_obs, perm, axis=1),
                )
                mbs = jax.tree_util.tree_map(_split_mb, batch)
                state, losses = jax.lax.scan(_update_minibatch, state, mbs)
                return (state, rng_e), losses

            rng, _rng = jax.random.split(rng)
            (ego_state, _), all_losses = jax.lax.scan(_update_epoch, (ego_state, _rng), None, UPDATE_EPOCHS)

            # Checkpoint
            update_step = update_step + 1
            selector = checkpoint_steps == update_step
            slot = jnp.argmax(selector)
            ck_buf = jax.lax.cond(
                jnp.any(selector),
                lambda b: jax.tree_util.tree_map(lambda bl, p: bl.at[slot].set(p), b, ego_state.params),
                lambda b: b, ck_buf,
            )

            # Metrics
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj.info)
            metric["total_loss"] = all_losses[0].mean()
            metric["update_step"] = update_step

            def _log(m):
                # vmap 아래서 leaf가 (num_seeds,) 배열일 수 있으므로 평균으로 집계.
                def _scalar(v):
                    return float(np.asarray(v).reshape(-1).mean())
                flat = {}
                for k, v in m.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}/{kk}"] = _scalar(vv)
                    else:
                        flat[k] = _scalar(v)
                wandb.log({f"gamma_s2_vae/{k}": v for k, v in flat.items()})
            jax.debug.callback(_log, metric)

            return (ego_state, env_state, last_obs, last_done,
                    ego_hstate, dec_carry, z_per_env,
                    ck_buf, vn_state, update_step, rng), metric

        init_runner = (ego_state, env_state, last_obs, last_done,
                       ego_hstate, dec_carry, z_per_env,
                       ck_buf, vn_state, jnp.int32(0), rng)
        final_runner, metrics = jax.lax.scan(_update_step, init_runner, None, NUM_UPDATES)
        return {"actor_params": final_runner[0].params, "actor_ckpts": final_runner[7]}

    return train


def _run_s2_vae_only(config, vae, vae_params, rng):
    """하위호환 wrapper: 단일 seed 순차 학습용."""
    train_fn = _build_s2_vae_train(config, vae, vae_params)
    return jax.jit(train_fn)(rng)


# =====================================================================
# Entry point
# =====================================================================

def run_gamma_s2_vae(config, pop_params, pop_dir):
    """GAMMA S2 VAE 전체 파이프라인: rollout → VAE 학습 → 순수 VAE RL."""
    model_config = config["model"]
    env_config = config["env"]

    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    env_raw = jaxmarl.make(env_name, **env_kwargs)
    ACTION_DIM = env_raw.action_space(env_raw.agents[0]).n
    model_config["ACTION_DIM"] = ACTION_DIM

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env_raw, replace_info=False)
    else:
        env = LogWrapper(env_raw, replace_info=False)

    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]
    network = ActorCriticRNN(action_dim=ACTION_DIM, config=model_config)

    # ---- Phase A: Rollout 수집 + VAE 학습 ----
    print("[GAMMA S2 VAE] Phase A: Collecting rollouts & training VAE")
    n_episodes = config.get("GAMMA_VAE_ROLLOUT_EPISODES", 100)
    rng = jax.random.PRNGKey(config["SEED"])
    rng, collect_rng = jax.random.split(rng)

    all_obs, all_actions = _collect_population_rollouts(
        pop_params, config, env, network, n_episodes, collect_rng,
    )

    obs_shape = all_obs[0].shape[1:]
    rng, vae_rng = jax.random.split(rng)
    vae, vae_params = _train_vae(config, all_obs, all_actions, obs_shape, vae_rng)

    # VAE params 저장
    run_base_dir = Path(config["RUN_BASE_DIR"])
    vae_dir = run_base_dir / "vae_model"
    vae_dir.mkdir(parents=True, exist_ok=True)
    with open(vae_dir / "vae_params.pkl", "wb") as f:
        pickle.dump(vae_params, f)
    print(f"[GAMMA S2 VAE] VAE saved to {vae_dir}")

    # ---- Phase B: 순수 VAE 파트너 RL (multi-seed 병렬) ----
    print("[GAMMA S2 VAE] Phase B: Pure VAE partner RL training (parallel)")
    num_seeds = config["NUM_SEEDS"]
    rng, rl_rng = jax.random.split(rng)
    rngs = jax.random.split(rl_rng, num_seeds)
    rngs = jax.device_put(rngs, jax.devices("cpu")[0])

    train_fn = _build_s2_vae_train(config, vae, vae_params)
    num_devices = get_num_devices()
    print(f"[GAMMA S2 VAE] num_seeds={num_seeds}, num_devices={num_devices}")

    if num_devices <= 1:
        train_jit = jax.jit(train_fn)
        if num_seeds == 1:
            out = train_jit(rngs[0])
        else:
            out = jax.vmap(train_jit)(rngs)
    else:
        if num_seeds == num_devices:
            out = jax.pmap(train_fn)(rngs)
        elif num_seeds % num_devices == 0:
            rngs_2d = rngs.reshape((num_devices, num_seeds // num_devices, *rngs.shape[1:]))
            out = jax.pmap(jax.vmap(train_fn))(rngs_2d)
            out = jax.tree_util.tree_map(
                lambda x: x.reshape((num_seeds, *x.shape[2:])), out
            )
        else:
            print(f"[warn] num_seeds({num_seeds}) % num_devices({num_devices}) != 0; fallback to vmap")
            train_jit = jax.jit(train_fn)
            out = jax.vmap(train_jit)(rngs)

    # 체크포인트 저장 (MEP/GAMMA rl S2 동일 패턴)
    num_checkpoints = config.get("NUM_CHECKPOINTS", 3)
    if num_checkpoints > 0:
        actor_ckpts = out["actor_ckpts"]
        sample_leaf = jax.tree_util.tree_leaves(actor_ckpts)[0]
        has_seed_axis = sample_leaf.shape[0] == num_seeds and num_seeds > 1

        for s in range(num_seeds):
            if has_seed_axis:
                seed_ckpts = jax.tree_util.tree_map(lambda x: x[s], actor_ckpts)
            else:
                seed_ckpts = actor_ckpts
            for slot in range(num_checkpoints - 1):
                params = jax.tree_util.tree_map(lambda x: x[slot], seed_ckpts)
                store_checkpoint(config, params, s, slot, final=False)
            params_final = jax.tree_util.tree_map(
                lambda x: x[num_checkpoints - 1], seed_ckpts
            )
            store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)

    print(f"[GAMMA S2 VAE] Saved {num_seeds} seeds to {config['RUN_BASE_DIR']}")
    return {"vae_params": vae_params, "rl_outs": out}
