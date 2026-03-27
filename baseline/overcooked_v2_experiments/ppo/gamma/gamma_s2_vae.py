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


class S2Transition(NamedTuple):
    done: jnp.ndarray
    ego_obs: jnp.ndarray
    ego_action: jnp.ndarray
    ego_log_prob: jnp.ndarray
    critic_value: jnp.ndarray
    reward: jnp.ndarray
    info: dict


# =====================================================================
# Phase A: Rollout 수집 + VAE 학습
# =====================================================================

def _collect_population_rollouts(pop_params, config, env, network, n_episodes, rng):
    """Population self-play rollout을 raw obs + action으로 수집."""
    model_config = config["model"]
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]
    N = jax.tree_util.tree_leaves(pop_params)[0].shape[0]
    max_steps = 400

    all_obs = []     # raw observations
    all_actions = []

    for member_idx in range(N):
        member_params = jax.tree_util.tree_map(lambda x: x[member_idx], pop_params)

        for ep in range(n_episodes // N):
            rng, k_reset = jax.random.split(rng)
            obs, state = env.reset(k_reset)

            h0 = ActorCriticRNN.initialize_carry(1, GRU_HIDDEN_DIM)
            h1 = ActorCriticRNN.initialize_carry(1, GRU_HIDDEN_DIM)
            done = jnp.zeros((1,), dtype=jnp.bool_)

            ep_obs_0, ep_obs_1, ep_act_0, ep_act_1 = [], [], [], []

            for step in range(max_steps):
                obs_0 = obs[env.agents[0]][jnp.newaxis, jnp.newaxis]  # (1,1,...)
                obs_1 = obs[env.agents[1]][jnp.newaxis, jnp.newaxis]

                rng, k0, k1, k_env = jax.random.split(rng, 4)
                h0, pi_0, _, _ = network.apply(member_params, h0, (obs_0, done[jnp.newaxis]))
                a0 = pi_0.sample(seed=k0).squeeze(0)

                h1, pi_1, _, _ = network.apply(member_params, h1, (obs_1, done[jnp.newaxis]))
                a1 = pi_1.sample(seed=k1).squeeze(0)

                # raw obs 저장 (VAE 자체 encoder가 처리)
                ep_obs_0.append(np.array(obs[env.agents[0]]))
                ep_obs_1.append(np.array(obs[env.agents[1]]))
                ep_act_0.append(int(a0.squeeze()))
                ep_act_1.append(int(a1.squeeze()))

                env_act = {env.agents[0]: a0.squeeze(), env.agents[1]: a1.squeeze()}
                obs, state, reward, done_dict, info = env.step(k_env, state, env_act)
                done = jnp.array([done_dict["__all__"]])

                if done_dict["__all__"]:
                    break

            if len(ep_obs_0) > 10:
                all_obs.append(np.stack(ep_obs_0))
                all_actions.append(np.array(ep_act_0))
                all_obs.append(np.stack(ep_obs_1))
                all_actions.append(np.array(ep_act_1))

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
    hidden_dim = config.get("GAMMA_VAE_HIDDEN_DIM", 64)
    action_dim = config["model"]["ACTION_DIM"]
    vae_lr = config.get("GAMMA_VAE_LR", 1e-3)
    vae_epochs = config.get("GAMMA_VAE_EPOCHS", 100)
    batch_size = config.get("GAMMA_VAE_BATCH_SIZE", 64)
    chunk_length = config.get("GAMMA_VAE_CHUNK_LENGTH", 100)
    kl_penalty_final = config.get("GAMMA_VAE_KL_PENALTY", 0.1)
    kl_penalty_init = config.get("GAMMA_VAE_KL_INIT", 0.01)
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

    tx = optax.adam(vae_lr)
    opt_state = tx.init(vae_params)

    chunks_obs, chunks_act = _prepare_chunks(all_obs, all_actions, chunk_length)
    n_chunks = len(chunks_obs)
    print(f"[VAE] {n_chunks} chunks, chunk_length={chunk_length}, obs_shape={obs_shape}")

    @jax.jit
    def vae_step(params, opt_state, obs_batch, act_batch, kl_w, rng):
        def loss_fn(params):
            T, B = obs_batch.shape[:2]
            enc_carry = jnp.zeros((B, hidden_dim))
            dec_carry = jnp.zeros((B, hidden_dim))
            logits, z_mean, z_logvar, z, _, _ = vae.apply(
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

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    for epoch in range(vae_epochs):
        kl_w = kl_penalty_init + (kl_penalty_final - kl_penalty_init) * epoch / max(vae_epochs - 1, 1)
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_chunks)

        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for batch_start in range(0, n_chunks, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            if len(batch_idx) < 2:
                continue
            obs_batch = jnp.stack([chunks_obs[int(i)] for i in batch_idx], axis=1)  # (T, B, ...)
            act_batch = jnp.stack([chunks_act[int(i)] for i in batch_idx], axis=1)

            rng, step_rng = jax.random.split(rng)
            vae_params, opt_state, loss, (recon, kl, acc) = vae_step(
                vae_params, opt_state, obs_batch, act_batch, kl_w, step_rng,
            )
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            n_batches += 1

        if n_batches > 0 and epoch % 10 == 0:
            print(f"[VAE] epoch {epoch}/{vae_epochs}: loss={epoch_loss/n_batches:.4f} acc={epoch_acc/n_batches:.3f}")
        if n_batches > 0:
            wandb.log({"vae/loss": epoch_loss / n_batches, "vae/accuracy": epoch_acc / n_batches, "vae/epoch": epoch})

    return vae, vae_params


# =====================================================================
# Phase B: 순수 VAE 파트너 RL (원본 GAMMA: train_coordinator_vs_vae)
# =====================================================================

def _run_s2_vae_only(config, vae, vae_params, rng):
    """
    순수 VAE decoder를 파트너로 사용한 PPO 학습 (원본 GAMMA와 동일).
    모든 파트너 = VAE decoder(raw_obs, z). Population 멤버 미사용.
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

    def train(rng):
        rng, _rng = jax.random.split(rng)
        sample_obs, _ = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        obs_shape = sample_obs[env.agents[0]].shape[1:]

        # Ego network init
        init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x = (jnp.zeros((1, NUM_ENVS, *obs_shape)), jnp.zeros((1, NUM_ENVS)))
        rng, _rng = jax.random.split(rng)
        ego_params = network.init(_rng, init_h, init_x)
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

        def _update_step(runner_state, unused):
            (ego_state, env_state, last_obs, last_done,
             ego_hstate, dec_carry, z_per_env,
             ck_buf, update_step, rng) = runner_state

            # ---- ROLLOUT: 순수 VAE 파트너 ----
            def _env_step(step_state, _):
                (env_state, last_obs, last_done,
                 ego_hs, dec_c, z_env, update_step, rng) = step_state

                ego_obs = last_obs[env.agents[0]]        # (E, ...)
                partner_obs = last_obs[env.agents[1]]     # (E, ...)
                done_t = last_done

                # 에피소드 리셋 시 새 z 샘플링 + decoder carry 초기화
                rng, z_rng = jax.random.split(rng)
                new_z = jax.random.normal(z_rng, z_env.shape)
                z_env = jnp.where(done_t[:, jnp.newaxis], new_z, z_env)
                dec_c = jnp.where(done_t[:, jnp.newaxis], jnp.zeros_like(dec_c), dec_c)

                # Ego agent
                rng, k_ego = jax.random.split(rng)
                ego_hs, ego_pi, crit_val, _ = network.apply(
                    ego_state.params, ego_hs,
                    (ego_obs[jnp.newaxis], done_t[jnp.newaxis]),
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

                # Step env
                env_act = {env.agents[0]: ego_action, env.agents[1]: partner_action}
                rng, k_env = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(k_env, NUM_ENVS), env_state, env_act,
                )
                done_env = done["__all__"]

                anneal = rew_shaping_anneal(update_step * NUM_STEPS * NUM_ENVS)
                reward = jax.tree_util.tree_map(lambda r, s: r + s * anneal, reward, info["shaped_reward"])
                ego_reward = reward[env.agents[0]]
                info = {k: v for k, v in info.items() if k not in ("shaped_reward", "shaped_reward_events")}
                info = jax.tree_util.tree_map(lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info)

                transition = S2Transition(
                    done=done_env, ego_obs=ego_obs, ego_action=ego_action,
                    ego_log_prob=ego_log_prob, critic_value=crit_val,
                    reward=ego_reward, info=info,
                )
                return (env_state, obsv, done_env, ego_hs, dec_c, z_env, update_step, rng), transition

            final, traj = jax.lax.scan(
                _env_step,
                (env_state, last_obs, last_done, ego_hstate, dec_carry, z_per_env, update_step, rng),
                None, NUM_STEPS,
            )
            (env_state, last_obs, last_done, ego_hstate, dec_carry, z_per_env, _, rng) = final

            # ---- GAE + PPO ----
            ego_obs_last = last_obs[env.agents[0]]
            init_h_bs = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            _, _, last_val, _ = network.apply(
                ego_state.params, init_h_bs,
                (ego_obs_last[jnp.newaxis], last_done[jnp.newaxis]),
            )
            last_val = last_val.squeeze(0)

            def _get_adv(carry, xs):
                gae, nv = carry
                done, value, reward = xs
                delta = reward + model_config["GAMMA"] * nv * (1 - done) - value
                gae = delta + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_adv, (jnp.zeros_like(last_val), last_val),
                (traj.done, traj.critic_value, traj.reward), reverse=True, unroll=16,
            )
            targets = advantages + traj.critic_value

            mb_size = NUM_ENVS // NUM_MINIBATCHES
            init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

            def _split_mb(x):
                return jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:])), 1, 0,
                )

            def _update_minibatch(state, mb):
                (mb_ah, mb_obs, mb_done, mb_act, mb_lp, mb_val, mb_adv, mb_tgt) = mb
                mb_ah = mb_ah.squeeze(0)
                def _loss(params):
                    _, pi, value, _ = network.apply(params, mb_ah, (mb_obs, mb_done))
                    lp = pi.log_prob(mb_act)
                    ratio = jnp.exp(lp - mb_lp)
                    adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    l1 = ratio * adv_n
                    l2 = jnp.clip(ratio, 1 - model_config["CLIP_EPS"], 1 + model_config["CLIP_EPS"]) * adv_n
                    actor_loss = -jnp.minimum(l1, l2).mean()
                    v_cl = mb_val + (value - mb_val).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                    critic_loss = 0.5 * jnp.maximum(jnp.square(value - mb_tgt), jnp.square(v_cl - mb_tgt)).mean()
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
                flat = {}
                for k, v in m.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}/{kk}"] = float(vv)
                    else:
                        flat[k] = float(v)
                wandb.log({f"gamma_s2_vae/{k}": v for k, v in flat.items()})
            jax.debug.callback(_log, metric)

            return (ego_state, env_state, last_obs, last_done,
                    ego_hstate, dec_carry, z_per_env,
                    ck_buf, update_step, rng), metric

        init_runner = (ego_state, env_state, last_obs, last_done,
                       ego_hstate, dec_carry, z_per_env,
                       ck_buf, jnp.int32(0), rng)
        final_runner, metrics = jax.lax.scan(_update_step, init_runner, None, NUM_UPDATES)
        return {"actor_params": final_runner[0].params, "actor_ckpts": final_runner[7]}

    return jax.jit(train)(rng)


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

    # ---- Phase B: 순수 VAE 파트너 RL (multi-seed) ----
    print("[GAMMA S2 VAE] Phase B: Pure VAE partner RL training")
    num_seeds = config["NUM_SEEDS"]
    rng, rl_rng = jax.random.split(rng)
    rngs = jax.random.split(rl_rng, num_seeds)

    all_outs = []
    for s in range(num_seeds):
        print(f"[GAMMA S2 VAE] Seed {s+1}/{num_seeds}")
        out_s = _run_s2_vae_only(config, vae, vae_params, rngs[s])
        all_outs.append(out_s)

        num_checkpoints = config.get("NUM_CHECKPOINTS", 3)
        if num_checkpoints > 0:
            ckpts = out_s["actor_ckpts"]
            for slot in range(num_checkpoints - 1):
                params = jax.tree_util.tree_map(lambda x: x[slot], ckpts)
                store_checkpoint(config, params, s, slot, final=False)
            params_final = jax.tree_util.tree_map(lambda x: x[num_checkpoints - 1], ckpts)
            store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)

    print(f"[GAMMA S2 VAE] Saved {num_seeds} seeds to {config['RUN_BASE_DIR']}")
    return {"vae_params": vae_params, "rl_outs": all_outs}
