"""
HSP Stage 1: N개 독립 정책을 random utility weight로 순차 학습.

각 정책은 self-play(두 에이전트가 동일 params 공유)로 학습하되,
보상 함수가 shaped_reward_events와 utility weight의 dot product로 구성됨.

원본 HSP 논문: "Hidden-utility Self-Play" (Yu et al., NeurIPS 2023)

핵심 차이 (vs MEP S1):
- vmap 대신 Python loop으로 N개 정책 순차 학습 (N=36 OOM 방지)
- entropy bonus 없음 — utility weight shaped reward만 사용
- REW_SHAPING_HORIZON=0: reward annealing 없음
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper, LogWrapper
import wandb

from overcooked_v2_experiments.ppo.models.rnn import ActorCriticRNN


class HSPTransition(NamedTuple):
    done: jnp.ndarray         # (NUM_STEPS, NUM_ENVS)
    ego_obs: jnp.ndarray      # (NUM_STEPS, NUM_ENVS, ...)
    ego_action: jnp.ndarray   # (NUM_STEPS, NUM_ENVS)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    critic_value: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    reward: jnp.ndarray       # (NUM_STEPS, NUM_ENVS)
    info: dict


def make_train_hsp_s1(config):
    """
    HSP Stage 1 학습 함수를 반환한다.

    Returns train(rng) → (all_ckpts, all_weights)
      - all_ckpts: list of N pytrees, 각 leaf shape (num_checkpoints, ...)
      - all_weights: jnp.array (N, EVENT_DIM+1) — 마지막 원소가 sparse reward weight
    """
    env_config = config["env"]
    model_config = config["model"]

    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    env_raw = jaxmarl.make(env_name, **env_kwargs)
    ACTION_DIM = env_raw.action_space(env_raw.agents[0]).n

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env_raw, replace_info=False)
    else:
        env = LogWrapper(env_raw, replace_info=False)

    N = config["HSP_POPULATION_SIZE"]        # 36
    EVENT_DIM = config["HSP_EVENT_DIM"]      # 5 (OV2) or 1 (ToyCoop)

    NUM_ENVS = model_config["NUM_ENVS"]
    NUM_STEPS = model_config["NUM_STEPS"]
    NUM_UPDATES = int(
        model_config["TOTAL_TIMESTEPS"] // NUM_STEPS // NUM_ENVS
    )
    NUM_MINIBATCHES = model_config["NUM_MINIBATCHES"]
    UPDATE_EPOCHS = model_config["UPDATE_EPOCHS"]
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]

    model_config["NUM_UPDATES"] = NUM_UPDATES
    model_config["ACTION_DIM"] = ACTION_DIM

    num_checkpoints = config.get("NUM_CHECKPOINTS", 3)

    network = ActorCriticRNN(action_dim=ACTION_DIM, config=model_config)

    # HSP S1은 REW_SHAPING_HORIZON=0이므로 annealing 사용하지 않음
    # 하지만 shaped_reward (기존 OV2 shaped reward)에 대해서도 annealing 적용할 수 있도록 유지
    rew_shaping_horizon = model_config.get("REW_SHAPING_HORIZON", 0)

    def _make_lr_fn():
        base_lr = model_config["LR"]
        if not model_config.get("ANNEAL_LR", True):
            return base_lr
        warmup_ratio = model_config.get("LR_WARMUP", 0.05)
        warmup_steps = int(warmup_ratio * NUM_UPDATES)
        steps_per_epoch = NUM_MINIBATCHES * UPDATE_EPOCHS
        warmup_fn = optax.linear_schedule(
            0.0, base_lr, max(warmup_steps * steps_per_epoch, 1)
        )
        cosine_fn = optax.cosine_decay_schedule(
            base_lr, max((NUM_UPDATES - warmup_steps) * steps_per_epoch, 1)
        )
        return optax.join_schedules(
            [warmup_fn, cosine_fn], [warmup_steps * steps_per_epoch]
        )

    def _make_tx(lr_fn):
        return optax.chain(
            optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
            optax.adam(lr_fn, eps=1e-5),
        )

    lr_fn = _make_lr_fn()

    def train(rng):
        # ----------------------------------------------------------------
        # Utility weight 생성 — 원본 HSP: (w0, w1) 두 벡터 per policy
        # w0: random event weights from {-10, 0, 10} + random sparse weight from {0, 1}
        # w1: 고정 [0,...,0,1] (event weights=0, sparse weight=1) — 순수 sparse reward
        # ----------------------------------------------------------------
        rng, wrng1, wrng2 = jax.random.split(rng, 3)
        weight_choices = jnp.array([-10.0, 0.0, 10.0])
        sparse_choices = jnp.array([0.0, 1.0])
        event_idxs = jax.random.randint(wrng1, (N, EVENT_DIM), 0, 3)
        event_weights = weight_choices[event_idxs]
        sparse_idxs = jax.random.randint(wrng2, (N, 1), 0, 2)
        sparse_weights = sparse_choices[sparse_idxs]
        all_w0 = jnp.concatenate([event_weights, sparse_weights], axis=-1)  # (N, EVENT_DIM+1)

        # w1: 고정 — event weight=0, sparse weight=1 (원본 HSP 패턴)
        all_w1 = jnp.zeros((N, EVENT_DIM + 1))
        all_w1 = all_w1.at[:, -1].set(1.0)  # (N, EVENT_DIM+1)

        all_ckpts = []

        for i in range(N):
            w0_i = all_w0[i]  # (EVENT_DIM+1,)
            w1_i = all_w1[i]  # (EVENT_DIM+1,)
            rng, srng = jax.random.split(rng)
            ckpts_i = _train_single_policy(srng, w0_i, w1_i)
            all_ckpts.append(ckpts_i)
            print(f"[HSP S1] Policy {i+1}/{N} 완료")

        # all_weights에 w0, w1 둘 다 저장 (w0만 random, w1은 고정이지만 기록용)
        all_weights = jnp.stack([all_w0, all_w1], axis=1)  # (N, 2, EVENT_DIM+1)
        return all_ckpts, all_weights

    def _train_single_policy(rng, w0, w1):
        """
        단일 self-play 정책 학습 (JIT 컴파일).
        두 에이전트가 동일 params를 공유하며, (w0, w1) utility weight 기반 보상으로 학습.
        원본 HSP: agent 0에 w0, agent 1에 w1 적용 (random_index로 swap 가능).
        반환: checkpoint buffer pytree, leaf shape (num_checkpoints, ...)
        """

        def _train_inner(rng):
            # ---- ENV shape inference ----
            rng, _rng = jax.random.split(rng)
            sample_obs, _ = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
            obs_shape = sample_obs[env.agents[0]].shape[1:]

            # ---- Network init ----
            init_hstate = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            init_x = (
                jnp.zeros((1, NUM_ENVS, *obs_shape)),
                jnp.zeros((1, NUM_ENVS)),
            )

            rng, _rng = jax.random.split(rng)
            params = network.init(_rng, init_hstate, init_x)
            actor_state = TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=_make_tx(lr_fn),
            )

            # ---- Init envs ----
            rng, _rng = jax.random.split(rng)
            last_obs, env_state = jax.vmap(env.reset)(
                jax.random.split(_rng, NUM_ENVS)
            )
            last_done = jnp.zeros((NUM_ENVS,), dtype=jnp.bool_)

            # ---- hstates for two agents (shared params, separate hidden states) ----
            hstate_0 = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            hstate_1 = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

            # ---- Checkpoint buffer ----
            checkpoint_steps = jnp.linspace(
                0, NUM_UPDATES, max(num_checkpoints, 1),
                endpoint=True, dtype=jnp.int32,
            )
            if num_checkpoints > 0:
                checkpoint_steps = checkpoint_steps.at[-1].set(NUM_UPDATES)
            ck_buf = jax.tree_util.tree_map(
                lambda p: jnp.zeros((max(num_checkpoints, 1),) + p.shape, p.dtype),
                actor_state.params,
            )

            # ---- Initial checkpoint at step 0 ----
            ck_buf = jax.tree_util.tree_map(
                lambda b, p: b.at[0].set(p), ck_buf, actor_state.params,
            )

            # random_index swap flags: 에피소드 시작 시 per-env 독립 swap (원본과 동일)
            rng, _swap_init_rng = jax.random.split(rng)
            swap_flags = jax.random.bernoulli(_swap_init_rng, 0.5, shape=(NUM_ENVS,))

            # ---- TRAIN LOOP ----
            def _update_step(runner_state, unused):
                (
                    actor_state, env_state, last_obs, last_done,
                    hstate_0, hstate_1, swap_flags,
                    ck_buf, update_step, rng,
                ) = runner_state

                # ---- ROLLOUT (self-play: 두 agent 모두 동일 params) ----
                def _env_step(step_state, _):
                    (
                        env_state, last_obs, last_done,
                        h0, h1, swap_flags, update_step, rng,
                    ) = step_state

                    obs_0 = last_obs[env.agents[0]]  # (E, ...)
                    obs_1 = last_obs[env.agents[1]]  # (E, ...)
                    done_t = last_done                 # (E,)

                    # Agent 0
                    rng, k0 = jax.random.split(rng)
                    h0, pi_0, val_0, _ = network.apply(
                        actor_state.params, h0,
                        (obs_0[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    action_0 = pi_0.sample(seed=k0).squeeze(0)
                    log_prob_0 = pi_0.log_prob(action_0[jnp.newaxis]).squeeze(0)
                    val_0 = val_0.squeeze(0)

                    # Agent 1 (동일 params)
                    rng, k1 = jax.random.split(rng)
                    h1, pi_1, _, _ = network.apply(
                        actor_state.params, h1,
                        (obs_1[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    action_1 = pi_1.sample(seed=k1).squeeze(0)

                    # Step envs
                    env_act = {
                        env.agents[0]: action_0,
                        env.agents[1]: action_1,
                    }
                    rng, k_env = jax.random.split(rng)
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        jax.random.split(k_env, NUM_ENVS), env_state, env_act,
                    )
                    done_env = done["__all__"]

                    # ---- random_index: 에피소드 시작 시 per-env swap 갱신 (원본 HSP와 동일) ----
                    rng, swap_rng = jax.random.split(rng)
                    new_swaps = jax.random.bernoulli(swap_rng, 0.5, shape=(NUM_ENVS,))
                    swap_flags = jnp.where(done_env, new_swaps, swap_flags)

                    # per-env w0/w1 선택: swap_flags (E,) broadcast → (E, EVENT_DIM+1)
                    eff_w0 = jnp.where(swap_flags[:, jnp.newaxis], w1, w0)  # (E, EVENT_DIM+1)
                    eff_w1 = jnp.where(swap_flags[:, jnp.newaxis], w0, w1)

                    # ---- Utility weight 기반 보상 (원본 HSP) ----
                    has_events = "shaped_reward_events" in info
                    if has_events:
                        ev_0 = info["shaped_reward_events"][env.agents[0]]  # (E, EVENT_DIM)
                        ev_1 = info["shaped_reward_events"][env.agents[1]]  # (E, EVENT_DIM)
                        r_0 = reward[env.agents[0]]  # sparse reward
                        r_1 = reward[env.agents[1]]
                        # agent별 독립 reward (per-env w0/w1)
                        r_agent0 = jnp.sum(ev_0 * eff_w0[:, :EVENT_DIM], axis=-1) + eff_w0[:, EVENT_DIM] * r_0
                        r_agent1 = jnp.sum(ev_1 * eff_w1[:, :EVENT_DIM], axis=-1) + eff_w1[:, EVENT_DIM] * r_1
                        ego_reward = (r_agent0 + r_agent1) / 2.0
                    else:
                        # ToyCoop fallback
                        r_0 = reward[env.agents[0]]
                        r_1 = reward[env.agents[1]]
                        ego_reward = (eff_w0[:, EVENT_DIM] * r_0 + eff_w1[:, EVENT_DIM] * r_1) / 2.0

                    # info에서 shaped_reward_events 제거 (wandb 로깅 시 혼란 방지)
                    info = {k: v for k, v in info.items()
                            if k not in ("shaped_reward", "shaped_reward_events")}
                    info = jax.tree_util.tree_map(
                        lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info,
                    )

                    transition = HSPTransition(
                        done=done_env,
                        ego_obs=obs_0,        # agent 0의 obs를 ego로 사용
                        ego_action=action_0,
                        ego_log_prob=log_prob_0,
                        critic_value=val_0,
                        reward=ego_reward,
                        info=info,
                    )
                    step_state = (
                        env_state, obsv, done_env,
                        h0, h1, swap_flags, update_step, rng,
                    )
                    return step_state, transition

                final_state, traj = jax.lax.scan(
                    _env_step,
                    (env_state, last_obs, last_done,
                     hstate_0, hstate_1, swap_flags, update_step, rng),
                    None,
                    NUM_STEPS,
                )
                (env_state, last_obs, last_done,
                 hstate_0, hstate_1, swap_flags, _, rng) = final_state

                # ---- GAE + PPO UPDATE ----
                ego_obs_last = last_obs[env.agents[0]]
                init_h_bs = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
                _, _, last_val, _ = network.apply(
                    actor_state.params, init_h_bs,
                    (ego_obs_last[jnp.newaxis], last_done[jnp.newaxis]),
                )
                last_val = last_val.squeeze(0)

                def _get_adv(carry, xs):
                    gae, nv = carry
                    done, value, reward = xs
                    delta = reward + model_config["GAMMA"] * nv * (1 - done) - value
                    gae = (
                        delta
                        + model_config["GAMMA"] * model_config["GAE_LAMBDA"]
                        * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_adv,
                    (jnp.zeros_like(last_val), last_val),
                    (traj.done, traj.critic_value, traj.reward),
                    reverse=True,
                    unroll=16,
                )
                targets = advantages + traj.critic_value

                mb_size = NUM_ENVS // NUM_MINIBATCHES
                init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def _split_mb(x):
                    return jnp.swapaxes(
                        jnp.reshape(
                            x, [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:])
                        ),
                        1, 0,
                    )

                def _update_minibatch(state, mb):
                    (mb_ah, mb_ego_obs, mb_done,
                     mb_action, mb_log_prob, mb_val_old,
                     mb_adv, mb_targets) = mb
                    mb_ah = mb_ah.squeeze(0)

                    def _loss(params):
                        _, pi, value, _ = network.apply(
                            params, mb_ah, (mb_ego_obs, mb_done)
                        )
                        lp = pi.log_prob(mb_action)
                        ratio = jnp.exp(lp - mb_log_prob)
                        adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        loss1 = ratio * adv_n
                        loss2 = jnp.clip(
                            ratio,
                            1 - model_config["CLIP_EPS"],
                            1 + model_config["CLIP_EPS"],
                        ) * adv_n
                        actor_loss = -jnp.minimum(loss1, loss2).mean()
                        v_cl = mb_val_old + (value - mb_val_old).clip(
                            -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                        )
                        critic_loss = 0.5 * jnp.maximum(
                            jnp.square(value - mb_targets),
                            jnp.square(v_cl - mb_targets),
                        ).mean()
                        entropy = pi.entropy().mean()
                        total = (
                            actor_loss
                            + model_config.get("VF_COEF", 0.5) * critic_loss
                            - model_config["ENT_COEF"] * entropy
                        )
                        return total, (actor_loss, critic_loss, entropy)

                    (total_loss, aux), grads = jax.value_and_grad(
                        _loss, has_aux=True
                    )(state.params)
                    return state.apply_gradients(grads=grads), (
                        total_loss, aux[0], aux[1], aux[2]
                    )

                def _update_epoch(epoch_state, _):
                    state, rng_e = epoch_state
                    rng_e, _rng_e = jax.random.split(rng_e)
                    perm = jax.random.permutation(_rng_e, NUM_ENVS)

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
                    minibatches = jax.tree_util.tree_map(_split_mb, batch)

                    state, losses = jax.lax.scan(
                        _update_minibatch, state, minibatches,
                    )
                    return (state, rng_e), losses

                rng, _rng = jax.random.split(rng)
                (actor_state, _), all_losses = jax.lax.scan(
                    _update_epoch,
                    (actor_state, _rng),
                    None,
                    UPDATE_EPOCHS,
                )

                # ---- Checkpoint buffer ----
                update_step = update_step + 1
                selector = checkpoint_steps == update_step
                slot = jnp.argmax(selector)
                ck_buf = jax.lax.cond(
                    jnp.any(selector),
                    lambda b: jax.tree_util.tree_map(
                        lambda bl, p: bl.at[slot].set(p), b, actor_state.params,
                    ),
                    lambda b: b,
                    ck_buf,
                )

                # ---- Metrics ----
                metric = jax.tree_util.tree_map(lambda x: x.mean(), traj.info)
                metric["total_loss"] = all_losses[0].mean()
                metric["actor_loss"] = all_losses[1].mean()
                metric["critic_loss"] = all_losses[2].mean()
                metric["entropy"] = all_losses[3].mean()
                metric["update_step"] = update_step
                metric["env_step"] = update_step * NUM_STEPS * NUM_ENVS

                def _log(m):
                    flat = {}
                    for k, v in m.items():
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                flat[f"{k}/{kk}"] = float(vv)
                        else:
                            flat[k] = float(v)
                    wandb.log({f"hsp_s1/{k}": v for k, v in flat.items()})

                jax.debug.callback(_log, metric)

                runner_state = (
                    actor_state, env_state, last_obs, last_done,
                    hstate_0, hstate_1, swap_flags,
                    ck_buf, update_step, rng,
                )
                return runner_state, metric

            init_runner = (
                actor_state, env_state, last_obs, last_done,
                hstate_0, hstate_1, swap_flags,
                ck_buf, jnp.int32(0), rng,
            )
            final_runner, metrics = jax.lax.scan(
                _update_step, init_runner, None, NUM_UPDATES,
            )

            return final_runner[7]  # ck_buf: leaf shape (num_checkpoints, ...)

        # JIT 컴파일하여 실행
        return jax.jit(_train_inner)(rng)

    return train
