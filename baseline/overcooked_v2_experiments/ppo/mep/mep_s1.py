"""
MEP Stage 1: Population training with entropy bonus.

Trains N IPPO agents jointly. Each agent i's reward is augmented:
    R_shaped = R + alpha * (-log(π_pop(a|s)))
    π_pop(a|s) = (1/N) Σ_k π_k(a|s)    [full N-member average]

Paper: "Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination"
       AAAI 2023
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


class MEPTransition(NamedTuple):
    done: jnp.ndarray         # (NUM_STEPS, NUM_ENVS)
    ego_obs: jnp.ndarray      # (NUM_STEPS, NUM_ENVS, H, W, C)
    ego_action: jnp.ndarray   # (NUM_STEPS, NUM_ENVS)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    critic_value: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    reward: jnp.ndarray       # (NUM_STEPS, NUM_ENVS)
    info: dict
    avail_actions: jnp.ndarray  # (NUM_STEPS, NUM_ENVS, ACTION_DIM) — GridSpread masking용, 타 env는 placeholder


def make_train_mep_s1(config):
    env_config = config["env"]
    model_config = config["model"]

    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    is_spread = (env_name == "GridSpread")
    env_raw = jaxmarl.make(env_name, **env_kwargs)
    ACTION_DIM = env_raw.action_space(env_raw.agents[0]).n

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env_raw, replace_info=False)
    else:
        env = LogWrapper(env_raw, replace_info=False)

    N = config["MEP_POPULATION_SIZE"]
    entropy_alpha = config.get("MEP_ENTROPY_ALPHA", 0.01)

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

    from overcooked_v2_experiments.ppo.utils.store import build_checkpoint_steps
    _ckpt_steps_arr, num_checkpoints = build_checkpoint_steps(config, NUM_UPDATES)

    network = ActorCriticRNN(action_dim=ACTION_DIM, config=model_config)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    def _make_lr_fn(base_lr_key):
        base_lr = model_config.get(base_lr_key, model_config["LR"])
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

    lr_fn = _make_lr_fn("LR")

    def train(rng):
        # ----------------------------------------------------------------
        # ENV shape inference
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        sample_obs, _ = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        obs_shape = sample_obs[env.agents[0]].shape[1:]  # (H, W, C)

        # ----------------------------------------------------------------
        # Network init
        # ----------------------------------------------------------------
        init_hstate = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x = (
            jnp.zeros((1, NUM_ENVS, *obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )

        # ----------------------------------------------------------------
        # Create N train states — stacked pytree with leaf shape (N, ...)
        # ----------------------------------------------------------------
        def _create_member(rng_m):
            params = network.init(rng_m, init_hstate, init_x)
            return TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=_make_tx(lr_fn),
            )

        rng, _rng = jax.random.split(rng)
        pop_states = jax.vmap(_create_member)(jax.random.split(_rng, N))

        # ----------------------------------------------------------------
        # Init N×NUM_ENVS environments
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        all_reset_rngs = jax.random.split(_rng, N * NUM_ENVS).reshape(N, NUM_ENVS, -1)
        pop_obs, pop_env_states = jax.vmap(
            lambda r: jax.vmap(env.reset)(r)
        )(all_reset_rngs)
        pop_done = jnp.zeros((N, NUM_ENVS), dtype=jnp.bool_)

        num_env_agents = env.num_agents  # 환경 에이전트 수 (2 or 3)
        num_partners = num_env_agents - 1  # 파트너 수

        # hstates: (N, NUM_ENVS, GRU_HIDDEN_DIM)
        _init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        pop_actor_hstates = jnp.stack([_init_h] * N)
        # partner hstates: (N, num_partners, NUM_ENVS, GRU_HIDDEN_DIM) for 3+ agents
        # 2-agent: backward compat → (N, NUM_ENVS, GRU_HIDDEN_DIM)
        if num_partners == 1:
            pop_partner_hstates = jnp.stack([_init_h] * N)
        else:
            pop_partner_hstates = jnp.stack([
                jnp.stack([_init_h] * num_partners) for _ in range(N)
            ])

        # Checkpoint buffer: leaf shape (N, num_checkpoints, ...member_param_shape...)
        checkpoint_steps = _ckpt_steps_arr
        ck_buf = jax.tree_util.tree_map(
            lambda p: jnp.zeros((N, max(num_checkpoints, 1)) + p.shape[1:], p.dtype),
            pop_states.params,
        )

        # ----------------------------------------------------------------
        # TRAIN LOOP
        # ----------------------------------------------------------------
        def _update_step(runner_state, unused):
            (
                pop_states,
                pop_env_states,
                pop_last_obs,
                pop_last_done,
                pop_actor_hstates,
                pop_partner_hstates,
                ck_buf,
                update_step,
                rng,
            ) = runner_state

            # -- Partner assignment --
            # Self-play warmup: 초기 SP_WARMUP_FRAC 비율 동안은 offset=0 (self-play).
            # 원본 MEP 는 pure self-play (round-robin per-episode) + MEP bonus 였으므로
            # MEP_S1_SP_WARMUP_FRAC=1.0 이면 원본 MEP 와 구조 동일 (항상 self-play).
            sp_warmup_frac = float(config.get("MEP_S1_SP_WARMUP_FRAC", 0.0))
            warmup_updates = int(sp_warmup_frac * NUM_UPDATES)
            is_sp = (update_step < warmup_updates) if warmup_updates > 0 else False

            rng, _rng = jax.random.split(rng)
            if num_partners == 1:
                offset = jax.random.randint(_rng, (), 1, max(N, 2))
                offset = jnp.where(is_sp, jnp.int32(0), offset)   # SP 구간엔 offset=0
                partner_idxs = (jnp.arange(N) + offset) % N  # (N,)
            else:
                # num_partners개의 partner를 각각 다른 offset으로 배정
                offsets = jax.random.choice(
                    _rng, jnp.arange(1, max(N, num_partners + 1)),
                    shape=(num_partners,), replace=False
                )
                partner_idxs = jnp.stack([
                    (jnp.arange(N) + offsets[p]) % N for p in range(num_partners)
                ])  # (num_partners, N)
                # num_partners>1 (3+ agent) 에선 SP warmup 미지원

            # All actor params (N stacked) — used as closure in inner fns
            all_actor_params = pop_states.params

            # ---- ROLLOUT ------------------------------------------------
            def _collect_member(
                ego_state,
                partner_idx,
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                partner_hstate,
                rng,
            ):
                def _env_step(step_state, _):
                    (
                        env_state, last_obs, last_done,
                        actor_hstate, partner_hstate,
                        update_step, rng,
                    ) = step_state

                    ego_obs_t = last_obs[env.agents[0]]     # (E, H, W, C)
                    done_t = last_done                       # (E,)

                    # === GridSpread 전용 ego action masking ===
                    if is_spread:
                        _avail_dict = jax.vmap(env._env.get_avail_actions)(env_state.env_state)
                        _avail_ego = _avail_dict[env.agents[0]]  # (E, ACTION_DIM)
                    else:
                        _avail_ego = jnp.zeros((NUM_ENVS, ACTION_DIM), dtype=jnp.int32)

                    # Ego: actor + critic in one forward pass
                    rng, _rng = jax.random.split(rng)
                    if is_spread:
                        actor_hstate, ego_pi, crit_val, _ = network.apply(
                            ego_state.params,
                            actor_hstate,
                            (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                            avail_actions=_avail_ego[jnp.newaxis],
                        )
                    else:
                        actor_hstate, ego_pi, crit_val, _ = network.apply(
                            ego_state.params,
                            actor_hstate,
                            (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                        )
                    ego_action = ego_pi.sample(seed=_rng).squeeze(0)   # (E,)
                    ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)
                    crit_val = crit_val.squeeze(0)                      # (E,)

                    # Partner actors (frozen) — N-agent 일반화
                    env_act = {env.agents[0]: ego_action}

                    if num_partners == 1:
                        # 2-agent: 기존 경로 유지
                        partner_obs_t = last_obs[env.agents[1]]
                        partner_params = jax.tree_util.tree_map(
                            lambda x: x[partner_idx], all_actor_params
                        )
                        rng, _rng2 = jax.random.split(rng)
                        partner_hstate, partner_pi, _, _ = network.apply(
                            partner_params,
                            partner_hstate,
                            (partner_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                        )
                        partner_action = partner_pi.sample(seed=_rng2).squeeze(0)
                        env_act[env.agents[1]] = partner_action
                    else:
                        # 3+ agent: 각 partner slot마다 독립적으로 population에서 행동 생성
                        new_partner_hstates = []
                        for p in range(num_partners):
                            p_obs = last_obs[env.agents[p + 1]]  # partner p의 관측
                            p_idx = partner_idx[p]  # population에서 배정된 인덱스
                            p_params = jax.tree_util.tree_map(
                                lambda x: x[p_idx], all_actor_params
                            )
                            rng, _rng_p = jax.random.split(rng)
                            p_h = partner_hstate[p]  # (E, GRU_HIDDEN_DIM)
                            new_p_h, p_pi, _, _ = network.apply(
                                p_params, p_h,
                                (p_obs[jnp.newaxis], done_t[jnp.newaxis]),
                            )
                            p_action = p_pi.sample(seed=_rng_p).squeeze(0)
                            env_act[env.agents[p + 1]] = p_action
                            new_partner_hstates.append(new_p_h)
                        partner_hstate = jnp.stack(new_partner_hstates)  # (num_partners, E, GRU_HIDDEN_DIM)
                    rng, _rng3 = jax.random.split(rng)
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        jax.random.split(_rng3, NUM_ENVS), env_state, env_act
                    )
                    done_env = done["__all__"]  # (E,)

                    # Reward shaping anneal
                    anneal_factor = rew_shaping_anneal(
                        update_step * NUM_STEPS * NUM_ENVS
                    )
                    if "shaped_reward" in info:
                        reward = jax.tree_util.tree_map(
                            lambda r, s: r + s * anneal_factor,
                            reward, info["shaped_reward"],
                        )
                    ego_reward = reward[env.agents[0]]  # (E,)
                    info.pop("shaped_reward", None)
                    info.pop("shaped_reward_events", None)
                    info = jax.tree_util.tree_map(
                        lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info
                    )

                    transition = MEPTransition(
                        done=done_env,
                        ego_obs=ego_obs_t,
                        ego_action=ego_action,
                        ego_log_prob=ego_log_prob,
                        critic_value=crit_val,
                        reward=ego_reward,
                        info=info,
                        avail_actions=_avail_ego,
                    )
                    step_state = (
                        env_state, obsv, done_env,
                        actor_hstate, partner_hstate,
                        update_step, rng,
                    )
                    return step_state, transition

                final_state, traj = jax.lax.scan(
                    _env_step,
                    (env_state, last_obs, last_done,
                     actor_hstate, partner_hstate,
                     update_step, rng),
                    None,
                    NUM_STEPS,
                )
                (fe, fo, fd, fah, fph, _, _) = final_state
                return traj, (fe, fo, fd, fah, fph)

            rng, _rng = jax.random.split(rng)
            # partner_idxs를 vmap 축에 맞게 정리:
            #   2-agent: (N,) → 그대로
            #   3+ agent: (num_partners, N) → (N, num_partners)
            vmap_partner_idxs = partner_idxs if num_partners == 1 else partner_idxs.T
            all_traj, finals = jax.vmap(
                _collect_member, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
            )(
                pop_states, vmap_partner_idxs,
                pop_env_states, pop_last_obs, pop_last_done,
                pop_actor_hstates, pop_partner_hstates,
                jax.random.split(_rng, N),
            )
            # all_traj leaf shapes: (N, NUM_STEPS, NUM_ENVS, ...)
            (pop_env_states, pop_last_obs, pop_last_done,
             pop_actor_hstates, pop_partner_hstates) = finals

            # ---- ENTROPY BONUS ------------------------------------------
            # jax.lax.map (not vmap) over N members → sequential forward
            # passes, peak memory = T*E per forward pass. Avoids OOM on
            # large layouts where vmap(vmap(...)) batches N²×T×E together.

            def compute_bonus(args):
                """Return -log π_pop(a|s) for member i. Shape: (T, E)."""
                ego_obs_i, ego_done_i, ego_act_i = args
                init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def fwd_k(ak_params):
                    _, pi, _, _ = network.apply(
                        ak_params, init_h, (ego_obs_i, ego_done_i)
                    )
                    return pi.probs  # (T, E, A)

                # Sequential over N actors → peak memory O(T*E) per forward pass
                all_probs = jax.lax.map(fwd_k, all_actor_params)  # (N, T, E, A)
                mean_prob = jnp.mean(all_probs, axis=0)             # (T, E, A)
                taken = jnp.take_along_axis(
                    mean_prob, ego_act_i[..., jnp.newaxis], axis=-1
                ).squeeze(-1)                                       # (T, E)
                return -jnp.log(taken + 1e-6)                      # (T, E)

            # Sequential over N members → peak memory O(T*E) per member
            all_bonuses = jax.lax.map(
                compute_bonus,
                (all_traj.ego_obs, all_traj.done, all_traj.ego_action),
            )  # (N, T, E)

            shaped_rewards = all_traj.reward + entropy_alpha * all_bonuses

            # ---- GAE + PPO UPDATE per member ----------------------------
            def _update_member(
                ego_state,
                traj_i,            # MEPTransition with shapes (T, E, ...)
                shaped_reward_i,   # (T, E)
                last_obs_i,        # {"agent_0": (E, H, W, C), ...}
                last_done_i,       # (E,)
                rng_m,
            ):
                # ---- Last value for GAE bootstrap (fresh hstate) ----
                ego_obs_last = last_obs_i[env.agents[0]]
                init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
                _, _, last_val, _ = network.apply(
                    ego_state.params,
                    init_h,
                    (ego_obs_last[jnp.newaxis], last_done_i[jnp.newaxis]),
                )
                last_val = last_val.squeeze(0)  # (E,)

                # ---- GAE ----
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
                    (traj_i.done, traj_i.critic_value, shaped_reward_i),
                    reverse=True,
                    unroll=16,
                )
                targets = advantages + traj_i.critic_value  # (T, E)

                # ---- PPO minibatch update ----
                mb_size = NUM_ENVS // NUM_MINIBATCHES
                init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def _split_minibatches(x):
                    """(T, E, ...) or (1, E, ...) → (NMB, T or 1, MB_SIZE, ...)"""
                    return jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:]),
                        ),
                        1, 0,
                    )

                def _update_minibatch(state, mb):
                    (mb_ah,
                     mb_ego_obs, mb_done,
                     mb_action, mb_log_prob, mb_val_old,
                     mb_adv, mb_targets, mb_avail) = mb

                    mb_ah = mb_ah.squeeze(0)  # (MB_SIZE, hidden)

                    def _loss(params):
                        # GridSpread: rollout과 동일한 masked pi 재생성
                        if is_spread:
                            _, pi, value, _ = network.apply(
                                params, mb_ah, (mb_ego_obs, mb_done),
                                avail_actions=mb_avail,
                            )
                        else:
                            _, pi, value, _ = network.apply(
                                params, mb_ah, (mb_ego_obs, mb_done)
                            )
                        log_prob_new = pi.log_prob(mb_action)
                        ratio = jnp.exp(log_prob_new - mb_log_prob)
                        adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        loss1 = ratio * adv_norm
                        loss2 = (
                            jnp.clip(ratio,
                                     1 - model_config["CLIP_EPS"],
                                     1 + model_config["CLIP_EPS"])
                            * adv_norm
                        )
                        actor_loss = -jnp.minimum(loss1, loss2).mean()
                        v_cl = mb_val_old + (value - mb_val_old).clip(
                            -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                        )
                        critic_loss = 0.5 * jnp.maximum(
                            jnp.square(value - mb_targets),
                            jnp.square(v_cl - mb_targets),
                        ).mean()
                        entropy = pi.entropy().mean()
                        # ENT_COEF 선형 스케줄링 (optional): ANNEAL_STEPS>0이면 START→END 선형.
                        _ent_anneal_steps = float(model_config.get("ENT_COEF_ANNEAL_STEPS", 0) or 0)
                        if _ent_anneal_steps > 0:
                            _ent_start = float(model_config.get("ENT_COEF_START", model_config["ENT_COEF"]))
                            _ent_end = float(model_config.get("ENT_COEF_END", model_config["ENT_COEF"]))
                            _env_steps_f = (
                                update_step * NUM_STEPS * NUM_ENVS
                            ).astype(jnp.float32)
                            _frac = jnp.clip(_env_steps_f / _ent_anneal_steps, 0.0, 1.0)
                            ent_coef_used = _ent_start + (_ent_end - _ent_start) * _frac
                        else:
                            ent_coef_used = model_config["ENT_COEF"]
                        total = (
                            actor_loss
                            + model_config.get("VF_COEF", 0.5) * critic_loss
                            - ent_coef_used * entropy
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
                        jnp.take(init_h_mb, perm, axis=0)[jnp.newaxis],  # (1, E, hid)
                        jnp.take(traj_i.ego_obs, perm, axis=1),           # (T, E, H, W, C)
                        jnp.take(traj_i.done, perm, axis=1),              # (T, E)
                        jnp.take(traj_i.ego_action, perm, axis=1),       # (T, E)
                        jnp.take(traj_i.ego_log_prob, perm, axis=1),     # (T, E)
                        jnp.take(traj_i.critic_value, perm, axis=1),     # (T, E)
                        jnp.take(advantages, perm, axis=1),               # (T, E)
                        jnp.take(targets, perm, axis=1),                  # (T, E)
                        jnp.take(traj_i.avail_actions, perm, axis=1),     # (T, E, ACTION_DIM)
                    )
                    minibatches = jax.tree_util.tree_map(_split_minibatches, batch)

                    state, losses = jax.lax.scan(
                        _update_minibatch, state, minibatches
                    )
                    return (state, rng_e), losses

                (state_out, _), all_losses = jax.lax.scan(
                    _update_epoch,
                    (ego_state, rng_m),
                    None,
                    UPDATE_EPOCHS,
                )
                # all_losses: (UPDATE_EPOCHS, NUM_MINIBATCHES, ...)
                total_loss = all_losses[0].mean()
                actor_loss = all_losses[1].mean()
                critic_loss = all_losses[2].mean()
                entropy = all_losses[3].mean()
                return state_out, total_loss, actor_loss, critic_loss, entropy

            rng, _rng = jax.random.split(rng)
            (
                pop_states,
                total_losses,
                actor_losses,
                critic_losses,
                entropies,
            ) = jax.vmap(_update_member)(
                pop_states,
                all_traj,
                shaped_rewards,
                pop_last_obs,
                pop_last_done,
                jax.random.split(_rng, N),
            )

            # ---- Checkpoint buffer update --------------------------------
            def _save_ck(buf, params, step):
                selector = checkpoint_steps == step
                slot = jnp.argmax(selector)
                return jax.lax.cond(
                    jnp.any(selector),
                    lambda b: jax.tree_util.tree_map(
                        lambda b_leaf, p_leaf: b_leaf.at[:, slot].set(p_leaf),
                        b, params,
                    ),
                    lambda b: b,
                    buf,
                )

            update_step = update_step + 1
            ck_buf = _save_ck(ck_buf, pop_states.params, update_step)

            # ---- WandB metrics -------------------------------------------
            metric = jax.tree_util.tree_map(lambda x: x.mean(), all_traj.info)
            # GridSpread success_rate 파생
            if is_spread and "success_at_done" in metric and "ep_done_flag" in metric:
                metric["success_rate"] = metric["success_at_done"] / (metric["ep_done_flag"] + 1e-8)
            metric["total_loss"] = total_losses.mean()
            metric["critic_loss"] = critic_losses.mean()
            metric["actor_loss"] = actor_losses.mean()
            metric["entropy"] = entropies.mean()
            metric["entropy_bonus"] = all_bonuses.mean()
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
                wandb.log({f"mep_s1/{k}": v for k, v in flat.items()})

            jax.debug.callback(_log, metric)

            runner_state = (
                pop_states,
                pop_env_states,
                pop_last_obs,
                pop_last_done,
                pop_actor_hstates,
                pop_partner_hstates,
                ck_buf,
                update_step,
                rng,
            )
            return runner_state, metric

        # ---- Initial checkpoint at step 0 --------------------------------
        ck_buf = jax.tree_util.tree_map(
            lambda b, p: b.at[:, 0].set(p),
            ck_buf, pop_states.params,
        )

        init_runner = (
            pop_states,
            pop_env_states,
            pop_obs,
            pop_done,
            pop_actor_hstates,
            pop_partner_hstates,
            ck_buf,
            jnp.int32(0),
            rng,
        )
        final_runner, metrics = jax.lax.scan(
            _update_step, init_runner, None, NUM_UPDATES
        )

        return {
            "runner_state": final_runner,
            "metrics": metrics,
            # Final actor params for each population member, shape (N, ...)
            "pop_actor_params": final_runner[0].params,
            # Checkpoint buffer: (N, num_checkpoints, ...)
            "pop_actor_ckpts": final_runner[6],
        }

    return train
