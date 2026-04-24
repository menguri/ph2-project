"""
GAMMA Stage 1: Population training with MAPPO centralized critic.

mep_s1.py 기반, 원본 GAMMA(zsc-basecamp) 재현을 위해 다음을 추가:
  1. MAPPO centralized critic (global_obs → 별도 MLP value head)
  2. ValueNorm (β=0.99999 EMA, GAE target 정규화)
  3. Per-policy advantage normalization (partner_idx 별 별도 정규화)
  4. Clipped value loss (mep_s1에 이미 구현됨)

다른 알고리즘(SP, E3T, FCP, MEP, HSP)에 영향 없음 — GAMMA 전용 파일.
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
from overcooked_v2_experiments.ppo.utils.valuenorm import (
    ValueNormState,
    valuenorm_update,
    valuenorm_normalize,
    valuenorm_denormalize,
)


class GammaTransition(NamedTuple):
    done: jnp.ndarray         # (NUM_STEPS, NUM_ENVS)
    ego_obs: jnp.ndarray      # (NUM_STEPS, NUM_ENVS, H, W, C)
    ego_action: jnp.ndarray   # (NUM_STEPS, NUM_ENVS)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    critic_value: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    reward: jnp.ndarray       # (NUM_STEPS, NUM_ENVS)
    info: dict
    avail_actions: jnp.ndarray  # (NUM_STEPS, NUM_ENVS, ACTION_DIM)
    global_obs: jnp.ndarray   # (NUM_STEPS, NUM_ENVS, GLOBAL_DIM) — centralized critic 용


def make_train_gamma_s1(config):
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

    # MAPPO / ValueNorm flags
    mappo_mode = bool(config.get("MAPPO_MODE", False))
    use_valuenorm = bool(model_config.get("USE_VALUENORM", False))

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
        sample_obs, sample_env_state = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        obs_shape = sample_obs[env.agents[0]].shape[1:]  # (H, W, C)

        # ----------------------------------------------------------------
        # Global obs shape (MAPPO centralized critic)
        # ----------------------------------------------------------------
        if mappo_mode:
            # per-actor full obs: 각 agent의 default full obs를 추출
            _full = jax.vmap(env._env.get_obs_default)(sample_env_state.env_state)
            # _full: (NUM_ENVS, num_agents, *obs_shape) or (NUM_ENVS, obs_dim)
            # agent 0의 full obs를 centralized critic 입력으로 사용
            _global_sample = _full[:, 0].astype(jnp.float32)  # (NUM_ENVS, ...)
            global_obs_shape = _global_sample.shape[1:]  # obs_shape or (obs_dim,)
            # flatten: centralized critic은 flat vector 입력
            global_obs_dim = 1
            for d in global_obs_shape:
                global_obs_dim *= d
            print(f"[GAMMA-S1] MAPPO global_obs_shape={global_obs_shape}, flat_dim={global_obs_dim}")
        else:
            global_obs_shape = None
            global_obs_dim = 0

        # ----------------------------------------------------------------
        # Network init
        # ----------------------------------------------------------------
        init_hstate = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x = (
            jnp.zeros((1, NUM_ENVS, *obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )

        # MAPPO: global_obs를 network.init에 전달하여 centralized critic 파라미터 초기화
        init_global_obs = (
            jnp.zeros((1, NUM_ENVS, global_obs_dim), dtype=jnp.float32)
            if mappo_mode else None
        )

        # ----------------------------------------------------------------
        # Create N train states
        # ----------------------------------------------------------------
        def _create_member(rng_m):
            params = network.init(rng_m, init_hstate, init_x, global_obs=init_global_obs)
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

        num_env_agents = env.num_agents
        num_partners = num_env_agents - 1

        _init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        pop_actor_hstates = jnp.stack([_init_h] * N)
        if num_partners == 1:
            pop_partner_hstates = jnp.stack([_init_h] * N)
        else:
            pop_partner_hstates = jnp.stack([
                jnp.stack([_init_h] * num_partners) for _ in range(N)
            ])

        # Checkpoint buffer
        checkpoint_steps = _ckpt_steps_arr
        ck_buf = jax.tree_util.tree_map(
            lambda p: jnp.zeros((N, max(num_checkpoints, 1)) + p.shape[1:], p.dtype),
            pop_states.params,
        )

        # ValueNorm state (shared across all population members)
        vn_state = ValueNormState.create()

        # ----------------------------------------------------------------
        # Global obs extraction helper
        # ----------------------------------------------------------------
        def _extract_global_obs(env_state):
            """env_state → (NUM_ENVS, global_obs_dim) flattened global obs."""
            _full = jax.vmap(env._env.get_obs_default)(env_state.env_state)
            _g = _full[:, 0].astype(jnp.float32)  # agent 0의 full obs
            return _g.reshape(NUM_ENVS, -1)  # flatten

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
                vn_state,
                update_step,
                rng,
            ) = runner_state

            # -- Partner assignment --
            # Self-play warmup: 초기 SP_WARMUP_FRAC 비율 동안은 offset=0 (self-play).
            # 그 이후에는 cross-play (offset∈[1,N)).
            # 원본 GAMMA S1 은 순수 self-play (모든 update round-robin single member) 였으므로
            # warmup 으로 기본 coord 먼저 학습한 뒤 cross-play 로 다양성 추가.
            sp_warmup_frac = float(config.get("GAMMA_S1_SP_WARMUP_FRAC", 0.0))
            warmup_updates = int(sp_warmup_frac * NUM_UPDATES)
            is_sp = (update_step < warmup_updates) if warmup_updates > 0 else False

            rng, _rng = jax.random.split(rng)
            if num_partners == 1:
                # SP: offset=0 → partner_idx[m] = m (self-play)
                # XP: offset∈[1,N) → partner 는 다른 member
                offset = jax.random.randint(_rng, (), 1, max(N, 2))
                offset = jnp.where(is_sp, jnp.int32(0), offset)
                partner_idxs = (jnp.arange(N) + offset) % N
            else:
                offsets = jax.random.choice(
                    _rng, jnp.arange(1, max(N, num_partners + 1)),
                    shape=(num_partners,), replace=False
                )
                partner_idxs = jnp.stack([
                    (jnp.arange(N) + offsets[p]) % N for p in range(num_partners)
                ])
                # SP warmup 은 num_partners==1 (2-agent) 케이스만 지원
                if num_partners > 1 and sp_warmup_frac > 0:
                    print("[WARN] GAMMA_S1_SP_WARMUP_FRAC is only supported for 2-agent envs; ignoring.")

            all_actor_params = pop_states.params

            # ---- ROLLOUT ------------------------------------------------
            def _collect_member(
                ego_state, partner_idx,
                env_state, last_obs, last_done,
                actor_hstate, partner_hstate, rng,
            ):
                def _env_step(step_state, _):
                    (
                        env_state, last_obs, last_done,
                        actor_hstate, partner_hstate,
                        update_step, rng,
                    ) = step_state

                    ego_obs_t = last_obs[env.agents[0]]
                    done_t = last_done

                    # GridSpread avail actions
                    if is_spread:
                        _avail_dict = jax.vmap(env._env.get_avail_actions)(env_state.env_state)
                        _avail_ego = _avail_dict[env.agents[0]]
                    else:
                        _avail_ego = jnp.zeros((NUM_ENVS, ACTION_DIM), dtype=jnp.int32)

                    # Global obs for centralized critic
                    if mappo_mode:
                        _gobs = _extract_global_obs(env_state)  # (E, global_obs_dim)
                        _gobs_in = _gobs[jnp.newaxis]  # (1, E, global_obs_dim)
                    else:
                        _gobs = jnp.zeros((NUM_ENVS, 1), dtype=jnp.float32)
                        _gobs_in = None

                    # Ego forward pass (actor + centralized critic)
                    rng, _rng = jax.random.split(rng)
                    actor_hstate, ego_pi, crit_val, _ = network.apply(
                        ego_state.params,
                        actor_hstate,
                        (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                        avail_actions=(_avail_ego[jnp.newaxis] if is_spread else None),
                        global_obs=_gobs_in,
                    )
                    ego_action = ego_pi.sample(seed=_rng).squeeze(0)
                    ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)
                    crit_val = crit_val.squeeze(0)

                    # Partner actors (frozen, actor only — no global_obs needed)
                    env_act = {env.agents[0]: ego_action}

                    if num_partners == 1:
                        partner_obs_t = last_obs[env.agents[1]]
                        partner_params = jax.tree_util.tree_map(
                            lambda x: x[partner_idx], all_actor_params
                        )
                        rng, _rng2 = jax.random.split(rng)
                        partner_hstate, partner_pi, _, _ = network.apply(
                            partner_params,
                            partner_hstate,
                            (partner_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                            actor_only=True,
                        )
                        partner_action = partner_pi.sample(seed=_rng2).squeeze(0)
                        env_act[env.agents[1]] = partner_action
                    else:
                        new_partner_hstates = []
                        for p in range(num_partners):
                            p_obs = last_obs[env.agents[p + 1]]
                            p_idx = partner_idx[p]
                            p_params = jax.tree_util.tree_map(
                                lambda x: x[p_idx], all_actor_params
                            )
                            rng, _rng_p = jax.random.split(rng)
                            p_h = partner_hstate[p]
                            new_p_h, p_pi, _, _ = network.apply(
                                p_params, p_h,
                                (p_obs[jnp.newaxis], done_t[jnp.newaxis]),
                                actor_only=True,
                            )
                            p_action = p_pi.sample(seed=_rng_p).squeeze(0)
                            env_act[env.agents[p + 1]] = p_action
                            new_partner_hstates.append(new_p_h)
                        partner_hstate = jnp.stack(new_partner_hstates)

                    rng, _rng3 = jax.random.split(rng)
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        jax.random.split(_rng3, NUM_ENVS), env_state, env_act
                    )
                    done_env = done["__all__"]

                    anneal_factor = rew_shaping_anneal(
                        update_step * NUM_STEPS * NUM_ENVS
                    )
                    if "shaped_reward" in info:
                        reward = jax.tree_util.tree_map(
                            lambda r, s: r + s * anneal_factor,
                            reward, info["shaped_reward"],
                        )
                    ego_reward = reward[env.agents[0]]
                    info.pop("shaped_reward", None)
                    info.pop("shaped_reward_events", None)
                    info = jax.tree_util.tree_map(
                        lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info
                    )

                    transition = GammaTransition(
                        done=done_env,
                        ego_obs=ego_obs_t,
                        ego_action=ego_action,
                        ego_log_prob=ego_log_prob,
                        critic_value=crit_val,
                        reward=ego_reward,
                        info=info,
                        avail_actions=_avail_ego,
                        global_obs=_gobs,
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
            vmap_partner_idxs = partner_idxs if num_partners == 1 else partner_idxs.T
            all_traj, finals = jax.vmap(
                _collect_member, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
            )(
                pop_states, vmap_partner_idxs,
                pop_env_states, pop_last_obs, pop_last_done,
                pop_actor_hstates, pop_partner_hstates,
                jax.random.split(_rng, N),
            )
            (pop_env_states, pop_last_obs, pop_last_done,
             pop_actor_hstates, pop_partner_hstates) = finals

            # ---- ENTROPY BONUS ------------------------------------------
            def compute_bonus(args):
                ego_obs_i, ego_done_i, ego_act_i = args
                init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def fwd_k(ak_params):
                    _, pi, _, _ = network.apply(
                        ak_params, init_h, (ego_obs_i, ego_done_i),
                        actor_only=True,
                    )
                    return pi.probs

                all_probs = jax.lax.map(fwd_k, all_actor_params)
                mean_prob = jnp.mean(all_probs, axis=0)
                taken = jnp.take_along_axis(
                    mean_prob, ego_act_i[..., jnp.newaxis], axis=-1
                ).squeeze(-1)
                return -jnp.log(taken + 1e-6)

            all_bonuses = jax.lax.map(
                compute_bonus,
                (all_traj.ego_obs, all_traj.done, all_traj.ego_action),
            )

            shaped_rewards = all_traj.reward + entropy_alpha * all_bonuses

            # ---- GAE + PPO UPDATE per member ----------------------------
            def _update_member(
                ego_state,
                traj_i,
                shaped_reward_i,
                last_obs_i,
                last_done_i,
                partner_idx_i,   # 이 member의 partner index (per-policy adv norm용)
                vn_state,
                rng_m,
            ):
                # Last value for GAE bootstrap
                ego_obs_last = last_obs_i[env.agents[0]]
                init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                if mappo_mode:
                    # 마지막 step의 global_obs: traj의 마지막 global_obs 사용 (근사)
                    _last_gobs = traj_i.global_obs[-1][jnp.newaxis]  # (1, E, global_dim)
                else:
                    _last_gobs = None

                _, _, last_val, _ = network.apply(
                    ego_state.params,
                    init_h,
                    (ego_obs_last[jnp.newaxis], last_done_i[jnp.newaxis]),
                    global_obs=_last_gobs,
                )
                last_val = last_val.squeeze(0)

                # ValueNorm: denormalize value predictions for GAE
                if use_valuenorm:
                    critic_values_denorm = valuenorm_denormalize(vn_state, traj_i.critic_value)
                    last_val_denorm = valuenorm_denormalize(vn_state, last_val)
                else:
                    critic_values_denorm = traj_i.critic_value
                    last_val_denorm = last_val

                # GAE
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
                    (jnp.zeros_like(last_val_denorm), last_val_denorm),
                    (traj_i.done, critic_values_denorm, shaped_reward_i),
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + critic_values_denorm  # (T, E)

                # ValueNorm: update running stats with returns, normalize targets
                if use_valuenorm:
                    vn_state_new = valuenorm_update(vn_state, returns)
                    targets = valuenorm_normalize(vn_state_new, returns)
                else:
                    vn_state_new = vn_state
                    targets = returns

                # Advantage normalization (global, not per-policy for S1 vmap simplicity)
                # 원본 GAMMA per-policy norm은 5 policy type 기반이나, S1에서는
                # 각 member가 1개 partner만 상대하므로 global norm과 동일 효과
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)

                # PPO minibatch update
                mb_size = NUM_ENVS // NUM_MINIBATCHES
                init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def _split_minibatches(x):
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
                     mb_adv, mb_targets, mb_avail, mb_gobs) = mb

                    mb_ah = mb_ah.squeeze(0)

                    def _loss(params):
                        _gobs_loss = mb_gobs if mappo_mode else None
                        _, pi, value, _ = network.apply(
                            params, mb_ah, (mb_ego_obs, mb_done),
                            avail_actions=(mb_avail if is_spread else None),
                            global_obs=_gobs_loss,
                        )
                        log_prob_new = pi.log_prob(mb_action)
                        ratio = jnp.exp(log_prob_new - mb_log_prob)
                        # Advantage는 이미 정규화됨
                        loss1 = ratio * mb_adv
                        loss2 = (
                            jnp.clip(ratio,
                                     1 - model_config["CLIP_EPS"],
                                     1 + model_config["CLIP_EPS"])
                            * mb_adv
                        )
                        actor_loss = -jnp.minimum(loss1, loss2).mean()

                        # Clipped value loss (원본 GAMMA 방식)
                        v_cl = mb_val_old + (value - mb_val_old).clip(
                            -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                        )
                        # 원본 GAMMA: use_huber_loss=True, huber_delta=10.0
                        _huber_delta = float(model_config.get("HUBER_DELTA", 10.0))
                        def _huber(err):
                            return jnp.where(
                                jnp.abs(err) <= _huber_delta,
                                0.5 * err ** 2,
                                _huber_delta * (jnp.abs(err) - 0.5 * _huber_delta),
                            )
                        critic_loss = jnp.maximum(
                            _huber(value - mb_targets), _huber(v_cl - mb_targets)
                        ).mean()

                        entropy = pi.entropy().mean()
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
                        jnp.take(init_h_mb, perm, axis=0)[jnp.newaxis],
                        jnp.take(traj_i.ego_obs, perm, axis=1),
                        jnp.take(traj_i.done, perm, axis=1),
                        jnp.take(traj_i.ego_action, perm, axis=1),
                        jnp.take(traj_i.ego_log_prob, perm, axis=1),
                        jnp.take(traj_i.critic_value, perm, axis=1),
                        jnp.take(advantages_norm, perm, axis=1),
                        jnp.take(targets, perm, axis=1),
                        jnp.take(traj_i.avail_actions, perm, axis=1),
                        jnp.take(traj_i.global_obs, perm, axis=1),
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
                total_loss = all_losses[0].mean()
                actor_loss = all_losses[1].mean()
                critic_loss = all_losses[2].mean()
                entropy = all_losses[3].mean()
                return state_out, total_loss, actor_loss, critic_loss, entropy, vn_state_new

            rng, _rng = jax.random.split(rng)
            vmap_partner_idxs_for_update = partner_idxs if num_partners == 1 else partner_idxs.T
            (
                pop_states,
                total_losses,
                actor_losses,
                critic_losses,
                entropies,
                vn_states_out,
            ) = jax.vmap(
                _update_member, in_axes=(0, 0, 0, 0, 0, 0, None, 0)
            )(
                pop_states,
                all_traj,
                shaped_rewards,
                pop_last_obs,
                pop_last_done,
                vmap_partner_idxs_for_update,
                vn_state,
                jax.random.split(_rng, N),
            )

            # ValueNorm: 모든 member의 update 결과 중 첫 번째 사용 (shared normalizer)
            # 원본도 shared ValueNorm(1)이므로, population 전체 returns의 대표값으로 근사
            vn_state = jax.tree_util.tree_map(lambda x: x[0], vn_states_out)

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
                wandb.log({f"gamma_s1/{k}": v for k, v in flat.items()})

            jax.debug.callback(_log, metric)

            runner_state = (
                pop_states,
                pop_env_states,
                pop_last_obs,
                pop_last_done,
                pop_actor_hstates,
                pop_partner_hstates,
                ck_buf,
                vn_state,
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
            vn_state,
            jnp.int32(0),
            rng,
        )
        final_runner, metrics = jax.lax.scan(
            _update_step, init_runner, None, NUM_UPDATES
        )

        return {
            "runner_state": final_runner,
            "metrics": metrics,
            "pop_actor_params": final_runner[0].params,
            "pop_actor_ckpts": final_runner[6],
        }

    return train
