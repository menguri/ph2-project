"""
MEP Stage 2: Adaptive agent training against MEP S1 population.

Trains a single MAPPO agent against the N-member population produced by Stage 1.
Partner selection uses return-based prioritized sampling:
    weights = softmax(-running_returns / temperature)
    lower return (harder partner) → higher sampling probability

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

from overcooked_v2_experiments.ppo.mep.models import MAPPOActorRNN, MAPPOCentralCriticRNN


class MEPTransition(NamedTuple):
    done: jnp.ndarray         # (NUM_STEPS, NUM_ENVS)
    ego_obs: jnp.ndarray      # (NUM_STEPS, NUM_ENVS, H, W, C)
    ego_action: jnp.ndarray   # (NUM_STEPS, NUM_ENVS)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    critic_value: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    reward: jnp.ndarray       # (NUM_STEPS, NUM_ENVS)
    global_obs: jnp.ndarray   # (NUM_STEPS, NUM_ENVS, H, W, 2C)
    info: dict


def make_train_mep_s2(config):
    """
    Returns train(rng, population=pop_actor_params) where
    pop_actor_params is a stacked pytree with leaf shape (N, ...) —
    the N actor params produced by MEP Stage 1.
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

    NUM_ENVS = model_config["NUM_ENVS"]
    NUM_STEPS = model_config["NUM_STEPS"]
    NUM_UPDATES = int(
        model_config["TOTAL_TIMESTEPS"] // NUM_STEPS // NUM_ENVS
    )
    NUM_MINIBATCHES = model_config["NUM_MINIBATCHES"]
    UPDATE_EPOCHS = model_config["UPDATE_EPOCHS"]
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]

    model_config["NUM_UPDATES"] = NUM_UPDATES

    prioritized_alpha = config.get("MEP_PRIORITIZED_ALPHA", 3.0)
    use_prioritized = config.get("MEP_USE_PRIORITIZED_SAMPLING", True)
    num_checkpoints = config.get("NUM_CHECKPOINTS", 0)

    actor_network = MAPPOActorRNN(action_dim=ACTION_DIM, config=model_config)
    critic_network = MAPPOCentralCriticRNN(config=model_config)

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

    actor_lr_fn = _make_lr_fn("LR")
    critic_lr_fn = _make_lr_fn("LR_CRITIC")

    def train(rng, population=None):
        """
        population: stacked actor params pytree, leaf shape (N, ...).
                    If None, falls back to self-play (no partner).
        """
        assert population is not None, "MEP S2 requires a population (MEP S1 output)"
        pop_actor_params = population

        N = jax.tree_util.tree_leaves(pop_actor_params)[0].shape[0]

        # ----------------------------------------------------------------
        # ENV shape inference
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        sample_obs, _ = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        obs_shape = sample_obs[env.agents[0]].shape[1:]
        global_obs_shape = (*obs_shape[:-1], obs_shape[-1] * 2)

        # ----------------------------------------------------------------
        # Init ego actor + critic
        # ----------------------------------------------------------------
        init_actor_h = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x_actor = (
            jnp.zeros((1, NUM_ENVS, *obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )
        init_critic_h = MAPPOCentralCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x_critic = (
            jnp.zeros((1, NUM_ENVS, *global_obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )

        rng, _rng = jax.random.split(rng)
        ra, rc = jax.random.split(_rng)
        a_params = actor_network.init(ra, init_actor_h, init_x_actor)
        c_params = critic_network.init(rc, init_critic_h, init_x_critic)

        actor_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=a_params,
            tx=_make_tx(actor_lr_fn),
        )
        critic_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=c_params,
            tx=_make_tx(critic_lr_fn),
        )

        # ----------------------------------------------------------------
        # Init envs
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        last_obs, env_state = jax.vmap(env.reset)(
            jax.random.split(_rng, NUM_ENVS)
        )
        last_done = jnp.zeros((NUM_ENVS,), dtype=jnp.bool_)

        # Running returns per population member (for prioritized sampling)
        running_returns = jnp.zeros((N,))

        # Checkpoint buffer: (num_checkpoints, ...)
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

        # ----------------------------------------------------------------
        # TRAIN LOOP
        # ----------------------------------------------------------------
        def _update_step(runner_state, unused):
            (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                partner_hstate,
                critic_hstate,
                running_returns,
                ck_buf,
                update_step,
                rng,
            ) = runner_state

            # -- Partner selection (prioritized or uniform) --
            rng, _rng = jax.random.split(rng)
            if use_prioritized:
                logits = -running_returns / prioritized_alpha
            else:
                logits = jnp.zeros((N,))
            partner_idx = jax.random.categorical(_rng, logits)

            partner_params = jax.tree_util.tree_map(
                lambda x: x[partner_idx], pop_actor_params
            )
            # Reset partner hstate at each update step so stale hidden state
            # from a previous (different) partner does not carry over.
            partner_hstate = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

            # ---- ROLLOUT ------------------------------------------------
            def _env_step(step_state, _):
                (
                    env_state, last_obs, last_done,
                    actor_hstate, partner_hstate, critic_hstate,
                    update_step, rng,
                ) = step_state

                ego_obs_t = last_obs[env.agents[0]]      # (E, H, W, C)
                partner_obs_t = last_obs[env.agents[1]]  # (E, H, W, C)
                global_obs_t = jnp.concatenate([ego_obs_t, partner_obs_t], axis=-1)
                done_t = last_done  # (E,)

                # Ego actor
                rng, _rng = jax.random.split(rng)
                actor_hstate, ego_pi = actor_network.apply(
                    actor_state.params,
                    actor_hstate,
                    (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                )
                ego_action = ego_pi.sample(seed=_rng).squeeze(0)
                ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)

                # Centralized critic
                critic_hstate, crit_val = critic_network.apply(
                    critic_state.params,
                    critic_hstate,
                    (global_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                )
                crit_val = crit_val.squeeze(0)

                # Partner action (frozen)
                rng, _rng2 = jax.random.split(rng)
                partner_hstate, partner_pi = actor_network.apply(
                    partner_params,
                    partner_hstate,
                    (partner_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                )
                partner_action = partner_pi.sample(seed=_rng2).squeeze(0)

                env_act = {
                    env.agents[0]: ego_action,
                    env.agents[1]: partner_action,
                }
                rng, _rng3 = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(_rng3, NUM_ENVS), env_state, env_act
                )
                done_env = done["__all__"]

                anneal_factor = rew_shaping_anneal(
                    update_step * NUM_STEPS * NUM_ENVS
                )
                reward = jax.tree_util.tree_map(
                    lambda r, s: r + s * anneal_factor,
                    reward, info["shaped_reward"],
                )
                ego_reward = reward[env.agents[0]]
                info = {k: v for k, v in info.items() if k != "shaped_reward"}
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
                    global_obs=global_obs_t,
                    info=info,
                )
                step_state = (
                    env_state, obsv, done_env,
                    actor_hstate, partner_hstate, critic_hstate,
                    update_step, rng,
                )
                return step_state, transition

            final_step, traj = jax.lax.scan(
                _env_step,
                (env_state, last_obs, last_done,
                 actor_hstate, partner_hstate, critic_hstate,
                 update_step, rng),
                None,
                NUM_STEPS,
            )
            (env_state, last_obs, last_done,
             actor_hstate, partner_hstate, critic_hstate, _, rng) = final_step

            # ---- Update running return for this partner ------------------
            # Use returned_episode_returns (unshaped cumulative return) so that
            # reward shaping early in training doesn't distort prioritized sampling.
            ep_returns = traj.info["returned_episode_returns"]  # (T, E)
            ep_done = traj.info["returned_episode"]             # (T, E) bool
            n_done = ep_done.sum()
            ep_return = jnp.where(
                n_done > 0,
                (ep_returns * ep_done).sum() / (n_done + 1e-6),
                running_returns[partner_idx],  # keep old if no episode completed
            )
            new_ret = running_returns.at[partner_idx].set(
                0.9 * running_returns[partner_idx] + 0.1 * ep_return
            )

            # ---- GAE + PPO UPDATE ----------------------------------------
            ego_obs_last = last_obs[env.agents[0]]
            partner_obs_last = last_obs[env.agents[1]]
            global_last = jnp.concatenate([ego_obs_last, partner_obs_last], axis=-1)
            _, last_val = critic_network.apply(
                critic_state.params,
                critic_hstate,
                (global_last[jnp.newaxis], last_done[jnp.newaxis]),
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
            init_ah = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            init_ch = MAPPOCentralCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

            def _split_mb(x):
                """(T, E, ...) or (1, E, ...) → (NMB, T or 1, MB_SIZE, ...)"""
                return jnp.swapaxes(
                    jnp.reshape(
                        x, [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:])
                    ),
                    1, 0,
                )

            def _update_minibatch(states, mb):
                a_st, c_st = states
                (mb_ah, mb_ch,
                 mb_ego_obs, mb_global, mb_done,
                 mb_action, mb_log_prob, mb_val_old,
                 mb_adv, mb_targets) = mb
                mb_ah = mb_ah.squeeze(0)  # (MB_SIZE, hidden)
                mb_ch = mb_ch.squeeze(0)

                def _actor_loss(a_params):
                    _, pi = actor_network.apply(a_params, mb_ah, (mb_ego_obs, mb_done))
                    lp = pi.log_prob(mb_action)
                    ratio = jnp.exp(lp - mb_log_prob)
                    adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    loss1 = ratio * adv_n
                    loss2 = jnp.clip(ratio,
                                     1 - model_config["CLIP_EPS"],
                                     1 + model_config["CLIP_EPS"]) * adv_n
                    al = -jnp.minimum(loss1, loss2).mean()
                    ent = pi.entropy().mean()
                    return al - model_config["ENT_COEF"] * ent, (al, ent)

                def _critic_loss(c_params):
                    _, val = critic_network.apply(c_params, mb_ch, (mb_global, mb_done))
                    v_cl = mb_val_old + (val - mb_val_old).clip(
                        -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                    )
                    return 0.5 * jnp.maximum(
                        jnp.square(val - mb_targets),
                        jnp.square(v_cl - mb_targets),
                    ).mean()

                (al, a_aux), ag = jax.value_and_grad(_actor_loss, has_aux=True)(a_st.params)
                cl, cg = jax.value_and_grad(_critic_loss)(c_st.params)
                return (a_st.apply_gradients(grads=ag),
                        c_st.apply_gradients(grads=cg)), (al, cl, a_aux[1])

            def _update_epoch(epoch_state, _):
                a_state, c_state, rng_e = epoch_state
                rng_e, _rng_e = jax.random.split(rng_e)
                perm = jax.random.permutation(_rng_e, NUM_ENVS)

                batch = (
                    jnp.take(init_ah, perm, axis=0)[jnp.newaxis],   # (1, E, hid)
                    jnp.take(init_ch, perm, axis=0)[jnp.newaxis],   # (1, E, hid)
                    jnp.take(traj.ego_obs, perm, axis=1),            # (T, E, H, W, C)
                    jnp.take(traj.global_obs, perm, axis=1),         # (T, E, H, W, 2C)
                    jnp.take(traj.done, perm, axis=1),               # (T, E)
                    jnp.take(traj.ego_action, perm, axis=1),        # (T, E)
                    jnp.take(traj.ego_log_prob, perm, axis=1),      # (T, E)
                    jnp.take(traj.critic_value, perm, axis=1),      # (T, E)
                    jnp.take(advantages, perm, axis=1),              # (T, E)
                    jnp.take(targets, perm, axis=1),                 # (T, E)
                )
                minibatches = jax.tree_util.tree_map(_split_mb, batch)

                (a_state, c_state), losses = jax.lax.scan(
                    _update_minibatch, (a_state, c_state), minibatches
                )
                return (a_state, c_state, rng_e), losses

            rng, _rng = jax.random.split(rng)
            (actor_state, critic_state, _), all_losses = jax.lax.scan(
                _update_epoch,
                (actor_state, critic_state, _rng),
                None,
                UPDATE_EPOCHS,
            )

            # ---- Checkpoint buffer ---------------------------------------
            update_step = update_step + 1
            selector = checkpoint_steps == update_step
            slot = jnp.argmax(selector)
            ck_buf = jax.lax.cond(
                jnp.any(selector),
                lambda b: jax.tree_util.tree_map(
                    lambda bl, p: bl.at[slot].set(p), b, actor_state.params
                ),
                lambda b: b,
                ck_buf,
            )

            # ---- Metrics -------------------------------------------------
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj.info)
            metric["total_loss"] = all_losses[0].mean()
            metric["critic_loss"] = all_losses[1].mean()
            metric["entropy"] = all_losses[2].mean()
            metric["partner_idx"] = partner_idx
            metric["running_return_mean"] = new_ret.mean()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * NUM_STEPS * NUM_ENVS

            def _log_s2(m):
                flat = {}
                for k, v in m.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}/{kk}"] = float(vv)
                    else:
                        flat[k] = float(v)
                wandb.log({f"mep_s2/{k}": v for k, v in flat.items()})

            jax.debug.callback(_log_s2, metric)

            runner_state = (
                actor_state, critic_state,
                env_state, last_obs, last_done,
                actor_hstate, partner_hstate, critic_hstate,
                new_ret, ck_buf, update_step, rng,
            )
            return runner_state, metric

        # ---- Initial checkpoint ------------------------------------------
        ck_buf = jax.tree_util.tree_map(
            lambda b, p: b.at[0].set(p), ck_buf, actor_state.params
        )

        init_runner = (
            actor_state, critic_state,
            env_state, last_obs, last_done,
            init_actor_h, init_actor_h, init_critic_h,
            running_returns, ck_buf, jnp.int32(0), rng,
        )
        final_runner, metrics = jax.lax.scan(
            _update_step, init_runner, None, NUM_UPDATES
        )

        return {
            "runner_state": final_runner,
            "metrics": metrics,
            "actor_params": final_runner[0].params,
            "actor_ckpts": final_runner[9],
        }

    return train
