"""
MEP Stage 1: Population training with entropy bonus.

Trains N MAPPO agents jointly. Each agent i's reward is augmented:
    R_shaped = R + alpha * (-log(π_pop(a|s)))
    π_pop(a|s) = (1/N) Σ_k π_k(a|s)    [full N-member average]

Paper: "Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination"
       AAAI 2023
"""

import jax
import jax.numpy as jnp
import numpy as np
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
    partner_obs: jnp.ndarray  # (NUM_STEPS, NUM_ENVS, H, W, C)
    ego_action: jnp.ndarray   # (NUM_STEPS, NUM_ENVS)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    critic_value: jnp.ndarray # (NUM_STEPS, NUM_ENVS)
    reward: jnp.ndarray       # (NUM_STEPS, NUM_ENVS)
    global_obs: jnp.ndarray   # (NUM_STEPS, NUM_ENVS, H, W, 2C)
    info: dict


def make_train_mep_s1(config):
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

    def train(rng):
        # ----------------------------------------------------------------
        # ENV shape inference
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        sample_obs, _ = jax.vmap(env.reset)(jax.random.split(_rng, NUM_ENVS))
        obs_shape = sample_obs[env.agents[0]].shape[1:]  # (H, W, C)
        global_obs_shape = (*obs_shape[:-1], obs_shape[-1] * 2)

        # ----------------------------------------------------------------
        # Network init shapes
        # ----------------------------------------------------------------
        init_actor_hstate = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x_actor = (
            jnp.zeros((1, NUM_ENVS, *obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )
        init_critic_hstate = MAPPOCentralCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x_critic = (
            jnp.zeros((1, NUM_ENVS, *global_obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )

        # ----------------------------------------------------------------
        # Create N (actor_state, critic_state) pairs — stacked pytree
        # ----------------------------------------------------------------
        def _create_member(rng_m):
            ra, rc = jax.random.split(rng_m)
            a_params = actor_network.init(ra, init_actor_hstate, init_x_actor)
            c_params = critic_network.init(rc, init_critic_hstate, init_x_critic)
            a_state = TrainState.create(
                apply_fn=actor_network.apply,
                params=a_params,
                tx=_make_tx(actor_lr_fn),
            )
            c_state = TrainState.create(
                apply_fn=critic_network.apply,
                params=c_params,
                tx=_make_tx(critic_lr_fn),
            )
            return a_state, c_state

        rng, _rng = jax.random.split(rng)
        # vmap _create_member over N random keys → stacked TrainStates
        pop_actor_states, pop_critic_states = jax.vmap(_create_member)(
            jax.random.split(_rng, N)
        )

        # ----------------------------------------------------------------
        # Init N×NUM_ENVS environments
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        # shape: (N, NUM_ENVS, 2) — PRNGKeys
        all_reset_rngs = jax.random.split(_rng, N * NUM_ENVS).reshape(N, NUM_ENVS, -1)
        pop_obs, pop_env_states = jax.vmap(
            lambda r: jax.vmap(env.reset)(r)
        )(all_reset_rngs)
        pop_done = jnp.zeros((N, NUM_ENVS), dtype=jnp.bool_)

        # Actor hstates: (N, NUM_ENVS, GRU_HIDDEN_DIM)
        _init_actor_h = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        pop_actor_hstates = jnp.stack([_init_actor_h] * N)
        pop_partner_hstates = jnp.stack([_init_actor_h] * N)
        # Critic hstates: (N, NUM_ENVS, GRU_HIDDEN_DIM)
        _init_critic_h = MAPPOCentralCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        pop_critic_hstates = jnp.stack([_init_critic_h] * N)

        # Checkpoint buffer: stores N actors' params, shape (N, num_checkpoints, ...)
        checkpoint_steps = jnp.linspace(
            0, NUM_UPDATES, max(num_checkpoints, 1),
            endpoint=True, dtype=jnp.int32,
        )
        if num_checkpoints > 0:
            checkpoint_steps = checkpoint_steps.at[-1].set(NUM_UPDATES)
        # stacked per-member checkpoint buffer: leaf (N, num_checkpoints, ...)
        ck_buf = jax.tree_util.tree_map(
            lambda p: jnp.zeros((N, max(num_checkpoints, 1)) + p.shape[1:], p.dtype),
            pop_actor_states.params,
        )

        # ----------------------------------------------------------------
        # TRAIN LOOP
        # ----------------------------------------------------------------
        def _update_step(runner_state, unused):
            (
                pop_actor_states,
                pop_critic_states,
                pop_env_states,
                pop_last_obs,
                pop_last_done,
                pop_actor_hstates,
                pop_partner_hstates,
                pop_critic_hstates,
                ck_buf,
                update_step,
                rng,
            ) = runner_state

            # -- Partner assignment (rotate by random offset != 0) --
            rng, _rng = jax.random.split(rng)
            offset = jax.random.randint(_rng, (), 1, max(N, 2))
            partner_idxs = (jnp.arange(N) + offset) % N

            # All actor params (N stacked) — used as closure in inner fns
            all_actor_params = pop_actor_states.params

            # ---- ROLLOUT ------------------------------------------------
            def _collect_member(
                ego_actor_state,
                ego_critic_state,
                partner_idx,
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                partner_hstate,
                critic_hstate,
                rng,
            ):
                def _env_step(step_state, _):
                    (
                        env_state, last_obs, last_done,
                        actor_hstate, partner_hstate, critic_hstate,
                        update_step, rng,
                    ) = step_state

                    ego_obs_t = last_obs[env.agents[0]]    # (E, H, W, C)
                    partner_obs_t = last_obs[env.agents[1]] # (E, H, W, C)
                    global_obs_t = jnp.concatenate([ego_obs_t, partner_obs_t], axis=-1)
                    done_t = last_done  # (E,)

                    # Ego actor
                    rng, _rng = jax.random.split(rng)
                    actor_hstate, ego_pi = actor_network.apply(
                        ego_actor_state.params,
                        actor_hstate,
                        (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    ego_action = ego_pi.sample(seed=_rng).squeeze(0)   # (E,)
                    ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)

                    # Centralized critic
                    critic_hstate, crit_val = critic_network.apply(
                        ego_critic_state.params,
                        critic_hstate,
                        (global_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    crit_val = crit_val.squeeze(0)  # (E,)

                    # Partner actor (frozen)
                    partner_params = jax.tree_util.tree_map(
                        lambda x: x[partner_idx], all_actor_params
                    )
                    rng, _rng2 = jax.random.split(rng)
                    partner_hstate, partner_pi = actor_network.apply(
                        partner_params,
                        partner_hstate,
                        (partner_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    partner_action = partner_pi.sample(seed=_rng2).squeeze(0)  # (E,)

                    # Step envs
                    env_act = {
                        env.agents[0]: ego_action,
                        env.agents[1]: partner_action,
                    }
                    rng, _rng3 = jax.random.split(rng)
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        jax.random.split(_rng3, NUM_ENVS), env_state, env_act
                    )
                    done_env = done["__all__"]  # (E,)

                    # Reward shaping anneal
                    anneal_factor = rew_shaping_anneal(
                        update_step * NUM_STEPS * NUM_ENVS
                    )
                    reward = jax.tree_util.tree_map(
                        lambda r, s: r + s * anneal_factor,
                        reward, info["shaped_reward"],
                    )
                    ego_reward = reward[env.agents[0]]  # (E,)
                    info = jax.tree_util.tree_map(lambda x: x.reshape((NUM_ENVS,)), info)

                    transition = MEPTransition(
                        done=done_env,
                        ego_obs=ego_obs_t,
                        partner_obs=partner_obs_t,
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

                final_state, traj = jax.lax.scan(
                    _env_step,
                    (env_state, last_obs, last_done,
                     actor_hstate, partner_hstate, critic_hstate,
                     update_step, rng),
                    None,
                    NUM_STEPS,
                )
                (fe, fo, fd, fah, fph, fch, _, _) = final_state
                return traj, (fe, fo, fd, fah, fph, fch)

            rng, _rng = jax.random.split(rng)
            all_traj, finals = jax.vmap(
                _collect_member, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )(
                pop_actor_states, pop_critic_states, partner_idxs,
                pop_env_states, pop_last_obs, pop_last_done,
                pop_actor_hstates, pop_partner_hstates, pop_critic_hstates,
                jax.random.split(_rng, N),
            )
            # all_traj leaf shapes: (N, NUM_STEPS, NUM_ENVS, ...)
            (pop_env_states, pop_last_obs, pop_last_done,
             pop_actor_hstates, pop_partner_hstates, pop_critic_hstates) = finals

            # ---- ENTROPY BONUS ------------------------------------------
            # ego_obs_all: (N, NUM_STEPS, NUM_ENVS, H, W, C)
            # Compute bonus for each member using all N actor probs

            def compute_bonus(ego_obs_i, ego_done_i, ego_act_i):
                """Return -log π_pop(a|s) for member i. Shape: (NUM_STEPS, NUM_ENVS)."""
                init_h = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def fwd_k(ak_params):
                    _, pi = actor_network.apply(
                        ak_params, init_h, (ego_obs_i, ego_done_i)
                    )
                    return pi.probs  # (NUM_STEPS, NUM_ENVS, A)

                all_probs = jax.vmap(fwd_k)(all_actor_params)  # (N, T, E, A)
                mean_prob = jnp.mean(all_probs, axis=0)          # (T, E, A)
                taken = jnp.take_along_axis(
                    mean_prob, ego_act_i[..., jnp.newaxis], axis=-1
                ).squeeze(-1)                                    # (T, E)
                return -jnp.log(taken + 1e-6)                   # (T, E)

            all_bonuses = jax.vmap(compute_bonus)(
                all_traj.ego_obs,    # (N, T, E, H, W, C)
                all_traj.done,       # (N, T, E)
                all_traj.ego_action, # (N, T, E)
            )  # (N, T, E)

            shaped_rewards = all_traj.reward + entropy_alpha * all_bonuses

            # ---- GAE + PPO UPDATE per member ----------------------------
            def _update_member(
                ego_actor_state,
                ego_critic_state,
                traj_i,            # MEPTransition with shapes (T, E, ...)
                shaped_reward_i,   # (T, E)
                last_obs_i,        # {"agent_0": (E, H, W, C), ...}
                last_done_i,       # (E,)
                critic_hstate_i,   # (E, hidden)
                rng_m,
            ):
                # ---- Last value for GAE bootstrap ----
                ego_obs_last = last_obs_i[env.agents[0]]
                partner_obs_last = last_obs_i[env.agents[1]]
                global_last = jnp.concatenate([ego_obs_last, partner_obs_last], axis=-1)
                _, last_val = critic_network.apply(
                    ego_critic_state.params,
                    critic_hstate_i,
                    (global_last[jnp.newaxis], last_done_i[jnp.newaxis]),
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
                # Following ippo.py: shuffle over ENV axis, reshape into minibatches,
                # then scan over pre-built minibatch arrays (no dynamic indexing).
                mb_size = NUM_ENVS // NUM_MINIBATCHES
                init_actor_h = MAPPOActorRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
                init_critic_h = MAPPOCentralCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

                def _split_minibatches(x):
                    """(T, E, ...) or (1, E, ...) → (NMB, T or 1, MB_SIZE, ...)"""
                    return jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:]),
                        ),
                        1, 0,
                    )

                def _update_minibatch(states, mb):
                    a_st, c_st = states
                    # mb is a tuple; each array has leading dim = T (or 1 for hstates)
                    (mb_ah, mb_ch,
                     mb_ego_obs, mb_global, mb_done,
                     mb_action, mb_log_prob, mb_val_old,
                     mb_adv, mb_targets) = mb

                    # hstates: (1, MB_SIZE, hidden) → squeeze time dim
                    mb_ah = mb_ah.squeeze(0)  # (MB_SIZE, hidden)
                    mb_ch = mb_ch.squeeze(0)  # (MB_SIZE, hidden)

                    def _actor_loss(a_params):
                        _, pi = actor_network.apply(
                            a_params, mb_ah, (mb_ego_obs, mb_done)
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
                        entropy = pi.entropy().mean()
                        return actor_loss - model_config["ENT_COEF"] * entropy, (actor_loss, entropy)

                    def _critic_loss(c_params):
                        _, value = critic_network.apply(
                            c_params, mb_ch, (mb_global, mb_done)
                        )
                        v_cl = mb_val_old + (value - mb_val_old).clip(
                            -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                        )
                        return 0.5 * jnp.maximum(
                            jnp.square(value - mb_targets),
                            jnp.square(v_cl - mb_targets),
                        ).mean()

                    (a_loss, a_aux), ag = jax.value_and_grad(_actor_loss, has_aux=True)(a_st.params)
                    c_loss, cg = jax.value_and_grad(_critic_loss)(c_st.params)
                    return (a_st.apply_gradients(grads=ag),
                            c_st.apply_gradients(grads=cg)), (a_loss, c_loss, a_aux[0], a_aux[1])

                def _update_epoch(epoch_state, _):
                    a_state, c_state, rng_e = epoch_state
                    rng_e, _rng_e = jax.random.split(rng_e)
                    perm = jax.random.permutation(_rng_e, NUM_ENVS)

                    # Shuffle over env axis (axis=1 for traj arrays, axis=0 for hstates)
                    batch = (
                        jnp.take(init_actor_h, perm, axis=0)[jnp.newaxis],  # (1, E, hid)
                        jnp.take(init_critic_h, perm, axis=0)[jnp.newaxis], # (1, E, hid)
                        jnp.take(traj_i.ego_obs, perm, axis=1),             # (T, E, H, W, C)
                        jnp.take(traj_i.global_obs, perm, axis=1),          # (T, E, H, W, 2C)
                        jnp.take(traj_i.done, perm, axis=1),                # (T, E)
                        jnp.take(traj_i.ego_action, perm, axis=1),         # (T, E)
                        jnp.take(traj_i.ego_log_prob, perm, axis=1),       # (T, E)
                        jnp.take(traj_i.critic_value, perm, axis=1),       # (T, E)
                        jnp.take(advantages, perm, axis=1),                 # (T, E)
                        jnp.take(targets, perm, axis=1),                    # (T, E)
                    )
                    # Split into (NMB, T or 1, MB_SIZE, ...)
                    minibatches = jax.tree_util.tree_map(_split_minibatches, batch)

                    (a_state, c_state), losses = jax.lax.scan(
                        _update_minibatch, (a_state, c_state), minibatches
                    )
                    return (a_state, c_state, rng_e), losses

                (a_state_out, c_state_out, _), all_losses = jax.lax.scan(
                    _update_epoch,
                    (ego_actor_state, ego_critic_state, rng_m),
                    None,
                    UPDATE_EPOCHS,
                )
                # all_losses: (UPDATE_EPOCHS, NUM_MINIBATCHES, ...)
                total_loss = all_losses[0].mean()
                critic_loss = all_losses[1].mean()
                actor_loss = all_losses[2].mean()
                entropy = all_losses[3].mean()
                return a_state_out, c_state_out, total_loss, critic_loss, actor_loss, entropy

            rng, _rng = jax.random.split(rng)
            (
                pop_actor_states,
                pop_critic_states,
                total_losses,
                critic_losses,
                actor_losses,
                entropies,
            ) = jax.vmap(_update_member)(
                pop_actor_states,
                pop_critic_states,
                all_traj,
                shaped_rewards,
                pop_last_obs,
                pop_last_done,
                pop_critic_hstates,
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
            ck_buf = _save_ck(ck_buf, pop_actor_states.params, update_step)

            # ---- WandB metrics -------------------------------------------
            metric = jax.tree_util.tree_map(lambda x: x.mean(), all_traj.info)
            metric["total_loss"] = total_losses.mean()
            metric["critic_loss"] = critic_losses.mean()
            metric["actor_loss"] = actor_losses.mean()
            metric["entropy"] = entropies.mean()
            metric["entropy_bonus"] = all_bonuses.mean()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * NUM_STEPS * NUM_ENVS

            def _log(m):
                wandb.log({f"mep_s1/{k}": float(v) for k, v in m.items()})

            jax.debug.callback(_log, metric)

            runner_state = (
                pop_actor_states,
                pop_critic_states,
                pop_env_states,
                pop_last_obs,
                pop_last_done,
                pop_actor_hstates,
                pop_partner_hstates,
                pop_critic_hstates,
                ck_buf,
                update_step,
                rng,
            )
            return runner_state, metric

        # ---- Initial checkpoint at step 0 --------------------------------
        ck_buf = jax.tree_util.tree_map(
            lambda b, p: b.at[:, 0].set(p),
            ck_buf, pop_actor_states.params,
        )

        init_runner = (
            pop_actor_states,
            pop_critic_states,
            pop_env_states,
            pop_obs,
            pop_done,
            pop_actor_hstates,
            pop_partner_hstates,
            pop_critic_hstates,
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
            "pop_actor_ckpts": final_runner[8],
        }

    return train
