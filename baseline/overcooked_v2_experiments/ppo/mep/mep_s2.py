"""
MEP Stage 2: Adaptive agent training against MEP S1 population.

Trains a single IPPO agent against the N-member population produced by Stage 1.
Partner selection uses return-based prioritized sampling:
    weights ∝ rank(1 / max(returns, 0) + ε)^alpha
    lower return (harder partner) → higher rank → higher sampling probability

Bidirectional: ego plays as agents[0] or agents[1], giving 2*N pairings.

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
from overcooked_v2_experiments.ppo.utils.valuenorm import (
    ValueNormState, valuenorm_update, valuenorm_normalize, valuenorm_denormalize,
)


class MEPTransition(NamedTuple):
    # 2-agent path:  all shapes use ENV dim = NUM_ENVS (E)
    # 3+ agent path: all shapes use ENV dim = num_env_agents * NUM_ENVS (N*E)
    done: jnp.ndarray         # (NUM_STEPS, E) or (NUM_STEPS, N*E)
    ego_obs: jnp.ndarray      # (NUM_STEPS, E, H, W, C) or (NUM_STEPS, N*E, H, W, C)
    ego_action: jnp.ndarray   # (NUM_STEPS, E) or (NUM_STEPS, N*E)
    ego_log_prob: jnp.ndarray # (NUM_STEPS, E) or (NUM_STEPS, N*E)
    critic_value: jnp.ndarray # (NUM_STEPS, E) or (NUM_STEPS, N*E)
    reward: jnp.ndarray       # (NUM_STEPS, E) or (NUM_STEPS, N*E)
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
    model_config["ACTION_DIM"] = ACTION_DIM

    prioritized_alpha = config.get("MEP_PRIORITIZED_ALPHA", 1.0)
    use_prioritized = config.get("MEP_USE_PRIORITIZED_SAMPLING", True)
    num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
    use_valuenorm = model_config.get("USE_VALUENORM", False)

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

    def train(rng, population=None):
        """
        population: stacked actor params pytree, leaf shape (N, ...).
                    Must be provided (MEP S1 output).
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

        # ----------------------------------------------------------------
        # Init ego network (single IPPO agent)
        # ----------------------------------------------------------------
        init_h = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
        init_x = (
            jnp.zeros((1, NUM_ENVS, *obs_shape)),
            jnp.zeros((1, NUM_ENVS)),
        )

        rng, _rng = jax.random.split(rng)
        params = network.init(_rng, init_h, init_x)
        actor_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=_make_tx(lr_fn),
        )

        # ----------------------------------------------------------------
        # Init envs
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        last_obs, env_state = jax.vmap(env.reset)(
            jax.random.split(_rng, NUM_ENVS)
        )
        last_done = jnp.zeros((NUM_ENVS,), dtype=jnp.bool_)

        num_env_agents = env.num_agents
        num_partners = num_env_agents - 1

        # Running returns per pairing
        # 2-agent: 2*N (ego=agents[0] × N, ego=agents[1] × N)
        # 3+ agent: N (ego 고정 agents[0], partner는 population에서 랜덤)
        num_pairings = 2 * N if num_env_agents == 2 else N
        running_returns = jnp.zeros((num_pairings,))

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
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                partner_hstate,
                running_returns,
                ck_buf,
                vn_state,
                update_step,
                rng,
            ) = runner_state

            # -- Partner selection: rank-based over pairings --
            rng, _rng = jax.random.split(rng)
            if use_prioritized:
                inv_returns = 1.0 / (jnp.maximum(running_returns, 0.0) + 1e-6)
                ranks = (jnp.argsort(jnp.argsort(inv_returns)) + 1).astype(jnp.float32)
                probs = jnp.power(ranks, prioritized_alpha)
                probs = probs / probs.sum()
                partner_idx = jax.random.categorical(_rng, jnp.log(probs + 1e-8))
            else:
                partner_idx = jax.random.randint(_rng, (), 0, num_pairings)

            if num_env_agents == 2:
                # 2-agent: bidirectional ego/partner role
                ego_role = jnp.where(partner_idx < N, 0, 1)
                pop_idx = jnp.where(partner_idx < N, partner_idx, partner_idx - N)
                partner_params = jax.tree_util.tree_map(
                    lambda x: x[pop_idx], pop_actor_params
                )
                partner_hstate = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            else:
                # 3+ agent (GridSpread / FCP-style):
                # - n_ego ~ Uniform[1, N] ego agents per update step
                # - is_ego_agents: (num_env_agents,) bool noise-rank mask
                # - actor_hstate / partner_hstate: flat (N*E, H) shape
                rng, _rng_n_ego, _rng_perm, _rng_pop = jax.random.split(rng, 4)
                pop_idx = partner_idx  # running_returns 업데이트용

                # Sample n_ego ∈ [1, num_env_agents] and build is_ego_agents mask
                # Use noise-rank: agents with rank < n_ego are ego (JAX-traceable)
                n_ego = jax.random.randint(_rng_n_ego, (), 1, num_env_agents + 1)
                noise = jax.random.uniform(_rng_perm, (num_env_agents,))
                # rank[i] = number of agents with strictly lower noise than agent i
                # is_ego[i] = (rank[i] < n_ego)  → top-n_ego agents by lowest noise
                noise_ranks = jnp.sum(
                    noise[jnp.newaxis, :] < noise[:, jnp.newaxis], axis=1
                )  # (num_env_agents,) — rank 0 = smallest noise
                is_ego_agents = noise_ranks < n_ego  # (num_env_agents,) bool
                # Flat (N*E,) mask: repeat each agent flag NUM_ENVS times (agent-major)
                is_ego_mask = jnp.repeat(is_ego_agents, NUM_ENVS)

                # Shared partner pop params (single idx, incoherence 방지)
                shared_pop_idx = jax.random.randint(_rng_pop, (), 0, N)
                shared_partner_params = jax.tree_util.tree_map(
                    lambda x: x[shared_pop_idx], pop_actor_params
                )
                # Flat hstates: (N*E, H) — reset each update step
                actor_hstate = ActorCriticRNN.initialize_carry(
                    num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
                )
                partner_hstate = ActorCriticRNN.initialize_carry(
                    num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
                )

            # ---- ROLLOUT ------------------------------------------------
            def _env_step(step_state, _):
                (
                    env_state, last_obs, last_done,
                    actor_hstate, partner_hstate,
                    update_step, rng,
                ) = step_state

                done_t = last_done  # (E,)

                if num_env_agents == 2:
                    # 2-agent: bidirectional ego/partner
                    agent0_obs_t = last_obs[env.agents[0]]
                    agent1_obs_t = last_obs[env.agents[1]]
                    ego_obs_t     = jnp.where(ego_role == 0, agent0_obs_t, agent1_obs_t)
                    partner_obs_t = jnp.where(ego_role == 0, agent1_obs_t, agent0_obs_t)

                    # Ego: actor + critic in one forward pass
                    rng, _rng = jax.random.split(rng)
                    actor_hstate, ego_pi, crit_val, _ = network.apply(
                        actor_state.params,
                        actor_hstate,
                        (ego_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    ego_action = ego_pi.sample(seed=_rng).squeeze(0)
                    ego_log_prob = ego_pi.log_prob(ego_action[jnp.newaxis]).squeeze(0)
                    crit_val = crit_val.squeeze(0)

                    # 2-agent: 단일 partner
                    rng, _rng2 = jax.random.split(rng)
                    partner_hstate, partner_pi, _, _ = network.apply(
                        partner_params,
                        partner_hstate,
                        (partner_obs_t[jnp.newaxis], done_t[jnp.newaxis]),
                    )
                    partner_action = partner_pi.sample(seed=_rng2).squeeze(0)
                    env_act = {}
                    env_act[env.agents[0]] = jnp.where(ego_role == 0, ego_action, partner_action)
                    env_act[env.agents[1]] = jnp.where(ego_role == 0, partner_action, ego_action)
                else:
                    # 3+ agent (FCP-style): all N agents processed in flat (N*E, H) batches
                    # all_obs_flat: concat all agents' obs → (N*E, obs_dim...)  agent-major
                    all_obs_flat = jnp.concatenate(
                        [last_obs[a] for a in env.agents], axis=0
                    )  # (N*E, ...)
                    # done_flat: tile done_t N times → (N*E,)
                    done_flat = jnp.tile(done_t, num_env_agents)  # (N*E,)

                    # Forward ALL N agents through train network (actor_hstate: (N*E, H))
                    rng, _rng_train, _rng_partner = jax.random.split(rng, 3)
                    new_actor_hstate, all_pi, all_values, _ = network.apply(
                        actor_state.params,
                        actor_hstate,
                        (all_obs_flat[jnp.newaxis], done_flat[jnp.newaxis]),
                    )
                    train_actions = all_pi.sample(seed=_rng_train).squeeze(0)  # (N*E,)
                    train_log_probs = all_pi.log_prob(
                        train_actions[jnp.newaxis]
                    ).squeeze(0)  # (N*E,)
                    all_crit_vals = all_values.squeeze(0)  # (N*E,)

                    # Forward ALL N agents through partner network (partner_hstate: (N*E, H))
                    new_partner_hstate, partner_pi, _, _ = network.apply(
                        shared_partner_params,
                        partner_hstate,
                        (all_obs_flat[jnp.newaxis], done_flat[jnp.newaxis]),
                    )
                    partner_actions = partner_pi.sample(seed=_rng_partner).squeeze(0)  # (N*E,)

                    # Select action: ego agents → train_actions, partners → partner_actions
                    actual_actions = jnp.where(is_ego_mask, train_actions, partner_actions)  # (N*E,)

                    # ego_obs_t, ego log_prob, crit_val: full (N*E,) flat arrays
                    ego_obs_t = all_obs_flat          # (N*E, ...)
                    ego_action = actual_actions        # (N*E,) stored for all agents
                    ego_log_prob = train_log_probs     # (N*E,) — partner slots unused in loss
                    crit_val = all_crit_vals           # (N*E,)

                    actor_hstate = new_actor_hstate
                    partner_hstate = new_partner_hstate

                    # Build env_act: agent-major slice [i*E:(i+1)*E]
                    env_act = {}
                    for i in range(num_env_agents):
                        env_act[env.agents[i]] = actual_actions[i * NUM_ENVS:(i + 1) * NUM_ENVS]

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
                if num_env_agents == 2:
                    ego_reward = jnp.where(
                        ego_role == 0,
                        reward[env.agents[0]],
                        reward[env.agents[1]],
                    )
                    tran_done = done_env          # (E,)
                else:
                    # 3+ agent: tile shared reward across all N agents → (N*E,) agent-major
                    # All agents share the cooperative reward (agents[0] representative)
                    ego_reward = jnp.tile(reward[env.agents[0]], num_env_agents)  # (N*E,)
                    # done: tile done_env → (N*E,) agent-major
                    tran_done = jnp.tile(done_env, num_env_agents)               # (N*E,)
                info.pop("shaped_reward", None)
                info.pop("shaped_reward_events", None)
                info = jax.tree_util.tree_map(
                    lambda x: x.mean(axis=-1) if x.ndim > 1 else x, info
                )

                transition = MEPTransition(
                    done=tran_done,
                    ego_obs=ego_obs_t,
                    ego_action=ego_action,
                    ego_log_prob=ego_log_prob,
                    critic_value=crit_val,
                    reward=ego_reward,
                    info=info,
                )
                step_state = (
                    env_state, obsv, done_env,
                    actor_hstate, partner_hstate,
                    update_step, rng,
                )
                return step_state, transition

            final_step, traj = jax.lax.scan(
                _env_step,
                (env_state, last_obs, last_done,
                 actor_hstate, partner_hstate,
                 update_step, rng),
                None,
                NUM_STEPS,
            )
            (env_state, last_obs, last_done,
             actor_hstate, partner_hstate, _, rng) = final_step

            # ---- Update running return for this partner ------------------
            ep_returns = traj.info["returned_episode_returns"]  # (T, E)
            ep_done = traj.info["returned_episode"]             # (T, E) bool
            n_done = ep_done.sum()
            ep_return = jnp.where(
                n_done > 0,
                (ep_returns * ep_done).sum() / (n_done + 1e-6),
                running_returns[partner_idx],
            )
            new_ret = running_returns.at[partner_idx].set(
                0.95 * running_returns[partner_idx] + 0.05 * ep_return
            )

            # ---- GAE + PPO UPDATE ----------------------------------------
            if num_env_agents == 2:
                a0_last = last_obs[env.agents[0]]
                a1_last = last_obs[env.agents[1]]
                ego_obs_last = jnp.where(ego_role == 0, a0_last, a1_last)
                init_h_bs = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
                done_last_flat = last_done  # (E,)
            else:
                # 3+ agent: concat all agents' last obs → (N*E, obs_dim...) agent-major
                ego_obs_last = jnp.concatenate(
                    [last_obs[a] for a in env.agents], axis=0
                )  # (N*E, ...)
                # tile last_done → (N*E,) agent-major
                done_last_flat = jnp.tile(last_done, num_env_agents)  # (N*E,)
                init_h_bs = ActorCriticRNN.initialize_carry(
                    num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
                )

            _, _, last_val, _ = network.apply(
                actor_state.params,
                init_h_bs,
                (ego_obs_last[jnp.newaxis], done_last_flat[jnp.newaxis]),
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

            # Reward normalization (HSP 논문 Table 9: use reward normalization=True)
            rewards = traj.reward
            if model_config.get("USE_REWARD_NORM", False):
                rewards = rewards / (rewards.std() + 1e-8)

            # ValueNorm: critic 출력을 원래 스케일로 denormalize 후 GAE 계산
            if use_valuenorm:
                gae_values = valuenorm_denormalize(vn_state, traj.critic_value)
                gae_last_val = valuenorm_denormalize(vn_state, last_val)
            else:
                gae_values = traj.critic_value
                gae_last_val = last_val

            _, advantages = jax.lax.scan(
                _get_adv,
                (jnp.zeros_like(gae_last_val), gae_last_val),
                (traj.done, gae_values, rewards),
                reverse=True,
                unroll=16,
            )
            # targets는 원래 스케일의 returns
            targets = advantages + gae_values

            # ValueNorm: 통계 업데이트 후 targets를 정규화
            if use_valuenorm:
                vn_state = valuenorm_update(vn_state, targets)
                targets = valuenorm_normalize(vn_state, targets)

            if num_env_agents == 2:
                # 2-agent: batch dim = E
                mb_size = NUM_ENVS // NUM_MINIBATCHES
                batch_dim = NUM_ENVS
                init_h_mb = ActorCriticRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)
            else:
                # 3+ agent: batch dim = N*E (agent-major flat)
                mb_size = (num_env_agents * NUM_ENVS) // NUM_MINIBATCHES
                batch_dim = num_env_agents * NUM_ENVS
                init_h_mb = ActorCriticRNN.initialize_carry(
                    num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
                )
                # is_ego_mask tiled over T for minibatch PPO weighting: (T, N*E)
                is_ego_mask_T = jnp.tile(is_ego_mask[jnp.newaxis], (NUM_STEPS, 1))

            def _split_mb(x):
                """(T, batch_dim, ...) or (1, batch_dim, ...) → (NMB, T or 1, MB_SIZE, ...)"""
                return jnp.swapaxes(
                    jnp.reshape(
                        x, [x.shape[0], NUM_MINIBATCHES, mb_size] + list(x.shape[2:])
                    ),
                    1, 0,
                )

            def _update_minibatch(state, mb):
                if num_env_agents == 2:
                    (mb_ah,
                     mb_ego_obs, mb_done,
                     mb_action, mb_log_prob, mb_val_old,
                     mb_adv, mb_targets) = mb
                else:
                    # 3+ agent: extra is_ego_mask_mb for actor loss weighting
                    (mb_ah,
                     mb_ego_obs, mb_done,
                     mb_action, mb_log_prob, mb_val_old,
                     mb_adv, mb_targets, mb_ego_mask) = mb
                mb_ah = mb_ah.squeeze(0)  # (MB_SIZE, hidden)

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
                    ppo_elem = -jnp.minimum(loss1, loss2)  # (T, MB_SIZE)
                    if num_env_agents == 2:
                        # 2-agent: simple mean over all samples
                        actor_loss = ppo_elem.mean()
                    else:
                        # 3+ agent: only ego samples contribute to actor loss
                        # mb_ego_mask: (T, MB_SIZE) bool, sum over ego, normalize by count
                        ego_count = mb_ego_mask.sum()
                        actor_loss = jnp.where(
                            ego_count > 0,
                            (ppo_elem * mb_ego_mask).sum() / (ego_count + 1e-8),
                            jnp.zeros(()),
                        )
                    v_cl = mb_val_old + (value - mb_val_old).clip(
                        -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                    )
                    # Huber loss (HSP 논문 Table 9: huber loss, delta=10.0)
                    huber_delta = model_config.get("HUBER_DELTA", 0.0)
                    if huber_delta > 0:
                        def _huber(x):
                            return jnp.where(
                                jnp.abs(x) <= huber_delta,
                                0.5 * x ** 2,
                                huber_delta * (jnp.abs(x) - 0.5 * huber_delta),
                            )
                        critic_loss = jnp.maximum(
                            _huber(value - mb_targets),
                            _huber(v_cl - mb_targets),
                        ).mean()
                    else:
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
                perm = jax.random.permutation(_rng_e, batch_dim)

                if num_env_agents == 2:
                    batch = (
                        jnp.take(init_h_mb, perm, axis=0)[jnp.newaxis],  # (1, E, hid)
                        jnp.take(traj.ego_obs, perm, axis=1),             # (T, E, H, W, C)
                        jnp.take(traj.done, perm, axis=1),                # (T, E)
                        jnp.take(traj.ego_action, perm, axis=1),         # (T, E)
                        jnp.take(traj.ego_log_prob, perm, axis=1),       # (T, E)
                        jnp.take(traj.critic_value, perm, axis=1),       # (T, E)
                        jnp.take(advantages, perm, axis=1),               # (T, E)
                        jnp.take(targets, perm, axis=1),                  # (T, E)
                    )
                else:
                    # 3+ agent: permute over N*E dimension, include ego mask
                    batch = (
                        jnp.take(init_h_mb, perm, axis=0)[jnp.newaxis],  # (1, N*E, hid)
                        jnp.take(traj.ego_obs, perm, axis=1),             # (T, N*E, ...)
                        jnp.take(traj.done, perm, axis=1),                # (T, N*E)
                        jnp.take(traj.ego_action, perm, axis=1),         # (T, N*E)
                        jnp.take(traj.ego_log_prob, perm, axis=1),       # (T, N*E)
                        jnp.take(traj.critic_value, perm, axis=1),       # (T, N*E)
                        jnp.take(advantages, perm, axis=1),               # (T, N*E)
                        jnp.take(targets, perm, axis=1),                  # (T, N*E)
                        jnp.take(is_ego_mask_T, perm, axis=1),            # (T, N*E) bool
                    )
                minibatches = jax.tree_util.tree_map(_split_mb, batch)

                state, losses = jax.lax.scan(
                    _update_minibatch, state, minibatches
                )
                return (state, rng_e), losses

            rng, _rng = jax.random.split(rng)
            (actor_state, _), all_losses = jax.lax.scan(
                _update_epoch,
                (actor_state, _rng),
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
            metric["actor_loss"] = all_losses[1].mean()
            metric["critic_loss"] = all_losses[2].mean()
            metric["entropy"] = all_losses[3].mean()
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
                actor_state,
                env_state, last_obs, last_done,
                actor_hstate, partner_hstate,
                new_ret, ck_buf, vn_state, update_step, rng,
            )
            return runner_state, metric

        # ---- Initial checkpoint ------------------------------------------
        ck_buf = jax.tree_util.tree_map(
            lambda b, p: b.at[0].set(p), ck_buf, actor_state.params
        )

        # hstate 초기화
        # 2-agent: actor_hstate (E, H), partner_hstate (E, H)
        # 3+ agent: actor_hstate (N*E, H) flat, partner_hstate (N*E, H) flat
        if num_partners == 1:
            # 2-agent path: unchanged
            init_partner_h = init_h
            init_actor_h = init_h
        else:
            # 3+ agent: flat (N*E, H) for both — reset inside _update_step each iteration
            init_actor_h = ActorCriticRNN.initialize_carry(
                num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
            )
            init_partner_h = ActorCriticRNN.initialize_carry(
                num_env_agents * NUM_ENVS, GRU_HIDDEN_DIM
            )

        vn_state = ValueNormState.create()

        init_runner = (
            actor_state,
            env_state, last_obs, last_done,
            init_actor_h, init_partner_h,
            running_returns, ck_buf, vn_state, jnp.int32(0), rng,
        )
        final_runner, metrics = jax.lax.scan(
            _update_step, init_runner, None, NUM_UPDATES
        )

        return {
            "runner_state": final_runner,
            "metrics": metrics,
            "actor_params": final_runner[0].params,
            "actor_ckpts": final_runner[7],
        }

    return train
