"""(b) Native v1 env evaluation inside ph2-project.

ph2-project ships its own legacy `jaxmarl.environments.overcooked` (same
schema as the version CEC was trained on, but without the 9x9 padded layout
factories). We supply the pre-computed CEC `forced_coord_9` layout dict from
`cec_integration.cec_layouts`, build a 9x9 v1 `Overcooked` env, and run the
ckpt via `CECRuntime` for a few full episodes.

Run with the ph2 venv:

    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. ./overcooked_v2/bin/python \
        cec_integration/scripts/cec_native_eval.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS

CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ckpts", "forced_coord_9"
)
LAYOUT = "forced_coord_9"
NUM_STEPS = 400  # CEC max_steps default for episode
NUM_TRAJS = 4
SEED = 0


def run_episode(rt0: CECRuntime, rt1: CECRuntime, env, rng):
    obs, env_state = env.reset(rng)
    hidden_0 = rt0.init_hidden(env.num_agents)
    hidden_1 = rt1.init_hidden(env.num_agents)
    done_batch = jnp.zeros(env.num_agents, dtype=bool)
    total_reward = 0.0
    for t in range(NUM_STEPS):
        rng, k1, k2, k_env = jax.random.split(rng, 4)
        # Replicate test_general.get_rollouts shapes exactly.
        obs_batch = jnp.stack([obs[a].flatten() for a in env.agents])
        agent_positions = jnp.stack([env_state.agent_pos for _ in env.agents])
        # CECRuntime.step expects (num_agents, *obs_dim)-ish; flat is fine.
        a0_all, hidden_0, _ = rt0.step(
            obs_batch, hidden_0, done_batch, k1, agent_positions=agent_positions
        )
        a1_all, hidden_1, _ = rt1.step(
            obs_batch, hidden_1, done_batch, k2, agent_positions=agent_positions
        )
        # Each runtime returns actions for BOTH slots; we use slot 0 for agent_0 and slot 1 for agent_1.
        env_act = {env.agents[0]: int(a0_all[0]), env.agents[1]: int(a1_all[1])}
        obs, env_state, reward, done, _info = env.step(k_env, env_state, env_act)
        done_batch = jnp.array([done[a] for a in env.agents])
        total_reward += float(reward["agent_0"])
        if bool(done["__all__"]):
            break
    return total_reward, t + 1


def main() -> int:
    print(f"[native] layout={LAYOUT} ckpt_dir={CKPT_DIR}")
    layout_dict = CEC_LAYOUTS[LAYOUT]
    env = jaxmarl.make(
        "overcooked",
        layout=layout_dict,
        random_reset=True,
        max_steps=NUM_STEPS,
    )
    print(
        f"[native] env num_agents={env.num_agents} "
        f"obs_shape={env.observation_space().shape} "
        f"action_dim={env.action_space().n}"
    )

    rt_ckpt0 = CECRuntime(os.path.join(CKPT_DIR, "seed11_ckpt0_improved"))
    rt_ckpt1 = CECRuntime(os.path.join(CKPT_DIR, "seed11_ckpt1_improved"))
    print(f"[native] loaded ckpt0 ({rt_ckpt0.ckpt_format}) ckpt1 ({rt_ckpt1.ckpt_format})")

    # Self-play: ckpt0 vs ckpt0
    print("\n[native] === self-play ckpt0 vs ckpt0 ===")
    rewards = []
    rng = jax.random.PRNGKey(SEED)
    for i in range(NUM_TRAJS):
        rng, sub = jax.random.split(rng)
        r, t = run_episode(rt_ckpt0, rt_ckpt0, env, sub)
        print(f"  traj {i}: return={r:6.2f} steps={t}")
        rewards.append(r)
    print(f"  mean return = {np.mean(rewards):.2f}")

    # Cross-play: ckpt0 vs ckpt1
    print("\n[native] === cross-play ckpt0 vs ckpt1 ===")
    rewards = []
    rng = jax.random.PRNGKey(SEED + 1)
    for i in range(NUM_TRAJS):
        rng, sub = jax.random.split(rng)
        r, t = run_episode(rt_ckpt0, rt_ckpt1, env, sub)
        print(f"  traj {i}: return={r:6.2f} steps={t}")
        rewards.append(r)
    print(f"  mean return = {np.mean(rewards):.2f}")

    print("\n[native] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
