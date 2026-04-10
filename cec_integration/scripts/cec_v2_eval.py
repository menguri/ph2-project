"""(a)+(b) End-to-end: drive ph2 overcooked_v2 forced_coord with CEC ckpt.

For each step the v2 env produces v2 state → `CECObsAdapter` builds the
(9,9,26) v1 obs that CEC was trained on → `CECRuntime` samples actions →
v2 env steps. We measure return to confirm the obs adapter produces
sensible inputs (vs the (b) native v1 baseline of mean return ≈ 30).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter import CECObsAdapter

CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ckpts", "forced_coord_9"
)
LAYOUT = "forced_coord"
NUM_STEPS = 400
NUM_TRAJS = 4
SEED = 0


def run_episode(rt0, rt1, env, adapter, rng):
    obs_v2_dict, env_state = env.reset(rng)
    hidden_0 = rt0.init_hidden(env.num_agents)
    hidden_1 = rt1.init_hidden(env.num_agents)
    done_batch = jnp.zeros(env.num_agents, dtype=bool)
    total_reward = 0.0
    for t in range(NUM_STEPS):
        rng, k1, k2, k_env = jax.random.split(rng, 4)
        cec_obs = adapter.get_cec_obs(env_state)  # {'agent_0': (9,9,26), 'agent_1': (9,9,26)}
        obs_batch = jnp.stack([cec_obs["agent_0"].flatten(), cec_obs["agent_1"].flatten()])
        # Use real v1-compatible agent positions for the input tuple shape
        agent_pos = jnp.stack(
            [
                jnp.stack([env_state.agents.pos.x[0], env_state.agents.pos.y[0]]),
                jnp.stack([env_state.agents.pos.x[1], env_state.agents.pos.y[1]]),
            ]
        )
        agent_positions = jnp.stack([agent_pos, agent_pos])
        a0_all, hidden_0, _ = rt0.step(
            obs_batch, hidden_0, done_batch, k1, agent_positions=agent_positions
        )
        a1_all, hidden_1, _ = rt1.step(
            obs_batch, hidden_1, done_batch, k2, agent_positions=agent_positions
        )
        env_act = {"agent_0": int(a0_all[0]), "agent_1": int(a1_all[1])}
        obs_v2_dict, env_state, reward, done, _info = env.step(k_env, env_state, env_act)
        done_batch = jnp.array([done["agent_0"], done["agent_1"]])
        total_reward += float(reward["agent_0"])
        if bool(done["__all__"]):
            break
    return total_reward, t + 1


def main() -> int:
    print(f"[v2] layout={LAYOUT} ckpt_dir={CKPT_DIR}")
    env = jaxmarl.make(
        "overcooked_v2",
        layout=LAYOUT,
        max_steps=NUM_STEPS,
        random_reset=True,
        random_agent_positions=True,
    )
    print(
        f"[v2] env num_agents={env.num_agents} h={env.height} w={env.width} "
        f"action_dim={env.action_space().n}"
    )

    adapter = CECObsAdapter(target_layout="forced_coord_9", max_steps=NUM_STEPS)
    print(f"[v2] adapter: target=({adapter._target_h},{adapter._target_w}) "
          f"padded={adapter._padded_shape}")

    rt0 = CECRuntime(os.path.join(CKPT_DIR, "seed11_ckpt0_improved"))
    rt1 = CECRuntime(os.path.join(CKPT_DIR, "seed11_ckpt1_improved"))
    print(f"[v2] loaded ckpt0 ({rt0.ckpt_format}) ckpt1 ({rt1.ckpt_format})")

    # Sanity: dump obs at reset, check basic invariants.
    rng0 = jax.random.PRNGKey(123)
    _, env_state0 = env.reset(rng0)
    obs0 = adapter.get_cec_obs(env_state0)
    a0_obs = np.asarray(obs0["agent_0"])
    a1_obs = np.asarray(obs0["agent_1"])
    print(f"[v2] obs shapes: a0={a0_obs.shape} a1={a1_obs.shape} dtype={a0_obs.dtype}")
    print(f"[v2] a0 self-pos sum (ch0)={a0_obs[:,:,0].sum()}  "
          f"other-pos sum (ch1)={a0_obs[:,:,1].sum()}  "
          f"wall sum (ch11)={a0_obs[:,:,11].sum()}")
    # Bob mirror check
    assert (a0_obs[:, :, 0] == a1_obs[:, :, 1]).all(), "agent pos channel mirror failed"
    assert (a0_obs[:, :, 1] == a1_obs[:, :, 0]).all(), "agent pos channel mirror failed"
    print("[v2] [OK] obs sanity / agent-pos mirror")

    # Self-play
    print("\n[v2] === self-play ckpt0 vs ckpt0 ===")
    rng = jax.random.PRNGKey(SEED)
    rewards = []
    for i in range(NUM_TRAJS):
        rng, sub = jax.random.split(rng)
        r, t = run_episode(rt0, rt0, env, adapter, sub)
        print(f"  traj {i}: return={r:6.2f} steps={t}")
        rewards.append(r)
    print(f"  mean return = {np.mean(rewards):.2f}")

    # Cross-play
    print("\n[v2] === cross-play ckpt0 vs ckpt1 ===")
    rng = jax.random.PRNGKey(SEED + 1)
    rewards = []
    for i in range(NUM_TRAJS):
        rng, sub = jax.random.split(rng)
        r, t = run_episode(rt0, rt1, env, adapter, sub)
        print(f"  traj {i}: return={r:6.2f} steps={t}")
        rewards.append(r)
    print(f"  mean return = {np.mean(rewards):.2f}")

    print("\n[v2] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
