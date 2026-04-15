"""V1 native env 에서 agent_0/1 의 시작 위치만 swap 해서 CEC 가 여전히 213 을 내는지 확인.

CEC_LAYOUTS['cramped_room_9'] 원래 agent_idx = [10, 12] 인데, [12, 10] 으로 바꿔 agent_0 이
(1,3) 에서 시작, agent_1 이 (1,1) 에서 시작. webapp (overcooked-ai) 의 agent 배치와 동일.

만약 결과가 여전히 ~213 이면: CEC 는 agent slot 배치에 무관. webapp 의 다른 이슈가 문제.
만약 결과가 0 이면: CEC 가 특정 agent_0 위치에 brittle. slot swap 이 필요.
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import jax, jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS


CKPT = os.path.join(PROJECT_ROOT, "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final")
NUM_STEPS = 400
NUM_EPS = 3


def run_ep(rt, env, rng):
    obs, state = env.reset(rng)
    h = rt.init_hidden(2)
    d = jnp.zeros((2,), dtype=jnp.bool_)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k1, ke = jax.random.split(rng, 3)
        batch = jnp.stack([obs[a].flatten() for a in env.agents])
        act, h, _ = rt.step(batch, h, d, k1)
        env_act = {env.agents[0]: int(act[0]), env.agents[1]: int(act[1])}
        obs, state, r, dn, _ = env.step(ke, state, env_act)
        d = jnp.array([dn[a] for a in env.agents])
        total += float(r["agent_0"])
        if bool(dn["__all__"]):
            break
    return total


def main():
    rt = CECRuntime(CKPT)
    print("loaded CEC runtime", flush=True)

    # Original layout (agent_0 at (1,1))
    layout_orig = CEC_LAYOUTS["cramped_room_9"]
    env_orig = V1Overcooked(layout=layout_orig, random_reset=False, max_steps=NUM_STEPS)

    # Swapped: agent_0 at (1,3), agent_1 at (1,1)
    orig_agent_idx = list(map(int, layout_orig["agent_idx"]))
    print(f"original agent_idx = {orig_agent_idx}", flush=True)
    swapped_idx = [orig_agent_idx[1], orig_agent_idx[0]]
    print(f"swapped agent_idx = {swapped_idx}", flush=True)
    layout_swap = FrozenDict({**dict(layout_orig), "agent_idx": jnp.array(swapped_idx, dtype=jnp.int32)})
    env_swap = V1Overcooked(layout=layout_swap, random_reset=False, max_steps=NUM_STEPS)

    rng = jax.random.PRNGKey(42)

    print("\n--- V1 native original agent_idx ---", flush=True)
    orig_rewards = []
    for ep in range(NUM_EPS):
        rng, sub = jax.random.split(rng)
        r = run_ep(rt, env_orig, sub)
        print(f"  ep{ep}: {r:.1f}", flush=True)
        orig_rewards.append(r)
    print(f"  mean = {np.mean(orig_rewards):.2f}", flush=True)

    print("\n--- V1 native swapped agent_idx ---", flush=True)
    swap_rewards = []
    for ep in range(NUM_EPS):
        rng, sub = jax.random.split(rng)
        r = run_ep(rt, env_swap, sub)
        print(f"  ep{ep}: {r:.1f}", flush=True)
        swap_rewards.append(r)
    print(f"  mean = {np.mean(swap_rewards):.2f}", flush=True)

    print("\n=====================================")
    print(f"original agent_idx: {np.mean(orig_rewards):.2f}")
    print(f"swapped  agent_idx: {np.mean(swap_rewards):.2f}")
    print("=====================================")


if __name__ == "__main__":
    main()
