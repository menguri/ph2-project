"""native vs v2 adapter 경로에서 CEC 가 각 step 에서 내는 action 비교.

같은 CEC 모델, 같은 초기 seed, 각 env step 을 동기화하여:
  1. 양 env 에서 obs 추출
  2. CEC 가 양쪽 obs 에 대해 고른 action 기록
  3. action 이 달라지는 첫 step 을 찾아서 그 시점의 obs 를 채널별 diff

의도: reward=0 이 adapter 의 어떤 채널/인코딩 차이 때문인지 좁히기.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl
from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec


CKPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final",
)
NUM_STEPS = 60  # 짧게만


CH_NAMES = [
    "self_pos","other_pos","self_E","self_S","self_W","self_N",
    "other_E","other_S","other_W","other_N",
    "pot","wall","onion_pile","tomato_pile","plate_pile","goal",
    "ons_in_pot","tom_in_pot","ons_in_soup","tom_in_soup",
    "cook_time","soup_ready","plate_on_grid","onion_on_grid","tomato_on_grid","urgency",
]


def main():
    print("loading runtime...", flush=True)
    rt = CECRuntime(CKPT_PATH)

    # Native env
    native_env = V1Overcooked(layout=CEC_LAYOUTS["cramped_room_9"], random_reset=False,
                              max_steps=NUM_STEPS)
    nat_obs, nat_state = native_env.reset(jax.random.PRNGKey(0))
    nat_h = rt.init_hidden(2)

    # V2 env
    v2_env = jaxmarl.make("overcooked_v2", layout="cramped_room", max_steps=NUM_STEPS,
                          random_reset=False, random_agent_positions=False)
    v2_obs, v2_state = v2_env.reset(jax.random.PRNGKey(0))
    v2_h = rt.init_hidden(2)

    done = jnp.zeros((2,), dtype=bool)
    rng_nat = jax.random.PRNGKey(100)
    rng_v2 = jax.random.PRNGKey(100)

    print(f"{'step':>4s} {'nat_acts':>10s} {'v2_acts':>10s} {'nat_rew':>8s} {'v2_rew':>8s}  status", flush=True)

    first_diff = None
    total_nat = 0.0
    total_v2 = 0.0
    for t in range(NUM_STEPS):
        # Native obs
        nat_batch = jnp.stack([nat_obs[a].flatten() for a in native_env.agents])
        rng_nat, k1, ke_n = jax.random.split(rng_nat, 3)
        nat_actions, nat_h, _ = rt.step(nat_batch, nat_h, done, k1)

        # V2 obs via adapter
        a0 = ov2_obs_to_cec(jnp.array(v2_obs["agent_0"], dtype=jnp.float32),
                             "cramped_room", t, NUM_STEPS)
        a1 = ov2_obs_to_cec(jnp.array(v2_obs["agent_1"], dtype=jnp.float32),
                             "cramped_room", t, NUM_STEPS)
        v2_batch = jnp.stack([a0, a1])
        rng_v2, k2, ke_v = jax.random.split(rng_v2, 3)
        v2_actions, v2_h, _ = rt.step(v2_batch, v2_h, done, k2)

        nat_a = (int(nat_actions[0]), int(nat_actions[1]))
        v2_a = (int(v2_actions[0]), int(v2_actions[1]))

        # Step envs
        nat_env_act = {native_env.agents[0]: nat_a[0], native_env.agents[1]: nat_a[1]}
        nat_obs, nat_state, nat_rew, nat_done, _ = native_env.step(ke_n, nat_state, nat_env_act)
        v2_env_act = {"agent_0": jnp.int32(v2_a[0]), "agent_1": jnp.int32(v2_a[1])}
        v2_obs, v2_state, v2_rew, v2_done, _ = v2_env.step(ke_v, v2_state, v2_env_act)

        total_nat += float(nat_rew["agent_0"])
        total_v2 += float(v2_rew["agent_0"])

        status = "match" if nat_a == v2_a else "DIFFER"
        if nat_a != v2_a and first_diff is None:
            first_diff = t
        print(f"{t:4d} {str(nat_a):>10s} {str(v2_a):>10s} {float(nat_rew['agent_0']):8.1f} "
              f"{float(v2_rew['agent_0']):8.1f}  {status}", flush=True)

    print(f"\ntotal native={total_nat} v2={total_v2}", flush=True)
    if first_diff is not None:
        print(f"first action diff at step={first_diff}", flush=True)

    # 마지막 step 의 obs 채널 diff
    # (양쪽 obs 를 다시 생성 — 이미 step 했으므로 마지막 입력 obs 는 손실)
    # 대신 원점 (t=0) obs 비교: 동일 seed 에서 양 env 의 CEC obs 가 채널별로 어떻게 다른지.
    print("\n--- reset 직후 obs 채널 diff (agent_0 입장) ---", flush=True)
    nat_env2 = V1Overcooked(layout=CEC_LAYOUTS["cramped_room_9"], random_reset=False,
                             max_steps=NUM_STEPS)
    nat_obs2, _ = nat_env2.reset(jax.random.PRNGKey(0))
    v2_env2 = jaxmarl.make("overcooked_v2", layout="cramped_room", max_steps=NUM_STEPS,
                            random_reset=False, random_agent_positions=False)
    v2_obs2, _ = v2_env2.reset(jax.random.PRNGKey(0))
    nat_a0 = np.asarray(nat_obs2["agent_0"])
    v2_a0 = np.asarray(ov2_obs_to_cec(jnp.array(v2_obs2["agent_0"], dtype=jnp.float32),
                                        "cramped_room", 0, NUM_STEPS))
    for ch in range(26):
        d = np.abs(nat_a0[:,:,ch] - v2_a0[:,:,ch])
        if d.max() > 0.01:
            coords = np.argwhere(d > 0.01)
            print(f"  ch{ch:2d} {CH_NAMES[ch]:18s} max={d.max():.2f} n_diff={len(coords)} "
                  f"e.g. {tuple(coords[0])} nat={nat_a0[tuple(coords[0])][ch]:.1f} v2={v2_a0[tuple(coords[0])][ch]:.1f}", flush=True)


if __name__ == "__main__":
    main()
