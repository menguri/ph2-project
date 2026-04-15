"""같은 CEC 모델을 (a) native v1 Overcooked + CEC_LAYOUTS (b) OV2 + v2 adapter 양쪽에서
돌려 reward 를 비교. adapter 의 행동 충실도를 직접 검증.

(a) 가 reward > 0 인데 (b) 가 0 이면 adapter 문제.
(a) 도 0 이면 모델 자체가 약함 (cross-eval baseline 을 webapp 이 만족 못 함).

Run (GPU 0):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/diagnose_cec_native_vs_v2.py
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
LAYOUT_OV2 = "cramped_room"
LAYOUT_CEC = "cramped_room_9"
NUM_STEPS = 400
NUM_EPISODES = 3


def run_native(rt, rng):
    """ph2 의 v1 Overcooked + CEC_LAYOUTS 에서 self-play rollout."""
    env = V1Overcooked(layout=CEC_LAYOUTS[LAYOUT_CEC], random_reset=False,
                       max_steps=NUM_STEPS)
    obs_dict, env_state = env.reset(rng)
    h = rt.init_hidden(env.num_agents)
    done = jnp.zeros(env.num_agents, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k1, ke = jax.random.split(rng, 3)
        obs_batch = jnp.stack([obs_dict[a].flatten() for a in env.agents])
        # agent_positions dummy (graph_net=False 인 경우)
        actions, h, _ = rt.step(obs_batch, h, done, k1)
        env_act = {env.agents[0]: int(actions[0]), env.agents[1]: int(actions[1])}
        obs_dict, env_state, reward, done_dict, _ = env.step(ke, env_state, env_act)
        done = jnp.array([done_dict[a] for a in env.agents])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total, t + 1


def run_v2(rt, rng, agent_swap=False, random_dir=False):
    """OV2 + v2 adapter 에서 self-play rollout.

    agent_swap=True: OV2 `swap_agents=True` 레이아웃(cramped_room, counter_circuit) 에서
    OV2 의 agent_0/1 순서를 뒤집어 CEC slot 에 매핑한다. CEC 는 native env (overcooked_ai /
    V1 Overcooked) 기준 slot 순서로 훈련되어 있어, OV2 의 swapped 순서를 그대로 넘기면
    slot 별 LSTM hidden state 와 obs 의 자세 매칭이 어긋난다.
    """
    env = jaxmarl.make("overcooked_v2", layout=LAYOUT_OV2, max_steps=NUM_STEPS,
                       random_reset=False, random_agent_positions=random_dir)
    obs_dict, env_state = env.reset(rng)
    h = rt.init_hidden(2)
    done = jnp.zeros((2,), dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k1, ke = jax.random.split(rng, 3)
        a0 = ov2_obs_to_cec(jnp.array(obs_dict["agent_0"], dtype=jnp.float32),
                             LAYOUT_OV2, t, NUM_STEPS)
        a1 = ov2_obs_to_cec(jnp.array(obs_dict["agent_1"], dtype=jnp.float32),
                             LAYOUT_OV2, t, NUM_STEPS)
        if agent_swap:
            obs_batch = jnp.stack([a1, a0])  # OV2 agent_1 → CEC slot 0, agent_0 → slot 1
        else:
            obs_batch = jnp.stack([a0, a1])
        actions, h, _ = rt.step(obs_batch, h, done, k1)
        if agent_swap:
            # CEC slot 0 action → OV2 agent_1, CEC slot 1 action → OV2 agent_0
            env_act = {"agent_0": jnp.int32(actions[1]), "agent_1": jnp.int32(actions[0])}
        else:
            env_act = {"agent_0": jnp.int32(actions[0]), "agent_1": jnp.int32(actions[1])}
        obs_dict, env_state, reward, done_dict, _ = env.step(ke, env_state, env_act)
        done = jnp.array([done_dict["agent_0"], done_dict["agent_1"]])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total, t + 1


def main():
    print("=" * 70)
    print(f"ckpt: {CKPT_PATH}")
    print(f"native layout: {LAYOUT_CEC}   v2 layout: {LAYOUT_OV2}")
    print(f"num_steps={NUM_STEPS} num_episodes={NUM_EPISODES}")
    print("=" * 70, flush=True)

    rt = CECRuntime(CKPT_PATH)
    print(f"loaded runtime (ckpt_format={rt.ckpt_format})", flush=True)
    # config hint
    cfg = rt.config
    print(f"config hints: ENV_NAME={cfg.get('ENV_NAME')} layout={cfg.get('ENV_KWARGS',{}).get('layout')} "
          f"LSTM={cfg.get('LSTM')} GRAPH_NET={cfg.get('GRAPH_NET')} "
          f"random_reset={cfg.get('ENV_KWARGS',{}).get('random_reset')}", flush=True)

    rng = jax.random.PRNGKey(0)
    print("\n--- NATIVE (v1 Overcooked + CEC_LAYOUTS[cramped_room_9]) self-play ---", flush=True)
    native_rewards = []
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, t = run_native(rt, sub)
        print(f"  native ep{ep}: reward={r:.1f} steps={t}", flush=True)
        native_rewards.append(r)
    print(f"  native mean = {np.mean(native_rewards):.2f}", flush=True)

    print("\n--- V2 ADAPTER (no swap) self-play ---", flush=True)
    v2_rewards = []
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, t = run_v2(rt, sub, agent_swap=False)
        print(f"  v2 ep{ep}: reward={r:.1f} steps={t}", flush=True)
        v2_rewards.append(r)
    print(f"  v2 mean = {np.mean(v2_rewards):.2f}", flush=True)

    print("\n--- V2 ADAPTER (agent_swap=True) self-play ---", flush=True)
    v2s_rewards = []
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, t = run_v2(rt, sub, agent_swap=True)
        print(f"  v2s ep{ep}: reward={r:.1f} steps={t}", flush=True)
        v2s_rewards.append(r)
    print(f"  v2+swap mean = {np.mean(v2s_rewards):.2f}", flush=True)

    print("\n--- V2 ADAPTER (random_dir=True) self-play ---", flush=True)
    v2r_rewards = []
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, t = run_v2(rt, sub, agent_swap=False, random_dir=True)
        print(f"  v2r ep{ep}: reward={r:.1f} steps={t}", flush=True)
        v2r_rewards.append(r)
    print(f"  v2+randdir mean = {np.mean(v2r_rewards):.2f}", flush=True)

    print("\n--- V2 ADAPTER (random_dir=True AND agent_swap=True) self-play ---", flush=True)
    v2rs_rewards = []
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, t = run_v2(rt, sub, agent_swap=True, random_dir=True)
        print(f"  v2rs ep{ep}: reward={r:.1f} steps={t}", flush=True)
        v2rs_rewards.append(r)
    print(f"  v2+rand+swap mean = {np.mean(v2rs_rewards):.2f}", flush=True)

    print("\n" + "=" * 70)
    print(f"NATIVE              : {np.mean(native_rewards):7.2f} ± {np.std(native_rewards):5.2f}")
    print(f"V2 ADP              : {np.mean(v2_rewards):7.2f} ± {np.std(v2_rewards):5.2f}")
    print(f"V2 ADP +swap        : {np.mean(v2s_rewards):7.2f} ± {np.std(v2s_rewards):5.2f}")
    print(f"V2 ADP +randdir     : {np.mean(v2r_rewards):7.2f} ± {np.std(v2r_rewards):5.2f}")
    print(f"V2 ADP +swap+randdir: {np.mean(v2rs_rewards):7.2f} ± {np.std(v2rs_rewards):5.2f}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
