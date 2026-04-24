"""V1 Overcooked engine 에서 CEC × CEC self-play — 4 layout reward 측정.

A1 핵심 검증: OV2 engine 을 V1 engine (CEC 학습 env) 으로 교체하면 CEC 가 native 수준 reward 를
회복하는지. `diagnose_cec_native_vs_v2.py` 의 NATIVE 경로를 4 layout 으로 확장한 버전.

Run (GPU 0):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/eval_cec_v1_engine_selfplay.py
"""
import os
import sys

sys.path.insert(0, "/home/mlic/mingukang/ph2-project")

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS


LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "counter_circuit"]
NUM_STEPS = 400
NUM_EPISODES = 3
WEBAPP_MODELS = "/home/mlic/mingukang/ph2-project/webapp/models"


def run_selfplay(rt, layout, rng):
    env = V1Overcooked(layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False, max_steps=NUM_STEPS)
    obs_dict, env_state = env.reset(rng)
    h = rt.init_hidden(env.num_agents)
    done = jnp.zeros(env.num_agents, dtype=bool)
    total = 0.0
    for t in range(NUM_STEPS):
        rng, k1, ke = jax.random.split(rng, 3)
        obs_batch = jnp.stack([obs_dict[a].flatten() for a in env.agents])
        actions, h, _ = rt.step(obs_batch, h, done, k1)
        env_act = {env.agents[i]: int(actions[i]) for i in range(env.num_agents)}
        obs_dict, env_state, reward, done_dict, _ = env.step(ke, env_state, env_act)
        done = jnp.array([done_dict[a] for a in env.agents])
        total += float(reward["agent_0"])
        if bool(done_dict["__all__"]):
            break
    return total, t + 1


def evaluate(layout):
    ckpt = os.path.join(WEBAPP_MODELS, layout, "cec", "run0", "ckpt_final")
    if not os.path.isdir(ckpt):
        print(f"[{layout}] SKIP ckpt missing: {ckpt}", flush=True)
        return None
    print(f"\n=== {layout} === (ckpt=run0)", flush=True)
    try:
        rt = CECRuntime(ckpt)
    except Exception as e:
        print(f"  [{layout}] ckpt load failed: {e}", flush=True)
        return None
    print(f"  loaded runtime", flush=True)

    rewards = []
    rng = jax.random.PRNGKey(42)
    for ep in range(NUM_EPISODES):
        rng, sub = jax.random.split(rng)
        r, steps = run_selfplay(rt, layout, sub)
        print(f"  ep{ep}: reward={r:.1f} steps={steps}", flush=True)
        rewards.append(r)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"  [{layout}] mean={mean:.2f} std={std:.2f}", flush=True)
    return mean, std


def main():
    print("=" * 72, flush=True)
    print("CEC × CEC self-play on V1 Overcooked engine + CEC_LAYOUTS (4 layouts)", flush=True)
    print(f"{NUM_EPISODES} episodes × {NUM_STEPS} steps", flush=True)
    print("=" * 72, flush=True)

    results = {}
    for layout in LAYOUTS:
        try:
            results[layout] = evaluate(layout)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[layout] = None

    print("\n" + "=" * 72)
    print(f"{'layout':20s} {'mean reward':>14s} {'std':>8s}")
    print("=" * 72)
    for name, r in results.items():
        if r is None:
            print(f"{name:20s} {'SKIP':>14s}")
        else:
            mean, std = r
            print(f"{name:20s} {mean:14.2f} {std:8.2f}")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main() or 0)
