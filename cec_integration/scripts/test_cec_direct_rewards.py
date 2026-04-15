"""OvercookedAIToCECAdapter 를 사용한 CEC self-play reward 측정.

overcooked-ai state → OvercookedAIToCECAdapter → V1 State → V1 env.get_obs → CEC.
OV2 포맷 경유하는 기존 `test_cec_webapp_rewards.py` 와 달리 V1 native obs 를 직접 제공.

비교 baseline:
- V1 native (CEC 훈련 env): 220
- cross-eval baseline: 156

Run:
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/test_cec_direct_rewards.py [layout]
"""
import os
import sys
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from app.game.engine import _load_custom_layout
from app.game.action_map import jaxmarl_to_overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_from_ai import OvercookedAIToCECAdapter


WEBAPP_MODELS_ROOT = os.path.join(PROJECT_ROOT, "webapp", "models")
NUM_STEPS = 400
NUM_EPISODES = 3

NATIVE_BASELINE = {
    "cramped_room":     156.73,
    "coord_ring":        94.23,
    "forced_coord":       7.78,
    "counter_circuit":    0.00,
    "asymm_advantages":   2.97,
}


def _auto_cook(state, mdp):
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup" and not obj.is_cooking and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()


def run_episode(rt, adapter, mdp, env, rng):
    """self-play: 한 CEC 런타임이 양 agent 제어."""
    env.reset()
    state = env.state
    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    total = 0.0
    for t in range(NUM_STEPS):
        # 양 agent obs 를 새 adapter 로 생성
        obs_dict = adapter.get_cec_obs_both(state, mdp, current_step=t)
        obs_batch = jnp.stack([obs_dict["agent_0"], obs_dict["agent_1"]])

        rng, sub = jax.random.split(rng)
        actions, hidden, _ = rt.step(obs_batch, hidden, done_arr, sub)
        a0, a1 = int(actions[0]), int(actions[1])

        joint = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, reward, done, _info = env.step(joint)
        _auto_cook(next_state, mdp)
        total += float(reward)
        state = next_state
        if done:
            break
    return total, t + 1


def evaluate_layout(layout, num_episodes=NUM_EPISODES):
    print(f"\n=== [{layout}] ===", flush=True)
    ckpt = os.path.join(WEBAPP_MODELS_ROOT, layout, "cec", "run0", "ckpt_final")
    if not os.path.isdir(ckpt):
        print(f"  [SKIP] ckpt missing: {ckpt}", flush=True)
        return None

    rt = CECRuntime(ckpt)
    print(f"  loaded runtime (format={rt.ckpt_format})", flush=True)

    adapter = OvercookedAIToCECAdapter(target_layout=layout, max_steps=NUM_STEPS)
    mdp = _load_custom_layout(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    print(f"  env + adapter ready", flush=True)

    rewards = []
    rng = jax.random.PRNGKey(42)
    for ep in range(num_episodes):
        rng, sub = jax.random.split(rng)
        r, steps = run_episode(rt, adapter, mdp, env, sub)
        print(f"  ep{ep}: reward={r:.1f} steps={steps}", flush=True)
        rewards.append(r)
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    baseline = NATIVE_BASELINE.get(layout, 0.0)
    ratio = mean / baseline if baseline > 0 else float("inf")
    ratio_str = f"{ratio:.0%}" if baseline > 0 else "n/a"
    print(f"  direct-adapter mean = {mean:.2f} ± {std:.2f}   "
          f"(native baseline={baseline:.2f}, ratio={ratio_str})", flush=True)
    return {"mean": mean, "std": std, "native": baseline, "ratio": ratio}


def main():
    print("=" * 70)
    print("CEC self-play via OvercookedAIToCECAdapter (overcooked-ai → V1 직접)")
    print(f"  {NUM_EPISODES} episodes × {NUM_STEPS} steps")
    print("=" * 70, flush=True)

    if len(sys.argv) > 1:
        layouts = sys.argv[1].split(",")
    else:
        layouts = ["cramped_room", "coord_ring", "forced_coord",
                   "counter_circuit", "asymm_advantages"]

    results = {}
    for layout in layouts:
        results[layout] = evaluate_layout(layout)

    print("\n" + "=" * 70)
    print(f"{'layout':20s} {'direct-adp':>14s} {'native':>10s} {'ratio':>8s}")
    for name, r in results.items():
        if r is None:
            print(f"{name:20s} SKIP")
            continue
        ratio_str = f"{r['ratio']:.0%}" if r["native"] > 0 else "n/a"
        print(f"{name:20s} {r['mean']:10.2f}±{r['std']:4.1f} "
              f"{r['native']:10.2f} {ratio_str:>8s}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
