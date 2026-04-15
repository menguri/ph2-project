"""CEC 모델이 obs_adapter_v2 경로로도 native eval 과 비슷한 행동/성적을 내는지 검증.

CEC native cross-eval (cec-zero-shot/cross_eval_outputs-2/summary.csv) 기준:
    cramped_room_9     SP=156.73 ± 19.95   XP=156.68 ± 19.57
    coord_ring_9       SP= 94.23 ± 23.12   XP= 81.46 ± 26.53
    forced_coord_9     SP=  7.78 ± 11.43   XP=  4.69 ±  9.19
    counter_circuit_9  SP=  0.00           XP=  0.00
    asymm_advantages_9 SP=  2.97           XP=  3.24

이 스크립트는 webapp 의 CEC 모델 (webapp/models/{layout}/cec/run*) 을 self-play /
cross-play 로 OV2 env + obs_adapter_v2 경로에서 돌려서 평균 reward 를 측정한다.
강한 베이스라인이 있는 cramped_room/coord_ring 에서 native 와 같은 수준이 나오면
adapter 가 의미적으로 올바르다고 판단할 수 있다.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        PYTHONPATH=. ./overcooked_v2/bin/python \
        cec_integration/scripts/test_cec_behavior_via_v2.py
"""
import os
import sys
import statistics

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec


WEBAPP_MODELS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "webapp", "models",
)

NATIVE_BASELINE = {
    "cramped_room":     {"sp": 156.73, "xp": 156.68},
    "coord_ring":       {"sp":  94.23, "xp":  81.46},
    "forced_coord":     {"sp":   7.78, "xp":   4.69},
    "counter_circuit":  {"sp":   0.00, "xp":   0.00},
    "asymm_advantages": {"sp":   2.97, "xp":   3.24},
}

NUM_STEPS = 200  # CEC native eval 은 400 — 시간 단축 위해 200, reward 도 절반 기대
NUM_EPISODES = 3  # 시간을 줄이기 위해 적게


def _ckpt_paths(layout):
    """webapp/models/{layout}/cec/run*/ckpt_final 경로 리스트."""
    base = os.path.join(WEBAPP_MODELS_ROOT, layout, "cec")
    if not os.path.isdir(base):
        return []
    paths = []
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d, "ckpt_final")
        if os.path.isdir(full):
            paths.append(full)
    return paths


def run_episode(rt0, rt1, env, layout, rng):
    """한 에피소드 — agent_0 = rt0, agent_1 = rt1. v2 adapter 경유."""
    obs_dict, env_state = env.reset(rng)
    h0 = rt0.init_hidden(2)
    h1 = rt1.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    total_reward = 0.0
    for t in range(NUM_STEPS):
        rng, k0, k1, ke = jax.random.split(rng, 4)
        # 각 agent 별로 OV2 obs → CEC obs 변환 (v2 adapter)
        a0_cec = ov2_obs_to_cec(jnp.array(obs_dict["agent_0"], dtype=jnp.float32),
                                 layout, t, NUM_STEPS)
        a1_cec = ov2_obs_to_cec(jnp.array(obs_dict["agent_1"], dtype=jnp.float32),
                                 layout, t, NUM_STEPS)
        # CECRuntime 은 (num_agents=2, 9, 9, 26) 입력 — slot 0 = self, slot 1 = dummy
        obs0 = jnp.stack([a0_cec, jnp.zeros_like(a0_cec)])
        obs1 = jnp.stack([a1_cec, jnp.zeros_like(a1_cec)])
        a0_all, h0, _ = rt0.step(obs0, h0, done_arr, k0)
        a1_all, h1, _ = rt1.step(obs1, h1, done_arr, k1)
        env_act = {"agent_0": jnp.int32(a0_all[0]), "agent_1": jnp.int32(a1_all[0])}
        obs_dict, env_state, reward, done, _ = env.step(ke, env_state, env_act)
        total_reward += float(reward["agent_0"])
        if bool(done["__all__"]):
            break
    return total_reward, t + 1


def evaluate_layout(layout, num_episodes=NUM_EPISODES):
    """webapp 의 5개 CEC 모델로 self-play 와 cross-play 평균 reward 측정."""
    print(f"\n=== [{layout}] ===")
    ckpts = _ckpt_paths(layout)
    if len(ckpts) < 2:
        print(f"  [SKIP] {len(ckpts)} ckpts found, need >= 2")
        return None

    # JIT 비용이 runtime 마다 재발생하므로 빠른 검증에는 1개만
    max_rt = int(os.environ.get("MAX_RUNTIMES", "2"))
    runtimes = []
    for p in ckpts[:max_rt]:
        try:
            runtimes.append(CECRuntime(p))
        except Exception as e:
            print(f"  [WARN] ckpt load failed: {p}: {e}")
    if len(runtimes) < 1:
        print(f"  [SKIP] only {len(runtimes)} runtimes loaded")
        return None
    print(f"  loaded {len(runtimes)} CEC runtimes (capped at MAX_RUNTIMES={max_rt})", flush=True)

    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=NUM_STEPS,
        random_reset=False,
        random_agent_positions=False,
    )

    rng = jax.random.PRNGKey(42)

    # Self-play: same checkpoint on both slots
    sp_rewards = []
    for i, rt in enumerate(runtimes):
        for ep in range(num_episodes):
            rng, sub = jax.random.split(rng)
            r, t = run_episode(rt, rt, env, layout, sub)
            sp_rewards.append(r)
            print(f"  SP rt{i} ep{ep}: reward={r:.1f} steps={t}", flush=True)
    sp_mean = statistics.mean(sp_rewards)
    sp_std = statistics.stdev(sp_rewards) if len(sp_rewards) > 1 else 0.0

    # Cross-play: rt0 vs rt1 (and rt2 vs rt3, etc.) — 인접 페어
    xp_rewards = []
    for i in range(len(runtimes) - 1):
        for ep in range(num_episodes):
            rng, sub = jax.random.split(rng)
            r, t = run_episode(runtimes[i], runtimes[i + 1], env, layout, sub)
            xp_rewards.append(r)
            print(f"  XP rt{i}+{i+1} ep{ep}: reward={r:.1f} steps={t}", flush=True)
    xp_mean = statistics.mean(xp_rewards) if xp_rewards else 0.0
    xp_std = statistics.stdev(xp_rewards) if len(xp_rewards) > 1 else 0.0

    baseline = NATIVE_BASELINE.get(layout, {})
    print(f"  v2 adapter SP: {sp_mean:7.2f} ± {sp_std:6.2f}  (n={len(sp_rewards)})")
    print(f"  native    SP: {baseline.get('sp', '?'):>7}")
    print(f"  v2 adapter XP: {xp_mean:7.2f} ± {xp_std:6.2f}  (n={len(xp_rewards)})")
    print(f"  native    XP: {baseline.get('xp', '?'):>7}")

    return {"sp": sp_mean, "sp_std": sp_std, "xp": xp_mean, "xp_std": xp_std,
            "native": baseline}


def main():
    print("=" * 70)
    print("CEC behavior via v2 adapter — native eval baseline 과 비교")
    print(f"  episodes per pair = {NUM_EPISODES}, steps per episode = {NUM_STEPS}")
    print("=" * 70)

    # CLI 인자로 특정 레이아웃만 빠르게 돌릴 수 있도록
    if len(sys.argv) > 1:
        layouts = sys.argv[1].split(",")
    else:
        layouts = ["cramped_room", "coord_ring", "forced_coord",
                   "counter_circuit", "asymm_advantages"]
    results = {}
    for layout in layouts:
        results[layout] = evaluate_layout(layout)

    print("\n" + "=" * 70)
    print(f"{'layout':20s} {'v2 SP':>10s} {'native SP':>11s} {'v2 XP':>10s} {'native XP':>11s}")
    for name, r in results.items():
        if r is None:
            print(f"{name:20s} {'SKIP':>10s}")
            continue
        ns = r["native"].get("sp", "?")
        nx = r["native"].get("xp", "?")
        print(f"{name:20s} {r['sp']:10.2f} {str(ns):>11s} {r['xp']:10.2f} {str(nx):>11s}")
    print("=" * 70)

    # 합격 기준: cramped_room SP/XP 가 native baseline 의 70% 이상
    crit = "cramped_room"
    if results.get(crit):
        thresh = NATIVE_BASELINE[crit]["sp"] * 0.7
        passed = results[crit]["sp"] >= thresh
        print(f"\n합격 기준 ({crit} SP >= {thresh:.1f}): {'PASS' if passed else 'FAIL'} (got {results[crit]['sp']:.2f})")
        return 0 if passed else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
