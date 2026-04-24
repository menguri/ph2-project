"""V1StateToOV2ObsAdapter 의 JIT 버전이 기존 numpy 버전과 byte-exact 일치하는지 검증.

5개 layout × 20 step random rollout 으로 get_ov2_obs vs get_ov2_obs_jit 비교.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked

from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter


LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "counter_circuit",
           "asymm_advantages"]
NUM_STEPS = 20
MAX_STEPS = 400


def compare_layout(layout: str) -> bool:
    print(f"\n── {layout} ──")
    try:
        env = V1Overcooked(
            layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False,
            max_steps=MAX_STEPS,
        )
    except KeyError:
        print(f"  [SKIP] CEC_LAYOUTS[{layout}_9] 없음")
        return True

    adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=MAX_STEPS)
    get_obs_jit = jax.jit(adapter.get_ov2_obs_jit)

    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    _, state = env.reset(k)
    ok = True
    max_diff_seen = 0.0
    for t in range(NUM_STEPS):
        old = adapter.get_ov2_obs(state, current_step=t)
        new = get_obs_jit(state, t)
        for agent in ("agent_0", "agent_1"):
            o = np.asarray(old[agent])
            n = np.asarray(new[agent])
            if o.shape != n.shape:
                print(f"  [FAIL] t={t} {agent} shape diff {o.shape} vs {n.shape}")
                ok = False
                break
            if not np.array_equal(o, n):
                diff = np.max(np.abs(o.astype(np.int32) - n.astype(np.int32)))
                max_diff_seen = max(max_diff_seen, float(diff))
                # find first differing channel
                diff_mask = (o != n).any(axis=(0, 1))
                diff_chans = np.where(diff_mask)[0].tolist()
                print(f"  [FAIL] t={t} {agent} max|diff|={diff} channels={diff_chans}")
                ok = False
                break
        if not ok:
            break
        # random step
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (env.num_agents,), 0, 6)
        action_dict = {f"agent_{i}": acts[i] for i in range(env.num_agents)}
        _, state, _, done_dict, _ = env.step(k_step, state, action_dict)
        if bool(done_dict["__all__"]):
            break
    if ok:
        print(f"  [OK] {NUM_STEPS} steps × 2 agents — byte-exact match")
    return ok


def main() -> int:
    print("V1StateToOV2ObsAdapter.get_ov2_obs_jit 검증")
    print("=" * 60)
    all_ok = True
    for L in LAYOUTS:
        if not compare_layout(L):
            all_ok = False
    print("=" * 60)
    print("RESULT:", "ALL OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
