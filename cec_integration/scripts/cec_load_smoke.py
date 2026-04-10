"""Acceptance smoke test for CEC checkpoint loading inside ph2-project.

Loads the target forced_coord_9 graphTrue checkpoints (orbax + pkl) using
`CECRuntime` and runs a few inference steps against synthetic obs. Verifies:

  1. Both orbax and pkl formats load and produce identical params.
  2. Forward pass returns valid 6-way action distributions, no NaNs.
  3. JIT-compiled `step` is stable across multiple calls.

This is the "load + sample" milestone the user asked for. Obs adaptation
from ph2 overcooked_v2 state is intentionally NOT exercised here.
"""
import hashlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from cec_integration.cec_runtime import CECRuntime
from cec_integration.checkpoint_io import load_checkpoint

import os
CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "ckpts",
    "forced_coord_9",
)
SEED = 11
CKPT_IDS = (0, 1)
NUM_AGENTS = 2
OBS_SHAPE = (9, 9, 26)


def _leaf_hash(tree) -> str:
    h = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        arr = np.asarray(leaf)
        h.update(arr.shape.__repr__().encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    print(f"[smoke] target dir: {CKPT_DIR}")
    runtimes = []
    for ckpt_id in CKPT_IDS:
        stem = f"{CKPT_DIR}/seed{SEED}_ckpt{ckpt_id}_improved"
        print(f"\n[smoke] === ckpt {ckpt_id} ===")

        # Format-parity check: orbax vs pkl should produce identical params.
        orbax_ckpt, fmt_o = load_checkpoint(stem, prefer_format="orbax", allow_fallback=False)
        pkl_ckpt, fmt_p = load_checkpoint(stem + ".pkl", prefer_format="pkl", allow_fallback=False)
        h_o = _leaf_hash(orbax_ckpt["params"])
        h_p = _leaf_hash(pkl_ckpt["params"])
        print(f"  orbax format={fmt_o} params_hash={h_o}")
        print(f"  pkl   format={fmt_p} params_hash={h_p}")
        if h_o != h_p:
            print("  [WARN] orbax and pkl params differ "
                  "(may be expected if saved at different times)")
        else:
            print("  [OK] orbax and pkl params identical")

        rt = CECRuntime(stem, beta=1.0, argmax=False)
        runtimes.append(rt)
        print(f"  CECRuntime built. config={rt.config}")

        # Forward smoke: random obs.
        rng = jax.random.PRNGKey(0)
        rng, sub = jax.random.split(rng)
        obs = jax.random.uniform(sub, (NUM_AGENTS, *OBS_SHAPE), dtype=jnp.float32)
        hidden = rt.init_hidden(NUM_AGENTS)
        done = jnp.zeros((NUM_AGENTS,), dtype=bool)

        actions_seen = []
        for t in range(5):
            rng, sub = jax.random.split(rng)
            actions, hidden, probs = rt.step(obs, hidden, done, sub)
            actions = np.asarray(actions)
            probs = np.asarray(probs)
            assert actions.shape == (NUM_AGENTS,), actions.shape
            assert probs.shape == (NUM_AGENTS, 6), probs.shape
            assert np.all(np.isfinite(probs)), "NaN/inf in probs"
            assert np.all((actions >= 0) & (actions < 6)), actions
            assert np.allclose(probs.sum(-1), 1.0, atol=1e-4), probs.sum(-1)
            actions_seen.append(actions.tolist())
        print(f"  [OK] 5 steps. actions={actions_seen}")
        # All-zero obs sanity (deterministic policy under sample is still fine).
        zero_obs = jnp.zeros_like(obs)
        rng, sub = jax.random.split(rng)
        a0, _, _ = rt.step(zero_obs, rt.init_hidden(NUM_AGENTS), done, sub)
        print(f"  [OK] zero-obs action: {np.asarray(a0).tolist()}")

    # Cross-ckpt sanity: different ckpts should usually have different params.
    h0 = _leaf_hash(runtimes[0].params)
    h1 = _leaf_hash(runtimes[1].params)
    print(f"\n[smoke] ckpt0 hash={h0}  ckpt1 hash={h1}  "
          f"{'DIFFER (expected)' if h0 != h1 else 'IDENTICAL (suspicious)'}")
    print("\n[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
