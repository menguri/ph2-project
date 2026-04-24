#!/usr/bin/env python3
"""BC × BC cross-play — pos_0 seeds × pos_1 seeds 매트릭스.

기존 eval_all.sh 의 BC×BC 블록은 seed_0 × seed_0 한 쌍만 평가.
여기서는 5×5 (또는 pos_0/pos_1 에 있는 seed 수만큼) 풀 매트릭스로 확장.

출력: results_final/bc_bc/scores.csv (bc_pos=0 기준 pos_0 seed, rl_seed=pos_1 seed)
"""
import argparse
import csv
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from policy import setup_pythonpath, BCPolicy


def load_bc_list(model_dir: Path, layout: str, pos: int):
    pos_dir = model_dir / layout / f"pos_{pos}"
    if not pos_dir.exists():
        return []
    pols = []
    for seed_dir in sorted(pos_dir.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_") and "tmp" not in seed_dir.name:
            try:
                pols.append((seed_dir.name, BCPolicy.from_pretrained(seed_dir)))
            except Exception as e:
                print(f"  [WARN] {seed_dir}: {e}")
    return pols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+",
                    default=["cramped_room", "asymm_advantages", "coord_ring",
                             "counter_circuit", "forced_coord"])
    ap.add_argument("--bc-model-dir", default="models")
    ap.add_argument("--num-eval-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--output-dir", default="results_final/bc_bc")
    args = ap.parse_args()

    setup_pythonpath("baseline")
    from overcooked_v2_experiments.eval.policy import PolicyPairing
    from overcooked_v2_experiments.eval.rollout import get_rollout
    from overcooked_v2_experiments.eval.utils import make_eval_env

    here = Path(__file__).resolve()
    bc_root = Path(args.bc_model_dir)
    if not bc_root.is_absolute():
        bc_root = (here.parent.parent / bc_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (here.parent.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # layout 별 요약
    summary_rows = [["layout", "mean_reward", "std_reward", "n_pairs"]]
    # 레이아웃 별 상세 scores
    for layout in args.layouts:
        print(f"\n=== {layout} ===", flush=True)
        bc0 = load_bc_list(bc_root, layout, 0)
        bc1 = load_bc_list(bc_root, layout, 1)
        if not bc0 or not bc1:
            print(f"  [SKIP] BC seeds 부족")
            summary_rows.append([layout, "-", "-", 0])
            continue
        print(f"  pos_0: {len(bc0)} seeds, pos_1: {len(bc1)} seeds")

        env, _, _ = make_eval_env(layout, {"max_steps": args.max_steps})

        rows = [["bc0_seed", "bc1_seed", "mean_reward", "std_reward"]]
        means = []
        for i0, (n0, p0) in enumerate(bc0):
            for i1, (n1, p1) in enumerate(bc1):
                pair = PolicyPairing(p0, p1)  # pos_0 agent × pos_1 agent
                key = jax.random.PRNGKey(0)
                eval_keys = jax.random.split(key, args.num_eval_seeds)
                rewards = []
                for ek in eval_keys:
                    roll = get_rollout(pair, env, ek, use_jit=True)
                    rewards.append(float(roll.total_reward))
                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                rows.append([n0, n1, f"{mean_r:.2f}", f"{std_r:.2f}"])
                means.append(mean_r)
                print(f"  BC0({n0}) × BC1({n1}): {mean_r:6.1f} ± {std_r:5.1f}", flush=True)

        # 상세 csv
        lay_dir = out_dir / layout
        lay_dir.mkdir(parents=True, exist_ok=True)
        with open(lay_dir / "scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

        mean_all = float(np.mean(means))
        std_all = float(np.std(means))
        print(f"  => layout mean: {mean_all:.1f} ± {std_all:.1f} (n={len(means)})")
        summary_rows.append([layout, f"{mean_all:.2f}", f"{std_all:.2f}", len(means)])

    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(summary_rows)
    print(f"\n저장: {out_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()
