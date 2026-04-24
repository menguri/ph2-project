#!/usr/bin/env python3
"""BC × CEC cross-play — forced_coord 전용 (cec_integration/ckpts 기반).

기존 evaluate_cec.py 는 webapp/models/*/cec 경로를 쓰지만, 여기서는
cec_integration/ckpts/forced_coord_9/seed{11,12,13,14,15,16,21,22}_ckpt2_improved
8 seeds 를 직접 지정해 V1 engine A1 경로로 cross-play 평가.

출력: scores.csv (bc_pos, bc_seed, cec_seed, mean_reward, std_reward)
"""
import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from policy import setup_pythonpath, BCPolicy, CECPolicy

CEC_SEEDS = [11, 12, 13, 14, 15, 16, 21, 22]
CKPT_SUFFIX = "ckpt2_improved"


def load_bc_policies(model_dir: Path, layout: str):
    bc_policies = {}
    for pos in [0, 1]:
        pos_dir = model_dir / layout / f"pos_{pos}"
        if not pos_dir.exists():
            continue
        pols = []
        for seed_dir in sorted(pos_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_") and "tmp" not in seed_dir.name:
                try:
                    pols.append(BCPolicy.from_pretrained(seed_dir))
                except Exception as e:
                    print(f"  [WARN] {seed_dir}: {e}")
        if pols:
            bc_policies[pos] = pols
            print(f"  BC pos_{pos}: {len(pols)} seeds")
    return bc_policies


def run_eval(bc_policies, cec_runtimes, layout, num_eval_seeds=5, max_steps=400):
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
    from cec_integration.cec_layouts import CEC_LAYOUTS
    from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter
    from cec_integration.webapp_v1_engine_helpers import ACTION_REMAP_OV2_TO_V1, BC_POS_TO_V1_SLOT

    env = V1Overcooked(layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False, max_steps=max_steps)
    ov2_adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=max_steps)

    slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
    results = []

    for bc_pos, bc_list in bc_policies.items():
        bc_slot_idx = slot_map[bc_pos]
        bc_slot = f"agent_{bc_slot_idx}"
        cec_slot_idx = 1 - bc_slot_idx
        cec_slot = f"agent_{cec_slot_idx}"
        for bc_idx, bc_policy in enumerate(bc_list):
            for cec_idx, (cec_name, cec_rt) in enumerate(cec_runtimes):
                key = jax.random.PRNGKey(0)
                eval_keys = jax.random.split(key, num_eval_seeds)
                rewards = []
                for ek in eval_keys:
                    obs_dict, env_state = env.reset(ek)
                    cec_h = cec_rt.init_hidden(env.num_agents)
                    bc_h = bc_policy.init_hstate(batch_size=1) if hasattr(bc_policy, "init_hstate") else None
                    done_arr = jnp.zeros(env.num_agents, dtype=bool)
                    total = 0.0
                    rng = ek
                    for t in range(max_steps):
                        rng, k_cec, k_bc, k_env = jax.random.split(rng, 4)
                        v1_obs_flat = jnp.stack([obs_dict[a].flatten() for a in env.agents])
                        cec_actions, cec_h, _ = cec_rt.step(v1_obs_flat, cec_h, done_arr, k_cec)
                        cec_action = int(cec_actions[cec_slot_idx])
                        ov2_obs_dict = ov2_adapter.get_ov2_obs(env_state, current_step=t)
                        bc_obs = jnp.asarray(np.asarray(ov2_obs_dict[bc_slot]).astype(np.uint8))
                        bc_done_scalar = jnp.array(bool(done_arr[bc_pos]))
                        bc_action, bc_h, _ = bc_policy.compute_action(bc_obs, bc_done_scalar, bc_h, k_bc)
                        bc_action_v1 = int(ACTION_REMAP_OV2_TO_V1[int(bc_action)])
                        env_act = {cec_slot: jnp.int32(cec_action), bc_slot: jnp.int32(bc_action_v1)}
                        obs_dict, env_state, reward, done_dict, _ = env.step(k_env, env_state, env_act)
                        done_arr = jnp.array([done_dict[a] for a in env.agents])
                        total += float(reward["agent_0"])
                        if bool(done_dict["__all__"]):
                            break
                    rewards.append(total)
                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                results.append({
                    "bc_pos": bc_pos, "bc_seed": bc_idx, "cec_seed": cec_idx,
                    "mean_reward": mean_r, "std_reward": std_r,
                })
                print(f"    BC(pos{bc_pos},s{bc_idx}) × CEC({cec_name}): {mean_r:6.1f} ± {std_r:5.1f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="forced_coord")
    ap.add_argument("--cec-ckpt-dir",
                    default=str(Path(__file__).resolve().parent.parent.parent
                                / "cec_integration/ckpts/forced_coord_9"))
    ap.add_argument("--bc-model-dir", default="models")
    ap.add_argument("--num-eval-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--output-dir", default="results_final/cec_forced_coord")
    ap.add_argument("--engine", choices=["v1", "ov2"], default="v1",
                    help="v1 (canonical, default) / ov2 (다른 dynamics)")
    ap.add_argument("--legacy", action="store_true",
                    help="Python per-step loop (느림, 호환 확인용)")
    args = ap.parse_args()

    setup_pythonpath("baseline")

    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    bc_root = Path(args.bc_model_dir)
    if not bc_root.is_absolute():
        bc_root = (here.parent.parent / bc_root).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (here.parent.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== BC × CEC Cross-Play: layout={args.layout} ===")
    print(f"  CEC ckpts: {args.cec_ckpt_dir}")
    print(f"  Seeds: {CEC_SEEDS}")

    # BC 로드
    print("BC 모델 로딩...")
    bc = load_bc_policies(bc_root, args.layout)
    if not bc:
        print("BC 없음")
        return 1

    # CEC 로드 — legacy V1 engine (Python loop) 이면 CECRuntime, 그 외 fast path 는 CECPolicy
    cec_root = Path(args.cec_ckpt_dir)
    if args.legacy:
        from cec_integration.cec_runtime import CECRuntime
        print("CEC runtimes 로딩 (legacy V1 engine 경로)...")
        cec_runtimes = []
        for seed in CEC_SEEDS:
            name = f"seed{seed}_{CKPT_SUFFIX}"
            ckpt = cec_root / name
            if not ckpt.is_dir():
                print(f"  [SKIP] {name} 없음")
                continue
            try:
                cec_runtimes.append((name, CECRuntime(str(ckpt))))
                print(f"  CEC {name}: loaded")
            except Exception as e:
                print(f"  [WARN] {name}: {e}")
        if not cec_runtimes:
            print("CEC 로드 실패")
            return 1
        print(f"\nCross-play 평가 (legacy, {args.num_eval_seeds} × {args.max_steps} steps)...")
        results = run_eval(bc, cec_runtimes, args.layout, args.num_eval_seeds, args.max_steps)
    else:
        print("CEC policies 로딩 (fast path — OV2 engine + JIT)...")
        cec_policies = []
        for seed in CEC_SEEDS:
            name = f"seed{seed}_{CKPT_SUFFIX}"
            ckpt = cec_root / name
            if not ckpt.is_dir():
                print(f"  [SKIP] {name} 없음")
                continue
            try:
                cec_policies.append(CECPolicy(str(ckpt), args.layout, stochastic=True))
                print(f"  CEC {name}: loaded")
            except Exception as e:
                print(f"  [WARN] {name}: {e}")
        if not cec_policies:
            print("CEC 로드 실패")
            return 1

        if args.engine == "v1":
            from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
            from cec_integration.cec_layouts import CEC_LAYOUTS
            from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter
            from fast_eval import run_cec_bc_crossplay_v1_fast
            # BC proxy eval 은 원본 canonical layout 에서 측정 (공정 비교).
            v1_env = V1Overcooked(layout=CEC_LAYOUTS[f"{args.layout}_9"],
                                  random_reset=False, max_steps=args.max_steps)
            adapter = V1StateToOV2ObsAdapter(target_layout=args.layout,
                                             max_steps=args.max_steps)
            print(f"\nCross-play (V1+JIT, {args.num_eval_seeds} × {args.max_steps} steps)...")
            results = run_cec_bc_crossplay_v1_fast(
                bc, cec_policies, v1_env, adapter, args.layout,
                num_eval_seeds=args.num_eval_seeds, max_steps=args.max_steps,
            )
        else:  # ov2
            from overcooked_v2_experiments.eval.utils import make_eval_env
            from fast_eval import run_cec_bc_crossplay_fast
            env, _, _ = make_eval_env(args.layout, {"max_steps": args.max_steps})
            print(f"\nCross-play (OV2+JIT, {args.num_eval_seeds} × {args.max_steps} steps)...")
            results = run_cec_bc_crossplay_fast(
                bc, cec_policies, env, args.layout,
                num_eval_seeds=args.num_eval_seeds, max_steps=args.max_steps,
            )

    # 저장
    out_csv = out_dir / "scores.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bc_pos", "bc_seed", "rl_seed", "mean_reward", "std_reward"])
        for r in results:
            w.writerow([r["bc_pos"], r["bc_seed"], r["cec_seed"],
                        f"{r['mean_reward']:.2f}", f"{r['std_reward']:.2f}"])
    print(f"저장: {out_csv}")
    if results:
        print(f"전체 평균 reward: {np.mean([r['mean_reward'] for r in results]):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
