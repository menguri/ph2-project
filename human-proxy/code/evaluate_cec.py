#!/usr/bin/env python3
"""BC × CEC cross-play 평가.

BC policy (human-proxy/models/{layout}/pos_{0,1}/seed_*/) vs
CEC policy (webapp/models/{layout}/cec/run*/ckpt_final) 매트릭스.

기존 evaluate.py 의 infrastructure (PolicyPairing, get_rollout, make_eval_env) 재사용.
CECPolicy 는 OV2 obs 를 자동으로 CEC 26ch 로 변환.

Usage:
    cd /home/mlic/mingukang/ph2-project/human-proxy && \
        PYTHONPATH=.:../ ./../overcooked_v2/bin/python \
        code/evaluate_cec.py --layout cramped_room
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# policy.py 경로 셋업
sys.path.insert(0, str(Path(__file__).parent))
from policy import (
    setup_pythonpath,
    BCPolicy,
    CECPolicy,
)


def load_bc_policies(model_dir: Path, layout: str):
    """Layout 의 BC policies (pos_0, pos_1 별) 로드."""
    bc_policies = {}
    layout_dir = model_dir / layout
    for pos in [0, 1]:
        pos_dir = layout_dir / f"pos_{pos}"
        if not pos_dir.exists():
            continue
        pols = []
        for seed_dir in sorted(pos_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_") and "tmp" not in seed_dir.name:
                try:
                    pols.append(BCPolicy.from_pretrained(seed_dir))
                except Exception as e:
                    print(f"  경고: {seed_dir} 로드 실패: {e}")
        if pols:
            bc_policies[pos] = pols
            print(f"  BC pos_{pos}: {len(pols)} seeds")
    return bc_policies


def load_cec_webapp(layout: str, webapp_root: Path):
    """webapp/models/{layout}/cec/run*/ckpt_final 에서 CECPolicy 로드."""
    cec_root = webapp_root / "models" / layout / "cec"
    if not cec_root.exists():
        return []
    policies = []
    for d in sorted(cec_root.iterdir()):
        ckpt_final = d / "ckpt_final"
        if d.is_dir() and ckpt_final.is_dir():
            try:
                policies.append(CECPolicy(str(ckpt_final), layout, stochastic=True))
                print(f"  CEC {d.name}: loaded")
            except Exception as e:
                print(f"  경고: {d.name} 로드 실패: {e}")
    return policies


# cec_integration (V1 engine A1 경로) 지원 레이아웃.
# asymm_advantages 는 CEC 학습 layout 과 OV2 가 구조적으로 달라 기존 OV2 경로로 fallback.
_CEC_V1_SUPPORTED_LAYOUTS = {"cramped_room", "coord_ring", "forced_coord", "counter_circuit"}


def _reload_cec_policies_as_v1_runtimes(cec_policies, layout):
    """CECPolicy 객체에서 체크포인트 경로를 꺼내 CECRuntime 을 다시 로드.

    CECPolicy 내부의 _runtime 을 직접 쓸 수도 있지만, hidden state 독립 제어가 필요해
    새 CECRuntime 을 만든다 (episode 간 reset 편의).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from cec_integration.cec_runtime import CECRuntime
    ckpt_root = project_root / "webapp" / "models" / layout / "cec"
    names = [d.name for d in sorted(ckpt_root.iterdir())
             if d.is_dir() and (d / "ckpt_final").is_dir()]
    runtimes = []
    for name in names[:len(cec_policies)]:
        ckpt = ckpt_root / name / "ckpt_final"
        runtimes.append((name, CECRuntime(str(ckpt))))
    return runtimes


def run_crossplay_v1_engine(bc_policies, cec_policies, layout, num_eval_seeds=5, max_steps=400):
    """V1 Overcooked engine + CEC_LAYOUTS 기반 crossplay (A1 구현).

    CEC 는 V1 native obs (9,9,26) 을 받고, BC 는 V1 state → OV2 obs (H,W,30) 을 받음.
    OV2 engine 경로는 CEC 가 학습 dynamics 와 달라 0 reward 였던 문제를 해결.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
    from cec_integration.cec_layouts import CEC_LAYOUTS
    from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter

    from cec_integration.webapp_v1_engine_helpers import ACTION_REMAP_OV2_TO_V1, BC_POS_TO_V1_SLOT

    env = V1Overcooked(layout=CEC_LAYOUTS[f"{layout}_9"], random_reset=False, max_steps=max_steps)
    ov2_adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=max_steps)

    # CECPolicy 객체 대신 직접 CECRuntime 사용 (hidden state 독립 reset 을 위해)
    cec_runtimes = _reload_cec_policies_as_v1_runtimes(cec_policies, layout)

    slot_map = BC_POS_TO_V1_SLOT.get(layout, {0: 0, 1: 1})
    results = []
    for bc_pos, bc_list in bc_policies.items():
        # BC pos → V1 slot 매핑 (V1 engine 의 agent 시작 위치와 OV2 의 BC 학습 분포 매칭)
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
                        # CEC: V1 obs 직접
                        v1_obs_flat = jnp.stack([obs_dict[a].flatten() for a in env.agents])
                        cec_actions, cec_h, _ = cec_rt.step(v1_obs_flat, cec_h, done_arr, k_cec)
                        cec_action = int(cec_actions[cec_slot_idx])
                        # BC: V1 state → OV2 obs
                        ov2_obs_dict = ov2_adapter.get_ov2_obs(env_state, current_step=t)
                        bc_obs = jnp.asarray(np.asarray(ov2_obs_dict[bc_slot]).astype(np.uint8))
                        bc_done_scalar = jnp.array(bool(done_arr[bc_pos]))
                        bc_action, bc_h, _ = bc_policy.compute_action(bc_obs, bc_done_scalar, bc_h, k_bc)
                        # BC outputs OV2 semantic action → V1 action 으로 remap (CEC 는 V1 학습 분포 그대로)
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
                    "rewards": rewards, "mean_reward": mean_r, "std_reward": std_r,
                })
                print(f"    BC(pos{bc_pos},s{bc_idx}) × CEC({cec_name}): "
                      f"{mean_r:6.1f} ± {std_r:5.1f}  [V1 engine]")
    return results


def run_crossplay(bc_policies, cec_policies, layout, num_eval_seeds=5, max_steps=400,
                  fast=True, engine="v1"):
    """Crossplay dispatcher.

    engine="v1", fast=True (default): V1 engine + JIT obs adapter + vmap.
      CEC 학습 dynamics 와 일치 + JIT 속도 — canonical + fast.
    engine="v1", fast=False: V1 engine legacy (Python per-step loop). 느림. 호환용.
    engine="ov2", fast=True: OV2 engine + JIT. CEC dynamics 달라 reward 낮음. 디버깅용.
    """
    if fast and engine == "v1":
        import sys as _sys, pathlib as _pl
        _root = _pl.Path(__file__).resolve().parent.parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
        from cec_integration.cec_layouts import CEC_LAYOUTS
        from cec_integration.obs_adapter_v1_to_ov2 import V1StateToOV2ObsAdapter
        from fast_eval import run_cec_bc_crossplay_v1_fast

        cec_name = f"{layout}_9"
        if cec_name not in CEC_LAYOUTS:
            print(f"  [FALLBACK] CEC_LAYOUTS[{cec_name}] 없음 → OV2 fast path")
            from overcooked_v2_experiments.eval.utils import make_eval_env
            from fast_eval import run_cec_bc_crossplay_fast
            env, _, _ = make_eval_env(layout, {"max_steps": max_steps})
            return run_cec_bc_crossplay_fast(
                bc_policies, cec_policies, env, layout,
                num_eval_seeds=num_eval_seeds, max_steps=max_steps,
            )

        v1_env = V1Overcooked(layout=CEC_LAYOUTS[cec_name], random_reset=False,
                              max_steps=max_steps)
        adapter = V1StateToOV2ObsAdapter(target_layout=layout, max_steps=max_steps)
        return run_cec_bc_crossplay_v1_fast(
            bc_policies, cec_policies, v1_env, adapter, layout,
            num_eval_seeds=num_eval_seeds, max_steps=max_steps,
        )

    if fast and engine == "ov2":
        from overcooked_v2_experiments.eval.utils import make_eval_env
        from fast_eval import run_cec_bc_crossplay_fast
        env, _, _ = make_eval_env(layout, {"max_steps": max_steps})
        return run_cec_bc_crossplay_fast(
            bc_policies, cec_policies, env, layout,
            num_eval_seeds=num_eval_seeds, max_steps=max_steps,
        )

    if layout in _CEC_V1_SUPPORTED_LAYOUTS:
        print(f"  [A1] V1 engine + V1StateToOV2ObsAdapter 경로 사용", flush=True)
        return run_crossplay_v1_engine(bc_policies, cec_policies, layout, num_eval_seeds, max_steps)

    # Legacy OV2 engine 경로 (asymm_advantages 등)
    print(f"  [legacy] OV2 engine + ov2_obs_to_cec 경로 (CEC 성능 제한 가능)", flush=True)
    from overcooked_v2_experiments.eval.policy import PolicyPairing
    from overcooked_v2_experiments.eval.rollout import get_rollout
    from overcooked_v2_experiments.eval.utils import make_eval_env

    env, _, _ = make_eval_env(layout, {"max_steps": max_steps})
    results = []

    for bc_pos, bc_list in bc_policies.items():
        for bc_idx, bc_policy in enumerate(bc_list):
            for cec_idx, cec_policy in enumerate(cec_policies):
                # bc_pos 0 → BC 가 slot 0, CEC 가 slot 1
                # bc_pos 1 → BC 가 slot 1, CEC 가 slot 0
                if bc_pos == 0:
                    pairing = PolicyPairing(bc_policy, cec_policy)
                else:
                    pairing = PolicyPairing(cec_policy, bc_policy)

                key = jax.random.PRNGKey(0)
                eval_keys = jax.random.split(key, num_eval_seeds)
                rewards = []
                for ek in eval_keys:
                    # CECPolicy 가 int() Python cast 하므로 use_jit=False 필수.
                    # 성능 저하 있지만 OV2 self-play 와 달리 cross-play 에서는 필연적.
                    rollout = get_rollout(pairing, env, ek, use_jit=False)
                    rewards.append(float(rollout.total_reward))
                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                results.append({
                    "bc_pos": bc_pos, "bc_seed": bc_idx, "cec_seed": cec_idx,
                    "rewards": rewards, "mean_reward": mean_r, "std_reward": std_r,
                })
                print(f"    BC(pos{bc_pos},s{bc_idx}) × CEC(run{cec_idx}): "
                      f"{mean_r:6.1f} ± {std_r:5.1f}")
    return results


def save_scores(results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bc_pos", "bc_seed", "cec_seed", "mean_reward", "std_reward"])
        for r in results:
            writer.writerow([r["bc_pos"], r["bc_seed"], r["cec_seed"],
                             f"{r['mean_reward']:.2f}", f"{r['std_reward']:.2f}"])
    print(f"점수 저장: {output_path}")


def save_heatmap(results, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음, 히트맵 스킵")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for pos in sorted(set(r["bc_pos"] for r in results)):
        pos_res = [r for r in results if r["bc_pos"] == pos]
        if not pos_res:
            continue
        bc_seeds = sorted(set(r["bc_seed"] for r in pos_res))
        cec_seeds = sorted(set(r["cec_seed"] for r in pos_res))
        mat = np.zeros((len(bc_seeds), len(cec_seeds)))
        for r in pos_res:
            mat[bc_seeds.index(r["bc_seed"]), cec_seeds.index(r["cec_seed"])] = r["mean_reward"]
        fig, ax = plt.subplots(figsize=(max(6, len(cec_seeds) + 2), max(4, len(bc_seeds) * 0.7 + 2)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean Reward")
        for i in range(len(bc_seeds)):
            for j in range(len(cec_seeds)):
                color = "white" if mat[i, j] > mat.max() * 0.7 else "black"
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color=color)
        ax.set_xticks(range(len(cec_seeds)))
        ax.set_xticklabels([f"CEC r{s}" for s in cec_seeds])
        ax.set_yticks(range(len(bc_seeds)))
        ax.set_yticklabels([f"BC s{s}" for s in bc_seeds])
        ax.set_xlabel("CEC Run")
        ax.set_ylabel("BC Seed")
        ax.set_title(f"BC × CEC cross-play (BC @ pos {pos})")
        mean_all = mat.mean()
        ax.text(0.02, 0.98, f"Mean: {mean_all:.1f}", transform=ax.transAxes,
                fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        plt.tight_layout()
        out = output_dir / f"heatmap_cec_pos{pos}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"히트맵: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--bc-model-dir", default="models")
    p.add_argument("--webapp-root", default=None,
                   help="webapp root (default: ../webapp)")
    p.add_argument("--num-eval-seeds", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--engine", choices=["v1", "ov2"], default="v1",
                   help="v1 (CEC canonical, default) / ov2 (다른 dynamics, 디버깅용)")
    p.add_argument("--legacy", action="store_true",
                   help="Python per-step loop (느림, 호환 확인용)")
    args = p.parse_args()

    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent  # human-proxy/code/evaluate_cec.py → ph2-project/
    webapp_root = Path(args.webapp_root) if args.webapp_root else project_root / "webapp"
    bc_model_dir = Path(args.bc_model_dir)
    if not bc_model_dir.is_absolute():
        bc_model_dir = (here.parent.parent / bc_model_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else bc_model_dir / args.layout / "eval_results_cec"

    # overcooked_v2_experiments PYTHONPATH 셋업 (baseline 소스 사용)
    setup_pythonpath("baseline")

    print(f"\n=== BC × CEC Cross-Play: layout={args.layout} ===")
    print("BC 모델 로딩...")
    bc = load_bc_policies(bc_model_dir, args.layout)
    if not bc:
        print(f"BC 모델 없음 in {bc_model_dir}/{args.layout}")
        return 1
    print("CEC 모델 로딩...")
    cec = load_cec_webapp(args.layout, webapp_root)
    if not cec:
        print(f"CEC 모델 없음 in {webapp_root}/models/{args.layout}/cec")
        return 1

    print(f"\nCross-play 평가 ({args.num_eval_seeds} seeds × {args.max_steps} steps per pair)...")
    results = run_crossplay(bc, cec, args.layout, args.num_eval_seeds, args.max_steps,
                            fast=not args.legacy, engine=args.engine)
    save_scores(results, output_dir / "scores_cec.csv")
    save_heatmap(results, output_dir)

    if results:
        mean_all = np.mean([r["mean_reward"] for r in results])
        print(f"\n=== 전체 평균 reward: {mean_all:.1f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
