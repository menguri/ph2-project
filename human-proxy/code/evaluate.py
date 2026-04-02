#!/usr/bin/env python3
"""
BC × RL 알고리즘 cross-play 평가 + 히트맵 생성.

BC는 layout당 1개 (position 구분 없음), 평가 시 pos_0/pos_1 양쪽에 배치.

사용법:
    python code/evaluate.py --algo-dir ../ph2/runs/..._cramped_room_ph2 --layout cramped_room
    python code/evaluate.py --algo-dir ../baseline/runs/..._cramped_room_sp --layout cramped_room

--source 자동 감지: 체크포인트의 ALG_NAME에 PH2가 포함되면 ph2, 아니면 baseline.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# policy.py에서 경로 설정 함수와 로더를 가져옴 (overcooked_v2_experiments import 전)
from policy import (
    setup_pythonpath,
    detect_source_from_run_dir,
    BCPolicy,
    load_ppo_policies_from_run_dir,
)


def load_bc_policies(model_dir: Path, layout: str):
    """레이아웃의 BC policy 로드. {pos: [BCPolicy, ...]} 반환."""
    bc_policies = {}
    layout_dir = model_dir / layout

    for pos in [0, 1]:
        pos_dir = layout_dir / f"pos_{pos}"
        if not pos_dir.exists():
            continue
        policies = []
        for seed_dir in sorted(pos_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_") and "tmp" not in seed_dir.name:
                try:
                    policy = BCPolicy.from_pretrained(seed_dir)
                    policies.append(policy)
                except Exception as e:
                    print(f"  경고: {seed_dir} 로드 실패: {e}")
        if policies:
            bc_policies[pos] = policies
            print(f"  BC pos_{pos}: {len(policies)} seeds 로드")

    return bc_policies


def run_crossplay(bc_policies, rl_policies, layout, num_eval_seeds=10, env_kwargs=None):
    """BC × RL cross-play 매트릭스 평가.

    bc_policies: dict[int, list[BCPolicy]] — position별 BC policy 리스트
    """
    from overcooked_v2_experiments.eval.policy import PolicyPairing
    from overcooked_v2_experiments.eval.rollout import get_rollout
    from overcooked_v2_experiments.eval.utils import make_eval_env

    if env_kwargs is None:
        env_kwargs = {}

    env, _, _ = make_eval_env(layout, env_kwargs)
    results = []

    for bc_pos, bc_list in bc_policies.items():
        for bc_idx, bc_policy in enumerate(bc_list):
            for rl_idx, rl_policy in enumerate(rl_policies):
                if bc_pos == 0:
                    pairing = PolicyPairing(bc_policy, rl_policy)
                else:
                    pairing = PolicyPairing(rl_policy, bc_policy)

                key = jax.random.PRNGKey(0)
                eval_keys = jax.random.split(key, num_eval_seeds)

                rewards = []
                for ek in eval_keys:
                    rollout = get_rollout(pairing, env, ek, use_jit=True)
                    rewards.append(float(rollout.total_reward))

                mean_r = np.mean(rewards)
                std_r = np.std(rewards)

                results.append({
                    "bc_pos": bc_pos,
                    "bc_seed": bc_idx,
                    "rl_seed": rl_idx,
                    "rewards": rewards,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                })

                print(f"    BC(pos{bc_pos},s{bc_idx}) × RL(s{rl_idx}): "
                      f"{mean_r:.1f} ± {std_r:.1f}")

    return results


def save_scores_csv(results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bc_pos", "bc_seed", "rl_seed", "mean_reward", "std_reward"])
        for r in results:
            writer.writerow([
                r["bc_pos"], r["bc_seed"], r["rl_seed"],
                f"{r['mean_reward']:.2f}", f"{r['std_reward']:.2f}",
            ])
    print(f"점수 저장: {output_path}")


def save_heatmaps(results, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음, 히트맵 스킵")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    positions = sorted(set(r["bc_pos"] for r in results))

    for pos in positions:
        pos_results = [r for r in results if r["bc_pos"] == pos]
        if not pos_results:
            continue

        bc_seeds = sorted(set(r["bc_seed"] for r in pos_results))
        rl_seeds = sorted(set(r["rl_seed"] for r in pos_results))

        matrix = np.zeros((len(bc_seeds), len(rl_seeds)))
        for r in pos_results:
            bi = bc_seeds.index(r["bc_seed"])
            ri = rl_seeds.index(r["rl_seed"])
            matrix[bi, ri] = r["mean_reward"]

        fig, ax = plt.subplots(figsize=(max(6, len(rl_seeds) * 0.8 + 2),
                                         max(4, len(bc_seeds) * 0.6 + 2)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean Reward")

        for i in range(len(bc_seeds)):
            for j in range(len(rl_seeds)):
                ax.text(j, i, f"{matrix[i, j]:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() * 0.7 else "black")

        ax.set_xticks(range(len(rl_seeds)))
        ax.set_xticklabels([f"RL s{s}" for s in rl_seeds])
        ax.set_yticks(range(len(bc_seeds)))
        ax.set_yticklabels([f"BC s{s}" for s in bc_seeds])
        ax.set_xlabel("RL Seed")
        ax.set_ylabel("BC Seed")
        ax.set_title(f"Cross-Play Rewards — BC Position {pos}")

        mean_all = matrix.mean()
        ax.text(0.02, 0.98, f"Mean: {mean_all:.1f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        fig_path = output_dir / f"heatmap_pos{pos}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"히트맵 저장: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="BC × RL Cross-Play 평가")
    parser.add_argument("--algo-dir", required=True,
                        help="RL 알고리즘 run 디렉토리 (run_0/, run_1/ 포함)")
    parser.add_argument("--layout", required=True, help="레이아웃 이름")
    parser.add_argument("--bc-model-dir", default="models", help="BC 모델 디렉토리")
    parser.add_argument("--num-eval-seeds", type=int, default=10, help="평가 시드 수")
    parser.add_argument("--max-steps", type=int, default=400, help="에피소드 최대 스텝 수")
    parser.add_argument("--source", default=None, choices=["ph2", "baseline"],
                        help="RL 코드 소스 (자동 감지 가능)")
    parser.add_argument("--output-dir", default=None, help="결과 저장 디렉토리")
    args = parser.parse_args()

    layout = args.layout
    bc_model_dir = Path(args.bc_model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else bc_model_dir / layout / "eval_results"

    # 1. 소스 감지 + PYTHONPATH 설정 (import 전에 반드시)
    if args.source:
        source = args.source
    else:
        print("알고리즘 소스 자동 감지 중...")
        source = detect_source_from_run_dir(args.algo_dir)
    print(f"소스: {source}")
    setup_pythonpath(source)

    print(f"\n=== Cross-Play 평가: layout={layout} ===")

    # 2. BC 모델 로드
    print("BC 모델 로딩...")
    bc_policies = load_bc_policies(bc_model_dir, layout)
    if not bc_policies:
        print("BC 모델 없음. train.py 먼저 실행하세요.")
        return

    # 3. RL 모델 로드
    print("RL 모델 로딩...")
    rl_policies = load_ppo_policies_from_run_dir(args.algo_dir)
    if not rl_policies:
        print("RL 모델 없음. algo-dir 확인하세요.")
        return

    # 4. Cross-play 평가
    print("Cross-play 평가 시작...")
    results = run_crossplay(
        bc_policies, rl_policies, layout,
        num_eval_seeds=args.num_eval_seeds,
        env_kwargs={"max_steps": args.max_steps},
    )

    # 5. 결과 저장
    save_scores_csv(results, output_dir / "scores.csv")
    save_heatmaps(results, output_dir)

    if results:
        mean_all = np.mean([r["mean_reward"] for r in results])
        print(f"\n=== 전체 평균 reward: {mean_all:.1f} ===")


if __name__ == "__main__":
    main()
