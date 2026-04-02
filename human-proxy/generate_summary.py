#!/usr/bin/env python3
"""
BC cross-play 평가 결과 요약 생성.

results/{algo}_{layout}/scores.csv 파일들을 읽어서
하나의 summary_bc_xp.csv로 합산.

사용법:
    cd human-proxy && python generate_summary.py

출력:
    results/summary_bc_xp.csv
"""
import csv
from pathlib import Path

import numpy as np


RESULTS_DIR = Path("results")
ALGOS = ["sp", "e3t", "fcp", "mep", "ph2"]
LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring", "counter_circuit", "forced_coord"]


def load_scores(algo, layout):
    """results/{algo}_{layout}/scores.csv → (mean, std) 또는 None."""
    scores_file = RESULTS_DIR / f"{algo}_{layout}" / "scores.csv"
    if not scores_file.exists():
        return None

    rewards = []
    with open(scores_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["mean_reward"]))

    if not rewards:
        return None
    return np.mean(rewards), np.std(rewards)


def main():
    print("=" * 80)
    print("BC Cross-Play 평가 요약")
    print("=" * 80)

    # 헤더
    header = f"{'layout':<20}"
    for algo in ALGOS:
        header += f" {algo:>12}"
    print(header)
    print("-" * (20 + 13 * len(ALGOS)))

    # CSV 출력 준비
    csv_rows = []

    for layout in LAYOUTS:
        row_str = f"{layout:<20}"
        csv_row = {"layout": layout}

        for algo in ALGOS:
            result = load_scores(algo, layout)
            if result is None:
                row_str += f" {'–':>12}"
                csv_row[f"{algo}_mean"] = ""
                csv_row[f"{algo}_std"] = ""
            else:
                mean, std = result
                row_str += f" {mean:>5.1f}±{std:<4.1f}"
                csv_row[f"{algo}_mean"] = f"{mean:.1f}"
                csv_row[f"{algo}_std"] = f"{std:.1f}"

        print(row_str)
        csv_rows.append(csv_row)

    # 알고리즘별 전체 평균
    print("-" * (20 + 13 * len(ALGOS)))
    avg_str = f"{'AVERAGE':<20}"
    for algo in ALGOS:
        all_means = []
        for layout in LAYOUTS:
            result = load_scores(algo, layout)
            if result:
                all_means.append(result[0])
        if all_means:
            avg_str += f" {np.mean(all_means):>5.1f}{'':>7}"
        else:
            avg_str += f" {'–':>12}"
    print(avg_str)

    # CSV 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "summary_bc_xp.csv"

    fieldnames = ["layout"]
    for algo in ALGOS:
        fieldnames.extend([f"{algo}_mean", f"{algo}_std"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n저장: {out_path}")

    # 개별 결과 파일 존재 여부
    print(f"\n{'='*80}")
    print("개별 결과 파일:")
    found = 0
    missing = 0
    for algo in ALGOS:
        for layout in LAYOUTS:
            p = RESULTS_DIR / f"{algo}_{layout}" / "scores.csv"
            if p.exists():
                found += 1
            else:
                missing += 1
                print(f"  ✗ {algo}_{layout}")
    print(f"  {found}/{found+missing} 존재, {missing} 미완료")


if __name__ == "__main__":
    main()
