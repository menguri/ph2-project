#!/usr/bin/env python3
"""results_final/ 내 scores.csv 들을 (layout, algo) 별 표로 정리.

각 cell 마다 (전부 BC × RL 점수):
  pos0      = BC pos_0 × RL mean
  pos0_std  = BC pos_0 × RL std (pair-mean 들의 std)
  pos1      = BC pos_1 × RL mean
  pos1_std  = BC pos_1 × RL std
  gap       = pos0 - pos1

layout × algo 표 출력 + CSV 저장.
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent / "results_final"
LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring", "counter_circuit", "forced_coord"]
ALGOS = ["sp", "e3t", "fcp", "mep", "gamma", "ph2", "cec"]


def load_scores(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "bc_pos": int(r["bc_pos"]),
                "mean_reward": float(r["mean_reward"]),
                "std_reward": float(r["std_reward"]),
            })
    return rows


def summarize(rows):
    """pos 별 mean of means, std of means 반환."""
    if not rows:
        return None
    pos0 = [r["mean_reward"] for r in rows if r["bc_pos"] == 0]
    pos1 = [r["mean_reward"] for r in rows if r["bc_pos"] == 1]
    sp_m = float(np.mean(pos0)) if pos0 else float("nan")
    sp_s = float(np.std(pos0)) if pos0 else float("nan")
    xp_m = float(np.mean(pos1)) if pos1 else float("nan")
    xp_s = float(np.std(pos1)) if pos1 else float("nan")
    gap = sp_m - xp_m if pos0 and pos1 else float("nan")
    return sp_m, sp_s, xp_m, xp_s, gap


def main():
    print(f"results_final root: {ROOT}")

    out_rows = []
    header = ["layout", "algo", "pos0", "pos0_std", "pos1", "pos1_std", "gap"]
    out_rows.append(header)

    # 표 출력
    print()
    print(f"{'layout':<20s} {'algo':<6s} {'pos0':>8s} {'pos0_std':>8s} {'pos1':>8s} {'pos1_std':>8s} {'gap':>8s}")
    print("=" * 75)

    for layout in LAYOUTS:
        for algo in ALGOS:
            # cec 는 forced_coord 만
            if algo == "cec":
                if layout != "forced_coord":
                    continue
                path = ROOT / "cec_forced_coord" / "scores.csv"
            else:
                path = ROOT / f"{algo}_{layout}" / "scores.csv"

            if not path.exists():
                row = [layout, algo, "-", "-", "-", "-", "-"]
                print(f"{layout:<20s} {algo:<6s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")
                out_rows.append(row)
                continue

            rows = load_scores(path)
            res = summarize(rows)
            if res is None:
                row = [layout, algo, "-", "-", "-", "-", "-"]
                print(f"{layout:<20s} {algo:<6s} {'empty':>8s}")
                out_rows.append(row)
                continue

            sp_m, sp_s, xp_m, xp_s, gap = res
            print(f"{layout:<20s} {algo:<6s} {sp_m:>8.1f} {sp_s:>8.1f} {xp_m:>8.1f} {xp_s:>8.1f} {gap:>8.1f}")
            out_rows.append([layout, algo, f"{sp_m:.2f}", f"{sp_s:.2f}",
                             f"{xp_m:.2f}", f"{xp_s:.2f}", f"{gap:.2f}"])
        print("-" * 75)

    # BC × BC reference
    bc_bc_path = ROOT / "bc_bc" / "scores.csv"
    print("\n── BC × BC reference ──")
    print(f"{'layout':<20s} {'mean':>8s} {'std':>8s} {'n_pairs':>8s}")
    print("=" * 50)
    bc_bc_rows = []
    if bc_bc_path.exists():
        with open(bc_bc_path) as f:
            for r in csv.DictReader(f):
                print(f"{r['layout']:<20s} {float(r['mean_reward']):>8.1f} "
                      f"{float(r['std_reward']):>8.1f} {r['n_pairs']:>8s}")
                bc_bc_rows.append([r["layout"], r["mean_reward"],
                                   r["std_reward"], r["n_pairs"]])

    # CSV 저장
    out_csv = ROOT / "summary_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)
        w.writerow([])
        w.writerow(["# BC × BC reference"])
        w.writerow(["layout", "mean", "std", "n_pairs"])
        w.writerows(bc_bc_rows)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
