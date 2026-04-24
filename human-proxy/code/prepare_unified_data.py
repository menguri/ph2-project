#!/usr/bin/env python3
"""Pos-unified BC 학습용 데이터 준비.

구조:
  1. data/bc/{layout}/pos_0 + pos_1 concat → ours_all (N개)
  2. Berkeley (있으면) 에서 N 개 random subsample → berk_sub
  3. combined = concat(ours_all, berk_sub)
  4. 저장: data/bc_unified/{layout}/{obs,actions}.npy (+ metadata.json)

Berkeley 가 없는 layout (counter_circuit, forced_coord) 은 ours_all 만 저장.
"""
import argparse
import json
from pathlib import Path

import numpy as np


LAYOUTS_BERKELEY = {
    "cramped_room":       "cramped_room",
    "asymm_advantages":   "asymmetric_advantages",
    "coord_ring":         "coordination_ring",
    "counter_circuit":    None,
    "forced_coord":       None,
}


def load_pos_dir(d: Path):
    o = d / "obs.npy"; a = d / "actions.npy"
    if not (o.exists() and a.exists()):
        return None, None
    return np.load(o), np.load(a)


def prepare_layout(layout: str, bc_root: Path, berk_root: Path, out_dir: Path, seed: int):
    rng = np.random.default_rng(seed)

    # 1) ours: pos_0 + pos_1 합치기
    ours_obs, ours_act = [], []
    for pos in (0, 1):
        o, a = load_pos_dir(bc_root / layout / f"pos_{pos}")
        if o is None:
            print(f"  [{layout}] ours pos_{pos} 없음 — skip")
            continue
        ours_obs.append(o); ours_act.append(a)
    if not ours_obs:
        print(f"  [{layout}] ours 데이터 없음")
        return
    ours_obs = np.concatenate(ours_obs, axis=0)
    ours_act = np.concatenate(ours_act, axis=0)
    n_ours = len(ours_act)

    # 2) berkeley: pos_0 + pos_1 합친 후 N 샘플 random pick
    berk_name = LAYOUTS_BERKELEY.get(layout)
    berk_obs_sub, berk_act_sub = None, None
    if berk_name is not None:
        bobs, bact = [], []
        for pos in (0, 1):
            o, a = load_pos_dir(berk_root / berk_name / f"pos_{pos}")
            if o is None:
                continue
            bobs.append(o); bact.append(a)
        if bobs:
            bobs = np.concatenate(bobs, axis=0)
            bact = np.concatenate(bact, axis=0)
            n_berk = len(bact)
            n_pick = min(n_ours, n_berk)
            idx = rng.choice(n_berk, size=n_pick, replace=False)
            berk_obs_sub = bobs[idx]
            berk_act_sub = bact[idx]

    # 3) concat
    if berk_obs_sub is not None:
        obs = np.concatenate([ours_obs, berk_obs_sub], axis=0)
        act = np.concatenate([ours_act, berk_act_sub], axis=0)
        src = f"ours({n_ours}) + berkeley({len(berk_act_sub)}) = {len(act)}"
    else:
        obs = ours_obs; act = ours_act
        src = f"ours only ({n_ours})"

    out = out_dir / layout
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "obs.npy", obs)
    np.save(out / "actions.npy", act)

    h = np.bincount(act, minlength=6).tolist()
    p = (np.array(h) / len(act) * 100).round(1).tolist()
    (out / "metadata.json").write_text(json.dumps({
        "layout": layout,
        "source": src,
        "n_total": int(len(act)),
        "n_ours": int(n_ours),
        "n_berkeley": int(len(berk_act_sub)) if berk_obs_sub is not None else 0,
        "action_hist": h,
        "action_pct": p,
        "obs_shape": list(obs.shape),
        "seed": int(seed),
    }, indent=2, ensure_ascii=False))
    print(f"  [{layout}] {src}  action%={p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-root", default="data/bc")
    ap.add_argument("--berk-root", default="data/berkeley")
    ap.add_argument("--out-dir", default="data/bc_unified")
    ap.add_argument("--layouts", nargs="+",
                    default=["cramped_room", "asymm_advantages", "coord_ring",
                             "counter_circuit", "forced_coord"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent  # human-proxy/
    bc_root = Path(args.bc_root); berk_root = Path(args.berk_root)
    out_dir = Path(args.out_dir)
    if not bc_root.is_absolute(): bc_root = here / bc_root
    if not berk_root.is_absolute(): berk_root = here / berk_root
    if not out_dir.is_absolute(): out_dir = here / out_dir

    print(f"bc_root:   {bc_root}")
    print(f"berk_root: {berk_root}")
    print(f"out_dir:   {out_dir}")

    for L in args.layouts:
        prepare_layout(L, bc_root, berk_root, out_dir, args.seed)


if __name__ == "__main__":
    main()
