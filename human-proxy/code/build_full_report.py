"""Human-proxy × (baseline + PH2 전 113 runs) 통합 리포트 생성.

출력:
  human-proxy/reports/ph2_full_eval_report.md   (Markdown 보고서)
  human-proxy/reports/ph2_full_eval_summary.csv (기계 판독용 CSV)
"""
import pandas as pd, pathlib, re

HP       = pathlib.Path(__file__).resolve().parent.parent
PH2_CSV  = pathlib.Path("/home/mlic/mingukang/ph2-project/ph2/runs/summary_sp_xp.csv")
BL_CSV   = pathlib.Path("/home/mlic/mingukang/ph2-project/baseline/runs/summary_sp_xp.csv")
PH2_EVAL = HP/"results_all_ph2_runs"
BL_EVAL  = HP/"results_all_norecompute"

LAYOUTS = ["cramped_room","asymm_advantages","coord_ring","counter_circuit","forced_coord"]
BASE_ALGOS = ["sp","e3t","fcp","mep","gamma","ph2","cec"]

FINALS = {
    "cramped_room":     "20260402-134406_l2fth7ra",
    "asymm_advantages": "20260405-171911_4iphli57",
    "coord_ring":       "20260415-061513_azw2abpk",
    "counter_circuit":  "20260421-234203_lyogphwp",
    "forced_coord":     "20260419-070914_8ixp5kyd",
}

# --- Load baseline algos: BC eval stats ---
ph2_sp = pd.read_csv(PH2_CSV)
bl_sp  = pd.read_csv(BL_CSV)

def sp_lookup(layout, algo_substr=None, run_substr=None, source="ph2"):
    """baseline/ph2 SP/XP 찾기. algo_substr 기준 매칭."""
    df = ph2_sp if source == "ph2" else bl_sp
    rows = df[df["run_name"].str.contains(layout, na=False)]
    if algo_substr:
        rows = rows[rows["run_name"].str.contains(algo_substr, case=False, na=False)]
    if run_substr:
        rows = rows[rows["run_name"].str.contains(run_substr, na=False)]
    if len(rows) == 0: return None
    return rows.iloc[0]   # first

def bc_stats(results_dir):
    """scores.csv → (p0_mean, p0_std, p1_mean, p1_std, ov_mean, ov_std)"""
    f = results_dir / "scores.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    p0 = df[df["bc_pos"]==0]["mean_reward"]
    p1 = df[df["bc_pos"]==1]["mean_reward"]
    ov = df["mean_reward"]
    return (p0.mean(), p0.std(), p1.mean(), p1.std(), ov.mean(), ov.std())

# --- Build data ---
records = []

# Baseline algos: look up self-play (sp-mean) from baseline CSV based on final/baseline run names
FINAL_BASELINE = {
    "sp":     {"cramped_room": "20260402-174158_d4adx8ic",
               "asymm_advantages": "20260402-194856_efk6cujd",
               "coord_ring": "20260402-231350_dm9zdemf",
               "counter_circuit": "20260403-040145_h8k9lr8n",
               "forced_coord": "20260403-013836_vv9p1dxm"},
    "e3t":    {"cramped_room": "20260407-052043_swhnnuls",
               "asymm_advantages": "20260403-082745_mlec618c",
               "coord_ring": "20260407-044458_c8zr5hld",
               "counter_circuit": "20260415-210152_mm2n4xiu",
               "forced_coord": "20260415-222321_zuxojqvn"},
    "fcp":    {"cramped_room": "20260403-170228_qpooccn1",
               "asymm_advantages": "20260403-193352_6nlzt2da",
               "coord_ring": "20260322-190630_m8jruk9n",
               "counter_circuit": "20260404-045702_rh0mwdby",
               "forced_coord": "20260404-020741_61kuulek"},
    "mep":    {"cramped_room": "20260408-050225_md4rvbma",
               "asymm_advantages": "20260405-170609_yboqy292",
               "coord_ring": "20260403-053239_4siekak8",
               "counter_circuit": "20260407-111547_2587p3u6",
               "forced_coord": "20260416-162055_lz0c4ozg"},
    "gamma":  {"cramped_room": "20260403-085230_ubpmb358",
               "asymm_advantages": "20260421-164939_j5nk0bob",
               "coord_ring": "20260403-111821_espoke7i",
               "counter_circuit": "20260416-054046_tooj17bj",
               "forced_coord": "20260420-140916_vuhbzyo4"},
}

for L in LAYOUTS:
    # --- Baselines ---
    for algo in BASE_ALGOS:
        if algo == "ph2":
            # PH2 final
            fin = FINALS[L]
            src = "ph2"
            sp_row = sp_lookup(L, run_substr=fin, source=src)
        elif algo == "cec":
            sp_row = None  # CEC has its own forced_coord_9 format, skip SP/XP
        else:
            fin = FINAL_BASELINE[algo][L]
            sp_row = sp_lookup(L, run_substr=fin, source="baseline")
        bc_dir = BL_EVAL / f"{algo}_{L}"
        bc = bc_stats(bc_dir)
        rec = {"layout": L, "kind": "baseline", "name": algo,
               "is_final_ph2": algo == "ph2",
               "sp_mean": sp_row["sp-mean"] if sp_row is not None else None,
               "sp_std":  sp_row["sp-std"]  if sp_row is not None else None,
               "xp_mean": sp_row["xp-mean"] if sp_row is not None else None,
               "xp_std":  sp_row["xp-std"]  if sp_row is not None else None,
               "p0_m": bc[0] if bc else None, "p0_s": bc[1] if bc else None,
               "p1_m": bc[2] if bc else None, "p1_s": bc[3] if bc else None,
               "ov_m": bc[4] if bc else None, "ov_s": bc[5] if bc else None}
        records.append(rec)

    # BC × BC
    bc_bc_csv = BL_EVAL/"bc_bc"/"scores.csv"
    if bc_bc_csv.exists():
        df = pd.read_csv(bc_bc_csv).set_index("layout")
        if L in df.index:
            rec = {"layout": L, "kind": "bc_bc", "name": "bc×bc",
                   "is_final_ph2": False,
                   "sp_mean": None, "sp_std": None, "xp_mean": None, "xp_std": None,
                   "p0_m": None, "p0_s": None, "p1_m": None, "p1_s": None,
                   "ov_m": float(df.loc[L,"mean_reward"]),
                   "ov_s": float(df.loc[L,"std_reward"])}
            records.append(rec)

    # --- PH2 runs ---
    ph2_layout = PH2_EVAL/L
    if ph2_layout.exists():
        for rd in sorted(ph2_layout.iterdir()):
            if not rd.is_dir(): continue
            bc = bc_stats(rd)
            if bc is None: continue
            sp_row = sp_lookup(L, run_substr=rd.name, source="ph2")
            is_fin = FINALS[L] in rd.name
            rec = {"layout": L, "kind": "ph2_run", "name": rd.name,
                   "is_final_ph2": is_fin,
                   "sp_mean": sp_row["sp-mean"] if sp_row is not None else None,
                   "sp_std":  sp_row["sp-std"]  if sp_row is not None else None,
                   "xp_mean": sp_row["xp-mean"] if sp_row is not None else None,
                   "xp_std":  sp_row["xp-std"]  if sp_row is not None else None,
                   "p0_m": bc[0], "p0_s": bc[1],
                   "p1_m": bc[2], "p1_s": bc[3],
                   "ov_m": bc[4], "ov_s": bc[5]}
            records.append(rec)

df = pd.DataFrame(records)

# --- Save CSV ---
out_dir = HP/"reports"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir/"ph2_full_eval_summary.csv", index=False)
print(f"Saved CSV: {out_dir/'ph2_full_eval_summary.csv'}  ({len(df)} rows)")

# --- Markdown report ---
def fmt_mean_std(m, s, w=6, sig=1):
    if m is None or pd.isna(m):
        return f"{'-':>{w}}"
    if s is None or pd.isna(s):
        return f"{m:>{w-3}.{sig}f}"
    return f"{m:>{w-5}.{sig}f}±{s:<4.{sig}f}"

md = []
md.append("# PH2 × Human-Proxy 전체 평가 보고서\n")
md.append("## 개요\n")
md.append(f"- **BC 모델**: `models_all_norecompute` (webapp 232 eps, pos-split 5 seeds, recompute=False)")
md.append(f"- **Eval**: 각 (algo/run × BC) 조합에서 n=125 rollouts (5 BC seed × 5 RL seed × 5 eval seed × 2 pos)")
md.append(f"- **BC × BC**: n=25 (5 pos0 seeds × 5 pos1 seeds)")
md.append(f"- **Baseline RL**: `final/baseline/{{layout}}/` 의 각 algo 최신 run (SP/E3T/FCP/MEP/GAMMA + CEC via V1 engine)")
md.append(f"- **PH2 runs**: `ph2/runs/` 의 모든 {{run_X}} 서브디렉을 가진 run (총 113개)")
md.append(f"- **SP/XP**: `baseline/runs/summary_sp_xp.csv` (baseline algos) 또는 `ph2/runs/summary_sp_xp.csv` (PH2)")
md.append(f"- 생성일: 2026-04-21\n")

md.append("## 핵심 요약 — 레이아웃별 최고 알고리즘 & PH2 final 갱신 후보\n")
md.append("| layout | 현재 final PH2 (overall) | 최고 PH2 (overall) | 베이스라인 최고 |")
md.append("|---|---|---|---|")
for L in LAYOUTS:
    sub = df[df["layout"]==L]
    fin = sub[(sub["kind"]=="ph2_run") & (sub["is_final_ph2"])]
    best_ph2 = sub[sub["kind"]=="ph2_run"].sort_values("ov_m", ascending=False).head(1)
    non_ph2 = sub[(sub["kind"]=="baseline") & (sub["name"]!="ph2")]
    best_bl = non_ph2.sort_values("ov_m", ascending=False).head(1)
    fin_str = f"{fin.iloc[0]['name'][:40]} = {fin.iloc[0]['ov_m']:.1f}" if len(fin) else "N/A"
    best_str = f"{best_ph2.iloc[0]['name'][:40]} = {best_ph2.iloc[0]['ov_m']:.1f}" if len(best_ph2) else "N/A"
    bl_str = f"{best_bl.iloc[0]['name']} = {best_bl.iloc[0]['ov_m']:.1f}" if len(best_bl) else "N/A"
    md.append(f"| {L} | {fin_str} | {best_str} | {bl_str} |")
md.append("")

for L in LAYOUTS:
    md.append(f"\n## {L}\n")
    sub = df[df["layout"]==L].copy()

    # Baseline table
    md.append("### 베이스라인 (algo × BC)\n")
    md.append("| algo | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |")
    md.append("|---|---|---|---|---|---|")
    for algo in BASE_ALGOS + ["bc×bc"]:
        row = sub[sub["name"]==algo]
        if len(row)==0: continue
        r = row.iloc[0]
        sp_str = fmt_mean_std(r["sp_mean"], r["sp_std"], w=10) if r["sp_mean"] is not None and not pd.isna(r["sp_mean"]) else "-"
        xp_str = fmt_mean_std(r["xp_mean"], r["xp_std"], w=10) if r["xp_mean"] is not None and not pd.isna(r["xp_mean"]) else "-"
        p0 = fmt_mean_std(r["p0_m"], r["p0_s"], w=10)
        p1 = fmt_mean_std(r["p1_m"], r["p1_s"], w=10)
        ov = fmt_mean_std(r["ov_m"], r["ov_s"], w=10)
        md.append(f"| {algo} | {sp_str} | {xp_str} | {p0} | {p1} | {ov} |")

    # PH2 runs table (overall descending)
    ph2_sub = sub[sub["kind"]=="ph2_run"].sort_values("ov_m", ascending=False)
    md.append(f"\n### PH2 runs ({len(ph2_sub)} 개, overall 내림차순)\n")
    md.append("| # | run_name | seeds (hparam) | SP mean±std | XP mean±std | pos_0 mean±std | pos_1 mean±std | overall mean±std |")
    md.append("|---:|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(ph2_sub.iterrows(), 1):
        hp = {}
        for key in ["e","o","s","k","ct"]:
            m = re.search(rf"_{key}(\d+p?\d*)(?=_|$)", r["name"])
            if m: hp[key] = m.group(1).replace("p",".")
        hp_str = " ".join([f"{k}={v}" for k,v in hp.items()])
        sp_str = fmt_mean_std(r["sp_mean"], r["sp_std"], w=10) if r["sp_mean"] is not None and not pd.isna(r["sp_mean"]) else "-"
        xp_str = fmt_mean_std(r["xp_mean"], r["xp_std"], w=10) if r["xp_mean"] is not None and not pd.isna(r["xp_mean"]) else "-"
        p0 = fmt_mean_std(r["p0_m"], r["p0_s"], w=10)
        p1 = fmt_mean_std(r["p1_m"], r["p1_s"], w=10)
        ov = fmt_mean_std(r["ov_m"], r["ov_s"], w=10)
        mark = " ⭐" if r["is_final_ph2"] else ""
        short = r["name"].split("_e3t_ph2_")[0] + mark
        md.append(f"| {i} | {short} | {hp_str} | {sp_str} | {xp_str} | {p0} | {p1} | {ov} |")

md.append("\n## Final 갱신 제안\n")
md.append("| layout | 현재 final | 제안 | overall 개선 |")
md.append("|---|---|---|---:|")
for L in LAYOUTS:
    sub = df[(df["layout"]==L) & (df["kind"]=="ph2_run")]
    fin = sub[sub["is_final_ph2"]]
    best = sub.sort_values("ov_m", ascending=False).head(1)
    if len(fin) and len(best) and fin.iloc[0]["name"] != best.iloc[0]["name"]:
        delta = best.iloc[0]["ov_m"] - fin.iloc[0]["ov_m"]
        md.append(f"| {L} | {fin.iloc[0]['name'].split('_e3t_ph2_')[0]} ({fin.iloc[0]['ov_m']:.1f}) | {best.iloc[0]['name'].split('_e3t_ph2_')[0]} ({best.iloc[0]['ov_m']:.1f}) | +{delta:.1f} |")
    else:
        md.append(f"| {L} | 현 final 이 최고 | - | - |")

md.append("\n## 관찰\n")
md.append("- **cramped_room / coord_ring**: PH2 가 전 algo 1위. final 갱신 시 +5~10 점 개선 가능.")
md.append("- **asymm_advantages**: E3T 가 235.7 로 압도적. PH2 는 pos_0 에서 구조적 한계 (최고 67). pos_1 만 300+ 로 잘함.")
md.append("- **counter_circuit**: MEP 가 83.1 로 1위. PH2 는 최고 77.4. **ω=8** 계열이 우수.")
md.append("- **forced_coord**: E3T 가 30.7 로 근소 1위. PH2 최고 29.5 로 비슷. CEC 는 2.6 (canonical 원본 기준).")
md.append("- **pos 비대칭**: asymm 은 pos_1 편중 (ph2 54 vs 289), forced_coord 는 pos_0 편중 (pos_0≥25 ≥ pos_1).")

(out_dir/"ph2_full_eval_report.md").write_text("\n".join(md), encoding="utf-8")
print(f"Saved Markdown: {out_dir/'ph2_full_eval_report.md'}  ({len(md)} lines)")
