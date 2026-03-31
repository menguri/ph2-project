#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

DRY_RUN=false
FORWARD_ARGS=()

# ---------------------------------------------------------------------
# Preset batch mode (used when CLI args are empty):
# - Auto-discover runs under AUTO_RUNS_DIR.
# - Include run dirs with date prefix >= AUTO_DATE_FROM_YYYYMMDD.
# - Or, if AUTO_AFTER_RUN is set, include run dirs strictly after that run
#   in sorted directory order.
# - Execute --eval-analysis in parallel batches of AUTO_BATCH_SIZE.
# ---------------------------------------------------------------------
AUTO_DISCOVER_PRESETS=false
AUTO_RUNS_DIR="../runs"
AUTO_DATE_FROM_YYYYMMDD=20260301
AUTO_AFTER_RUN=""
AUTO_BEFORE_RUN=""
AUTO_FROM_RUN="20260303-000027_31hadyvh_grounded_coord_simple_e3t_ph2"
AUTO_TO_RUN="20260303-192651_a0t7uvpt_grounded_coord_simple_e3t_ph2_ct1_e0p2_o20_s3"
AUTO_GPU_IDX=2
AUTO_BATCH_SIZE=2
AUTO_PRESET_EVAL_MODE="cross-play" # cross-play | eval-analysis | eval-viz
AUTO_MAX_STEPS=400
AUTO_PH1_CROSS_NUM_SEEDS=3
AUTO_PH1_CROSS_NUM_RECENT_TILDES=10
AUTO_PH2_CROSS_NUM_SEEDS=5

# Optional manual fallback commands (used only when AUTO_DISCOVER_PRESETS=false).
PRESET_FACTORY_COMMANDS=(
  # ToyCoop_Ran PH2 cross-play re-eval (ind policy, 10 seeds) — GPU 6,7
  "./run_visualize.sh --gpu 6 --dir runs/20260328-133416_qfkxpo37_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-143817_wo2r0bqt_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260328-153110_m7op22zg_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-162318_3goextvw_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260328-171532_o7pc4sou_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-180752_ri4wgq22_ToyCoop_Ran_e3t_ph2_e0p2_o0p1_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260328-190000_286uynm6_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-195200_ox12wgdz_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260328-204418_nadbm4qx_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-213618_usj8zw39_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260328-222810_hg0fdek8_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260328-232029_bzaihl92_ToyCoop_Ran_e3t_ph2_e0p2_o0p5_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-001226_6u9s4bpt_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-010433_3a4jvzht_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-015621_0ksq0soq_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-024909_q0t4mf0z_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-034059_utnfavax_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-043243_cn3l3tdp_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-052431_iq455xaw_ToyCoop_Ran_e3t_ph2_e0p2_o3_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-061626_qu45k9sl_ToyCoop_Ran_e3t_ph2_e0p2_o3_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-070824_7oi7g824_ToyCoop_Ran_e3t_ph2_e0p2_o3_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-080008_49m90m7a_ToyCoop_Ran_e3t_ph2_e0p2_o3_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-085150_uarm4qv7_ToyCoop_Ran_e3t_ph2_e0p2_o3_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-090658_03xfkxgb_ToyCoop_Ran_e3t_ph2_e0p2_o2_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-094344_eu6l69a0_ToyCoop_Ran_e3t_ph2_e0p2_o3_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-100103_n4jymnlu_ToyCoop_Ran_e3t_ph2_e0p2_o2_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-105725_e7qby54p_ToyCoop_Ran_e3t_ph2_e0p2_o2_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-122407_vy56jwcp_ToyCoop_Ran_e3t_ph2_e0p2_o100_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-131850_scpukz5b_ToyCoop_Ran_e3t_ph2_e0p2_o100_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-141312_ywvfofff_ToyCoop_Ran_e3t_ph2_e0p2_o100_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-150845_hs570crn_ToyCoop_Ran_e3t_ph2_e0p2_o100_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-160427_vmgw147h_ToyCoop_Ran_e3t_ph2_e0p2_o100_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-170016_bca8txn4_ToyCoop_Ran_e3t_ph2_e0p2_o100_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-175520_9o54b9rm_ToyCoop_Ran_e3t_ph2_e0p2_o100_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-185021_ezxmt0rg_ToyCoop_Ran_e3t_ph2_e0p2_o100_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-194524_0uxuv0ak_ToyCoop_Ran_e3t_ph2_e0p2_o100_s7_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-204053_b310oa3o_ToyCoop_Ran_e3t_ph2_e0p2_o100_s7_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-213559_zx67wzf0_ToyCoop_Ran_e3t_ph2_e0p2_o100_s9_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260329-223131_v6ylwytr_ToyCoop_Ran_e3t_ph2_e0p2_o100_s9_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260329-232707_uuonjxuj_ToyCoop_Ran_e3t_ph2_e0p2_o400_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-002211_705giyzf_ToyCoop_Ran_e3t_ph2_e0p2_o400_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-011758_zdfrfq6i_ToyCoop_Ran_e3t_ph2_e0p2_o400_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-021335_a5yq2ubb_ToyCoop_Ran_e3t_ph2_e0p2_o400_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-030916_4xgq8gru_ToyCoop_Ran_e3t_ph2_e0p2_o400_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-034641_0n8zn42h_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-044207_knmd3wyf_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-053717_acp343h9_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-063246_0rjv0q2v_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-072740_m3jvspvh_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-082241_tghwswy3_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-091711_76t1yru5_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-101251_q6myqegu_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-110800_fj0lnjae_ToyCoop_Ran_e3t_ph2_e0p2_o2_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-120355_8daxrou1_ToyCoop_Ran_e3t_ph2_e0p2_o2_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-125913_2frd7roi_ToyCoop_Ran_e3t_ph2_e0p2_o2_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-132356_b0qco6yw_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-135323_o3nv3604_ToyCoop_Ran_e3t_ph2_e0p2_o2_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-141618_20rmxmq8_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-144414_0zv4p7vh_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-144729_to88amdc_ToyCoop_Ran_e3t_ph2_e0p2_o10_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-153754_n0qllad6_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-154307_x91zajuq_ToyCoop_Ran_e3t_ph2_e0p2_o10_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-163123_ck4b41zl_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-163911_rlj8yg24_ToyCoop_Ran_e3t_ph2_e0p2_o10_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-172452_3wztma91_ToyCoop_Ran_e3t_ph2_e0p2_o1_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-173512_n8scx9su_ToyCoop_Ran_e3t_ph2_e0p2_o10_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-181805_nqe2zoz0_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-183101_vuq4dg42_ToyCoop_Ran_e3t_ph2_e0p2_o10_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-191103_9jun3fgf_ToyCoop_Ran_e3t_ph2_e0p2_o1_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-192534_dcrksf1a_ToyCoop_Ran_e3t_ph2_e0p2_o10_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-200531_7cuaalsc_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-202031_1zz8w0q7_ToyCoop_Ran_e3t_ph2_e0p2_o10_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-205814_rfy7eiwt_ToyCoop_Ran_e3t_ph2_e0p2_o1_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-211500_hlkvw540_ToyCoop_Ran_e3t_ph2_e0p2_o10_s1_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-215203_mckgt1sh_ToyCoop_Ran_e3t_ph2_e0p2_o2_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-221051_blc7rjgh_ToyCoop_Ran_e3t_ph2_e0p2_o20_s5_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-224505_0jglasjc_ToyCoop_Ran_e3t_ph2_e0p2_o2_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260330-230553_cs9h4dqk_ToyCoop_Ran_e3t_ph2_e0p2_o20_s5_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260330-233727_96mqsu94_ToyCoop_Ran_e3t_ph2_e0p2_o2_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260331-000101_dp1w1gd7_ToyCoop_Ran_e3t_ph2_e0p2_o20_s3_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260331-003126_x3d6jo0x_ToyCoop_Ran_e3t_ph2_e0p2_o2_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260331-005600_acd7i8jj_ToyCoop_Ran_e3t_ph2_e0p2_o20_s3_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260331-012502_9wqct816_ToyCoop_Ran_e3t_ph2_e0p2_o2_s2_k1_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260331-014838_xl3dsy0v_ToyCoop_Ran_e3t_ph2_e0p2_o1_s5_k1_ct0_pair --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260331-021736_oo9t59j5_ToyCoop_Ran_e3t_ph2_e0p2_o2_s2_k2_ct0 --cross --num_seeds 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260331-031007_vibbzqx6_ToyCoop_Ran_e3t_ph2_e0p2_o2_s1_k1_ct0 --cross --num_seeds 10 --no_viz"
)

usage() {
  cat <<EOF
Usage:
  $0                            # run preset batch (auto-discovered by default)
  $0 <run_visualize.sh options...>

Factory options:
  --dry-run                      Print commands without executing
  --cross-play                   Preset batch mode selector
  --eval-analysis                Preset batch mode selector (default)
  --eval-viz                     Preset batch mode selector
  --after-run <run_dir_name>     Use runs strictly after this folder (sorted order)
  --before-run <run_dir_name>    Use runs up to this folder (inclusive, sorted order)
  --max-steps <N>                Eval rollout max steps override for preset commands

Preset batch defaults:
  runs dir:                      $AUTO_RUNS_DIR
  date from:                     $AUTO_DATE_FROM_YYYYMMDD
  after run:                     ${AUTO_AFTER_RUN:-<disabled>}
  before run:                    ${AUTO_BEFORE_RUN:-<disabled>}
  from run (inclusive):          ${AUTO_FROM_RUN:-<disabled>}
  to run (inclusive):            ${AUTO_TO_RUN:-<disabled>}
  gpu:                           $AUTO_GPU_IDX
  parallel batch size:           $AUTO_BATCH_SIZE
  preset eval mode:              $AUTO_PRESET_EVAL_MODE
  max steps:                     $AUTO_MAX_STEPS
  ph1 cross:                     num_seeds=$AUTO_PH1_CROSS_NUM_SEEDS num_recent_tildes=$AUTO_PH1_CROSS_NUM_RECENT_TILDES
  ph2 cross:                     num_seeds=$AUTO_PH2_CROSS_NUM_SEEDS

Forwarded run_visualize.sh options (examples):
  --cross --num_seeds N          Standard visualize/eval mode
  --latent_analysis              Latent analysis mode
  --value_analysis               Value analysis mode
  --eval-analysis                Offline eval analysis csv mode (video disabled)
                                 Equivalent output: <run_dir>/eval/offline_eval_analysis.csv
  --eval-viz                     Offline eval final-checkpoint video mode
                                 Equivalent output: <run_dir>/eval/video/*.gif

Examples:
  $0 --cross-play                # run preset batch with phase-aware cross-play
  $0 --eval-viz                  # run preset batch with eval-viz
  $0 --eval-analysis --after-run 20260216-104647_04d61n7i_grounded_coord_simple_e3t_ph1
  $0 --eval-analysis --before-run 20260219-114500_abcd1234_counter_circuit_e3t_ph1
  $0 --dry-run --eval-analysis   # print preset eval-analysis commands
  $0 --gpu 0 --dir runs/20260128-015425_9usv82dt_counter_circuit_fcp --eval-analysis
  $0 --gpu 0 --dir runs/20260210-143629_wsvfmd7x_counter_circuit_e3t_ph1 --cross --num_seeds 5 --no_viz
EOF
}

build_auto_preset_commands() {
  local runs_root="$AUTO_RUNS_DIR"
  local date_from="$AUTO_DATE_FROM_YYYYMMDD"
  local after_run="$AUTO_AFTER_RUN"
  local before_run="$AUTO_BEFORE_RUN"
  local from_run="$AUTO_FROM_RUN"
  local to_run="$AUTO_TO_RUN"
  local gpu_idx="$AUTO_GPU_IDX"
  local preset_mode="$AUTO_PRESET_EVAL_MODE"
  if [[ ! -d "$runs_root" ]]; then
    return 0
  fi

  local use_after=false
  local use_before=false
  local use_from=false
  local use_to=false
  local after_found=false
  local before_found=false
  local from_found=false
  local to_found=false
  if [[ -n "$after_run" ]]; then
    use_after=true
  fi
  if [[ -n "$before_run" ]]; then
    use_before=true
  fi
  if [[ -n "$from_run" ]]; then
    use_from=true
  fi
  if [[ -n "$to_run" ]]; then
    use_to=true
  fi

  local run_dir=""
  local run_base=""
  local run_date=""
  local run_base_lc=""
  while IFS= read -r run_dir; do
    run_base="$(basename "$run_dir")"
    if [[ "$use_from" == "true" ]]; then
      if [[ "$from_found" == "false" ]]; then
        if [[ "$run_base" == "$from_run" ]]; then
          from_found=true
        else
          continue
        fi
      fi
    elif [[ "$use_after" == "true" ]]; then
      if [[ "$after_found" == "false" ]]; then
        if [[ "$run_base" == "$after_run" ]]; then
          after_found=true
        fi
        continue
      fi
    else
      run_date="${run_base:0:8}"
      if [[ ! "$run_date" =~ ^[0-9]{8}$ ]] || (( run_date < date_from )); then
        continue
      fi
    fi

    run_base_lc="$(echo "$run_base" | tr '[:upper:]' '[:lower:]')"
    # ToyCoop은 max_steps=100
    local _max_steps="$AUTO_MAX_STEPS"
    if [[ "$run_base_lc" == *"toycoop"* ]]; then
      _max_steps=100
    fi
    case "$preset_mode" in
      cross-play)
        if [[ "$run_base_lc" == *"ph1"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --cross --num_seeds $AUTO_PH1_CROSS_NUM_SEEDS --num_recent_tildes $AUTO_PH1_CROSS_NUM_RECENT_TILDES --max_steps $_max_steps"
        elif [[ "$run_base_lc" == *"ph2"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --cross --num_seeds $AUTO_PH2_CROSS_NUM_SEEDS --max_steps $_max_steps"
        fi
        ;;
      eval-viz)
        if [[ "$run_base_lc" == *"ph1"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --$preset_mode --max_steps $_max_steps"
        fi
        ;;
      eval-analysis)
        echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --$preset_mode --max_steps $_max_steps"
        ;;
      *)
        echo "[WARN] unsupported AUTO_PRESET_EVAL_MODE: $preset_mode" >&2
        return 0
        ;;
    esac

    if [[ "$use_to" == "true" && "$run_base" == "$to_run" ]]; then
      to_found=true
      break
    fi
    if [[ "$use_before" == "true" && "$run_base" == "$before_run" ]]; then
      before_found=true
      break
    fi
  done < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ "$use_after" == "true" && "$after_found" == "false" ]]; then
    echo "[WARN] --after-run target not found: $after_run" >&2
  fi
  if [[ "$use_from" == "true" && "$from_found" == "false" ]]; then
    echo "[WARN] from-run target not found: $from_run" >&2
  fi
  if [[ "$use_before" == "true" && "$before_found" == "false" ]]; then
    echo "[WARN] --before-run target not found: $before_run" >&2
  fi
  if [[ "$use_to" == "true" && "$to_found" == "false" ]]; then
    echo "[WARN] to-run target not found: $to_run" >&2
  fi
}

run_preset_factory_commands() {
  local total=0
  local launched=0
  local completed=0
  local success=0
  local failed=0
  local batch_size="$AUTO_BATCH_SIZE"
  local -a commands=()
  local -a pids=()
  local -a pid_cmds=()

  if [[ ! -f "./run_visualize.sh" ]]; then
    echo "[ERROR] run_visualize.sh not found in $(pwd)"
    exit 1
  fi

  if [[ "$AUTO_DISCOVER_PRESETS" == "true" ]]; then
    mapfile -t commands < <(build_auto_preset_commands)
  else
    # Accept both styles:
    # 1) each entry is a full command string (recommended)
    # 2) accidentally tokenized entries (unquoted command in array)
    mapfile -t commands < <(normalize_manual_preset_commands)
  fi

  echo "=== Preset Factory Commands ==="
  echo "[INFO] auto_discover=$AUTO_DISCOVER_PRESETS runs_dir=$AUTO_RUNS_DIR date_from=$AUTO_DATE_FROM_YYYYMMDD after_run=${AUTO_AFTER_RUN:-<disabled>} before_run=${AUTO_BEFORE_RUN:-<disabled>} from_run=${AUTO_FROM_RUN:-<disabled>} to_run=${AUTO_TO_RUN:-<disabled>} batch_size=$batch_size mode=$AUTO_PRESET_EVAL_MODE max_steps=$AUTO_MAX_STEPS"

  for cmd in "${commands[@]}"; do
    [[ -z "$cmd" ]] && continue
    if [[ "$cmd" != *"--max_steps"* && "$cmd" != *"--max-steps"* ]]; then
      # ToyCoop은 max_steps=100
      local _fallback_steps="$AUTO_MAX_STEPS"
      local _cmd_lc="$(echo "$cmd" | tr '[:upper:]' '[:lower:]')"
      if [[ "$_cmd_lc" == *"toycoop"* ]]; then
        _fallback_steps=100
      fi
      cmd="$cmd --max_steps $_fallback_steps"
    fi
    total=$((total + 1))
    echo "[CMD] $cmd"
    if [[ "$DRY_RUN" != "true" ]]; then
      bash -lc "$cmd" &
      pids+=("$!")
      pid_cmds+=("$cmd")
      launched=$((launched + 1))

      if [[ ${#pids[@]} -ge "$batch_size" ]]; then
        for i in "${!pids[@]}"; do
          if wait "${pids[$i]}"; then
            success=$((success + 1))
          else
            failed=$((failed + 1))
            echo "[WARN] command failed: ${pid_cmds[$i]}"
          fi
          completed=$((completed + 1))
        done
        pids=()
        pid_cmds=()
      fi
    fi
  done

  if [[ "$DRY_RUN" != "true" && ${#pids[@]} -gt 0 ]]; then
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        success=$((success + 1))
      else
        failed=$((failed + 1))
        echo "[WARN] command failed: ${pid_cmds[$i]}"
      fi
      completed=$((completed + 1))
    done
  fi

  echo "==============================="
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] preset commands: total=$total (dry-run)"
  else
    echo "[INFO] preset commands: total=$total launched=$launched completed=$completed success=$success failed=$failed"
  fi
  if [[ "$total" -eq 0 ]]; then
    echo "[INFO] no preset commands found."
  fi
}

normalize_manual_preset_commands() {
  local -a raw=("${PRESET_FACTORY_COMMANDS[@]}")
  local token=""
  local has_split_tokens=false

  for token in "${raw[@]}"; do
    if [[ "$token" == "./run_visualize.sh" || "$token" == "run_visualize.sh" ]]; then
      has_split_tokens=true
      break
    fi
  done

  if [[ "$has_split_tokens" != "true" ]]; then
    printf '%s\n' "${raw[@]}"
    return
  fi

  local current_cmd=""
  for token in "${raw[@]}"; do
    [[ -z "$token" ]] && continue
    if [[ "$token" == "./run_visualize.sh" || "$token" == "run_visualize.sh" ]]; then
      if [[ -n "$current_cmd" ]]; then
        printf '%s\n' "$current_cmd"
      fi
      current_cmd="$token"
    else
      if [[ -z "$current_cmd" ]]; then
        current_cmd="$token"
      else
        current_cmd+=" $token"
      fi
    fi
  done
  if [[ -n "$current_cmd" ]]; then
    printf '%s\n' "$current_cmd"
  fi
}

if [[ $# -eq 0 ]]; then
  run_preset_factory_commands
  exit 0
fi

# If only factory flags are given, run preset batch mode.
factory_only=true
idx=1
while [[ $idx -le $# ]]; do
  arg="${!idx}"
  case "$arg" in
    --dry-run|--cross-play|--eval-analysis|--eval-viz)
      ;;
    --after-run)
      idx=$((idx + 1))
      if [[ $idx -gt $# ]]; then
        echo "[ERROR] --after-run requires a run directory name"
        exit 1
      fi
      ;;
    --before-run)
      idx=$((idx + 1))
      if [[ $idx -gt $# ]]; then
        echo "[ERROR] --before-run requires a run directory name"
        exit 1
      fi
      ;;
    --max-steps)
      idx=$((idx + 1))
      if [[ $idx -gt $# ]]; then
        echo "[ERROR] --max-steps requires a value"
        exit 1
      fi
      ;;
    *)
      factory_only=false
      break
      ;;
  esac
  idx=$((idx + 1))
done

if [[ "$factory_only" == "true" ]]; then
  boundary_overridden=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --eval-analysis)
        AUTO_PRESET_EVAL_MODE="eval-analysis"
        shift
        ;;
      --cross-play)
        AUTO_PRESET_EVAL_MODE="cross-play"
        shift
        ;;
      --eval-viz)
        AUTO_PRESET_EVAL_MODE="eval-viz"
        shift
        ;;
      --after-run)
        if [[ "$boundary_overridden" == "false" ]]; then
          AUTO_AFTER_RUN=""
          AUTO_BEFORE_RUN=""
          boundary_overridden=true
        fi
        shift
        if [[ $# -eq 0 ]]; then
          echo "[ERROR] --after-run requires a run directory name"
          exit 1
        fi
        AUTO_AFTER_RUN="$1"
        shift
        ;;
      --before-run)
        if [[ "$boundary_overridden" == "false" ]]; then
          AUTO_AFTER_RUN=""
          AUTO_BEFORE_RUN=""
          boundary_overridden=true
        fi
        shift
        if [[ $# -eq 0 ]]; then
          echo "[ERROR] --before-run requires a run directory name"
          exit 1
        fi
        AUTO_BEFORE_RUN="$1"
        shift
        ;;
      --max-steps)
        shift
        if [[ $# -eq 0 ]]; then
          echo "[ERROR] --max-steps requires a value"
          exit 1
        fi
        AUTO_MAX_STEPS="$1"
        shift
        ;;
    esac
  done
  run_preset_factory_commands
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval|--training-eval|--eval_analysis)
      echo "[ERROR] $1 is removed from viz_factory.sh."
      echo "        Use run_visualize.sh --eval-analysis instead."
      echo "        Example: ./run_visualize.sh --gpu 0 --dir runs/<run_dir> --eval-analysis"
      exit 1
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "./run_visualize.sh" ]]; then
  echo "[ERROR] run_visualize.sh not found in $(pwd)"
  exit 1
fi

if [[ ${#FORWARD_ARGS[@]} -eq 0 ]]; then
  echo "[ERROR] no run_visualize.sh arguments given"
  usage
  exit 1
fi

cmd=(./run_visualize.sh "${FORWARD_ARGS[@]}")
echo "[CMD] ${cmd[*]}"
if [[ "$DRY_RUN" != "true" ]]; then
  "${cmd[@]}"
fi
