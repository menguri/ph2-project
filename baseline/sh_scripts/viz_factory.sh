#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

# GridSpread baseline level-one cross-play — sparse (기본) + combined
PRESET_FACTORY_COMMANDS=(
  # GridSpread baseline level-one cross-play — combined (step_cost 미포함)
  # --per_ckpt_cross 
  "./run_visualize.sh --gpu 4 --dir runs/20260407-123622_s5iplg7d_GridSpread_e3t_h256_e128 --cross --per_ckpt_cross --num_seeds 5 --no_viz --max_steps 100"
  # "./run_visualize.sh --gpu 1 --dir runs/20260407-180000_dt6clc2k_GridSpread_sp_e128 --cross --num_seeds 5 --no_viz --max_steps 100"
  # "./run_visualize.sh --gpu 0 --dir runs/20260405-195201_fpre9e74_GridSpread_e3t --cross --cross_mode level_one --num_seeds 5 --no_viz --max_steps 100"
  # "./run_visualize.sh --gpu 1 --dir runs/20260405-204455_qtvxc1xs_GridSpread_mep_h64_pop5 --cross --cross_mode level_one --num_seeds 5 --no_viz --max_steps 100"
)

BATCH_SIZE=2

echo "=== Baseline Viz Factory: ${#PRESET_FACTORY_COMMANDS[@]} commands, batch_size=${BATCH_SIZE} ==="

pids=()
for i in "${!PRESET_FACTORY_COMMANDS[@]}"; do
  cmd="${PRESET_FACTORY_COMMANDS[$i]}"
  echo "[$(( i + 1 ))/${#PRESET_FACTORY_COMMANDS[@]}] $cmd"
  bash -lc "$cmd" &
  pids+=($!)

  if (( ${#pids[@]} >= BATCH_SIZE )); then
    for pid in "${pids[@]}"; do
      wait "$pid" || echo "[WARN] PID $pid exited with error"
    done
    pids=()
  fi
done

# 남은 프로세스 대기
for pid in "${pids[@]}"; do
  wait "$pid" || echo "[WARN] PID $pid exited with error"
done

echo "=== Baseline Viz Factory complete ==="
