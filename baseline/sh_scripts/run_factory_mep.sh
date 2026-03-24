#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_BASE="${PROJECT_DIR}/runs"

# -----------------------------------------------------------------------------
# MEP Experiment Factory
# S1 → S2 순차 실행. 밑에 run_mep_s1 / run_mep_s2 호출만 조정하면 됨.
# -----------------------------------------------------------------------------

S1_SEEDS=1       # S1은 풀 하나 (seed 고정)
S2_SEEDS=10      # S2는 다른 baseline과 동일한 multi-seed

run_mep_s1() {
  local gpus=$1
  local layout=$2

  echo "[MEP S1] gpus=${gpus} layout=${layout}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-mep-s1 \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds "${S1_SEEDS}"
}

find_latest_pop_dir() {
  find "${RUNS_BASE}" -maxdepth 3 -type d -name "mep_population" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}'
}

run_mep_s2() {
  local gpus=$1
  local layout=$2
  local pop_dir=$3

  if [[ -z "${pop_dir}" || ! -d "${pop_dir}" ]]; then
    echo "[ERROR] run_mep_s2: invalid pop_dir='${pop_dir}'" >&2
    exit 1
  fi

  echo "[MEP S2] gpus=${gpus} layout=${layout} seeds=${S2_SEEDS}"
  echo "[MEP S2] pop_dir=${pop_dir}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-mep-s2 \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds "${S2_SEEDS}" \
      --mep-pop-dir "${pop_dir}"
}

# -----------------------------------------------------------------------------
# Sweep 설정 — 여기만 수정하면 됨
# S1: single GPU (vmap으로 population 병렬화, multi-GPU 불필요)
# S2: multi-GPU (multi-seed pmap+vmap)
# -----------------------------------------------------------------------------
S1_GPU="6"
S2_GPUS="6,7"

echo "[MEP] layout=cramped_room"
run_mep_s1 "$S1_GPU" "cramped_room"
# POP_DIR="${RUNS_BASE}/20260319-034649_ztpeud5n_cramped_room_m1/mep_population"
# run_mep_s2 "$S2_GPUS" "cramped_room" "$POP_DIR"

echo "[MEP] layout=asymm_advantages"
run_mep_s1 "$S1_GPU" "asymm_advantages"
# POP_DIR="${RUNS_BASE}/20260319-051106_05r4fmok_asymm_advantages_m1/mep_population"
# run_mep_s2 "$S2_GPUS" "asymm_advantages" "$POP_DIR"

echo "[MEP] layout=coord_ring"
run_mep_s1 "$S1_GPU" "coord_ring"
# POP_DIR="${RUNS_BASE}/20260319-071343_1gan15ov_coord_ring_m1/mep_population"
# run_mep_s2 "$S2_GPUS" "coord_ring" "$POP_DIR"

# echo "[MEP] layout=counter_circuit"
# run_mep_s1 "6" "counter_circuit" &
# POP_DIR="${RUNS_BASE}/20260320-072007_v34ebc7q_counter_circuit_m1/mep_population"
# run_mep_s2 "$S2_GPUS" "counter_circuit" "$POP_DIR"

# echo "[MEP] layout=forced_coord" & 
# run_mep_s1 "7" "forced_coord" &
# POP_DIR="${RUNS_BASE}/20260319-061341_zq0tpuaw_forced_coord_m1/mep_population"
# run_mep_s2 "$S2_GPUS" "forced_coord" "$POP_DIR"

# --- OV2 Layouts ---
# echo "[MEP] layout=grounded_coord_simple"
# run_mep_s1 "$S1_GPU" "grounded_coord_simple"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "grounded_coord_simple" "$POP_DIR"

# echo "[MEP] layout=grounded_coord_ring"
# run_mep_s1 "$S1_GPU" "grounded_coord_ring"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "grounded_coord_ring" "$POP_DIR"

# echo "[MEP] layout=demo_cook_simple"
# run_mep_s1 "$S1_GPU" "demo_cook_simple"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "demo_cook_simple" "$POP_DIR"

# echo "[MEP] layout=demo_cook_wide"
# run_mep_s1 "$S1_GPU" "demo_cook_wide"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "demo_cook_wide" "$POP_DIR"

# echo "[MEP] layout=test_time_simple"
# run_mep_s1 "$S1_GPU" "test_time_simple"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "test_time_simple" "$POP_DIR"

# echo "[MEP] layout=test_time_wide"
# run_mep_s1 "$S1_GPU" "test_time_wide"
# POP_DIR=""
# run_mep_s2 "$S2_GPUS" "test_time_wide" "$POP_DIR"

# echo "[MEP] all layout jobs finished."
