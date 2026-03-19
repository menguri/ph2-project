#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
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
      "+MEP_POPULATION_DIR=${pop_dir}"
}

# -----------------------------------------------------------------------------
# Sweep 설정 — 여기만 수정하면 됨
# S1: single GPU (vmap으로 population 병렬화, multi-GPU 불필요)
# S2: multi-GPU (multi-seed pmap+vmap)
# -----------------------------------------------------------------------------
S1_GPU="0"
S2_GPUS="0,1,2,3,4"

echo "[MEP] layout=cramped_room"
run_mep_s1 "$S1_GPU" "cramped_room"
POP_DIR=$(find_latest_pop_dir)
run_mep_s2 "$S2_GPUS" "cramped_room" "$POP_DIR"

echo "[MEP] layout=asymm_advantages"
run_mep_s1 "$S1_GPU" "asymm_advantages"
POP_DIR=$(find_latest_pop_dir)
run_mep_s2 "$S2_GPUS" "asymm_advantages" "$POP_DIR"

echo "[MEP] layout=coord_ring"
run_mep_s1 "$S1_GPU" "coord_ring"
POP_DIR=$(find_latest_pop_dir)
run_mep_s2 "$S2_GPUS" "coord_ring" "$POP_DIR"

echo "[MEP] layout=forced_coord"
run_mep_s1 "$S1_GPU" "forced_coord"
POP_DIR=$(find_latest_pop_dir)
run_mep_s2 "$S2_GPUS" "forced_coord" "$POP_DIR"

echo "[MEP] layout=counter_circuit"
run_mep_s1 "$S1_GPU" "counter_circuit"
POP_DIR=$(find_latest_pop_dir)
run_mep_s2 "$S2_GPUS" "counter_circuit" "$POP_DIR"

echo "[MEP] all layout jobs finished."
