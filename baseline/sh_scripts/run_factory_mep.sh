#!/usr/bin/env bash
# MEP 통합 파이프라인: S1 + S2를 하나의 run에서 자동 진행.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_BASE="${PROJECT_DIR}/runs"

S1_GPU="6"
S2_GPUS="6,7"

# === 통합 실행 (권장) ===
run_mep() {
  local gpus=$1
  local layout=$2

  echo "[MEP] gpus=${gpus} layout=${layout}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-mep \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1
}

# === 개별 S1/S2 (레거시) ===
run_mep_s1() {
  local gpus=$1; local layout=$2
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-mep-s1 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 1
}

find_latest_pop_dir() {
  find "${RUNS_BASE}" -maxdepth 3 -type d -name "mep_population" \
    -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | awk '{print $2}'
}

run_mep_s2() {
  local gpus=$1; local layout=$2; local pop_dir=$3
  [[ -z "${pop_dir}" || ! -d "${pop_dir}" ]] && { echo "[ERROR] invalid pop_dir" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-mep-s2 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 10 \
    --mep-pop-dir "${pop_dir}"
}

# --- Unified runs ---
for layout in cramped_room asymm_advantages coord_ring; do
  echo "===== MEP - ${layout} ====="
  run_mep "$S1_GPU" "$layout"
done
