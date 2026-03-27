#!/usr/bin/env bash
# HSP 통합 파이프라인: S1 + Greedy Selection + S2를 하나의 run에서 자동 진행.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

S1_GPU="6"
S2_GPUS="6,7"

run_hsp() {
  local gpus=$1
  local layout=$2

  echo "[HSP] gpus=${gpus} layout=${layout}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-hsp \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1
}

# 개별 S1/S2도 여전히 지원 (레거시)
run_hsp_s1() {
  local gpus=$1; local layout=$2
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-hsp-s1 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 1
}

run_hsp_s2() {
  local gpus=$1; local layout=$2; local pop_dir=$3
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-hsp-s2 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 10 \
    --hsp-pop-dir "${pop_dir}"
}

# --- Unified runs ---
for layout in cramped_room asymm_advantages coord_ring; do
  echo "===== HSP - ${layout} ====="
  run_hsp "$S1_GPU" "$layout"
done
