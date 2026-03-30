#!/usr/bin/env bash
# GAMMA 통합 파이프라인: S1 + S2를 하나의 run에서 자동 진행.
# method=rl (기본) 또는 method=vae
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

METHOD="${METHOD:-rl}"     # rl | vae
S1_GPU="6"
S2_GPUS="6,7"

run_gamma() {
  local gpus=$1
  local layout=$2
  local method=${3:-$METHOD}

  echo "[GAMMA] gpus=${gpus} layout=${layout} method=${method}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-gamma \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1 \
      --extra "++GAMMA_S2_METHOD=${method}"
}

# 개별 S1/S2도 여전히 지원 (레거시)
run_gamma_s1() {
  local gpus=$1; local layout=$2
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-gamma-s1 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 1
}

run_gamma_s2() {
  local gpus=$1; local layout=$2; local pop_dir=$3
  CUDA_VISIBLE_DEVICES="${gpus}" ./run_user_wandb.sh \
    --exp rnn-gamma-s2 --env "${layout}" --layout "${layout}" --gpus "${gpus}" --seeds 10 \
    --gamma-pop-dir "${pop_dir}"
}

# --- Unified runs ---
for layout in cramped_room asymm_advantages coord_ring; do
  echo "===== GAMMA ${METHOD} - ${layout} ====="
  run_gamma "$S1_GPU" "$layout" "$METHOD"
done
