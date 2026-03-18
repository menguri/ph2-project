#!/usr/bin/env bash
# run_factory_mep_s1.sh — MEP Stage 1 단독 실행
# Usage: ./run_factory_mep_s1.sh [--layout LAYOUT] [--gpus GPU_ID] [extra args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAYOUT="${LAYOUT:-cramped_room}"
GPU="${GPU:-0}"

# 나머지 인자는 run_user_wandb.sh로 전달
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout) LAYOUT="$2"; shift 2;;
    --gpus)   GPU="$2"; shift 2;;
    *)        EXTRA_ARGS+=("$1"); shift;;
  esac
done

echo "[MEP S1] Layout=${LAYOUT} GPU=${GPU}"
echo "[MEP S1] Starting Stage 1 training..."

CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${SCRIPT_DIR}/run_user_wandb.sh" \
    --exp rnn-mep-s1 \
    --layout "${LAYOUT}" \
    --gpus "${GPU}" \
    --seeds 1 \
    "${EXTRA_ARGS[@]}"

echo "[MEP S1] Done."
