#!/usr/bin/env bash
# run_factory_mep_s2.sh — MEP Stage 2 단독 실행 (population 디렉토리 필요)
# Usage: ./run_factory_mep_s2.sh --pop-dir <path> [--layout LAYOUT] [--gpus GPU_ID]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAYOUT="${LAYOUT:-cramped_room}"
GPU="${GPU:-0}"
POP_DIR=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout)  LAYOUT="$2"; shift 2;;
    --gpus)    GPU="$2"; shift 2;;
    --pop-dir) POP_DIR="$2"; shift 2;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${POP_DIR}" ]]; then
  echo "[ERROR] --pop-dir <mep_population_dir> is required for MEP S2" >&2
  exit 1
fi

echo "[MEP S2] Layout=${LAYOUT} GPU=${GPU}"
echo "[MEP S2] Population dir: ${POP_DIR}"
echo "[MEP S2] Starting Stage 2 training..."

CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${SCRIPT_DIR}/run_user_wandb.sh" \
    --exp rnn-mep-s2 \
    --layout "${LAYOUT}" \
    --gpus "${GPU}" \
    --seeds 1 \
    "${EXTRA_ARGS[@]}" \
    "+MEP_POPULATION_DIR=${POP_DIR}"

echo "[MEP S2] Done."
