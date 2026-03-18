#!/usr/bin/env bash
# run_factory_mep.sh — MEP S1 → S2 자동 순차 실행
# Usage: ./run_factory_mep.sh [--layout LAYOUT] [--gpus GPU_ID]
#
# 1) MEP S1 training → saves population to RUN_BASE_DIR/mep_population/
# 2) Reads the latest run's mep_population dir
# 3) MEP S2 training using that population
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAYOUT="${LAYOUT:-cramped_room}"
GPU="${GPU:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout) LAYOUT="$2"; shift 2;;
    --gpus)   GPU="$2"; shift 2;;
    *)        EXTRA_ARGS+=("$1"); shift;;
  esac
done

echo "=========================================="
echo "[MEP] Layout=${LAYOUT}  GPU=${GPU}"
echo "=========================================="

# ----------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------
echo ""
echo "[MEP] ===== Stage 1: Population Training ====="
CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${SCRIPT_DIR}/run_user_wandb.sh" \
    --exp rnn-mep-s1 \
    --layout "${LAYOUT}" \
    --gpus "${GPU}" \
    --seeds 1 \
    "${EXTRA_ARGS[@]}"

echo "[MEP] Stage 1 complete."

# ----------------------------------------------------------------
# Find the latest run directory to locate mep_population/
# The RUN_BASE_DIR is written by run.py into the config; we locate it
# by finding the most recently modified run directory.
# ----------------------------------------------------------------
RUNS_BASE="${PROJECT_DIR}/runs"

# Find the most recently modified run directory that contains mep_population/
POP_DIR=$(
  find "${RUNS_BASE}" -maxdepth 3 -type d -name "mep_population" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}'
)

if [[ -z "${POP_DIR}" ]]; then
  echo "[ERROR] Could not find mep_population/ under ${RUNS_BASE}" >&2
  echo "        Please run Stage 2 manually with:"
  echo "        ./run_factory_mep_s2.sh --pop-dir <path> --layout ${LAYOUT} --gpus ${GPU}"
  exit 1
fi

echo "[MEP] Found S1 population: ${POP_DIR}"

# ----------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------
echo ""
echo "[MEP] ===== Stage 2: Adaptive Agent Training ====="
CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${SCRIPT_DIR}/run_user_wandb.sh" \
    --exp rnn-mep-s2 \
    --layout "${LAYOUT}" \
    --gpus "${GPU}" \
    --seeds 1 \
    "${EXTRA_ARGS[@]}" \
    "+MEP_POPULATION_DIR=${POP_DIR}"

echo "[MEP] Stage 2 complete."
echo "[MEP] ===== MEP Pipeline Done ====="
