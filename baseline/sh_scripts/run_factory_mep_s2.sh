#!/usr/bin/env bash
# run_factory_mep_s2.sh — MEP Stage 2 단독 실행 (population 디렉토리 필요)
# Usage: ./run_factory_mep_s2.sh --pop-dir <path> [--layout LAYOUT] [--gpus GPU_IDs] [--seeds N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAYOUT="${LAYOUT:-cramped_room}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-10}"
POP_DIR=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout)  LAYOUT="$2"; shift 2;;
    --gpus)    GPU="$2"; shift 2;;
    --seeds)   SEEDS="$2"; shift 2;;
    --pop-dir) POP_DIR="$2"; shift 2;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${POP_DIR}" ]]; then
  echo "[ERROR] --pop-dir <mep_population_dir> is required for MEP S2" >&2
  exit 1
fi

# ---- Pool existence check ----
if [[ ! -d "${POP_DIR}" ]]; then
  echo "[ERROR] Population dir does not exist: ${POP_DIR}" >&2
  echo "        Run MEP Stage 1 first:" >&2
  echo "        ./run_factory_mep_s1.sh --layout ${LAYOUT} --gpus ${GPU}" >&2
  exit 1
fi

# Check that at least one member_{i}/ckpt_init_actor.pkl exists
MEMBER_COUNT=$(find "${POP_DIR}" -maxdepth 2 -name "ckpt_init_actor.pkl" 2>/dev/null | wc -l)
if [[ "${MEMBER_COUNT}" -eq 0 ]]; then
  echo "[ERROR] No ckpt_init_actor.pkl found under ${POP_DIR}" >&2
  echo "        The population may be incomplete — re-run Stage 1." >&2
  exit 1
fi

echo "=========================================="
echo "[MEP S2] Layout=${LAYOUT}  GPU=${GPU}  Seeds=${SEEDS}"
echo "[MEP S2] Population dir: ${POP_DIR} (${MEMBER_COUNT} init ckpts found)"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${SCRIPT_DIR}/run_user_wandb.sh" \
    --exp rnn-mep-s2 \
    --env "${LAYOUT}" \
    --layout "${LAYOUT}" \
    --gpus "${GPU}" \
    --seeds "${SEEDS}" \
    "${EXTRA_ARGS[@]}" \
    "+MEP_POPULATION_DIR=${POP_DIR}"

echo "[MEP S2] Done."
