#!/bin/bash

set -euo pipefail

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Default values
GPU_IDX=5
NUM_SEEDS=10
SEED=1
ALL_CKPT=false
CROSS=false
ALL=false
NO_VIZ=false
NO_RESET=false
PAIRING_POLICY=""
CROSS_MODE=""
EVAL_REWARD=""
DIRECTORY=""
MAX_STEPS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)            GPU_IDX="$2";          shift 2 ;;
        --dir)            DIRECTORY="$2";        shift 2 ;;
        --num_seeds)      NUM_SEEDS="$2";        shift 2 ;;
        --seed)           SEED="$2";             shift 2 ;;
        --all_ckpt)       ALL_CKPT=true;         shift ;;
        --cross)          CROSS=true;            shift ;;
        --all)            ALL=true;              shift ;;
        --no_viz)         NO_VIZ=true;           shift ;;
        --no_reset)       NO_RESET=true;         shift ;;
        --pairing_policy) PAIRING_POLICY="$2";   shift 2 ;;
        --cross_mode)     CROSS_MODE="$2";       shift 2 ;;
        --eval_reward)    EVAL_REWARD="$2";      shift 2 ;;
        --max_steps|--max-steps) MAX_STEPS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dir <directory> [options]"
            echo "Options:"
            echo "  --gpu <id>             GPU device ID (default: 0)"
            echo "  --dir <path>           Directory with checkpoints (required)"
            echo "  --num_seeds <n>        Number of evaluation seeds (default: 10)"
            echo "  --seed <n>             Random seed (default: 42)"
            echo "  --all_ckpt             Evaluate all checkpoints, not just final"
            echo "  --cross                Run cross-play evaluation"
            echo "  --all                  Run both self-play and cross-play"
            echo "  --no_viz               Skip video generation (only compute metrics)"
            echo "  --no_reset             Disable random reset and permutations"
            echo "  --pairing_policy <id>  Policy index for pairing in cross-play"
            echo "  --max_steps <n>        Eval rollout max steps override"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DIRECTORY" ]; then
    echo "Error: --dir <directory> is required"
    exit 1
fi

# Set GPU / CUDA environment
export CUDA_VISIBLE_DEVICES=$GPU_IDX
: "${CUDA_HOME:=/usr/local/cuda-12.9}"
export CUDA_HOME
export CUDA_ROOT="${CUDA_HOME}"   # nvidia-cuda-nvcc-cu12 __file__=None 버그 우회

# Disable WANDB for visualization
export WANDB_MODE=disabled

# JAX 플랫폼 설정: GPU 사용
export JAX_PLATFORMS=cuda

# Print configuration
echo "=== Visualization Configuration ==="
echo "GPU: $GPU_IDX"
echo "Directory: $DIRECTORY"
echo "Number of seeds: $NUM_SEEDS"
echo "Random seed: $SEED"
echo "All checkpoints: $ALL_CKPT"
echo "Cross-play: $CROSS"
echo "All modes: $ALL"
echo "No visualization: $NO_VIZ"
echo "No reset: $NO_RESET"
[ -n "$MAX_STEPS" ] && echo "Max steps override: $MAX_STEPS"
[ -n "$PAIRING_POLICY" ] && echo "Pairing policy: $PAIRING_POLICY"
echo "===================================="

# Build command arguments
ARGS=( --d "$DIRECTORY" --seed "$SEED" --num_seeds "$NUM_SEEDS" )

[ "$ALL_CKPT" = true ] && ARGS+=( --all_ckpt )
[ "$CROSS" = true ] && ARGS+=( --cross )
[ "$ALL" = true ] && ARGS+=( --all )
[ "$NO_VIZ" = true ] && ARGS+=( --no_viz )
[ "$NO_RESET" = true ] && ARGS+=( --no_reset )
[ -n "$PAIRING_POLICY" ] && ARGS+=( --pairing_policy "$PAIRING_POLICY" )
[ -n "$CROSS_MODE" ] && ARGS+=( --cross_mode "$CROSS_MODE" )
[ -n "$EVAL_REWARD" ] && ARGS+=( --eval_reward "$EVAL_REWARD" )
[ -n "$MAX_STEPS" ] && ARGS+=( --max_steps "$MAX_STEPS" )

# Change to experiments directory
SCRIPT_DIR_VIZ="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_VIZ="$(cd "${SCRIPT_DIR_VIZ}/../.." && pwd)"
PROJECT_DIR_VIZ="$(cd "${SCRIPT_DIR_VIZ}/.." && pwd)"

# Resolve python executable from project venv first.
# Avoid sourcing activate scripts because moved/broken venv paths can leave stale VIRTUAL_ENV values.
VENV_DIR_VIZ="${REPO_ROOT_VIZ}/overcooked_v2"
LEGACY_VENV_DIR_VIZ="${REPO_ROOT_VIZ}/overcookedv2"
PYTHON_BIN_VIZ=""

if [[ -x "${VENV_DIR_VIZ}/bin/python" ]]; then
    PYTHON_BIN_VIZ="${VENV_DIR_VIZ}/bin/python"
elif [[ -x "${LEGACY_VENV_DIR_VIZ}/bin/python" ]]; then
    PYTHON_BIN_VIZ="${LEGACY_VENV_DIR_VIZ}/bin/python"
else
    echo "[ERROR] Could not find venv python executable."
    echo "        Expected one of:"
    echo "        - ${VENV_DIR_VIZ}/bin/python"
    echo "        - ${LEGACY_VENV_DIR_VIZ}/bin/python"
    exit 1
fi

# Set PYTHONPATH to this project only — prevents stale ex-overcookedv2 modules from leaking in
export PYTHONPATH="${PROJECT_DIR_VIZ}:${REPO_ROOT_VIZ}/JaxMARL"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"
echo "[INFO] Using python: ${PYTHON_BIN_VIZ}"

# Sanity check: jax import (same env cleanup as actual execution)
env -u LD_LIBRARY_PATH -u XLA_FLAGS "${PYTHON_BIN_VIZ}" -c "import jax" >/dev/null 2>&1 || {
    echo "[WARN] jax import check failed for ${PYTHON_BIN_VIZ} — proceeding anyway"
}

cd "${PROJECT_DIR_VIZ}" || exit 1

# Run visualization
env -u LD_LIBRARY_PATH -u XLA_FLAGS \
    "${PYTHON_BIN_VIZ}" overcooked_v2_experiments/ppo/utils/visualize_ppo.py "${ARGS[@]}"

echo ""
echo "Visualization complete!"
echo "Results saved in: $DIRECTORY"





# # experiments 폴더에서 실행
# cd /home/mlic/mingukang/ex-overcookedv2/experiments

# # 기본 사용 (self-play, 10 seeds)
# ./run_visualize.sh --gpu 0 --dir runs/20251107-064612_a8lag2vo_test_time_simple_avs-2

# # Cross-play 평가 (20 seeds)
# ./run_visualize.sh --gpu 1 --dir runs/cramped_room_run --num_seeds 20 --cross

# # 비디오 생성 없이 메트릭만 계산 (500 seeds)
# ./run_visualize.sh --gpu 0 --dir runs/my_run --num_seeds 500 --no_viz --cross

# # 모든 체크포인트 평가 (self-play만)
# ./run_visualize.sh --gpu 2 --dir runs/my_run --all_ckpt

# # Self-play와 Cross-play 둘 다
# ./run_visualize.sh --gpu 0 --dir runs/my_run --all --num_seeds 100 --no_viz

# # 특정 정책과 페어링하여 평가
# ./run_visualize.sh --gpu 0 --dir runs/my_run --cross --pairing_policy 0 --num_seeds 50
