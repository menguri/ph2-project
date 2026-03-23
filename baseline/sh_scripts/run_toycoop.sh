#!/bin/bash
# =============================================================================
# ToyCoop (Dual Destination) — SP → E3T 순차 실행
# baseline/ 에서 실행
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

GPUS="${1:-0,1}"
echo "============================================="
echo "  ToyCoop Pipeline: SP → E3T"
echo "  GPUs: $GPUS"
echo "============================================="

# --- 공통 설정 ---
ENV_DEVICE="cpu"
TOYCOOP_NENVS=256
TOYCOOP_NSTEPS=100

# =============================================================================
# 1) SP (Self-Play)
# =============================================================================
echo ""
echo "[1/2] ====== ToyCoop SP ======"
./run_user_wandb.sh \
    --gpus "$GPUS" \
    --env "toy_coop" \
    --exp "rnn-sp-toycoop" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$TOYCOOP_NENVS" \
    --nsteps "$TOYCOOP_NSTEPS" \
    --tags "toycoop,sp"

echo "[1/2] ====== ToyCoop SP 완료 ======"

# =============================================================================
# 2) E3T
# =============================================================================
echo ""
echo "[2/2] ====== ToyCoop E3T ======"
./run_user_wandb.sh \
    --gpus "$GPUS" \
    --env "toy_coop" \
    --exp "rnn-e3t-toycoop" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$TOYCOOP_NENVS" \
    --nsteps "$TOYCOOP_NSTEPS" \
    --e3t-epsilon 0.2 \
    --use-partner-modeling True \
    --pred-loss-coef 1.0 \
    --tags "toycoop,e3t"

echo "[2/2] ====== ToyCoop E3T 완료 ======"

echo ""
echo "============================================="
echo "  ToyCoop Pipeline 완료"
echo "============================================="
