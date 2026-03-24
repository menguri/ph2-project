#!/bin/bash
# =============================================================================
# ToyCoop (Dual Destination) — PH2 실행
# ph2/ 에서 실행
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR="$(pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- 기본 설정 ---
GPUS="${1:-2,3}"
: "${RANDOM_RESET:=false}"     # true면 매 에피소드 랜덤 배치 (procedural generation)
ENV_DEVICE="cpu"
NUM_SEEDS=5
NENVS=512
NSTEPS=100
FIXED_SEED=41

# PH2 experiment config
EXP="rnn-ph2-toycoop"

# CT 모드: USE_CT=1 이면 rnn-ct experiment config 사용 (토이쿱에선 기본 off)
: "${USE_CT:=0}"

# PH1 blocked-target 파라미터
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=False}"
: "${PH1_OMEGA:=0.1}"
: "${PH1_SIGMA:=1.0}"
: "${PH1_DIST_THRESH:=0.1}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.5}"
: "${PH1_MULTI_PENALTY_ENABLED:=False}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=2.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.0}"
: "${PH2_EPSILON:=-1.0}"
: "${PH1_WARMUP_STEPS:=0}"
: "${ACTION_PREDICTION:=True}"
: "${CYCLE_LOSS_ENABLED:=False}"
: "${CYCLE_LOSS_COEF:=0.1}"
: "${LATENT_MODE:=False}"

# PH2 schedule
PH2_RATIO_STAGE1=2
PH2_RATIO_STAGE2=1
PH2_RATIO_STAGE3=2
PH2_FIXED_IND_PROB=0.5

# random_reset override 인자
RANDOM_RESET_ARGS=""
if [[ "$RANDOM_RESET" == "true" ]]; then
  RANDOM_RESET_ARGS="--random-reset true"
fi

echo "============================================="
echo "  ToyCoop PH2 Pipeline"
echo "  GPUs: $GPUS"
echo "  random_reset: $RANDOM_RESET"
echo "  USE_CT: $USE_CT"
echo "============================================="

# =============================================================================
# PH2
# =============================================================================
echo ""
echo "[1/1] ====== ToyCoop PH2 ======"

CT_ARGS=""
if [[ "$USE_CT" == "1" ]]; then
  EXP="rnn-ct"
  CT_ARGS="--transformer-action True"
fi

./run_user_wandb.sh \
    --gpus "$GPUS" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "toy_coop" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$NENVS" \
    --nsteps "$NSTEPS" \
    --ph1-omega "$PH1_OMEGA" \
    --ph1-sigma "$PH1_SIGMA" \
    --ph1-dist "$PH1_DIST_THRESH" \
    --ph1-pool-size "$PH1_POOL_SIZE" \
    --ph1-normal-prob "$PH1_NORMAL_PROB" \
    --ph1-epsilon "$PH1_EPSILON" \
    --ph1-warmup-steps "$PH1_WARMUP_STEPS" \
    --ph2-ratio-stage1 "$PH2_RATIO_STAGE1" \
    --ph2-ratio-stage2 "$PH2_RATIO_STAGE2" \
    --ph2-ratio-stage3 "$PH2_RATIO_STAGE3" \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --ph2-epsilon "$PH2_EPSILON" \
    --action-prediction "$ACTION_PREDICTION" \
    --cycle-loss-enabled "$CYCLE_LOSS_ENABLED" \
    --cycle-loss-coef "$CYCLE_LOSS_COEF" \
    --latent-mode "$LATENT_MODE" \
    --tags "toycoop,ph2" \
    $RANDOM_RESET_ARGS \
    $CT_ARGS

echo "[1/1] ====== ToyCoop PH2 완료 ======"

echo ""
echo "============================================="
echo "  ToyCoop PH2 Pipeline 완료"
echo "============================================="
