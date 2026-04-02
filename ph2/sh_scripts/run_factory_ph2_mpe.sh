#!/bin/bash
# =============================================================================
# PH2 × MPE 환경 (SimpleSpread, SimpleReference)
# penalty_count = 1, 2, 3, 4 스윕
# =============================================================================

cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2-mpe"
ENV_DEVICE="gpu"

# MPE 기본 하이퍼파라미터
: "${GPUS:=3,4,5,6,7}"
: "${NUM_SEEDS:=10}"
: "${FIXED_SEED:=42}"
: "${NENVS:=256}"
: "${NSTEPS:=128}"

# PH1 파라미터
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_BETA_START:=0.0}"
: "${PH1_BETA_END:=1.0}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=-1}"
: "${PH1_OMEGA:=1.0}"
: "${PH1_SIGMA:=1.0}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=500000}"
: "${ACTION_PREDICTION:=True}"
: "${SAVE_EVAL_CHECKPOINTS:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"

MPE_ENVS=(mpe_spread mpe_reference)
PENALTY_COUNTS=(1 2 3 4)

run_ph2_mpe() {
  local gpus=$1
  local env=$2
  local penalty_count=$3

  local tags="ph2,mpe,${env},k${penalty_count}"

  echo "================================================================================"
  echo "[PH2-MPE] env=${env} | penalty_count=${penalty_count} | GPU ${gpus} | seeds=${NUM_SEEDS}"
  echo "================================================================================"

  ./run_user_wandb.sh \
    --gpus "$gpus" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "$env" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$NENVS" \
    --nsteps "$NSTEPS" \
    --tags "$tags" \
    --ph1-beta $PH1_BETA \
    --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
    --ph1-beta-start $PH1_BETA_START \
    --ph1-beta-end $PH1_BETA_END \
    --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
    --ph1-omega $PH1_OMEGA \
    --ph1-sigma $PH1_SIGMA \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob $PH1_NORMAL_PROB \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count $penalty_count \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph2-epsilon $PH2_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS"

  echo "[PH2-MPE] env=${env} k=${penalty_count} 완료"
  echo ""
}

# =============================================================================
# 실행: 환경 × penalty_count 스윕
# =============================================================================
echo "============================================================"
echo "  PH2 × MPE 환경 실험 시작"
echo "  환경: ${MPE_ENVS[*]}"
echo "  penalty_count: ${PENALTY_COUNTS[*]}"
echo "  GPU: ${GPUS} | Seeds: ${NUM_SEEDS}"
echo "============================================================"

for env in "${MPE_ENVS[@]}"; do
  for k in "${PENALTY_COUNTS[@]}"; do
    run_ph2_mpe "$GPUS" "$env" "$k"
  done
done

echo "============================================================"
echo "  PH2 × MPE 환경 실험 완료"
echo "============================================================"
