#!/bin/bash
# =============================================================================
# PH2 × MPE 3-Agent 환경 (SimpleSpread, 3 agents)
# 2-agent run_factory_ph2_mpe.sh와 동일 파라미터, ACTION_PREDICTION만 false
# penalty_count = 1, 2, 3, 4 스윕
# =============================================================================

cd "$(dirname "$0")" || exit 1

: "${GPUS:=5,7}"
: "${SMOKE:=0}"

EXP="rnn-ph2-mpe-3a"
ENV="mpe_spread_Na"
ENV_DEVICE="gpu"

# MPE 기본 하이퍼파라미터 (2-agent와 동일)
: "${NUM_SEEDS:=10}"
: "${FIXED_SEED:=42}"
: "${NENVS:=256}"
: "${NSTEPS:=128}"

# PH1 파라미터 (2-agent와 동일)
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
: "${SAVE_EVAL_CHECKPOINTS:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"

# 3-agent에서는 action prediction 비활성화
ACTION_PREDICTION="false"

PENALTY_COUNTS=(1 2 3 4)

if [[ "$SMOKE" == "1" ]]; then
  TOTAL_TS="--extra model.TOTAL_TIMESTEPS=100000"
  NUM_SEEDS=1
  PENALTY_COUNTS=(1)
  echo "[SMOKE TEST MODE] TOTAL_TIMESTEPS=100000, SEEDS=1, k=1"
else
  TOTAL_TS=""
fi

run_ph2_mpe_3a() {
  local gpus=$1
  local penalty_count=$2

  local tags="ph2,mpe,3a,${ENV},k${penalty_count}"

  echo "================================================================================"
  echo "[PH2-MPE-3A] env=${ENV} | penalty_count=${penalty_count} | GPU ${gpus} | seeds=${NUM_SEEDS}"
  echo "================================================================================"

  ./run_user_wandb.sh \
    --gpus "$gpus" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "$ENV" \
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
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
    $TOTAL_TS

  echo "[PH2-MPE-3A] k=${penalty_count} 완료"
  echo ""
}

# =============================================================================
# 실행: penalty_count 스윕
# =============================================================================
echo "============================================================"
echo "  PH2 × MPE 3-Agent 실험 시작"
echo "  환경: ${ENV}"
echo "  penalty_count: ${PENALTY_COUNTS[*]}"
echo "  GPU: ${GPUS} | Seeds: ${NUM_SEEDS}"
echo "  ACTION_PREDICTION: ${ACTION_PREDICTION}"
echo "============================================================"

for k in "${PENALTY_COUNTS[@]}"; do
  run_ph2_mpe_3a "$GPUS" "$k"
done

echo "============================================================"
echo "  PH2 × MPE 3-Agent 실험 완료"
echo "============================================================"

# =============================================================================
# ACTION_PREDICTION true/false 스윕
# 위의 기본 실행은 ACTION_PREDICTION=false.
# 아래 주석 해제 시 prediction=true 버전도 실행.
# =============================================================================
# echo "============================================================"
# echo "  PH2 × MPE 3A — ACTION_PREDICTION=true 스윕"
# echo "============================================================"
ACTION_PREDICTION="true"
for k in "${PENALTY_COUNTS[@]}"; do
  run_ph2_mpe_3a "$GPUS" "$k"
done
