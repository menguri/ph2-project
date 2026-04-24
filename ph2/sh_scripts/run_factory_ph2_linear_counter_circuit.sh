#!/bin/bash
# -----------------------------------------------------------------------------
# PH2 Linear Penalty Mode 실험 — counter_circuit
#   penalty = α / (ε + latent_dist)   (PH1_PENALTY_LINEAR_MODE=true)
#
# 환경변수 override 예시:
#   PH1_LINEAR_ALPHA=0.05 bash run_factory_ph2_linear_counter_circuit.sh
#   GPUS="1,2,3" NUM_SEEDS=3 bash run_factory_ph2_linear_counter_circuit.sh
# -----------------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NSTEPS=256

: "${GPUS:=1,2,3,4,5}"
: "${NUM_SEEDS:=10}"
: "${FIXED_SEED:=42}"

# Linear penalty params (사용자 지정값)
: "${PH1_PENALTY_LINEAR_MODE:=true}"
: "${PH1_LINEAR_ALPHA:=0.02}"
: "${PH1_LINEAR_EPSILON:=0.001}"

# counter_circuit best 기본 셋팅 (linear 모드에서 omega/sigma는 무시되지만 호환 위해 유지)
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_BETA_START:=0.0}"
: "${PH1_BETA_END:=1.0}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=-1}"
: "${PH1_OMEGA:=10.0}"
: "${PH1_SIGMA:=4.0}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MAX_PENALTY_COUNT:=1}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=2000000}"
: "${ACTION_PREDICTION:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"

ALPHA_TAG=$(echo "$PH1_LINEAR_ALPHA" | tr '.' 'p')
TAGS="ph2,e3t,linear_penalty,linear_alpha${ALPHA_TAG},counter_circuit"

echo "============================================================"
echo "  PH2 Linear Penalty — counter_circuit"
echo "  GPUs            : $GPUS"
echo "  Seeds           : $NUM_SEEDS  (base seed=$FIXED_SEED)"
echo "  LINEAR_MODE     : $PH1_PENALTY_LINEAR_MODE"
echo "  LINEAR_ALPHA    : $PH1_LINEAR_ALPHA"
echo "  LINEAR_EPSILON  : $PH1_LINEAR_EPSILON"
echo "  Max penalty cap : $(awk "BEGIN{print $PH1_LINEAR_ALPHA/$PH1_LINEAR_EPSILON}")"
echo "  Warmup steps    : $PH1_WARMUP_STEPS"
echo "============================================================"

./run_user_wandb.sh \
  --gpus "$GPUS" \
  --seeds "$NUM_SEEDS" \
  --seed "$FIXED_SEED" \
  --env "counter_circuit" \
  --exp "$EXP" \
  --env-device "$ENV_DEVICE" \
  --nenvs "$NENVS" \
  --nsteps "$NSTEPS" \
  --tags "$TAGS" \
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
  --ph1-max-penalty-count $PH1_MAX_PENALTY_COUNT \
  --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
  --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
  --ph1-epsilon $PH1_EPSILON \
  --ph1-warmup-steps $PH1_WARMUP_STEPS \
  --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
  --ph2-epsilon "$PH2_EPSILON" \
  --action-prediction "$ACTION_PREDICTION" \
  --ph1-penalty-linear-mode "$PH1_PENALTY_LINEAR_MODE" \
  --ph1-linear-alpha "$PH1_LINEAR_ALPHA" \
  --ph1-linear-epsilon "$PH1_LINEAR_EPSILON"
