#!/bin/bash
# =============================================================================
# PH2 GridSpread 스윕 2: k × ω × β_end × σ  (agentwise penalty ON)
# TOTAL_TS=5e7
# 총 조합: 3 × 3 × 3 × 4 = 108
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="${ENV_DEVICE:-cpu}"
GPUS="${GPUS:-0,2,6,7}"
NUM_SEEDS="${NUM_SEEDS:-12}"
FIXED_SEED="${FIXED_SEED:-42}"
TOTAL_TS="${TOTAL_TS:-5e7}"

# PH1/PH2 고정 파라미터
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_BETA_START:=0.0}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=-1}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=2000000}"
: "${ACTION_PREDICTION:=True}"
: "${SAVE_EVAL_CHECKPOINTS:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"
: "${PH1_AGENTWISE_PENALTY:=True}"

# 에이전트 수: CLI 인자 또는 기본값 4
if [ $# -eq 0 ]; then
  AGENT_COUNTS=(4)
else
  AGENT_COUNTS=("$@")
fi

# 스윕 변수
PENALTY_COUNTS=(1)
OMEGAS=(50.0 30.0 20.0 10.0 5.0 1.0)
BETA_ENDS=(0.0 0.5)
SIGMAS=(3.0)

for N in "${AGENT_COUNTS[@]}"; do
  ENV="gridspread"
  for K in "${PENALTY_COUNTS[@]}"; do
    for O in "${OMEGAS[@]}"; do
      for BE in "${BETA_ENDS[@]}"; do
        for S in "${SIGMAS[@]}"; do
          echo "================================================================"
          echo "  PH2 GridSpread SWEEP2  N=${N}  k=${K}  omega=${O}  beta_end=${BE}  sigma=${S}"
          echo "================================================================"
          ./run_user_wandb.sh \
            --gpus "$GPUS" \
            --seeds "$NUM_SEEDS" \
            --seed "$FIXED_SEED" \
            --env "$ENV" \
            --exp "$EXP" \
            --env-device "$ENV_DEVICE" \
            --tags "ph2,spread,N${N},k${K},o${O},be${BE},s${S},aw" \
            --ph1-beta $PH1_BETA \
            --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
            --ph1-beta-start $PH1_BETA_START \
            --ph1-beta-end "$BE" \
            --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
            --ph1-omega "$O" \
            --ph1-sigma "$S" \
            --ph1-pool-size $PH1_POOL_SIZE \
            --ph1-normal-prob $PH1_NORMAL_PROB \
            --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
            --ph1-max-penalty-count "$K" \
            --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
            --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
            --ph1-epsilon $PH1_EPSILON \
            --ph1-warmup-steps $PH1_WARMUP_STEPS \
            --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
            --ph2-epsilon "$PH2_EPSILON" \
            --action-prediction "$ACTION_PREDICTION" \
            --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
            --ph1-agentwise-penalty "$PH1_AGENTWISE_PENALTY" \
            --extra "++model.TOTAL_TIMESTEPS=${TOTAL_TS}" \
            --extra "++model.OBS_ENCODER=MLP" \
            --extra "++env.ENV_KWARGS.n_agents=${N}"
        done
      done
    done
  done
done

echo "================================================================"
echo "  PH2 GridSpread SWEEP2 전체 완료"
echo "================================================================"
