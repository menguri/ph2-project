#!/bin/bash
# =============================================================================
# PH2 GridSpread 실험
# 사용법:
#   bash run_factory_ph2_spread.sh                  # 기본 N=4
#   bash run_factory_ph2_spread.sh 2 4 6 8 10       # 여러 N 순차 실행
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="${ENV_DEVICE:-cpu}"
GPUS="${GPUS:-2,3}"
NENVS="${NENVS:-128}"
NSTEPS="${NSTEPS:-512}"
NUM_SEEDS="${NUM_SEEDS:-10}"
FIXED_SEED="${FIXED_SEED:-42}"
TOTAL_TS="${TOTAL_TS:-1e8}"     # 100M

# PH1/PH2 파라미터 (기본값)
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_BETA_START:=0.0}"
: "${PH1_BETA_END:=1.0}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=-1}"
: "${PH1_OMEGA:=10.0}"
: "${PH1_SIGMA:=2.0}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=5000000}"
: "${ACTION_PREDICTION:=True}"
: "${SAVE_EVAL_CHECKPOINTS:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"

# 에이전트 수: CLI 인자 또는 기본값 4
if [ $# -eq 0 ]; then
  AGENT_COUNTS=(4)
else
  AGENT_COUNTS=("$@")
fi

run_ph2_spread() {
  local gpus=$1
  local env=$2
  local n_agents=$3

  local tags="ph2,spread,N${n_agents}"

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
    --ph1-max-penalty-count 1 \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --ph2-epsilon "$PH2_EPSILON" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
    --extra "++model.TOTAL_TIMESTEPS=${TOTAL_TS}" \
    --extra "++model.OBS_ENCODER=MLP" \
    --extra "++model.ENT_COEF_START=0.5" \
    --extra "++model.ENT_COEF_END=0.01" \
    --extra "++model.ENT_COEF_ANNEAL_STEPS=1e7" \
    --extra "++env.ENV_KWARGS.dist_shaping_coef=0.01" \
    --extra "++env.ENV_KWARGS.early_terminate=true" \
    --extra "++env.ENV_KWARGS.n_agents=${n_agents}"
}

# for N in "${AGENT_COUNTS[@]}"; do
#   ENV="gridspread"
#   echo "================================================================"
#   echo "  PH2 GridSpread N=${N}  (env=${ENV})"
#   echo "================================================================"

#   # --- PH2 학습 (spec/ind 동시 학습) ---
#   echo "[PH2] N=${N}"
#   run_ph2_spread "${GPUS}" "${ENV}" "${N}"
# done

# ===========================================================================
# 스윕: PH1_MAX_PENALTY_COUNT × PH1_OMEGA × PH1_SIGMA
# ===========================================================================
PENALTY_COUNTS=(2)
OMEGAS=(100.0 10.0 5.0 1.0)
SIGMAS=(2.0 3.0 4.0 5.0 6.0 7.0 10.0)

for N in "${AGENT_COUNTS[@]}"; do
  ENV="gridspread"
  for K in "${PENALTY_COUNTS[@]}"; do
    for O in "${OMEGAS[@]}"; do
      for S in "${SIGMAS[@]}"; do
        echo "================================================================"
        echo "  PH2 GridSpread SWEEP  N=${N}  k=${K}  omega=${O}  sigma=${S}"
        echo "================================================================"
        ./run_user_wandb.sh \
          --gpus "$GPUS" \
          --seeds "$NUM_SEEDS" \
          --seed "$FIXED_SEED" \
          --env "$ENV" \
          --exp "$EXP" \
          --env-device "$ENV_DEVICE" \
          --nenvs "$NENVS" \
          --nsteps "$NSTEPS" \
          --tags "ph2,spread,N${N},k${K},o${O},s${S}" \
          --ph1-beta $PH1_BETA \
          --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
          --ph1-beta-start $PH1_BETA_START \
          --ph1-beta-end $PH1_BETA_END \
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
          --extra "++model.TOTAL_TIMESTEPS=${TOTAL_TS}" \
          --extra "++model.OBS_ENCODER=MLP" \
          --extra "++model.ENT_COEF_START=0.5" \
          --extra "++model.ENT_COEF_END=0.01" \
          --extra "++model.ENT_COEF_ANNEAL_STEPS=1e7" \
    --extra "++env.ENV_KWARGS.dist_shaping_coef=0.01" \
    --extra "++env.ENV_KWARGS.early_terminate=true" \
          --extra "++env.ENV_KWARGS.n_agents=${N}"
      done
    done
  done
done

echo "================================================================"
echo "  PH2 GridSpread 전체 완료"
echo "================================================================"
