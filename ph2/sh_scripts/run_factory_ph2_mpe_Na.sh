#!/bin/bash
# =============================================================================
# PH2 × MPE N-Agent 환경 (SimpleSpread)
# 2-agent run_factory_ph2_mpe.sh와 동일 파라미터
# penalty_count = 1, 2, 3, 4 스윕 + ACTION_PREDICTION true/false 스윕
#
# 사용법:
#   bash run_factory_ph2_mpe_Na.sh 3          # 3-agent (기본)
#   bash run_factory_ph2_mpe_Na.sh 5          # 5-agent
#   bash run_factory_ph2_mpe_Na.sh 10         # 10-agent
#   CROSS_PLAY_SEEDS=1 bash run_factory_ph2_mpe_Na.sh 10
#   SMOKE=1 bash run_factory_ph2_mpe_Na.sh 5
# =============================================================================

cd "$(dirname "$0")" || exit 1

# Agent 수 (첫 번째 인자, 기본 3)
N_AGENTS=${1:-10}

: "${GPUS:=5,7}"
: "${SMOKE:=0}"
: "${CROSS_PLAY_SEEDS:=1}"
: "${SKIP_EVAL:=1}"

# 5a 이상에서는 cross-play eval seed 기본 1개
if [[ "$N_AGENTS" -ge 5 && -z "${CROSS_PLAY_SEEDS_SET:-}" ]]; then
  CROSS_PLAY_SEEDS=1
fi

EXP="rnn-ph2-mpe-3a"
NAME="rnn-ph2-mpe-${N_AGENTS}a"
ENV="mpe_spread_Na"
ENV_DEVICE="gpu"
NAGENTS_EXTRA_A="--extra ++env.ENV_KWARGS.num_agents=${N_AGENTS}"
NAGENTS_EXTRA_L="--extra ++env.ENV_KWARGS.num_landmarks=${N_AGENTS}"

# MPE 기본 하이퍼파라미터 (2-agent와 동일)
: "${NUM_SEEDS:=10}"
: "${FIXED_SEED:=42}"
: "${NENVS:=64}"
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

# 3+ agent에서는 action prediction 기본 비활성화
: "${ACTION_PREDICTION:=false}"

PENALTY_COUNTS=(1 2 3 4)

# cross-play seeds / eval skip hydra override
XPLAY_ARG="--extra +CROSS_PLAY_SEEDS=${CROSS_PLAY_SEEDS}"
if [[ "$SKIP_EVAL" == "1" ]]; then
  EVAL_ARG="--extra +EVAL.ENABLED=False"
else
  EVAL_ARG=""
fi

if [[ "$SMOKE" == "1" ]]; then
  TOTAL_TS="--extra model.TOTAL_TIMESTEPS=100000"
  NUM_SEEDS=1
  PENALTY_COUNTS=(1)
  echo "[SMOKE TEST MODE] TOTAL_TIMESTEPS=100000, SEEDS=1, k=1"
else
  TOTAL_TS=""
fi

run_ph2_mpe_Na() {
  local gpus=$1
  local penalty_count=$2

  local tags="ph2,mpe,${N_AGENTS}a,${ENV},k${penalty_count},pred_${ACTION_PREDICTION}"

  echo "================================================================================"
  echo "[PH2-MPE-${N_AGENTS}A] k=${penalty_count} | pred=${ACTION_PREDICTION} | GPU ${gpus} | seeds=${NUM_SEEDS}"
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
    --extra "++wandb.name=${NAME}" \
    $NAGENTS_EXTRA_A $NAGENTS_EXTRA_L \
    $XPLAY_ARG $EVAL_ARG $TOTAL_TS

  echo "[PH2-MPE-${N_AGENTS}A] k=${penalty_count} pred=${ACTION_PREDICTION} 완료"
  echo ""
}

# =============================================================================
# 실행: penalty_count 스윕
# =============================================================================
echo "============================================================"
echo "  PH2 × MPE ${N_AGENTS}-Agent 실험"
echo "  환경: ${ENV}"
echo "  penalty_count: ${PENALTY_COUNTS[*]}"
echo "  GPU: ${GPUS} | Seeds: ${NUM_SEEDS}"
echo "  ACTION_PREDICTION: ${ACTION_PREDICTION}"
echo "  Cross-play eval seeds: ${CROSS_PLAY_SEEDS}"
echo "============================================================"

# --- ACTION_PREDICTION=true 스윕 ---
# 주석 해제 시 prediction=true 버전도 실행
ACTION_PREDICTION="true"
for k in "${PENALTY_COUNTS[@]}"; do
  run_ph2_mpe_Na "$GPUS" "$k"
done

echo "============================================================"
echo "  PH2 × MPE ${N_AGENTS}-Agent 실험 완료"
echo "============================================================"
