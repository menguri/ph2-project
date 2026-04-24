#!/bin/bash
# =============================================================================
# PH2 Component Ablation 실험
#
# 3 레이아웃 × 3 ablation = 9 셀, 각 5 seeds
#
# Best params (final/ph2 등록된 모델의 run_metadata.json 기준):
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=2 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=4 k=1 normal_prob=0.0
#   공통: ent=0.01 (default 사용)
#
# Ablation 조건:
#   A1: w/o Penalty Filtering   — beta_start=0, beta_end=0 (V-gap 비활성화 → uniform sampling)
#   A2: w/o Action Prediction   — ACTION_PREDICTION=False
#   A3: w/o Spec-Ind Coop       — PH2_EPSILON=1.0 (spec-ind에서 spec이 항상 random)
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=10
NSTEPS=256
FIXED_SEED=42

# 공통 PH1/PH2 파라미터
PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_POOL_SIZE=128
PH1_MULTI_PENALTY_ENABLED=True
PH1_MULTI_PENALTY_SINGLE_WEIGHT=1.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
PH1_EPSILON=0.2
PH1_WARMUP_STEPS=2000000
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
PH2_FIXED_IND_PROB=0.5
SAVE_EVAL_CHECKPOINTS=True

# GPU 설정
GPUS="${GPUS:-1,2,3,4,5}"

# =============================================================================
# run_ablation  <env> <omega> <sigma> <k> <normal_prob> <ablation_tag> [overrides...]
# =============================================================================
run_ablation() {
  local env=$1
  local omega=$2
  local sigma=$3
  local k=$4
  local normal_prob=$5
  local ablation_tag=$6
  shift 6

  # 나머지 인자는 추가 오버라이드
  local beta_start="${PH1_BETA_START:-0.0}"
  local beta_end="${PH1_BETA_END:-1.0}"
  local action_pred="${ACTION_PREDICTION:-True}"
  local ph2_eps="${PH2_EPSILON:-0.2}"

  # 오버라이드 적용
  for arg in "$@"; do
    case "$arg" in
      BETA_START=*)    beta_start="${arg#*=}" ;;
      BETA_END=*)      beta_end="${arg#*=}" ;;
      ACTION_PRED=*)   action_pred="${arg#*=}" ;;
      PH2_EPS=*)       ph2_eps="${arg#*=}" ;;
    esac
  done

  local tags="ph2,ablation,${ablation_tag},${env}"

  echo "================================================================"
  echo "  ABLATION: ${ablation_tag}  env=${env}  ω=${omega} σ=${sigma} k=${k}"
  echo "  beta_start=${beta_start} beta_end=${beta_end} action_pred=${action_pred} ph2_eps=${ph2_eps}"
  echo "================================================================"

  ./run_user_wandb.sh \
    --gpus "$GPUS" \
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
    --ph1-beta-start "$beta_start" \
    --ph1-beta-end "$beta_end" \
    --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
    --ph1-omega "$omega" \
    --ph1-sigma "$sigma" \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob "$normal_prob" \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count "$k" \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --ph2-epsilon "$ph2_eps" \
    --action-prediction "$action_pred" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS"
}

# =============================================================================
# A1: w/o Penalty Filtering (beta=0 → uniform sampling, V-gap 비활성화)
# =============================================================================
echo "#################### A1: w/o Penalty Filtering ####################"
run_ablation "cramped_room"     10.0 2.0 1 0.5  "wo_filtering"  BETA_START=0.0 BETA_END=0.0
run_ablation "coord_ring"       10.0 2.0 2 0.0  "wo_filtering"  BETA_START=0.0 BETA_END=0.0
run_ablation "counter_circuit"  10.0 4.0 1 0.0  "wo_filtering"  BETA_START=0.0 BETA_END=0.0

# =============================================================================
# A2: w/o Action Prediction (E3T partner predictor 비활성화)
# =============================================================================
echo "#################### A2: w/o Action Prediction ####################"
run_ablation "cramped_room"     10.0 2.0 1 0.5  "wo_action_pred"  ACTION_PRED=False
run_ablation "coord_ring"       10.0 2.0 2 0.0  "wo_action_pred"  ACTION_PRED=False
run_ablation "counter_circuit"  10.0 4.0 1 0.0  "wo_action_pred"  ACTION_PRED=False

# =============================================================================
# A3: w/o Spec-Ind Cooperation (spec이 ind와 매칭 시 항상 random action)
# =============================================================================
echo "#################### A3: w/o Spec-Ind Coop ####################"
run_ablation "cramped_room"     10.0 2.0 1 0.5  "wo_specind"  PH2_EPS=1.0
run_ablation "coord_ring"       10.0 2.0 2 0.0  "wo_specind"  PH2_EPS=1.0
run_ablation "counter_circuit"  10.0 4.0 1 0.0  "wo_specind"  PH2_EPS=1.0

echo "================================================================"
echo "  Ablation 전체 완료 (9 셀)"
echo "================================================================"
