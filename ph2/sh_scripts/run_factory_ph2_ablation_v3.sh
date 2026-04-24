#!/bin/bash
# =============================================================================
# PH2 Ablation V3 — forced_coord 전용 sweep (omega × k grid)
#
# 고정:
#   - env        = forced_coord
#   - sigma      = 8
#   - epsilon    = 0.3  (PH1, PH2 둘 다)
#   - normal_prob = 0.0
#
# Sweep:
#   - omega ∈ {2, 3, 4, 8, 10}
#   - k     ∈ {1, 2, 3}
#   => 총 5 × 3 = 15 cell
#
# NOTE: k≥3 은 multi-penalty buffer 가 커서 OOM 위험 → NENVS=32 로 감량
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
PH1_BETA_START=0.0
PH1_BETA_END=1.0
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
PH1_POOL_SIZE=128
PH1_NORMAL_PROB=0.0
PH1_MULTI_PENALTY_ENABLED=True
PH1_MULTI_PENALTY_SINGLE_WEIGHT=1.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
PH1_EPSILON=0.3
PH2_EPSILON=0.3
PH1_WARMUP_STEPS=2000000
PH2_FIXED_IND_PROB=0.5
ACTION_PREDICTION=True
SAVE_EVAL_CHECKPOINTS=True

# GPU 설정
GPUS="${GPUS:-1,2,3,4,5}"

# 고정 파라미터
ENV="forced_coord"
FIXED_SIGMA=8.0
FIXED_NP=0.0

# Sweep 공간
OMEGAS=(2 3 4 8 10)
KS=(1 2 3)

# =============================================================================
# run_cell  <env> <omega> <sigma> <k> <normal_prob> <tag>
# =============================================================================
run_cell() {
  local env=$1
  local omega=$2
  local sigma=$3
  local k=$4
  local normal_prob=$5
  local tag=$6

  local tags="ph2,ablation_v3,${tag},${env}"

  echo "================================================================"
  echo "  [${tag}]  env=${env}  ω=${omega} σ=${sigma} k=${k} np=${normal_prob}  eps=${PH1_EPSILON}  NENVS=${NENVS_OVERRIDE:-$NENVS}"
  echo "================================================================"

  ./run_user_wandb.sh \
    --gpus "$GPUS" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "$env" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "${NENVS_OVERRIDE:-$NENVS}" \
    --nsteps "$NSTEPS" \
    --tags "$tags" \
    --ph1-beta $PH1_BETA \
    --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
    --ph1-beta-start $PH1_BETA_START \
    --ph1-beta-end $PH1_BETA_END \
    --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
    --ph1-omega "$omega" \
    --ph1-sigma "$sigma" \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob "$normal_prob" \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count "$k" \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon "$PH1_EPSILON" \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --ph2-epsilon "$PH2_EPSILON" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS"
}

# =============================================================================
# Sweep: omega × k grid (sigma=8, eps=0.3 고정)
# =============================================================================
echo "#################### Sweep: omega × k grid (σ=8, eps=0.3 fixed) ####################"
for omega in "${OMEGAS[@]}"; do
  for k in "${KS[@]}"; do
    tag="w${omega}_k${k}"
    if [[ $k -ge 3 ]]; then
      NENVS_OVERRIDE=32 \
        run_cell "$ENV" "$omega" "$FIXED_SIGMA" "$k" "$FIXED_NP" "$tag"
    else
      run_cell "$ENV" "$omega" "$FIXED_SIGMA" "$k" "$FIXED_NP" "$tag"
    fi
  done
done

echo "================================================================"
echo "  Ablation V3 완료 — forced_coord 전용 (총 ${#OMEGAS[@]} × ${#KS[@]} = $(( ${#OMEGAS[@]} * ${#KS[@]} )) cell)"
echo "================================================================"
