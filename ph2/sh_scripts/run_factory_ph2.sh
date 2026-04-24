#!/bin/bash

cd "$(dirname "$0")" || exit 1

# -----------------------------------------------------------------------------
# PH2 OV1 전 레이아웃 학습
# -----------------------------------------------------------------------------
EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=5
NSTEPS=256
FIXED_SEED=42

# PH1/PH2 파라미터
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
: "${PH1_WARMUP_STEPS:=2000000}"
: "${ACTION_PREDICTION:=True}"
: "${PH2_FIXED_IND_PROB:=0.5}"

# Penalty Linear Mode (Overcooked 전용 실험; default false=원본 exp 유지)
: "${PH1_PENALTY_LINEAR_MODE:=false}"
: "${PH1_LINEAR_ALPHA:=0.02}"
: "${PH1_LINEAR_EPSILON:=0.001}"

run_ph2() {
  local gpus=$1
  local env=$2
  local ph1_omega=${3:-$PH1_OMEGA}
  local ph1_sigma=${4:-$PH1_SIGMA}
  local ph1_max_penalty_count=${5:-1}

  local tags="ph2,e3t,multi_penalty_max${ph1_max_penalty_count}"

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
    --ph1-omega $ph1_omega \
    --ph1-sigma $ph1_sigma \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob $PH1_NORMAL_PROB \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count $ph1_max_penalty_count \
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
}

# =============================================================================
# OV1 전 레이아웃 PH2 학습 — 12 seeds (seed 41×5 + seed 42×5 + seed 43×2)
#
# 각 레이아웃별 best 파라미터 (final/ph2 등록된 모델의 run_metadata.json 기준):
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   asymm_advantages: omega=5  sigma=3 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=2 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=4 k=1 normal_prob=0.0
#   forced_coord:     omega=4  sigma=8 k=1 normal_prob=0.0
# 공통: beta=1.0, beta_end=1.0, eps=0.2, ent=0.01(default), warmup=2M
# =============================================================================
OV1_GPUS="1,2,4,5,6"

run_ov1_all() {
  local seed=$1
  local nseeds=$2

  FIXED_SEED=$seed
  NUM_SEEDS=$nseeds

  echo "============================================================"
  echo "  OV1 전 레이아웃 — SEED=${seed}, ${nseeds} seeds"
  echo "============================================================"

  # cramped_room: omega=10 sigma=2 k=1 normal_prob=0.5
  PH1_NORMAL_PROB=0.5
  run_ph2 "$OV1_GPUS" "cramped_room" 10.0 2.0 1

  # asymm_advantages: omega=5 sigma=3 k=1 normal_prob=0.5
  PH1_NORMAL_PROB=0.5
  run_ph2 "$OV1_GPUS" "asymm_advantages" 5.0 3.0 1

  # coord_ring: omega=10 sigma=2 k=2 normal_prob=0.0
  PH1_NORMAL_PROB=0.0
  run_ph2 "$OV1_GPUS" "coord_ring" 10.0 2.0 2

  # counter_circuit: omega=10 sigma=4 k=1 normal_prob=0.0
  PH1_NORMAL_PROB=0.0
  run_ph2 "$OV1_GPUS" "counter_circuit" 10.0 4.0 1

  # forced_coord: omega=4 sigma=8 k=1 normal_prob=0.0
  PH1_NORMAL_PROB=0.0
  run_ph2 "$OV1_GPUS" "forced_coord" 4.0 8.0 1
}

# --- Round 1: SEED=41, 5 seeds ---
run_ov1_all 41 5

# --- Round 2: SEED=42, 5 seeds ---
run_ov1_all 42 5

# --- Round 3: SEED=43, 2 seeds ---
run_ov1_all 43 2
