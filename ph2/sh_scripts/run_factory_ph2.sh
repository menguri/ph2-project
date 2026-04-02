#!/bin/bash

cd "$(dirname "$0")" || exit 1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# -----------------------------------------------------------------------------
# PH2 Experiment Factory
# -----------------------------------------------------------------------------
EXP="rnn-ph2"
# CT 모드: USE_CT=1 이면 rnn-ct experiment config 사용
: "${USE_CT:=0}"
if [[ "$USE_CT" == "1" ]]; then
  EXP="rnn-ct"
fi
# SHARED_PREDICTION: USE_SHARED=1 이면 spec/ind가 CT + predictor 파라미터 공유
: "${USE_SHARED:=0}"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=5
NSTEPS=256

FIXED_SEED=42
while [[ $# -gt 0 ]]; do
  case "$1" in
    *)
      echo "[WARN] Unknown option ignored: $1"
      shift
      ;;
  esac
done

# Shared PH1 blocked-target params
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
: "${SAVE_EVAL_CHECKPOINTS:=True}"

# ToyCoop random_reset (procedural generation)
: "${RANDOM_RESET:=false}"

# CycleTransformer (CT) 파라미터 — USE_CT=1일 때 유효 (rnn-ct.yaml 기본값 오버라이드용)
: "${TRANSFORMER_RECON_COEF:=1.0}"
: "${TRANSFORMER_PRED_COEF:=1.0}"
: "${TRANSFORMER_CYCLE_COEF:=0.5}"
: "${TRANSFORMER_WINDOW_SIZE:=10}"
: "${TRANSFORMER_D_C:=64}"
: "${TRANSFORMER_V2:=0}"
: "${TRANSFORMER_V3:=0}"

# PH2 ind 매칭 확률 (0.5 = spec-spec 50%, spec-ind/ind-ind 50%)
# PH2_FIXED_IND_PROB 설정 시 PH2_RATIO_STAGE1/2/3 무시됨
: "${PH2_FIXED_IND_PROB:=0.5}"

run_ph2() {
  local gpus=$1
  local env=$2
  local ph1_omega=${3:-$PH1_OMEGA}
  local ph1_sigma=${4:-$PH1_SIGMA}
  local ph1_max_penalty_count=${5:-1}

  local ct_tag="ct0"; [[ "$USE_CT" == "1" ]] && ct_tag="ct1"
  local shared_tag="shared0"; [[ "$USE_SHARED" == "1" ]] && shared_tag="shared1"
  local tags="ph2,e3t,multi_penalty_max${ph1_max_penalty_count},${ct_tag},${shared_tag}"

  local -a cmd=("./run_user_wandb.sh"
    --gpus "$gpus"
    --seeds "$NUM_SEEDS"
    --seed "$FIXED_SEED"
    --env "$env"
    --exp "$EXP"
    --env-device "$ENV_DEVICE"
    --nenvs "$NENVS"
    --nsteps "$NSTEPS"
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
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS")
  if [[ -n "$PH2_EPSILON" ]]; then
    cmd+=(--ph2-epsilon "$PH2_EPSILON")
  fi
  if [[ "$USE_CT" == "1" ]]; then
    cmd+=(--transformer-action True)
    cmd+=(--transformer-window-size "$TRANSFORMER_WINDOW_SIZE")
    cmd+=(--transformer-d-c "$TRANSFORMER_D_C")
    cmd+=(--transformer-recon-coef "$TRANSFORMER_RECON_COEF")
    cmd+=(--transformer-pred-coef "$TRANSFORMER_PRED_COEF")
    cmd+=(--transformer-cycle-coef "$TRANSFORMER_CYCLE_COEF")
    if [[ "$TRANSFORMER_V2" == "1" ]]; then
      cmd+=(--transformer-v2 True)
    fi
    if [[ "$TRANSFORMER_V3" == "1" ]]; then
      cmd+=(--transformer-v3 True)
    fi
  fi
  if [[ "$USE_SHARED" == "1" ]]; then
    cmd+=(--shared-prediction True)
  fi
  if [[ "$RANDOM_RESET" == "true" ]]; then
    cmd+=(--random-reset true)
  fi

  echo "Executing: ${cmd[*]}"
  "${cmd[@]}"
}

# =============================================================================
# OV1 전 레이아웃 PH2 학습 — 12 seeds (seed 41×5 + seed 42×5 + seed 43×2)
# GPU 5개에서 5 seeds/run, 순차 3회
#
# 각 레이아웃별 best 파라미터:
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   asymm_advantages: omega=5  sigma=3 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=3 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=2 k=1 normal_prob=0.0
#   forced_coord:     omega=4  sigma=8 k=1 normal_prob=0.0
# 공통: beta=1.0, beta_end=1.0, eps=0.2, ent=0.01, warmup=2M
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

  # coord_ring: omega=10 sigma=2 k=3 normal_prob=0.0
  PH1_NORMAL_PROB=0.0
  run_ph2 "$OV1_GPUS" "coord_ring" 10.0 2.0 3

  # counter_circuit: omega=10 sigma=2 k=1 normal_prob=0.0
  PH1_NORMAL_PROB=0.0
  run_ph2 "$OV1_GPUS" "counter_circuit" 10.0 2.0 1

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

