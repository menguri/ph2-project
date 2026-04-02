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

# -----------------------------------------------------------------------------
# Sweep 파라미터
# -----------------------------------------------------------------------------
SWEEP_GPUS="3,7,0,1,2"

# =============================================================================
# cramped_room Sweep — partner diversity 향상 탐색
# 5축: PH1_BETA_END, PH1_NORMAL_PROB, PH1_EPSILON, PH2_EPSILON, ENT_COEF
#
# 사용법: 필요한 sweep 블록만 주석 해제하여 실행
# =============================================================================

sweep_cramped() {
  local gpus=$1
  local beta_end=$2
  local normal_prob=$3
  local ph1_eps=$4
  local ph2_eps=$5
  local ent_coef=$6

  local tags="ph2,sweep,cramped_room,be${beta_end},np${normal_prob},e1_${ph1_eps},e2_${ph2_eps},ent${ent_coef}"

  echo "================================================================================"
  echo "[SWEEP] cramped_room | beta_end=${beta_end} normal_prob=${normal_prob}"
  echo "        ph1_eps=${ph1_eps} ph2_eps=${ph2_eps} ent_coef=${ent_coef} | GPU ${gpus}"
  echo "================================================================================"

  ./run_user_wandb.sh \
    --gpus "$gpus" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "cramped_room" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$NENVS" \
    --nsteps "$NSTEPS" \
    --tags "$tags" \
    --layout "cramped_room" \
    --ph1-beta "$PH1_BETA" \
    --ph1-beta-schedule-enabled "$PH1_BETA_SCHEDULE_ENABLED" \
    --ph1-beta-start "$PH1_BETA_START" \
    --ph1-beta-end "$beta_end" \
    --ph1-beta-schedule-horizon-env-steps "$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS" \
    --ph1-omega "$PH1_OMEGA" \
    --ph1-sigma "$PH1_SIGMA" \
    --ph1-pool-size "$PH1_POOL_SIZE" \
    --ph1-normal-prob "$normal_prob" \
    --ph1-multi-penalty-enabled "$PH1_MULTI_PENALTY_ENABLED" \
    --ph1-max-penalty-count 1 \
    --ph1-multi-penalty-single-weight "$PH1_MULTI_PENALTY_SINGLE_WEIGHT" \
    --ph1-multi-penalty-other-weight "$PH1_MULTI_PENALTY_OTHER_WEIGHT" \
    --ph1-epsilon "$ph1_eps" \
    --ph2-epsilon "$ph2_eps" \
    --ph1-warmup-steps "$PH1_WARMUP_STEPS" \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
    --extra "model.ENT_COEF=$ent_coef"
}

# sweep_cramped  GPUS  BETA_END  NORMAL_PROB  PH1_EPS  PH2_EPS  ENT_COEF
# ─────────────────────────────────────────────────────────────────────────

# --- Sweep 1: ENT_COEF × PH1_NORMAL_PROB (epsilon 고정 0.2) ---
sweep_cramped "$SWEEP_GPUS"  0.0  0.0  0.3  0.3  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.0  0.4  0.4  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.0  0.5  0.5  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.3  0.2  0.2  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.3  0.3  0.3  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.3  0.5  0.5  0.05
sweep_cramped "$SWEEP_GPUS"  0.0  0.5  0.2  0.2  0.1
sweep_cramped "$SWEEP_GPUS"  0.0  0.5  0.3  0.3  0.1
sweep_cramped "$SWEEP_GPUS"  0.0  0.5  0.5  0.5  0.1
sweep_cramped "$SWEEP_GPUS"  0.0  0.0  0.3  0.3  0.05

# --- Sweep 2: PH1_BETA_END × PH1_NORMAL_PROB (entropy/epsilon 고정) ---
# sweep_cramped "$SWEEP_GPUS"  0.5  0.0  0.2  0.2  0.01
# sweep_cramped "$SWEEP_GPUS"  0.5  0.3  0.2  0.2  0.01
# sweep_cramped "$SWEEP_GPUS"  0.5  0.5  0.2  0.2  0.01
# sweep_cramped "$SWEEP_GPUS"  0.5  0.0  0.2  0.2  0.01
# sweep_cramped "$SWEEP_GPUS"  0.5  0.3  0.2  0.2  0.01
# sweep_cramped "$SWEEP_GPUS"  0.5  0.5  0.2  0.2  0.01

# --- Sweep 3: PH1_EPSILON × PH2_EPSILON (나머지 고정) ---
# sweep_cramped "$SWEEP_GPUS"  1.0  0.0  0.1  0.1  0.01
# sweep_cramped "$SWEEP_GPUS"  1.0  0.0  0.1  0.3  0.01
# sweep_cramped "$SWEEP_GPUS"  1.0  0.0  0.3  0.1  0.01
# sweep_cramped "$SWEEP_GPUS"  1.0  0.0  0.3  0.3  0.01

# --- Sweep 4: 유망 조합 정밀 탐색 (Sweep 1~3 결과 보고 채울 것) ---
# sweep_cramped "$SWEEP_GPUS"  ...  ...  ...  ...  ...

