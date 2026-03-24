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

FIXED_SEED=41
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
: "${PH1_DIST_THRESH:=0.1}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=2000000}"
: "${ACTION_PREDICTION:=True}"
: "${CYCLE_LOSS_ENABLED:=False}"
: "${CYCLE_LOSS_COEF:=0.1}"
: "${LATENT_MODE:=False}"

# ToyCoop random_reset (procedural generation)
: "${RANDOM_RESET:=false}"

# CycleTransformer (CT) 파라미터 — USE_CT=1일 때 유효 (rnn-ct.yaml 기본값 오버라이드용)
: "${TRANSFORMER_RECON_COEF:=1.0}"
: "${TRANSFORMER_PRED_COEF:=1.0}"
: "${TRANSFORMER_CYCLE_COEF:=0.5}"
: "${TRANSFORMER_WINDOW_SIZE:=16}"
: "${TRANSFORMER_D_C:=64}"
: "${TRANSFORMER_V2:=0}"

# PH2 schedule configs
PH2_RATIO_STAGE1=2
PH2_RATIO_STAGE2=1
PH2_RATIO_STAGE3=2
PH2_FIXED_IND_PROB=""

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
    --ph1-dist $PH1_DIST_THRESH \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob $PH1_NORMAL_PROB \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count $ph1_max_penalty_count \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-ratio-stage1 $PH2_RATIO_STAGE1 \
    --ph2-ratio-stage2 $PH2_RATIO_STAGE2 \
    --ph2-ratio-stage3 $PH2_RATIO_STAGE3 \
    --action-prediction "$ACTION_PREDICTION"
    --cycle-loss-enabled "$CYCLE_LOSS_ENABLED"
    --cycle-loss-coef "$CYCLE_LOSS_COEF"
    --latent-mode "$LATENT_MODE")

  if [[ -n "$PH2_FIXED_IND_PROB" ]]; then
    cmd+=(--ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB")
  fi
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
SWEEP_GPUS="1,2,3,4,5"
TARGET_OMEGA=10.0
TARGET_SIGMA=2.0
TARGET_MAX_COUNT=1

# -----------------------------------------------------------------------------
# 레이아웃별 실행 커멘드 — 원하는 줄만 주석 해제해서 사용
#
# OV1 (full observation, agent_view_size 없음):
# -----------------------------------------------------------------------------

# echo "[PH2] counter_circuit"
# run_ph2 "$SWEEP_GPUS" "counter_circuit" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] asymm_advantages"
# run_ph2 "$SWEEP_GPUS" "asymm_advantages" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] cramped_room"
# run_ph2 "$SWEEP_GPUS" "cramped_room" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] coord_ring"
# run_ph2 "$SWEEP_GPUS" "coord_ring" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

TARGET_OMEGA=5.0
TARGET_SIGMA=2.0
TARGET_MAX_COUNT=1

echo "[PH2] forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"


TARGET_OMEGA=5.0
TARGET_SIGMA=3.0
TARGET_MAX_COUNT=1


echo "[PH2] forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"


TARGET_OMEGA=3.0
TARGET_SIGMA=3.0
TARGET_MAX_COUNT=1


echo "[PH2] forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"


TARGET_OMEGA=3.0
TARGET_SIGMA=2.0
TARGET_MAX_COUNT=1


echo "[PH2] forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"


TARGET_OMEGA=2.0
TARGET_SIGMA=2.0
TARGET_MAX_COUNT=1


echo "[PH2] forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# -----------------------------------------------------------------------------
# OV2 (partial observation, agent_view_size=2):
# USE_CT=1로 실행 권장 — get_obs_default() full obs가 CT recon target으로 사용됨
# -----------------------------------------------------------------------------

# echo "[PH2] grounded_coord_simple"
# run_ph2 "$SWEEP_GPUS" "grounded_coord_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] grounded_coord_ring"
# run_ph2 "$SWEEP_GPUS" "grounded_coord_ring" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] test_time_simple"
# run_ph2 "$SWEEP_GPUS" "test_time_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] test_time_wide"
# run_ph2 "$SWEEP_GPUS" "test_time_wide" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] demo_cook_simple"
# run_ph2 "$SWEEP_GPUS" "demo_cook_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# echo "[PH2] demo_cook_wide"
# run_ph2 "$SWEEP_GPUS" "demo_cook_wide" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# =============================================================================
# CT v1 / v2 비교 실험: grounded_coord_simple (OV2), counter_circuit (OV1)
# GPU: 0,1,2,3,4
# =============================================================================
# SWEEP_GPUS="3,4,5,6,7"

# # --- grounded_coord_simple: CT v1 ---
# USE_CT=1
# TRANSFORMER_V2=0
# EXP="rnn-ct"
# echo "[PH2-CT-v1] grounded_coord_simple"
# run_ph2 "$SWEEP_GPUS" "grounded_coord_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"
# wait
# # --- grounded_coord_simple: CT v2 ---
# USE_CT=1
# TRANSFORMER_V2=1
# EXP="rnn-ct"
# echo "[PH2-CT-v2] grounded_coord_simple"
# run_ph2 "$SWEEP_GPUS" "grounded_coord_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"
# wait
# # --- grounded_coord_simple: original ---
# USE_CT=0
# TRANSFORMER_V2=0
# EXP="rnn-ph2"
# echo "[PH2-CT-original] grounded_coord_simple"
# run_ph2 "$SWEEP_GPUS" "grounded_coord_simple" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"
# # --- counter_circuit: CT v1 ---
# USE_CT=1
# TRANSFORMER_V2=0
# EXP="rnn-ct"
# echo "[PH2-CT-v1] counter_circuit"
# run_ph2 "$SWEEP_GPUS" "counter_circuit" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"
# wait
# # --- counter_circuit: CT v2 ---
# USE_CT=1
# TRANSFORMER_V2=1
# EXP="rnn-ct"
# echo "[PH2-CT-v2] counter_circuit"
# run_ph2 "$SWEEP_GPUS" "counter_circuit" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# =============================================================================
# ToyCoop (Dual Destination) — MLP encoder, full obs
# RANDOM_RESET 기본값은 스크립트 상단에서 설정 (true/false)
# =============================================================================

# ToyCoop 공통: 10시드, MLP, env_device=cpu
# NUM_SEEDS=5
# NENVS=512
# NSTEPS=100

# # --- ToyCoop CT=0, random_reset (기본값 따름) ---
# EXP="rnn-ph2-toycoop"
# USE_CT=0
# echo "[PH2] toy_coop (CT=0, random_reset=$RANDOM_RESET)"
# run_ph2 "$SWEEP_GPUS" "toy_coop" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# # --- ToyCoop CT=1, random_reset (기본값 따름) ---
# EXP="rnn-ph2-toycoop"
# USE_CT=1
# echo "[PH2] toy_coop (CT=1, random_reset=$RANDOM_RESET)"
# run_ph2 "$SWEEP_GPUS" "toy_coop" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# # ToyCoop 공통: 10시드, MLP, env_device=cpu
# NUM_SEEDS=5
# NENVS=512
# NSTEPS=100
# FIXED_SEED=41

# # --- ToyCoop CT=0, random_reset (기본값 따름) ---
# EXP="rnn-ph2-toycoop"
# USE_CT=0
# echo "[PH2] toy_coop (CT=0, random_reset=$RANDOM_RESET)"
# run_ph2 "$SWEEP_GPUS" "toy_coop" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

# # --- ToyCoop CT=1, random_reset (기본값 따름) ---
# EXP="rnn-ph2-toycoop"
# USE_CT=1
# echo "[PH2] toy_coop (CT=1, random_reset=$RANDOM_RESET)"
# run_ph2 "$SWEEP_GPUS" "toy_coop" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"
