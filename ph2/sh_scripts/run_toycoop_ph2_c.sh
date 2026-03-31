#!/bin/bash
# =============================================================================
# ToyCoop (Dual Destination) — PH2
# ph2/ 에서 실행
#
# 사용법:
#   bash run_toycoop_ph2.sh                            # 기본 실행
#   bash run_toycoop_ph2.sh 6,7                        # GPU 지정
#   PH1_OMEGA=0.5 PH1_SIGMA=2.0 bash run_toycoop_ph2.sh  # 파라미터 오버라이드
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

# =============================================================================
# 공통 설정
# =============================================================================
DEFAULT_GPUS="${1:-0,3}"
: "${RANDOM_RESET:=true}"
ENV_DEVICE="cpu"
NENVS=512
NSTEPS=100
NUM_SEEDS=10
FIXED_SEED=42

RANDOM_RESET_ARGS=""
[[ "$RANDOM_RESET" == "true" ]] && RANDOM_RESET_ARGS="--random-reset true"

# =============================================================================
# PH2 기본 파라미터 (환경변수로 오버라이드 가능)
# =============================================================================
# PH1 blocked-target
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_OMEGA:=0.1}"
: "${PH1_SIGMA:=1.0}"
: "${PH1_POOL_SIZE:=64}"
: "${PH1_NORMAL_PROB:=0.0}"
: "${PH1_MULTI_PENALTY_ENABLED:=False}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=10000000}"

: "${PH2_EPSILON:=0.2}"
: "${PH2_FIXED_IND_PROB:=0.5}"   # ind 매칭 확률 (0.5 = spec-spec 50%, spec-ind/ind-ind 50%)

# 보조 모듈 (실제 사용되는 것만)
: "${ACTION_PREDICTION:=True}"
: "${SAVE_EVAL_CHECKPOINTS:=False}"

# CT 모드
: "${USE_CT:=0}"

# =============================================================================
# 실행 함수
# =============================================================================

run_ph2() {
  local gpus=${1:-$DEFAULT_GPUS}
  local omega=${2:-$PH1_OMEGA}
  local sigma=${3:-$PH1_SIGMA}
  local max_k=${4:-1}
  local seeds=${5:-$NUM_SEEDS}

  local exp="rnn-ph2-toycoop"
  local ct_tag="ct0"
  local ct_args=""
  if [[ "$USE_CT" == "1" ]]; then
    exp="rnn-ct"
    ct_tag="ct1"
    ct_args="--transformer-action True"
  fi

  echo "====== [PH2] ToyCoop  gpus=$gpus o=$omega s=$sigma k=$max_k ct=$ct_tag ======"
  ./run_user_wandb.sh \
      --gpus "$gpus" \
      --seeds "$seeds" \
      --seed "$FIXED_SEED" \
      --env toy_coop \
      --exp "$exp" \
      --env-device "$ENV_DEVICE" \
      --nenvs "$NENVS" \
      --nsteps "$NSTEPS" \
      --ph1-beta "$PH1_BETA" \
      --ph1-beta-schedule-enabled "$PH1_BETA_SCHEDULE_ENABLED" \
      --ph1-omega "$omega" \
      --ph1-sigma "$sigma" \
      --ph1-pool-size "$PH1_POOL_SIZE" \
      --ph1-normal-prob "$PH1_NORMAL_PROB" \
      --ph1-multi-penalty-enabled "$PH1_MULTI_PENALTY_ENABLED" \
      --ph1-max-penalty-count "$max_k" \
      --ph1-multi-penalty-single-weight "$PH1_MULTI_PENALTY_SINGLE_WEIGHT" \
      --ph1-multi-penalty-other-weight "$PH1_MULTI_PENALTY_OTHER_WEIGHT" \
      --ph1-epsilon "$PH1_EPSILON" \
      --ph1-warmup-steps "$PH1_WARMUP_STEPS" \
      --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
      --ph2-epsilon "$PH2_EPSILON" \
      --action-prediction "$ACTION_PREDICTION" \
      --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
      --tags "toycoop,ph2,k${max_k},${ct_tag}" \
      $RANDOM_RESET_ARGS \
      $ct_args
}

# =============================================================================
# 실행 — 원하는 줄만 주석 해제하거나 파라미터 조정
#
# run_ph2 <gpus> <omega> <sigma> <max_penalty_count> [seeds]
# =============================================================================
echo "============================================="
echo "  ToyCoop PH2 Pipeline"
echo "  GPUs: $DEFAULT_GPUS  |  CT: $USE_CT"
echo "  random_reset: $RANDOM_RESET"
echo "============================================="

# omega × sigma × k sweep (action prediction 모드, CT OFF)
for omega in 1 2 3 4 5; do
  for sigma in 5.0 3.0 2.0 1.0; do
    for k in 1 2; do
      run_ph2 "$DEFAULT_GPUS" "$omega" "$sigma" "$k"
    done
  done
done

echo "============================================="
echo "  ToyCoop PH2 Pipeline 완료"
echo "============================================="
