#!/bin/bash
# =============================================================================
# ToyCoop (Dual Destination) — SP / E3T / FCP / MEP / GAMMA / HSP
# baseline/ 에서 실행
#
# 사용법:
#   bash run_toycoop.sh                          # 전체 (주석 해제된 것만)
#   bash run_toycoop.sh 6,7                      # GPU 지정
#   RANDOM_RESET=false bash run_toycoop.sh       # 랜덤 리셋 비활성화
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR="$(pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_BASE="${PROJECT_DIR}/runs"

# =============================================================================
# 공통 설정 (모든 알고리즘 공유)
# =============================================================================
DEFAULT_GPUS="${1:-4,5}"
: "${RANDOM_RESET:=true}"
ENV_DEVICE="cpu"
TOYCOOP_NENVS=512
TOYCOOP_NSTEPS=100

RANDOM_RESET_ARGS=""
[[ "$RANDOM_RESET" == "true" ]] && RANDOM_RESET_ARGS="--random-reset true"

# 공통 인자 배열
_common_args() {
  local gpus=${1:-$DEFAULT_GPUS}
  echo --gpus "$gpus" \
       --env toy_coop \
       --env-device "$ENV_DEVICE" \
       --nenvs "$TOYCOOP_NENVS" \
       --nsteps "$TOYCOOP_NSTEPS" \
       $RANDOM_RESET_ARGS
}

# =============================================================================
# 알고리즘별 함수
# =============================================================================

run_sp() {
  local gpus=${1:-$DEFAULT_GPUS}
  local seeds=${2:-10}
  echo "====== [SP] ToyCoop  gpus=$gpus seeds=$seeds ======"
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-sp-toycoop \
      --seeds "$seeds" \
      --tags "toycoop,sp"
}

run_e3t() {
  local gpus=${1:-$DEFAULT_GPUS}
  local seeds=${2:-10}
  local epsilon=${3:-0.2}
  echo "====== [E3T] ToyCoop  gpus=$gpus eps=$epsilon ======"
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-e3t-toycoop \
      --seeds "$seeds" \
      --e3t-epsilon "$epsilon" \
      --use-partner-modeling True \
      --pred-loss-coef 1.0 \
      --tags "toycoop,e3t"
}

run_fcp() {
  local gpus=${1:-$DEFAULT_GPUS}
  local seeds=${2:-10}
  local pop_path=${3:-fcp_populations/toy_coop_sp}
  local fcp_device=${4:-gpu}
  echo "====== [FCP] ToyCoop  gpus=$gpus pop=$pop_path ======"
  if [[ ! -d "$PROJECT_DIR/$pop_path" ]]; then
    echo "  [스킵] $pop_path 없음. SP를 먼저 실행하세요."
    return 1
  fi
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-fcp-toycoop \
      --seeds "$seeds" \
      --fcp "$pop_path" \
      --fcp-device "$fcp_device" \
      --tags "toycoop,fcp"
}

run_mep() {
  local gpus=${1:-$DEFAULT_GPUS}
  echo "====== [MEP] ToyCoop  gpus=$gpus (S1→S2 자동) ======"
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-mep-toycoop \
      --seeds 1 \
      --tags "toycoop,mep"
}

run_gamma() {
  local gpus=${1:-$DEFAULT_GPUS}
  local method=${2:-rl}   # rl | vae
  echo "====== [GAMMA] ToyCoop  gpus=$gpus method=$method (S1→S2 자동) ======"
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-gamma-toycoop \
      --seeds 1 \
      --tags "toycoop,gamma" \
      --extra "++GAMMA_S2_METHOD=$method"
}

run_hsp() {
  local gpus=${1:-$DEFAULT_GPUS}
  echo "====== [HSP] ToyCoop  gpus=$gpus (S1→Greedy→S2 자동, EVENT_DIM=3) ======"
  ./run_user_wandb.sh $(_common_args "$gpus") \
      --exp rnn-hsp-dualdest \
      --seeds 1 \
      --tags "toycoop,hsp,event3"
}

# =============================================================================
# 실행 — SP → FCP population 복사 → E3T → MEP → HSP → GAMMA 순차
# =============================================================================
echo "============================================="
echo "  ToyCoop Baseline Pipeline (전체 순차)"
echo "  GPUs: $DEFAULT_GPUS"
echo "  random_reset: $RANDOM_RESET"
echo "============================================="

# # --- (1) SP: 10 seeds (FCP population 용) ---
# run_sp "$DEFAULT_GPUS" 10

# # --- (2) SP → FCP population 복사 ---
# SP_RUN_NAME=$(ls -td "${RUNS_BASE}"/*ToyCoop*sp* 2>/dev/null | head -1 | xargs basename)
# if [[ -z "$SP_RUN_NAME" ]]; then
#   echo "[ERROR] SP run not found in ${RUNS_BASE}!" >&2
#   exit 1
# fi
# echo "[INFO] SP run: ${SP_RUN_NAME}"

# cd "$PROJECT_DIR" || exit 1
# bash sh_scripts/copy_fcp.sh "${SP_RUN_NAME}"

# # fcp_populations/{SP_RUN_NAME} → fcp_populations/toy_coop_sp 으로 이름 변경
# FCP_SRC="fcp_populations/${SP_RUN_NAME}"
# FCP_DST="fcp_populations/toy_coop_sp"
# if [[ -d "$FCP_DST" ]]; then
#   echo "[WARN] ${FCP_DST} already exists, removing..."
#   rm -rf "$FCP_DST"
# fi
# mv "$FCP_SRC" "$FCP_DST"
# echo "[INFO] FCP population ready: ${FCP_DST}"
# cd "$SCRIPT_DIR" || exit 1

# # --- (3) FCP ---
# run_fcp "$DEFAULT_GPUS" 10

# # --- (4) E3T ---
# run_e3t "$DEFAULT_GPUS" 10 0.2

# --- (5) MEP (S1 → S2 통합) ---
run_mep "$DEFAULT_GPUS"

# --- (7) GAMMA (S1 → S2 VAE) ---
run_gamma "$DEFAULT_GPUS" "vae"

# --- (6) HSP (S1 → Greedy Selection → S2 통합, EVENT_DIM=3) ---
run_hsp "$DEFAULT_GPUS"

echo "============================================="
echo "  ToyCoop Baseline Pipeline 완료"
echo "  PH2는 ph2/sh_scripts/run_toycoop_ph2.sh 에서 실행"
echo "============================================="
