#!/usr/bin/env bash
# =============================================================================
# MPE 3-Agent Baselines — SP, E3T, FCP, MEP
# SimpleSpread (3 agents, 3 landmarks)
# 2-agent run_mpe_baselines_all.sh와 동일 구조, 파라미터 통일
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# GPU 할당
: "${GPUS:=0,4}"
: "${NUM_SEEDS:=10}"
: "${SMOKE:=0}"

ENV="mpe_spread_3a"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=128
FCP_POP="fcp_populations/mpe_spread_3a_sp"

if [[ "$SMOKE" == "1" ]]; then
  SMOKE_EXTRA="--extra model.TOTAL_TIMESTEPS=100000"
  NUM_SEEDS=1
  echo "[SMOKE TEST MODE] TOTAL_TIMESTEPS=100000, SEEDS=1"
else
  SMOKE_EXTRA=""
fi

# =============================================================================
# SP
# =============================================================================
run_sp() {
  echo "============================================================"
  echo "  [SP] MPE 3A | GPU ${GPUS} | seeds=${NUM_SEEDS}"
  echo "============================================================"
  ./run_user_wandb.sh \
    --exp "rnn-sp-mpe-3a" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "sp,mpe,3a" \
    $SMOKE_EXTRA
}

# =============================================================================
# FCP population 자동 구성 (SP 학습 완료 후)
# 최신 SP run을 찾아서 fcp_populations/mpe_spread_3a_sp/ 로 복사
# =============================================================================
build_fcp_population() {
  # sh_scripts/ → baseline/ 기준으로 경로 설정
  local POP_DIR="../${FCP_POP}"
  echo "============================================================"
  echo "  FCP population 구성 (SP → ${POP_DIR})"
  echo "============================================================"

  SP_RUN=$(ls -dt ../runs/*MPE_simple_spread*_sp* 2>/dev/null | head -1)

  if [[ -z "$SP_RUN" ]]; then
    echo "[ERROR] SP run을 찾을 수 없습니다. SP를 먼저 학습하세요."
    return 1
  fi

  echo "[FCP-POP] SP run: ${SP_RUN}"

  rm -rf "$POP_DIR"
  mkdir -p "$POP_DIR"

  counter=0
  subdir_counter=0
  for run in "$SP_RUN"/run_*; do
    [[ ! -d "$run" ]] && continue
    if (( counter % 8 == 0 )); then
      subdir_counter=$((subdir_counter + 1))
      mkdir -p "$POP_DIR/fcp_$subdir_counter"
    fi
    cp -r "$run" "$POP_DIR/fcp_$subdir_counter/"
    counter=$((counter + 1))
  done

  echo "[FCP-POP] ${counter} runs → ${subdir_counter} groups → ${POP_DIR}"
}

# =============================================================================
# E3T (no action prediction, mixed policy only)
# =============================================================================
run_e3t() {
  echo "============================================================"
  echo "  [E3T] MPE 3A | GPU ${GPUS} | seeds=${NUM_SEEDS}"
  echo "============================================================"
  ./run_user_wandb.sh \
    --exp "rnn-e3t-mpe-3a" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "e3t,mpe,3a" \
    $SMOKE_EXTRA
}

# =============================================================================
# MEP (S1 + S2 unified)
# =============================================================================
run_mep() {
  echo "============================================================"
  echo "  [MEP] MPE 3A | GPU ${GPUS} | seeds=1 (S2: ${NUM_SEEDS})"
  echo "============================================================"
  ./run_user_wandb.sh \
    --exp "rnn-mep-mpe-3a" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "mep,mpe,3a" \
    $SMOKE_EXTRA
}

# =============================================================================
# FCP — SP population 사용
# =============================================================================
run_fcp() {
  if [[ ! -d "../${FCP_POP}" ]]; then
    echo "[ERROR] FCP population 없음 (../${FCP_POP}). build_fcp_population 먼저 실행하세요."
    return 1
  fi

  echo "============================================================"
  echo "  [FCP] MPE 3A | population = ${FCP_POP}"
  echo "============================================================"
  ./run_user_wandb.sh \
    --exp "rnn-fcp-mpe-3a" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --fcp "${FCP_POP}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "fcp,mpe,3a" \
    $SMOKE_EXTRA
}

# =============================================================================
# 실행 — 필요한 것만 주석 해제
# 권장 순서:
#   (1) SP 학습 (FCP population 원본)
#   (2) SP 체크포인트 → fcp_populations/mpe_spread_3a_sp/ 자동 복사
#   (3) E3T, FCP, MEP 학습 (병렬 가능)
# =============================================================================
# --- (1) SP ---
run_sp

# --- (2) SP → FCP population 복사 ---
build_fcp_population

# --- (3) E3T ---
run_e3t

# --- (3) FCP (build_fcp_population 완료 후) ---
run_fcp

# --- (3) MEP (S1+S2 자동) ---
run_mep

echo ""
echo "============================================================"
echo "  모든 MPE 3A baseline 실험 완료"
echo "============================================================"
