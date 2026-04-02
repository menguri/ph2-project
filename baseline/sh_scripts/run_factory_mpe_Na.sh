#!/usr/bin/env bash
# =============================================================================
# MPE N-Agent Baselines — SP, E3T, FCP, MEP
# SimpleSpread (N agents, N landmarks)
#
# 사용법:
#   bash run_factory_mpe_Na.sh 3             # 3-agent (기본)
#   bash run_factory_mpe_Na.sh 5             # 5-agent
#   bash run_factory_mpe_Na.sh 10            # 10-agent
#   CROSS_PLAY_SEEDS=1 bash run_factory_mpe_Na.sh 10   # eval seed 1개
#   SMOKE=1 bash run_factory_mpe_Na.sh 5     # smoke test
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Agent 수 (첫 번째 인자, 기본 3)
N_AGENTS=${1:-5}

# GPU / 공통 설정
: "${GPUS:=1,2}"
: "${NUM_SEEDS:=10}"
: "${SMOKE:=0}"
: "${CROSS_PLAY_SEEDS:=10}"

# 5a 이상에서는 cross-play eval seed 기본 1개
if [[ "$N_AGENTS" -ge 5 && -z "${CROSS_PLAY_SEEDS_SET:-}" ]]; then
  CROSS_PLAY_SEEDS=1
fi

ENV="mpe_spread_${N_AGENTS}a"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=128
FCP_POP="fcp_populations/${ENV}_sp"

# 3a 이상 공통 experiment config (하이퍼파라미터 동일)
EXP_SP="rnn-sp-mpe-3a"
EXP_E3T="rnn-e3t-mpe-3a"
EXP_FCP="rnn-fcp-mpe-3a"
EXP_MEP="rnn-mep-mpe-3a"

# cross-play seeds를 hydra override로 전달
XPLAY_ARG="--extra +CROSS_PLAY_SEEDS=${CROSS_PLAY_SEEDS}"

if [[ "$SMOKE" == "1" ]]; then
  SMOKE_EXTRA="--extra model.TOTAL_TIMESTEPS=100000"
  NUM_SEEDS=1
  echo "[SMOKE TEST MODE] TOTAL_TIMESTEPS=100000, SEEDS=1"
else
  SMOKE_EXTRA=""
fi

echo "============================================================"
echo "  MPE ${N_AGENTS}-Agent Baselines"
echo "  ENV: ${ENV} | GPU: ${GPUS} | Seeds: ${NUM_SEEDS}"
echo "  Cross-play eval seeds: ${CROSS_PLAY_SEEDS}"
echo "============================================================"

# =============================================================================
# SP
# =============================================================================
run_sp() {
  echo "===== [SP] MPE ${N_AGENTS}A ====="
  ./run_user_wandb.sh \
    --exp "${EXP_SP}" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "sp,mpe,${N_AGENTS}a" \
    $XPLAY_ARG $SMOKE_EXTRA
}

# =============================================================================
# FCP population 자동 구성 (SP → fcp_populations/)
# =============================================================================
build_fcp_population() {
  echo "===== FCP population 구성 (SP → ${FCP_POP}) ====="

  SP_RUN=$(ls -dt ../runs/*MPE_simple_spread*${N_AGENTS}a*_sp* 2>/dev/null | head -1)
  if [[ -z "$SP_RUN" ]]; then
    echo "[ERROR] ${N_AGENTS}a SP run을 찾을 수 없습니다."
    return 1
  fi

  echo "[FCP-POP] SP run: ${SP_RUN}"
  rm -rf "$FCP_POP"
  mkdir -p "$FCP_POP"

  counter=0
  subdir_counter=0
  for run in "$SP_RUN"/run_*; do
    [[ ! -d "$run" ]] && continue
    if (( counter % 8 == 0 )); then
      subdir_counter=$((subdir_counter + 1))
      mkdir -p "$FCP_POP/fcp_$subdir_counter"
    fi
    cp -r "$run" "$FCP_POP/fcp_$subdir_counter/"
    counter=$((counter + 1))
  done
  echo "[FCP-POP] ${counter} runs → ${subdir_counter} groups"
}

# =============================================================================
# E3T (no action prediction, mixed policy only)
# =============================================================================
run_e3t() {
  echo "===== [E3T] MPE ${N_AGENTS}A ====="
  ./run_user_wandb.sh \
    --exp "${EXP_E3T}" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "e3t,mpe,${N_AGENTS}a" \
    $XPLAY_ARG $SMOKE_EXTRA
}

# =============================================================================
# MEP (S1 + S2 unified)
# =============================================================================
run_mep() {
  echo "===== [MEP] MPE ${N_AGENTS}A ====="
  ./run_user_wandb.sh \
    --exp "${EXP_MEP}" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "mep,mpe,${N_AGENTS}a" \
    $XPLAY_ARG $SMOKE_EXTRA
}

# =============================================================================
# FCP
# =============================================================================
run_fcp() {
  if [[ ! -d "$FCP_POP" ]]; then
    echo "[ERROR] FCP population 없음 (${FCP_POP}). build_fcp_population 먼저."
    return 1
  fi
  echo "===== [FCP] MPE ${N_AGENTS}A ====="
  ./run_user_wandb.sh \
    --exp "${EXP_FCP}" \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --fcp "${FCP_POP}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --tags "fcp,mpe,${N_AGENTS}a" \
    $XPLAY_ARG $SMOKE_EXTRA
}

# =============================================================================
# 실행 — 필요한 것만 주석 해제
# 권장 순서: SP → build_fcp_population → E3T, FCP, MEP
# =============================================================================
# --- (1) SP ---
run_sp

# --- (2) SP → FCP population 복사 ---
build_fcp_population

# --- (3) E3T ---
run_e3t

# --- (3) FCP ---
run_fcp

# --- (3) MEP ---
run_mep

echo ""
echo "============================================================"
echo "  모든 MPE ${N_AGENTS}A baseline 실험 완료"
echo "============================================================"
