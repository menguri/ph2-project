#!/usr/bin/env bash
# =============================================================================
# MPE Baselines — SP, E3T, FCP, MEP, GAMMA
# SimpleSpread + SimpleReference 환경에서 순차 실행
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# GPU 할당
: "${GPUS:=0,1}"
: "${NUM_SEEDS:=10}"

MPE_ENVS=(mpe_spread mpe_reference)

# =============================================================================
# SP
# =============================================================================
echo "============================================================"
echo "  SP — MPE 환경"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  echo "[SP] env=${env} | GPU ${GPUS} | seeds=${NUM_SEEDS}"
  ./run_user_wandb.sh \
    --exp rnn-sp-mpe \
    --env "${env}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --tags "sp,mpe,${env}"
done

# =============================================================================
# FCP population 자동 구성 (SP 학습 완료 후)
# 최신 SP run을 찾아서 fcp_populations/{env}_sp 로 복사
# =============================================================================
echo "============================================================"
echo "  FCP population 구성 (SP → fcp_populations)"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  # 환경 이름 → MPE run dir 패턴
  if [[ "$env" == "mpe_spread" ]]; then
    SP_RUN=$(ls -dt ../runs/*MPE_simple_spread*_sp 2>/dev/null | head -1)
  elif [[ "$env" == "mpe_reference" ]]; then
    SP_RUN=$(ls -dt ../runs/*MPE_simple_reference*_sp 2>/dev/null | head -1)
  fi

  if [[ -z "$SP_RUN" ]]; then
    echo "[WARN] ${env}: SP run 없음, population 구성 건너뜀"
    continue
  fi

  FCP_DEST="fcp_populations/${env}_sp"
  echo "[FCP-POP] ${env}: ${SP_RUN} → ${FCP_DEST}"

  # 기존 population 삭제 후 재구성
  rm -rf "$FCP_DEST"
  mkdir -p "$FCP_DEST"

  counter=0
  subdir_counter=0
  for run in "$SP_RUN"/run_*; do
    [[ ! -d "$run" ]] && continue
    if (( counter % 8 == 0 )); then
      subdir_counter=$((subdir_counter + 1))
      mkdir -p "$FCP_DEST/fcp_$subdir_counter"
    fi
    cp -r "$run" "$FCP_DEST/fcp_$subdir_counter/"
    counter=$((counter + 1))
  done
  echo "[FCP-POP] ${env}: ${counter} runs → ${subdir_counter} groups"
done

# =============================================================================
# E3T
# =============================================================================
echo "============================================================"
echo "  E3T — MPE 환경"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  echo "[E3T] env=${env} | GPU ${GPUS} | seeds=${NUM_SEEDS}"
  ./run_user_wandb.sh \
    --exp rnn-e3t-mpe \
    --env "${env}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --tags "e3t,mpe,${env}"
done

# =============================================================================
# MEP (S1 + S2 unified)
# =============================================================================
echo "============================================================"
echo "  MEP — MPE 환경"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  echo "[MEP] env=${env} | GPU ${GPUS} | seeds=1 (S2: ${NUM_SEEDS})"
  ./run_user_wandb.sh \
    --exp rnn-mep-mpe \
    --env "${env}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags "mep,mpe,${env}"
done

# =============================================================================
# GAMMA (S1 + S2 unified, rl mode)
# =============================================================================
echo "============================================================"
echo "  GAMMA — MPE 환경"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  echo "[GAMMA] env=${env} | GPU ${GPUS} | seeds=1 (S2: ${NUM_SEEDS})"
  ./run_user_wandb.sh \
    --exp rnn-gamma-mpe \
    --env "${env}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags "gamma,mpe,${env}"
done

# =============================================================================
# FCP — SP population 사용 (환경명 기반 폴더: fcp_populations/{env}_sp)
# =============================================================================
echo "============================================================"
echo "  FCP — MPE 환경 (환경명 기반 population)"
echo "============================================================"
for env in "${MPE_ENVS[@]}"; do
  FCP_POP="fcp_populations/${env}_sp"
  if [[ ! -d "$FCP_POP" ]]; then
    echo "[WARN] ${env}: population 디렉토리 없음 (${FCP_POP}), FCP 건너뜀"
    continue
  fi

  echo "[FCP] ${env}: population = ${FCP_POP}"

  ./run_user_wandb.sh \
    --exp rnn-fcp-mpe \
    --env "${env}" \
    --gpus "${GPUS}" \
    --seeds "${NUM_SEEDS}" \
    --fcp "${FCP_POP}" \
    --tags "fcp,mpe,${env}"
done

echo ""
echo "============================================================"
echo "  모든 MPE baseline 실험 완료"
echo "============================================================"
