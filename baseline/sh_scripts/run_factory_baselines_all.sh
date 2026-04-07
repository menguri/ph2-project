#!/usr/bin/env bash
# =============================================================================
# 전체 Baseline 실험: SP, E3T, FCP, MEP, GAMMA, HSP — OV1 전 레이아웃
# 각 알고리즘의 config에 파라미터가 전부 정의되어 있으므로 별도 override 불필요.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="4,5"
ENV_DEVICE="cpu"

# Eval 체크포인트: eval/ 서브폴더에 1M step 단위로 저장 (전부 30M → 31 ckpts)
EVAL_CKPT_DIR_EXTRA="++SAVE_TO_EVAL_DIR=true"
EVAL_NUM_CKPT_EXTRA="++NUM_CHECKPOINTS=31"
EVAL_NUM_CKPT_SP_EXTRA="${EVAL_NUM_CKPT_EXTRA}"

# OV1_LAYOUTS=(cramped_room asymm_advantages coord_ring forced_coord counter_circuit)
OV1_LAYOUTS=(asymm_advantages)

# 레이아웃별 nenvs override (OOM 방지용). 미설정 레이아웃은 config 기본값 사용.
declare -A NENVS_OVERRIDE=(
  [asymm_advantages]=32
  [counter_circuit]=64
)

# =============================================================================
# 1. SP (COLE-sync)
# =============================================================================
echo "============================================================"
echo "  SP (COLE-sync) — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[SP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-sp-cole \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --env-device "${ENV_DEVICE}" \
    --tags sp-cole \
    --extra "${EVAL_CKPT_DIR_EXTRA}" \
    --extra "${EVAL_NUM_CKPT_SP_EXTRA}"
done

# =============================================================================
# 2. E3T (원본 E3T-Overcooked sync)
# =============================================================================
echo "============================================================"
echo "  E3T — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[E3T] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-e3t \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --env-device "${ENV_DEVICE}" \
    --tags e3t \
    --extra "${EVAL_CKPT_DIR_EXTRA}" \
    --extra "${EVAL_NUM_CKPT_SP_EXTRA}"
done

# =============================================================================
# 3. FCP (COLE-sync) — SP population 필요
# =============================================================================
echo "============================================================"
echo "  FCP (COLE-sync) — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  FCP_DIR="fcp_populations/${layout}_sp"
  if [ ! -d "${FCP_DIR}" ]; then
    echo "[FCP] SKIP ${layout} — population 없음: ${FCP_DIR}"
    continue
  fi
  echo "[FCP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-fcp-cole \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --env-device "${ENV_DEVICE}" \
    --fcp "${FCP_DIR}" \
    --seeds 1 \
    --tags fcp-cole \
    --extra "${EVAL_CKPT_DIR_EXTRA}" \
    --extra "${EVAL_NUM_CKPT_SP_EXTRA}"
done

# =============================================================================
# 4. MEP (원본 MEP sync)
# =============================================================================
echo "============================================================"
echo "  MEP — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[MEP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-mep \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --extra "${EVAL_CKPT_DIR_EXTRA}" \
    --extra "${EVAL_NUM_CKPT_EXTRA}"
done

# =============================================================================
# 5. GAMMA (논문 sync, VAE mode)
#    gamma_population이 이미 저장된 레이아웃은 S1 스킵 (--gamma-pop-dir)
#    asymm_advantages는 population 없음 → 처음부터 전체 실행
# =============================================================================
echo "============================================================"
echo "  GAMMA (vae) — OV1"
echo "============================================================"

declare -A GAMMA_POP_DIRS=(
  [cramped_room]="runs/20260403-030117_cxscqq3s_cramped_room_gamma-vae_h64_pop5_z16_e100/gamma_population"
  [coord_ring]="runs/20260403-031556_2lwp3hfs_coord_ring_gamma-vae_h64_pop5_z16_e100/gamma_population"
  [forced_coord]="runs/20260403-033139_fydlylv5_forced_coord_gamma-vae_h64_pop5_z16_e100/gamma_population"
  [counter_circuit]="runs/20260403-034716_fhmsbh93_counter_circuit_gamma-vae_h64_pop5_z16_e100/gamma_population"
)

for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[GAMMA vae] ${layout}"
  NENVS_ARG=""
  [[ -n "${NENVS_OVERRIDE[$layout]+_}" ]] && NENVS_ARG="--nenvs ${NENVS_OVERRIDE[$layout]}"
  if [[ -n "${GAMMA_POP_DIRS[$layout]+_}" ]]; then
    # S1 스킵: 기존 population 재사용
    ./run_user_wandb.sh \
      --exp rnn-gamma \
      --env "${layout}" \
      --gpus "${GPUS}" \
      --seeds 1 \
      --gamma-pop-dir "${GAMMA_POP_DIRS[$layout]}" \
      --extra "++GAMMA_S2_METHOD=vae" \
      --extra "${EVAL_CKPT_DIR_EXTRA}" \
      --extra "${EVAL_NUM_CKPT_EXTRA}" \
      ${NENVS_ARG}
  else
    # asymm_advantages: population 없어 S1부터 전체 실행
    ./run_user_wandb.sh \
      --exp rnn-gamma \
      --env "${layout}" \
      --gpus "${GPUS}" \
      --seeds 1 \
      --extra "++GAMMA_S2_METHOD=vae" \
      --extra "${EVAL_CKPT_DIR_EXTRA}" \
      --extra "${EVAL_NUM_CKPT_EXTRA}" \
      ${NENVS_ARG}
  fi
done