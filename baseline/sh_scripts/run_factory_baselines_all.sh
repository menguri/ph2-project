#!/usr/bin/env bash
# =============================================================================
# 전체 Baseline 실험: SP, E3T, FCP, MEP, GAMMA, HSP — OV1 전 레이아웃
# 각 알고리즘의 config에 파라미터가 전부 정의되어 있으므로 별도 override 불필요.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="3,7"
ENV_DEVICE="cpu"

OV1_LAYOUTS=(cramped_room asymm_advantages coord_ring forced_coord counter_circuit)

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
    --tags sp-cole
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
    --tags e3t
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
    --tags fcp-cole
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
    --seeds 1
done

# =============================================================================
# 5. GAMMA (논문 sync, VAE mode)
# =============================================================================
echo "============================================================"
echo "  GAMMA (vae) — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[GAMMA vae] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-gamma \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --extra "++GAMMA_S2_METHOD=vae"
done

# =============================================================================
# 6. HSP (논문 sync, S1→Greedy→S2 통합)
# =============================================================================
echo "============================================================"
echo "  HSP — OV1"
echo "============================================================"
for layout in "${OV1_LAYOUTS[@]}"; do
  echo "[HSP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-hsp \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags hsp
done

