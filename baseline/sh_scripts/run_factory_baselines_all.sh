#!/usr/bin/env bash
# =============================================================================
# MEP, GAMMA (vae), HSP — OV1 + OV2 전 레이아웃 실험
# GPU 6,7 / S2 10 seeds
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# GPU 할당 (S1+S2 모두 같은 프로세스에서 순차 실행 — 동일 GPU 사용)
GPUS="6,7"

# --- OV1 layouts ---
OV1_LAYOUTS=(cramped_room asymm_advantages coord_ring forced_coord counter_circuit)

# --- OV2 layouts ---
OV2_LAYOUTS=(grounded_coord_simple grounded_coord_ring demo_cook_simple demo_cook_wide test_time_simple test_time_wide)

ALL_LAYOUTS=("${OV1_LAYOUTS[@]}" "${OV2_LAYOUTS[@]}")

# =============================================================================
# GAMMA (vae)
# =============================================================================
run_gamma() {
  local layout=$1
  echo "[GAMMA vae] layout=${layout} | GPU ${GPUS} (S2: 10 seeds)"
  ./run_user_wandb.sh \
    --exp rnn-gamma \
    --env "${layout}" \
    --layout "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --extra "++GAMMA_S2_METHOD=vae"
}

echo "============================================================"
echo "  GAMMA (vae) — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_gamma "$layout"
done

# =============================================================================
# HSP
# =============================================================================
run_hsp() {
  local layout=$1
  echo "[HSP] layout=${layout} | GPU ${GPUS} (S2: 10 seeds)"
  ./run_user_wandb.sh \
    --exp rnn-hsp \
    --env "${layout}" \
    --layout "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1
}

echo "============================================================"
echo "  HSP — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_hsp "$layout"
done

# =============================================================================
# MEP
# =============================================================================
run_mep() {
  local layout=$1
  echo "[MEP] layout=${layout} | GPU ${GPUS} (S2: 10 seeds)"
  ./run_user_wandb.sh \
    --exp rnn-mep \
    --env "${layout}" \
    --layout "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1
}

echo "============================================================"
echo "  MEP — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_mep "$layout"
done