#!/usr/bin/env bash
# =============================================================================
# (1) MEP, GAMMA, HSP — OV1 + OV2 전 레이아웃 실험
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# GPU 할당 (필요에 따라 수정)
GPU_S1="0"        # S1 (population training, 순차)
GPU_S2="0,1"      # S2 (adaptive, multi-seed)

# --- OV1 layouts (original overcooked, full observation) ---
OV1_LAYOUTS=(cramped_room asymm_advantages coord_ring forced_coord counter_circuit)

# --- OV2 layouts (partial observation, agent_view_size=2) ---
OV2_LAYOUTS=(grounded_coord_simple grounded_coord_ring)

ALL_LAYOUTS=("${OV1_LAYOUTS[@]}" "${OV2_LAYOUTS[@]}")

# =============================================================================
# MEP (원본 MEP 기준 파라미터, rnn-mep config)
# =============================================================================
run_mep() {
  local gpus=$1; local layout=$2
  echo "[MEP] layout=${layout} gpus=${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-mep \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1
}

echo "============================================================"
echo "  MEP — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_mep "$GPU_S1" "$layout"
done

# =============================================================================
# GAMMA (GAMMA 기준 파라미터, rnn-gamma config, method=rl/vae)
# =============================================================================
GAMMA_METHOD="${GAMMA_METHOD:-rl}"   # rl | vae

run_gamma() {
  local gpus=$1; local layout=$2; local method=${3:-$GAMMA_METHOD}
  echo "[GAMMA ${method}] layout=${layout} gpus=${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-gamma \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1 \
      --extra "++GAMMA_S2_METHOD=${method}"
}

echo "============================================================"
echo "  GAMMA (${GAMMA_METHOD}) — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_gamma "$GPU_S1" "$layout" "$GAMMA_METHOD"
done

# =============================================================================
# HSP (원본 HSP 기준 파라미터, rnn-hsp config)
# =============================================================================
run_hsp() {
  local gpus=$1; local layout=$2
  echo "[HSP] layout=${layout} gpus=${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp rnn-hsp \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds 1
}

echo "============================================================"
echo "  HSP — All layouts"
echo "============================================================"
for layout in "${ALL_LAYOUTS[@]}"; do
  run_hsp "$GPU_S1" "$layout"
done
