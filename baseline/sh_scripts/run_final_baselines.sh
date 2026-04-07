#!/usr/bin/env bash
# =============================================================================
# Final Baseline 실험: E3T, MEP, HSP, Gamma
# GPU 5,6,7 사용, 순차 실행
# MEP/HSP/Gamma: S1=10M, S2=100M, nenvs=32, S2_NUM_SEEDS=12, 전 레이아웃
# E3T: 30M, NUM_SEEDS=12, 누락 레이아웃만
# 모든 알고리즘 S1부터 처음부터 실행
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="0,2,3,4,5"
ALL_LAYOUTS=(counter_circuit forced_coord asymm_advantages coord_ring cramped_room)

# =============================================================================
# 1. E3T — counter_circuit, forced_coord, asymm_advantages (누락분)
# =============================================================================
# echo "============================================================"
# echo "  E3T (12 seeds, 30M)"
# echo "============================================================"

# for layout in "${ALL_LAYOUTS[@]}"; do
#   echo "[E3T] ${layout}"
#   ./run_user_wandb.sh \
#     --exp rnn-e3t \
#     --env "${layout}" \
#     --gpus "${GPUS}" \
#     --seeds 10 \
#     --tags e3t,final
# done

# =============================================================================
# 2. MEP — 전 레이아웃
#    S1=10M, S2=100M, nenvs=32, S2 12 seeds
# =============================================================================
echo "============================================================"
echo "  MEP (S1=10M, S2=100M, nenvs=32, 12 seeds) — 전 레이아웃"
echo "============================================================"

for layout in "${ALL_LAYOUTS[@]}"; do
  echo "[MEP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-mep \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --nenvs 32 \
    --tags mep,final \
    --extra "model.TOTAL_TIMESTEPS=1.5e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=3e7" \
    --extra "S2_NUM_SEEDS=10"
done

# =============================================================================
# 4. Gamma — 전 레이아웃
#    S1=10M, S2=100M, nenvs=32, S2 12 seeds (전부 S1부터 새로 실행)
# =============================================================================
echo "============================================================"
echo "  Gamma vae (S1=10M, S2=100M, nenvs=32, 12 seeds) — 전 레이아웃"
echo "============================================================"

for layout in "${ALL_LAYOUTS[@]}"; do
  echo "[GAMMA vae] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-gamma \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --nenvs 32 \
    --tags gamma,final \
    --extra "++GAMMA_S2_METHOD=vae" \
    --extra "model.TOTAL_TIMESTEPS=1e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
    --extra "S2_NUM_SEEDS=10"
done

echo "============================================================"
echo "  전체 완료"
echo "============================================================"
