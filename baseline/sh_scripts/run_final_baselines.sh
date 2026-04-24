#!/usr/bin/env bash
# =============================================================================
# Final Baseline 실험 — counter_circuit / forced_coord, PH2 tuned
# 순서: GAMMA → MEP → E3T
# PH2 IPPO 하이퍼파라미터 이식 (NUM_ENVS 제외)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="0,6"
HARD_LAYOUTS=(counter_circuit forced_coord)

# PH2 IPPO 공통 하이퍼파라미터 — ph2/config/model/rnn.yaml 기준으로 전 알고리즘 정렬
# 제외: TOTAL_TIMESTEPS(원래 설정 유지), OBS_ENCODER(GAMMA는 CNNGamma — 알고리즘별 세팅).
# NUM_ENVS=256, NUM_STEPS=256, NUM_MINIBATCHES=64 공통 (batch=65,536 / mb_size=1024).
PH2_OVERRIDES=(
  --extra "model.LR=2.5e-4"
  --extra "model.ANNEAL_LR=True"
  --extra "model.LR_WARMUP=0.05"
  --extra "model.NUM_ENVS=64"
  --extra "model.NUM_STEPS=256"
  --extra "model.NUM_MINIBATCHES=16"
  --extra "model.UPDATE_EPOCHS=4"
  --extra "model.GAMMA=0.99"
  --extra "model.GAE_LAMBDA=0.95"
  --extra "model.CLIP_EPS=0.2"
  --extra "model.ENT_COEF=0.01"
  --extra "model.VF_COEF=0.5"
  --extra "model.MAX_GRAD_NORM=0.25"
  --extra "model.GRU_HIDDEN_DIM=128"
  --extra "model.FC_DIM_SIZE=128"
  --extra "model.CNN_FEATURES=32"
  --extra "model.ACTIVATION=relu"
)

# =============================================================================
# 1. GAMMA — counter_circuit / forced_coord (MAPPO, S1=30M, S2=100M VAE)
# =============================================================================
echo "============================================================"
echo "  GAMMA (MAPPO, S1=30M, S2=100M VAE, 10 seeds, PH2 tuned)"
echo "============================================================"

for layout in "${HARD_LAYOUTS[@]}"; do
  echo "[GAMMA] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-gamma \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags gamma,final,ph2tuned \
    --extra "++GAMMA_S2_METHOD=vae" \
    --extra "model.TOTAL_TIMESTEPS=3e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
    --extra "S2_NUM_SEEDS=10" \
    --extra "model.OBS_ENCODER=CNNGamma" \
    "${PH2_OVERRIDES[@]}"
done

# =============================================================================
# 2. MEP — counter_circuit / forced_coord (S1=30M, S2=100M)
# =============================================================================
echo "============================================================"
echo "  MEP (S1=30M, S2=100M, 10 seeds, PH2 tuned)"
echo "============================================================"

for layout in "${HARD_LAYOUTS[@]}"; do
  echo "[MEP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-mep \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags mep,final,ph2tuned \
    --extra "model.TOTAL_TIMESTEPS=3e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
    --extra "S2_NUM_SEEDS=10" \
    --extra "model.OBS_ENCODER=CNN" \
    "${PH2_OVERRIDES[@]}"
done

# =============================================================================
# 3. E3T — counter_circuit / forced_coord (30M, 10 seeds)
# =============================================================================
# echo "============================================================"
# echo "  E3T (30M, 10 seeds, PH2 tuned)"
# echo "============================================================"

# for layout in "${HARD_LAYOUTS[@]}"; do
#   echo "[E3T] ${layout}"
#   ./run_user_wandb.sh \
#     --exp rnn-e3t \
#     --env "${layout}" \
#     --gpus "${GPUS}" \
#     --seeds 10 \
#     --tags e3t,final,ph2tuned \
#     --extra "model.TOTAL_TIMESTEPS=3e7" \
#     --extra "model.OBS_ENCODER=CNN" \
#     "${PH2_OVERRIDES[@]}"
# done

echo "============================================================"
echo "  전체 완료"
echo "============================================================"
