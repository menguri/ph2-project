#!/usr/bin/env bash
# =============================================================================
# Final Baseline 실험
# 순서: GAMMA → SP → E3T → MEP (→ FCP는 주석 처리)
# Eval 체크포인트: base.yaml에서 EVAL_CKPT_EVERY_ENV_STEPS=500000 기본 적용
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="0,5"
ALL_LAYOUTS=(counter_circuit forced_coord asymm_advantages coord_ring cramped_room)

# =============================================================================
# 1. GAMMA — 전 레이아웃 (MAPPO, S1=10M, S2=100M VAE)
#    counter_circuit: 기존 S1 population 재사용 (S1 스킵)
# =============================================================================
echo "============================================================"
echo "  GAMMA (MAPPO, S1=10M, S2=100M VAE, nenvs=32, 10 seeds)"
echo "============================================================"

# counter_circuit: 기존 S1 재사용
echo "[GAMMA] counter_circuit (S1 재사용)"
./run_user_wandb.sh \
  --exp rnn-gamma \
  --env counter_circuit \
  --gpus "${GPUS}" \
  --seeds 1 \
  --nenvs 32 \
  --tags gamma,final \
  --extra "++GAMMA_S2_METHOD=vae" \
  --extra "model.TOTAL_TIMESTEPS=1e7" \
  --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
  --extra "S2_NUM_SEEDS=10" \
  --extra "++GAMMA_POPULATION_DIR=runs/20260410-062333_jmdif7z5_counter_circuit_gamma-vae_h64_pop8_z16_e32/gamma_population"

# 나머지 레이아웃: S1부터 새로 실행
for layout in forced_coord asymm_advantages coord_ring cramped_room; do
  echo "[GAMMA] ${layout}"
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

# =============================================================================
# 2. SP — 전 레이아웃
# =============================================================================
echo "============================================================"
echo "  SP (10 seeds, 30M)"
echo "============================================================"

for layout in "${ALL_LAYOUTS[@]}"; do
  echo "[SP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-sp \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 10 \
    --tags sp,final
done

# =============================================================================
# 3. E3T — 전 레이아웃
# =============================================================================
echo "============================================================"
echo "  E3T (10 seeds, 30M)"
echo "============================================================"

for layout in "${ALL_LAYOUTS[@]}"; do
  echo "[E3T] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-e3t \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 10 \
    --tags e3t,final
done

# =============================================================================
# 4. MEP — 전 레이아웃
# =============================================================================
echo "============================================================"
echo "  MEP (S1=15M, S2=30M, nenvs=32, 10 seeds)"
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
    --extra "model.TOTAL_TIMESTEPS=1e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
    --extra "S2_NUM_SEEDS=10"
done

=============================================================================
5. FCP — 주석 처리 (필요 시 해제)
=============================================================================
echo "============================================================"
echo "  FCP (10 seeds, 30M)"
echo "============================================================"

for layout in "${ALL_LAYOUTS[@]}"; do
  echo "[FCP] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-fcp \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 10 \
    --tags fcp,final
done

echo "============================================================"
echo "  전체 완료"
echo "============================================================"
