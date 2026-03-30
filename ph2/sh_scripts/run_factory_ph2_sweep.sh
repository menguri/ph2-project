#!/bin/bash
# =============================================================================
# (2) PH2 + PH2-CT v3 — grounded_coord_simple sweep
#     penalty_count × omega × sigma
# (3) PH2 — forced_coord 추가 sweep (패널티 영향 축소 방향)
# =============================================================================

cd "$(dirname "$0")" || exit 1

# GPU 할당
SWEEP_GPUS="1,2,3,4,5"

# 공통 파라미터
NUM_SEEDS=5
FIXED_SEED=41
NENVS=64
NSTEPS=256
ENV_DEVICE="gpu"

PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=1.0
PH1_POOL_SIZE=128
PH1_NORMAL_PROB=0.0
PH1_MULTI_PENALTY_ENABLED=True
PH1_EPSILON=0.2
PH2_EPSILON=0.2
PH1_WARMUP_STEPS=2000000
PH2_FIXED_IND_PROB=0.5
ACTION_PREDICTION=True
SAVE_EVAL_CHECKPOINTS=True

run_ph2() {
  local gpus=$1 env=$2 omega=$3 sigma=$4 max_k=$5
  local exp=${6:-rnn-ph2}
  local extra_args=("${@:7}")

  local tags="ph2,sweep,o${omega}_s${sigma}_k${max_k}"

  local -a cmd=("./run_user_wandb.sh"
    --gpus "$gpus" --seeds "$NUM_SEEDS" --seed "$FIXED_SEED"
    --env "$env" --exp "$exp"
    --env-device "$ENV_DEVICE" --nenvs "$NENVS" --nsteps "$NSTEPS"
    --tags "$tags"
    --ph1-beta $PH1_BETA
    --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED
    --ph1-beta-start $PH1_BETA_START --ph1-beta-end $PH1_BETA_END
    --ph1-omega "$omega" --ph1-sigma "$sigma"
    --ph1-pool-size $PH1_POOL_SIZE --ph1-normal-prob $PH1_NORMAL_PROB
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED
    --ph1-max-penalty-count "$max_k"
    --ph1-epsilon $PH1_EPSILON --ph1-warmup-steps $PH1_WARMUP_STEPS
    --ph2-epsilon "$PH2_EPSILON"
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB"
    --action-prediction "$ACTION_PREDICTION"
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS"
    "${extra_args[@]}")

  echo ">>> $exp | $env | omega=$omega sigma=$sigma k=$max_k"
  "${cmd[@]}"
}

run_ph2_ct_v3() {
  local gpus=$1 env=$2 omega=$3 sigma=$4 max_k=$5
  run_ph2 "$gpus" "$env" "$omega" "$sigma" "$max_k" "rnn-ct" \
    --transformer-action True \
    --transformer-v3 True \
    --transformer-window-size 10 \
    --transformer-d-c 64 \
    --transformer-recon-coef 1.0 \
    --transformer-pred-coef 1.0 \
    --transformer-cycle-coef 0.5
}

# =============================================================================
# (2) grounded_coord_simple sweep: PH2 + PH2-CT v3
#     penalty_count ∈ {1, 2, 3, 4}
#     omega ∈ {3.0, 10.0}
#     sigma ∈ {2.0, 3.0}
# =============================================================================
echo "============================================================"
echo "  (2) PH2 — grounded_coord_simple sweep"
echo "============================================================"
for omega in 10.0 3.0; do
  for sigma in 2.0 3.0; do
    for k in 1 2 3 4; do
      run_ph2 "$SWEEP_GPUS" "grounded_coord_simple" "$omega" "$sigma" "$k"
    done
  done
done

echo "============================================================"
echo "  (2) PH2-CT v3 — grounded_coord_simple sweep"
echo "============================================================"
for omega in 10.0 3.0; do
  for sigma in 2.0 3.0; do
    for k in 1 2 3 4; do
      run_ph2_ct_v3 "$SWEEP_GPUS" "grounded_coord_simple" "$omega" "$sigma" "$k"
    done
  done
done

# =============================================================================
# (3) forced_coord — 패널티 영향 축소 방향 sweep
#     omega를 작게 (패널티 강도 ↓): {1.0, 3.0, 5.0}
#     sigma를 크게 (패널티 범위 ↓): {3.0, 5.0, 10.0}
#     penalty_count: {1, 2}
# =============================================================================
echo "============================================================"
echo "  (3) PH2 — forced_coord sweep (패널티 축소 방향)"
echo "============================================================"
for omega in 1.0 3.0 5.0; do
  for sigma in 3.0 5.0 10.0; do
    for k in 1 2; do
      run_ph2 "$SWEEP_GPUS" "forced_coord" "$omega" "$sigma" "$k"
    done
  done
done

# =============================================================================
# (4) PH2 Partner Prediction + Cycle Loss — OV2 전체 레이아웃
#     Z_PREDICTION_ENABLED=True, CYCLE_LOSS_ENABLED=True
#     CT v3와 동일한 policy input dim (262D)이지만 CT 없이 동작
#     omega=10, sigma=2, penalty_count=1 (기본값)
# =============================================================================
OV2_LAYOUTS=(grounded_coord_simple grounded_coord_ring)

run_ph2_zp() {
  local gpus=$1 env=$2 omega=$3 sigma=$4 max_k=$5
  run_ph2 "$gpus" "$env" "$omega" "$sigma" "$max_k" "rnn-ph2" \
    --z-prediction-enabled True \
    --cycle-loss-enabled True \
    --cycle-loss-coef 0.1
}

echo "============================================================"
echo "  (4) PH2 Z-Prediction + Cycle Loss — OV2 all layouts"
echo "============================================================"
for layout in "${OV2_LAYOUTS[@]}"; do
  echo "[PH2-ZP+Cycle] ${layout}"
  run_ph2_zp "$SWEEP_GPUS" "$layout" "10.0" "2.0" "1"
done

# Z-Prediction + Cycle Loss: grounded_coord_simple penalty sweep
echo "============================================================"
echo "  (4b) PH2 Z-Prediction + Cycle — grounded_coord_simple sweep"
echo "============================================================"
for omega in 10.0 3.0; do
  for sigma in 2.0 3.0; do
    for k in 1 2 3 4; do
      run_ph2_zp "$SWEEP_GPUS" "grounded_coord_simple" "$omega" "$sigma" "$k"
    done
  done
done
