#!/bin/bash
# =============================================================================
# PH2 Ablation V5 — Linear Penalty (α × k) sweep × {coord_ring, forced_coord}
#
# Linear mode penalty:  penalty = α / (ε + latent_dist),   ε=0.001 고정
#
# Sweep:
#   k=1:     α ∈ {0.01, 0.05, 0.5, 1.0}        (0.1은 이미 완료 → 스킵)
#   k=2,3,4: α ∈ {0.01, 0.05, 0.1, 0.5, 1.0}   (0.1 포함)
#   layouts: coord_ring, forced_coord
#   → 2 layouts × (4 + 5 + 5 + 5) α_cells = 38 cells × 12 seeds = 456 runs
#   순서: layout → k → α (각 k마다 α 전부 돌린 뒤 다음 k로)
#
# layout 별 omega/sigma/normal_prob 는 final/ph2 best 유지 (linear에선 unused이지만
# 호환성 위해 CLI 로 전달).
#
# Usage:
#   bash run_factory_ph2_ablation_v5.sh
#   GPUS="0,1,2,3" bash run_factory_ph2_ablation_v5.sh
#   LAYOUTS="coord_ring" bash run_factory_ph2_ablation_v5.sh
#   ALPHAS="0.01 0.05" KS="2 3" bash run_factory_ph2_ablation_v5.sh
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=${NUM_SEEDS:-12}
NSTEPS=256
FIXED_SEED=${FIXED_SEED:-42}

# 공통 PH1/PH2 파라미터
PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=0.5
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
PH1_POOL_SIZE=128
PH1_MULTI_PENALTY_ENABLED=True
PH1_MULTI_PENALTY_SINGLE_WEIGHT=1.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
PH1_EPSILON=0.2
PH2_EPSILON=0.2
PH1_WARMUP_STEPS=2000000
PH2_FIXED_IND_PROB=0.5
ACTION_PREDICTION=True
SAVE_EVAL_CHECKPOINTS=True

# Linear penalty (α, k sweep)
PH1_PENALTY_LINEAR_MODE=true
PH1_LINEAR_EPSILON=0.001
KS=(${KS:-1 2 3 4})
# k별 α 리스트. 기본은 k=1은 0.1 제외(이미 완료), k=2,3,4는 0.1 포함.
# override 예시:  ALPHAS_K1="0.01 0.5" ALPHAS_DEFAULT="0.01 0.05 0.1 0.5 1.0"
ALPHAS_K1=(${ALPHAS_K1:-0.01 0.05 0.5 1.0})
ALPHAS_DEFAULT=(${ALPHAS_DEFAULT:-0.01 0.05 0.1 0.5 1.0})

# GPU 설정 (default: 0,5,6,7)
GPUS="${GPUS:-0,5,6,7}"

# =============================================================================
# 레이아웃별 best omega/sigma/normal_prob (linear mode 에선 omega/sigma unused)
#   cramped_room:     omega=10 sigma=2  np=0.5
#   asymm_advantages: omega=5  sigma=3  np=0.5
#   coord_ring:       omega=10 sigma=2  np=0.0
#   forced_coord:     omega=4  sigma=8  np=0.0
# k 는 전부 1 로 고정 (v5 규칙)
# =============================================================================
declare -A BEST_OMEGA=( [cramped_room]=10.0 [asymm_advantages]=5.0  [coord_ring]=10.0 [forced_coord]=4.0 )
declare -A BEST_SIGMA=( [cramped_room]=2.0  [asymm_advantages]=3.0  [coord_ring]=2.0  [forced_coord]=8.0 )
declare -A BEST_NP=(    [cramped_room]=0.5  [asymm_advantages]=0.5  [coord_ring]=0.0  [forced_coord]=0.0 )

# Layouts (환경변수로 override 가능) — 기본은 coord_ring, forced_coord
LAYOUTS=(${LAYOUTS:-forced_coord})

# =============================================================================
# run_cell <env> <alpha> <k>
# =============================================================================
run_cell() {
  local env=$1
  local alpha=$2
  local k=$3

  local omega=${BEST_OMEGA[$env]}
  local sigma=${BEST_SIGMA[$env]}
  local np=${BEST_NP[$env]}

  local alpha_tag=$(echo "$alpha" | tr '.' 'p')
  local tags="ph2,ablation_v5,linear,alpha${alpha_tag},k${k},${env}"

  echo "================================================================"
  echo "  [v5 linear]  env=${env}  α=${alpha} ε=${PH1_LINEAR_EPSILON}  k=${k} np=${np}"
  echo "  max_penalty_cap = α/ε = $(awk "BEGIN{print $alpha/$PH1_LINEAR_EPSILON}")"
  echo "  seeds=${NUM_SEEDS}  gpus=${GPUS}"
  echo "================================================================"

  ./run_user_wandb.sh \
    --gpus "$GPUS" \
    --seeds "$NUM_SEEDS" \
    --seed "$FIXED_SEED" \
    --env "$env" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "${NENVS_OVERRIDE:-$NENVS}" \
    --nsteps "$NSTEPS" \
    --tags "$tags" \
    --ph1-beta $PH1_BETA \
    --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
    --ph1-beta-start $PH1_BETA_START \
    --ph1-beta-end $PH1_BETA_END \
    --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
    --ph1-omega "$omega" \
    --ph1-sigma "$sigma" \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob "$np" \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count "$k" \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --ph2-epsilon "$PH2_EPSILON" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
    --ph1-penalty-linear-mode "$PH1_PENALTY_LINEAR_MODE" \
    --ph1-linear-alpha "$alpha" \
    --ph1-linear-epsilon "$PH1_LINEAR_EPSILON"
}

# =============================================================================
# Sweep 실행
# =============================================================================
# k별 α 리스트 선택 (k=1 은 ALPHAS_K1, 그 외는 ALPHAS_DEFAULT)
alphas_for_k() {
  local k=$1
  if [ "$k" = "1" ]; then
    echo "${ALPHAS_K1[@]}"
  else
    echo "${ALPHAS_DEFAULT[@]}"
  fi
}

# total cells 계산
TOTAL_CELLS=0
for k in "${KS[@]}"; do
  alphas_k=( $(alphas_for_k "$k") )
  TOTAL_CELLS=$(( TOTAL_CELLS + ${#LAYOUTS[@]} * ${#alphas_k[@]} ))
done

echo "============================================================"
echo "  V5 Linear — α × k sweep"
echo "  LAYOUTS:        ${LAYOUTS[*]}"
echo "  KS:             ${KS[*]}"
echo "  ALPHAS_K1:      ${ALPHAS_K1[*]}   (k=1 전용)"
echo "  ALPHAS_DEFAULT: ${ALPHAS_DEFAULT[*]}  (k>=2)"
echo "  seeds=${NUM_SEEDS}  GPUs=${GPUS}"
echo "  total cells: ${TOTAL_CELLS}"
echo "============================================================"

CELL_IDX=0
for env in "${LAYOUTS[@]}"; do
  for k in "${KS[@]}"; do
    alphas_k=( $(alphas_for_k "$k") )
    for alpha in "${alphas_k[@]}"; do
      CELL_IDX=$((CELL_IDX + 1))
      echo ""
      echo ">>> cell ${CELL_IDX}/${TOTAL_CELLS}: env=${env} k=${k} α=${alpha}"
      run_cell "$env" "$alpha" "$k"
    done
  done
done

echo ""
echo "============================================================"
echo "  V5 완료  $(date)"
echo "============================================================"
