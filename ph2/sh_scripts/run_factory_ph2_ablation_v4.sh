#!/bin/bash
# =============================================================================
# PH2 Ablation V4 — Linear Penalty α sweep × 5 Overcooked layouts
#
# Linear mode penalty:  penalty = α / (ε + latent_dist),   ε=0.001 고정
#
# Sweep (작은 α 부터 순차 실행):
#   α ∈ {0.05, 0.1, 0.5, 1.0}
#   layouts: cramped_room, asymm_advantages, coord_ring, counter_circuit, forced_coord
#   → 4 × 5 = 20 cells × NUM_SEEDS seeds
#
# 레이아웃별 비-linear 파라미터는 final/ph2 등록 best 기준 (k/np 유지; omega/sigma는
# linear 모드에선 unused이지만 호환성 위해 CLI 로 전달).
#
# 폴더 이름 규칙 (linear mode):
#   {ts}_{wid}_{layout}_e3t_ph2_e{eps}_a{alpha}_k{k}_ct0
#
# Usage:
#   bash run_factory_ph2_ablation_v4.sh                    # default GPUs + all α + all layouts
#   GPUS="0,1,2,3" bash run_factory_ph2_ablation_v4.sh
#   ALPHAS="0.1" LAYOUTS="counter_circuit" bash run_factory_ph2_ablation_v4.sh   # 부분 sweep
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=${NUM_SEEDS:-10}
NSTEPS=256
FIXED_SEED=${FIXED_SEED:-42}

# 공통 PH1/PH2 파라미터
PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=1.0
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

# Linear penalty (고정) — α 만 sweep
PH1_PENALTY_LINEAR_MODE=true
PH1_LINEAR_EPSILON=0.001

# GPU 설정 (override: GPUS="1,2,3,4" bash ...)
GPUS="${GPUS:-1,2,3,4,5}"

# =============================================================================
# 레이아웃별 Best non-linear 파라미터 (k/np; omega/sigma는 linear에선 unused)
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   asymm_advantages: omega=5  sigma=3 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=2 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=4 k=1 normal_prob=0.0
#   forced_coord:     omega=4  sigma=8 k=1 normal_prob=0.0
# =============================================================================
declare -A BEST_OMEGA=( [cramped_room]=10.0 [asymm_advantages]=5.0  [coord_ring]=10.0 [counter_circuit]=10.0 [forced_coord]=4.0 )
declare -A BEST_SIGMA=( [cramped_room]=2.0  [asymm_advantages]=3.0  [coord_ring]=2.0  [counter_circuit]=4.0  [forced_coord]=8.0 )
declare -A BEST_K=(     [cramped_room]=1    [asymm_advantages]=1    [coord_ring]=2    [counter_circuit]=1    [forced_coord]=1   )
declare -A BEST_NP=(    [cramped_room]=0.0  [asymm_advantages]=0.0  [coord_ring]=0.0  [counter_circuit]=0.0  [forced_coord]=0.0 )

# 스윕 축 (환경변수로 override 가능)
ALPHAS=(${ALPHAS:-0.05 0.1 0.5 1.0})
LAYOUTS=(${LAYOUTS:-counter_circuit cramped_room asymm_advantages coord_ring forced_coord})

# =============================================================================
# run_cell <env> <alpha>
# =============================================================================
run_cell() {
  local env=$1
  local alpha=$2

  local omega=${BEST_OMEGA[$env]}
  local sigma=${BEST_SIGMA[$env]}
  local k=${BEST_K[$env]}
  local np=${BEST_NP[$env]}

  local alpha_tag=$(echo "$alpha" | tr '.' 'p')
  local tags="ph2,ablation_v4,linear,alpha${alpha_tag},${env}"

  echo "================================================================"
  echo "  [v4 linear]  env=${env}  α=${alpha} ε=${PH1_LINEAR_EPSILON}  k=${k} np=${np}"
  echo "  max_penalty_cap = α/ε = $(awk "BEGIN{print $alpha/$PH1_LINEAR_EPSILON}")"
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
echo "============================================================"
echo "  V4 Linear α sweep — ${#LAYOUTS[@]} layouts × ${#ALPHAS[@]} alphas = $((${#LAYOUTS[@]} * ${#ALPHAS[@]})) cells"
echo "  GPUs=$GPUS  NUM_SEEDS=$NUM_SEEDS"
echo "  ALPHAS: ${ALPHAS[*]}"
echo "  LAYOUTS: ${LAYOUTS[*]}"
echo "============================================================"

for env in "${LAYOUTS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    run_cell "$env" "$alpha"
  done
done

echo ""
echo "============================================================"
echo "  V4 sweep 완료  $(date)"
echo "============================================================"
