#!/bin/bash
# =============================================================================
# PH2 forced_coordination 안정화 sweep
# -----------------------------------------------------------------------------
# 목적: forced_coord에서 시드 1~2개가 학습 붕괴하는 문제를 HP-only로 완화.
# 스윕 대상 (다른 knob은 모두 default 유지):
#   - ENT_COEF              (현재 0.01)
#   - LR_WARMUP             (현재 0.05)
#   - PH1_WARMUP_STEPS      (현재 2e6)
#   - PH1_EPSILON           (현재 0.2)
#   - PH2_EPSILON           (현재 0.2)
# 고정: omega=4 sigma=8 k=1 normal_prob=0.0  (기존 forced_coord 전용 PH1 셋업)
# =============================================================================

cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NSTEPS=256
ENV="forced_coord"
GPUS="${GPUS:-2,3,4,5,6}"
NUM_SEEDS="${NUM_SEEDS:-5}"
# 시드 3개 × 시드당 5개 = 총 15 runs
SEED_LIST="${SEED_LIST:-41 42 43}"

# forced_coord 고정 PH1 (run_factory_ph2.sh와 동일)
PH1_OMEGA=4.0
PH1_SIGMA=8.0
PH1_MAX_PENALTY_COUNT=1
PH1_NORMAL_PROB=0.0

# PH2 공통 (run_factory_ph2.sh defaults 그대로)
PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=1.0
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
PH1_POOL_SIZE=128
PH1_MULTI_PENALTY_ENABLED=True
PH1_MULTI_PENALTY_SINGLE_WEIGHT=1.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
ACTION_PREDICTION=True
SAVE_EVAL_CHECKPOINTS=True
PH2_FIXED_IND_PROB=0.5

# -----------------------------------------------------------------------------
# Sweep 정의
#   각 row: name | ENT_COEF | LR_WARMUP | PH1_WARMUP_STEPS | PH1_EPSILON | PH2_EPSILON
# -----------------------------------------------------------------------------
SWEEP=(
  # fc_ent030 — 검증된 설정. SEED_LIST의 각 시드마다 NUM_SEEDS개씩 run.
  "fc_ent030        | 0.03 | 0.05 | 2000000 | 0.2 | 0.2"
)

run_one() {
  local name="$1"
  local ent_coef="$2"
  local lr_warmup="$3"
  local ph1_warmup="$4"
  local ph1_eps="$5"
  local ph2_eps="$6"
  local base_seed="$7"

  local tags="ph2,sweep,forced_coord,${name},seed${base_seed}"

  echo "============================================================"
  echo "  [SWEEP] $name  (base_seed=$base_seed)"
  echo "    ENT_COEF=$ent_coef  LR_WARMUP=$lr_warmup"
  echo "    PH1_WARMUP_STEPS=$ph1_warmup  PH1_EPSILON=$ph1_eps  PH2_EPSILON=$ph2_eps"
  echo "    SEED=$base_seed  NUM_SEEDS=$NUM_SEEDS"
  echo "============================================================"

  ./run_user_wandb.sh \
    --gpus "$GPUS" \
    --seeds "$NUM_SEEDS" \
    --seed "$base_seed" \
    --env "$ENV" \
    --exp "$EXP" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$NENVS" \
    --nsteps "$NSTEPS" \
    --tags "$tags" \
    --ph1-beta "$PH1_BETA" \
    --ph1-beta-schedule-enabled "$PH1_BETA_SCHEDULE_ENABLED" \
    --ph1-beta-start "$PH1_BETA_START" \
    --ph1-beta-end "$PH1_BETA_END" \
    --ph1-beta-schedule-horizon-env-steps "$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS" \
    --ph1-omega "$PH1_OMEGA" \
    --ph1-sigma "$PH1_SIGMA" \
    --ph1-pool-size "$PH1_POOL_SIZE" \
    --ph1-normal-prob "$PH1_NORMAL_PROB" \
    --ph1-multi-penalty-enabled "$PH1_MULTI_PENALTY_ENABLED" \
    --ph1-max-penalty-count "$PH1_MAX_PENALTY_COUNT" \
    --ph1-multi-penalty-single-weight "$PH1_MULTI_PENALTY_SINGLE_WEIGHT" \
    --ph1-multi-penalty-other-weight "$PH1_MULTI_PENALTY_OTHER_WEIGHT" \
    --ph1-epsilon "$ph1_eps" \
    --ph1-warmup-steps "$ph1_warmup" \
    --ph2-epsilon "$ph2_eps" \
    --ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB" \
    --action-prediction "$ACTION_PREDICTION" \
    --save-eval-checkpoints "$SAVE_EVAL_CHECKPOINTS" \
    --extra "++model.ENT_COEF=$ent_coef" \
    --extra "++model.LR_WARMUP=$lr_warmup"
}

# -----------------------------------------------------------------------------
# Sweep 실행
#   특정 항목만 돌리려면: ONLY="fc_ent020,fc_warmup_long" bash ph2_sweep.sh
# -----------------------------------------------------------------------------
ONLY="${ONLY:-}"

for row in "${SWEEP[@]}"; do
  IFS='|' read -r name ent lrw phw phe pe <<< "$row"
  name=$(echo "$name" | xargs)
  ent=$(echo "$ent" | xargs)
  lrw=$(echo "$lrw" | xargs)
  phw=$(echo "$phw" | xargs)
  phe=$(echo "$phe" | xargs)
  pe=$(echo "$pe" | xargs)

  if [[ -n "$ONLY" ]]; then
    if [[ ",$ONLY," != *",$name,"* ]]; then
      echo "[skip] $name (not in ONLY=$ONLY)"
      continue
    fi
  fi

  for base_seed in $SEED_LIST; do
    run_one "$name" "$ent" "$lrw" "$phw" "$phe" "$pe" "$base_seed"
  done
done

echo "================================================================"
echo "  PH2 forced_coord sweep 완료"
echo "================================================================"
