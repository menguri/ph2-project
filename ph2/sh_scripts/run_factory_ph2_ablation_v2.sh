#!/bin/bash
# =============================================================================
# PH2 Ablation V2 — penalty_state / sigma / omega sweep
#
# 4 레이아웃: counter_circuit, coord_ring, cramped_room, forced_coord
#
# 각 레이아웃별 best params 기반 (final/ph2 등록된 모델의 run_metadata.json 기준),
# sweep 대상 파라미터만 변경:
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=2 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=4 k=1 normal_prob=0.0
#   forced_coord:     omega=4  sigma=8 k=1 normal_prob=0.0
#   공통: ent=0.01 (default 사용)
#
# Sweep 1: penalty_state = {0, 1, 2, 3, 4}   (best omega/sigma/np 유지, k 변경)
#   - penalty_state=0 → PH1_ENABLED=False (ph1 비활성화)
#   - penalty_state=1~4 → PH1_MAX_PENALTY_COUNT=k
#
# Sweep 2: sigma = {0, 2, 4, 6, 8, 10}       (best omega/k/np 유지, sigma 변경)
#
# Sweep 3: omega = {0, 2, 4, 6, 8, 10}        (best sigma/k/np 유지, omega 변경)
#
# 총 셀: (5 + 6 + 6) × 4 레이아웃 = 68 셀, 각 10 seeds
# =============================================================================
cd "$(dirname "$0")" || exit 1

EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=10
NSTEPS=256
FIXED_SEED=42

# 공통 PH1/PH2 파라미터
PH1_BETA=1.0
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=1.0
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
PH1_POOL_SIZE=128
PH1_NORMAL_PROB=0.0
PH1_MULTI_PENALTY_ENABLED=True
PH1_MULTI_PENALTY_SINGLE_WEIGHT=1.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
PH1_EPSILON=0.2
PH2_EPSILON=0.2
PH1_WARMUP_STEPS=2000000
PH2_FIXED_IND_PROB=0.5
ACTION_PREDICTION=True
SAVE_EVAL_CHECKPOINTS=True

# GPU 설정
GPUS="${GPUS:-1,3,4,5,6}"

# =============================================================================
# 레이아웃별 Best 파라미터 (final/ph2 등록 모델 기준)
#   cramped_room:     omega=10 sigma=2 k=1 normal_prob=0.5
#   coord_ring:       omega=10 sigma=2 k=2 normal_prob=0.0
#   counter_circuit:  omega=10 sigma=4 k=1 normal_prob=0.0
#   forced_coord:     omega=4  sigma=8 k=1 normal_prob=0.0
#   공통: ent=0.01 (default 사용)
# =============================================================================
declare -A BEST_OMEGA=( [cramped_room]=10.0 [coord_ring]=10.0 [counter_circuit]=10.0 [forced_coord]=4.0 )
declare -A BEST_SIGMA=( [cramped_room]=2.0  [coord_ring]=2.0  [counter_circuit]=4.0  [forced_coord]=8.0 )
declare -A BEST_K=(     [cramped_room]=1    [coord_ring]=2    [counter_circuit]=1    [forced_coord]=1 )
declare -A BEST_NP=(    [cramped_room]=0.5  [coord_ring]=0.0  [counter_circuit]=0.0  [forced_coord]=0.0 )

LAYOUTS=("counter_circuit" "coord_ring" "cramped_room" "forced_coord")

# =============================================================================
# run_cell  <env> <omega> <sigma> <k> <normal_prob> <tag> [--ph1-disabled]
# =============================================================================
run_cell() {
  local env=$1
  local omega=$2
  local sigma=$3
  local k=$4
  local normal_prob=$5
  local tag=$6
  local ph1_disabled=$7   # "true" 이면 --ph1-enabled False

  local tags="ph2,ablation_v2,${tag},${env}"

  echo "================================================================"
  echo "  [${tag}]  env=${env}  ω=${omega} σ=${sigma} k=${k} np=${normal_prob}  ph1_disabled=${ph1_disabled:-false}"
  echo "================================================================"

  local extra_args=()
  if [[ "$ph1_disabled" == "true" ]]; then
    extra_args+=(--ph1-enabled False)
  fi

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
    --ph1-normal-prob "$normal_prob" \
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
    "${extra_args[@]}"
}

# =============================================================================
# Sweep 1: penalty_state — 보강 실행 (미완료 셀만)
#   - counter_circuit: ps3, ps4 만 (ps0~ps2 완료, ps3/ps4 크래시)
#   - coord_ring:      ps1~ps4 (ps0만 학습됨)
#   - cramped_room, forced_coord: 전체 완료 → 스킵
# =============================================================================
echo "#################### Sweep 1: penalty_state (보강) ####################"

# --- counter_circuit: ps3, ps4 재실행 ---
#   k≥3에서 OOM으로 silent kill 발생 → NENVS를 64 → 32로 감량.
#   PH1 multi-penalty 버퍼가 k에 선형 증가하므로 NENVS를 절반으로 보상.
#   (best k=1, sigma=4, omega=10 / np=0.0)
cc_omega=${BEST_OMEGA[counter_circuit]}
cc_sigma=${BEST_SIGMA[counter_circuit]}
cc_np=${BEST_NP[counter_circuit]}
for k in 3 4; do
  NENVS_OVERRIDE=32 \
    run_cell "counter_circuit" "$cc_omega" "$cc_sigma" "$k" "$cc_np" "ps${k}"
done

# --- coord_ring: ps1~ps4 (best k=2) ---
cr_omega=${BEST_OMEGA[coord_ring]}
cr_sigma=${BEST_SIGMA[coord_ring]}
cr_np=${BEST_NP[coord_ring]}
for k in 1 2 3 4; do
  run_cell "coord_ring" "$cr_omega" "$cr_sigma" "$k" "$cr_np" "ps${k}"
done

# =============================================================================
# Sweep 2: sigma = {0, 2, 4, 6, 8, 10}
#   각 레이아웃 best omega/k/normal_prob 유지, sigma만 변경
# =============================================================================
echo "#################### Sweep 2: sigma ####################"
for env in "${LAYOUTS[@]}"; do
  local_omega=${BEST_OMEGA[$env]}
  local_k=${BEST_K[$env]}
  local_np=${BEST_NP[$env]}

  for sigma in 0 2 4 6 8 10; do
    run_cell "$env" "$local_omega" "$sigma" "$local_k" "$local_np" "sig${sigma}"
  done
done

# # =============================================================================
# # Sweep 3: omega = {0, 2, 4, 6, 8, 10}
# #   각 레이아웃 best sigma/k/normal_prob 유지, omega만 변경
# # =============================================================================
# echo "#################### Sweep 3: omega ####################"
# for env in "${LAYOUTS[@]}"; do
#   local_sigma=${BEST_SIGMA[$env]}
#   local_k=${BEST_K[$env]}
#   local_np=${BEST_NP[$env]}

#   for omega in 0 2 4 6 8 10; do
#     run_cell "$env" "$omega" "$local_sigma" "$local_k" "$local_np" "omg${omega}"
#   done
# done

echo "================================================================"
echo "  Ablation V2 전체 완료 (68 셀)"
echo "================================================================"
