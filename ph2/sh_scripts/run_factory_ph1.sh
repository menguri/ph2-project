#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/overcooked_v2_experiments" ]]; then
  REPO_ROOT="/home/mlic/mingukang/ex-overcookedv2/experiments-stablock"
fi

# ==============================================================================
# STA-PH1 Experiment Factory Script
# ==============================================================================

# Common Configuration
EXP="rnn-ph1"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=128
NUM_SEEDS=5
while [[ $# -gt 0 ]]; do
    case "$1" in
        *)
            echo "[WARN] Unknown option ignored: $1"
            shift
            ;;
    esac
done

# PH1 Default Settings
BETA=0.5 # Default value, can be overridden
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=2.0
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
OMEGA=1.0 # Default value, can be overridden
SIGMA=1.0 # Default value, can be overridden
DIST_THRESH=0.1
POOL_SIZE=128
NORMAL_PROB=0.5
PH1_MULTI_PENALTY_ENABLED=False
PH1_MAX_PENALTY_COUNT=1
PH1_MULTI_PENALTY_SINGLE_WEIGHT=2.0
PH1_MULTI_PENALTY_OTHER_WEIGHT=1.0
PH1_EPSILON=0.2 # With probability epsilon, one agent takes random action
WARMUP_STEPS=1000000 # 5m steps for warm-up (normal interaction)
# E3T Settings
EPSILON=0.0 # 사용자 요청으로 0.0

run_ph1() {
    local gpus=$1
    local env=$2
    local beta=$3
    local omega=$4
    local normal_prob=$5
    local ph1_epsilon=$6
    local warmup_steps=$7
    local sigma=$8
    local beta_end=$9

    # Defaults check
    if [ -z "$beta" ]; then beta=$BETA; fi
    if [ -z "$omega" ]; then omega=$OMEGA; fi
    if [ -z "$normal_prob" ]; then normal_prob=$NORMAL_PROB; fi
    if [ -z "$ph1_epsilon" ]; then ph1_epsilon=$PH1_EPSILON; fi
    if [ -z "$warmup_steps" ]; then warmup_steps=$WARMUP_STEPS; fi
    if [ -z "$sigma" ]; then sigma=$SIGMA; fi
    if [ -z "$beta_end" ]; then beta_end=$PH1_BETA_END; fi

    echo "================================================================================"
    echo "STARTING STA-PH1 EXPERIMENT"
    echo "ENV: $env"
    echo "GPUS: $gpus"
    echo "BETA: $beta, OMEGA: $omega, NORMAL_PROB: $normal_prob, EPSILON: $ph1_epsilon"
    echo "BETA_SCHEDULE: enabled=$PH1_BETA_SCHEDULE_ENABLED start=$PH1_BETA_START end=$beta_end horizon=$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS"
    echo "SIGMA: $sigma"
    echo "WARMUP: $warmup_steps"
    echo "================================================================================"

    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --seeds $NUM_SEEDS \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3t-epsilon $EPSILON \
        --tags ph1,e3t \
        --ph1-beta $beta \
        --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
        --ph1-beta-start $PH1_BETA_START \
        --ph1-beta-end $beta_end \
        --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
        --ph1-omega $omega \
        --ph1-sigma $sigma \
        --ph1-dist $DIST_THRESH \
        --ph1-pool-size $POOL_SIZE \
        --ph1-normal-prob $normal_prob \
        --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
        --ph1-max-penalty-count $PH1_MAX_PENALTY_COUNT \
        --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
        --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
        --ph1-epsilon $ph1_epsilon \
        --ph1-warmup-steps $warmup_steps"
    echo "Executing: $cmd"
    local run_log_file
    run_log_file=$(mktemp)
    $cmd | tee "$run_log_file"

    rm -f "$run_log_file"

    echo "================================================================================"
    echo "FINISHED STA-PH1 EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List
# Usage:
#   run_ph1 <GPUS> <ENV> <BETA> <OMEGA> <NORMAL_PROB> <PH1_EPSILON> <WARMUP_STEPS> <SIGMA> <BETA_END>
#
# Policy:
# - no for-loop
# - sequential code blocks
# - two GPU groups in parallel: 0,1,2 and 3,4,5
# - each run uses NUM_SEEDS=3
# ==============================================================================

# grounded_coord_simple: (omega,sigma) = (10,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "1.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "1.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "10.0" "0.0" "" "" "1.0" "1.0" &
# wait
run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "10.0" "0.0" "" "" "3.0" "1.0" &
wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "1.0" "1.0" &
# wait
run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "10.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "10.0" "0.5" "" "" "0.5" "2.0" &
# wait

# # grounded_coord_simple: (omega,sigma) = (20,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "0.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "1.0" &
# wait
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "2.0" &
# wait

# counter_circuit: (omega,sigma) = (90,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "0.5" &
# run_ph1 "3,4,5" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "1.0" &
# wait
# run_ph1 "0,1,2" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "2.0" &
# wait

echo "All jobs finished."
