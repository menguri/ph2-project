#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# E3T Experiment Factory Script
# Runs E3T experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-e3t"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=128

# E3T Specific Settings
# 본래 0.3 / 0.8 <-> counter circuit 0.2 / 1.0
EPSILON=0.2
USE_PM=True
PRED_COEF=1.0

# Function to run experiment
run_e3t() {
    local gpus=$1
    local env=$2
    local layout=$3
    
    echo "================================================================================"
    echo "STARTING E3T EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3t-epsilon $EPSILON \
        --use-partner-modeling $USE_PM \
        --pred-loss-coef $PRED_COEF \
        --tags e3t"

    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED E3T EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# Usage: run_e3t <GPUS> <ENV_GROUP> <LAYOUT>
# ==============================================================================

# # # 5. Cramped Room (Original)
# run_e3t "0,1" "cramped_room" ""

# # # 7. Coordination Ring (Original)
# run_e3t "0,1" "coord_ring" ""

# # # 8. Forced Coordination (Original)
# run_e3t "0,1" "forced_coord" ""

# # 9. Counter Circuit (Original)
# run_e3t "0,1" "counter_circuit" ""

# # 6. Asymmetric Advantages (Original)
# run_e3t "0,1" "asymm_advantages" ""

# # 1. Grounded Coord Simple
# run_e3t "0,1" "grounded_coord_simple" ""

# # # 2. Grounded Coord Ring
# run_e3t "0,1" "grounded_coord_ring" ""

# # # # 3. Demo Cook Simple
# run_e3t "0,1" "demo_cook_simple" ""

# # # 4. Demo Cook Wide
# run_e3t "0,1" "demo_cook_wide" ""

# # # 5. Test Time Simple
# run_e3t "0,1" "test_time_simple" ""

# # 6. Test Time Wide
# run_e3t "0,1" "test_time_wide" ""

--- ToyCoop (Dual Destination) ---
EXP="rnn-e3t-toycoop" NENVS=512 NSTEPS=100 run_e3t "0,1" "toy_coop" ""