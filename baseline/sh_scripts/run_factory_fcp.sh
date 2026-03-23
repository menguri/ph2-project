#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# FCP Experiment Factory Script
# Runs FCP experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-fcp"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=128

# FCP Specific Settings
FCP_DEVICE="gpu"
SEEDS=10

# Population annealing settings for FCP mixed training
: "${POP_ANNEAL_ENABLE:=0}"
: "${POP_ANNEAL_HORIZON:=1000000}"
: "${POP_ANNEAL_BEGIN:=0}"

# Function to get FCP path based on env
get_fcp_path() {
    local env=$1
    case $env in
        "cramped_room")
            echo "fcp_populations/cramped_room_sp"
            ;;
        "asymm_advantages")
            echo "fcp_populations/asymm_advantages_sp"
            ;;
        "coord_ring")
            echo "fcp_populations/coord_ring_sp"
            ;;
        "forced_coord")
            echo "fcp_populations/forced_coord_sp"
            ;;
        "counter_circuit")
            echo "fcp_populations/counter_circuit_sp"
            ;;
        "grounded_coord_simple")
            echo "fcp_populations/grounded_coord_simple_sp"
            ;;
        "grounded_coord_ring")
            echo "fcp_populations/grounded_coord_ring_sp"
            ;;
        "demo_cook_simple")
            echo "fcp_populations/demo_cook_simple_sp"
            ;;
        "demo_cook_wide")
            echo "fcp_populations/demo_cook_wide_sp"
            ;;
        "test_time_simple")
            echo "fcp_populations/test_time_simple_sp"
            ;;
        "test_time_wide")
            echo "fcp_populations/test_time_wide_sp"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run experiment
run_fcp() {
    local gpus=$1
    local env=$2
    local layout=$3
    
    echo "================================================================================"
    echo "STARTING FCP EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    if [[ "$POP_ANNEAL_ENABLE" == "1" ]]; then
        echo "Population Anneal: ON (horizon=$POP_ANNEAL_HORIZON, begin=$POP_ANNEAL_BEGIN)"
    else
        echo "Population Anneal: OFF"
    fi
    echo "================================================================================"
    
    local fcp_path=$(get_fcp_path $env)
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --seeds $SEEDS \
        --fcp-device $FCP_DEVICE"

    if [ -n "$fcp_path" ]; then
        cmd="$cmd --fcp $fcp_path"
    fi
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi

    if [[ "$POP_ANNEAL_ENABLE" == "1" ]]; then
        cmd="$cmd --pop-anneal-horizon $POP_ANNEAL_HORIZON --pop-anneal-begin $POP_ANNEAL_BEGIN"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED FCP EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# ==============================================================================

# --- OV1 Layouts ---
# run_fcp "1,2,3,6,7" "cramped_room" ""
# run_fcp "1,2,3,6,7" "asymm_advantages" ""
# run_fcp "1,2,3,6,7" "coord_ring" ""
# run_fcp "1,2,3,6,7" "forced_coord" ""
# run_fcp "1,2,3,6,7" "counter_circuit" ""

# # --- OV2 Layouts ---
# run_fcp "1,2,3,6,7" "grounded_coord_simple" ""
# run_fcp "1,2,3,6,7" "grounded_coord_ring" ""
# run_fcp "1,2,3,6,7" "demo_cook_simple" ""
# run_fcp "1,2,3,6,7" "demo_cook_wide" ""
# run_fcp "1,2,3,6,7" "test_time_simple" ""
# run_fcp "1,2,3,6,7" "test_time_wide" ""

# --- ToyCoop (Dual Destination) ---
EXP="rnn-fcp-toycoop" NENVS=512 NSTEPS=100 run_fcp "0,1" "toy_coop" ""
