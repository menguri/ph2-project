#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# E3T Experiment Factory Script
# 모든 E3T 파라미터는 config/experiment/rnn-e3t.yaml + config/model/rnn-e3t.yaml에
# 원본 E3T-Overcooked 기준으로 설정되어 있으므로, 별도 CLI override 불필요.
# ==============================================================================

EXP="rnn-e3t"
ENV_DEVICE="cpu"

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

# --- Original Overcooked Layouts ---
# run_e3t "0,1" "cramped_room" ""
# run_e3t "0,1" "coord_ring" ""
# run_e3t "0,1" "forced_coord" ""
# run_e3t "0,1" "counter_circuit" ""
# run_e3t "0,1" "asymm_advantages" ""

# --- Overcooked V2 Layouts ---
# run_e3t "0,1" "grounded_coord_simple" ""
# run_e3t "0,1" "grounded_coord_ring" ""
# run_e3t "0,1" "demo_cook_simple" ""
# run_e3t "0,1" "demo_cook_wide" ""
# run_e3t "0,1" "test_time_simple" ""
# run_e3t "0,1" "test_time_wide" ""

# --- ToyCoop (별도 experiment config 사용) ---
# EXP="rnn-e3t-toycoop" run_e3t "0,1" "toy_coop" ""
