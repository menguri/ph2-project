#!/bin/bash

cd "$(dirname "$0")" || exit 1

# ===============================
# SP Experiment Factory Script
# ===============================
EXP="rnn-sp"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=128

run_sp() {
    local gpus=$1
    local env=$2
    local layout=$3
    local extra=$4
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS $extra"

    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    echo "Executing: $cmd"
    $cmd
}

# 실행 목록
# run_sp "0,1" "counter_circuit" "" 
# run_sp "0,1" "asymm_advantages" "" 
# run_sp "0,1" "coord_ring" "" 
# run_sp "0,1" "forced_coord" "" 
# run_sp "0,1" "cramped_room" "" 
# run_sp "0,1" "grounded_coord_simple" ""  
# run_sp "0,1" "grounded_coord_ring" "" 
# run_sp "0,1" "demo_cook_simple" "" 
# run_sp "0,1" "demo_cook_wide" "" 
# run_sp "0,1" "test_time_simple" "" 
# run_sp "0,1" "test_time_wide" "" 
# run_sp "5,6" "cramped_room" "" 
# run_sp "1,2,4,5,6" "asymm_advantages" "" 
# run_sp "5,6" "coord_ring" "" 
# run_sp "5,6" "forced_coord" "" 
# run_sp "1,2,4,5,6" "counter_circuit" "" 
# --- ToyCoop (Dual Destination) ---
# ToyCoop은 별도 experiment config 사용 (MLP encoder, CEC 하이퍼파라미터)
run_sp "0,1" "toy_coop" "" "--exp rnn-sp-toycoop"

# run_sp "5,6" "cramped_room" ""
# run_sp "5,6" "asymm_advantages" ""
# run_sp "5,6" "coord_ring" ""
# run_sp "5,6" "forced_coord" ""
# run_sp "5,6" "counter_circuit" ""
