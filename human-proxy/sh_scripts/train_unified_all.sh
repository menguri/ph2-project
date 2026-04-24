#!/bin/bash
# Pos-unified BC 5 layouts × 3 seeds 학습
set -e
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-6}
SEEDS=${2:-3}
EPOCHS=${3:-200}
PY=/home/mlic/mingukang/ph2-project/overcooked_v2/bin/python

LOGDIR=models_new/logs
mkdir -p "$LOGDIR"

for L in cramped_room asymm_advantages coord_ring counter_circuit forced_coord; do
    echo "── $L ──"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/train_unified.py \
        --layout "$L" --num-seeds "$SEEDS" --epochs "$EPOCHS" \
        > "$LOGDIR/$L.log" 2>&1
    rc=$?
    tail -4 "$LOGDIR/$L.log"
    echo "rc=$rc"
done
