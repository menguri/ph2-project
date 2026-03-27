#!/bin/bash
# BC × RL 알고리즘 cross-play 평가
#
# 사용법:
#   bash sh_scripts/eval_with_bc.sh ../baseline/runs/20260318-..._cramped_room_sp cramped_room
#   bash sh_scripts/eval_with_bc.sh ALGO_DIR LAYOUT [NUM_EVAL_SEEDS]
cd "$(dirname "$0")/.." || exit 1

ALGO_DIR=${1:?"사용법: eval_with_bc.sh ALGO_DIR LAYOUT [NUM_EVAL_SEEDS]"}
LAYOUT=${2:?"사용법: eval_with_bc.sh ALGO_DIR LAYOUT [NUM_EVAL_SEEDS]"}
NUM_EVAL_SEEDS=${3:-10}

echo "========================================"
echo "Cross-Play 평가"
echo "  Layout: $LAYOUT"
echo "  Algo:   $ALGO_DIR"
echo "  Seeds:  $NUM_EVAL_SEEDS"
echo "========================================"

python code/evaluate.py \
    --algo-dir "$ALGO_DIR" \
    --layout "$LAYOUT" \
    --num-eval-seeds "$NUM_EVAL_SEEDS"
