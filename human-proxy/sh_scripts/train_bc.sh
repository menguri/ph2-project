#!/bin/bash
# BC 모델 훈련 — 레이아웃별, 포지션(0/1)별, 5 seeds
#
# 사용법:
#   bash sh_scripts/train_bc.sh cramped_room          # 단일 레이아웃
#   bash sh_scripts/train_bc.sh                        # 기본: cramped_room
cd "$(dirname "$0")/.." || exit 1

LAYOUT=${1:-"cramped_room"}

for pos in 0 1; do
    echo "========================================"
    echo "Training BC: layout=$LAYOUT, position=$pos"
    echo "========================================"
    python code/train.py \
        --layout "$LAYOUT" \
        --position "$pos" \
        --num-seeds 5
    echo ""
done

echo "완료: models/$LAYOUT/pos_0/ 및 pos_1/ 확인"
