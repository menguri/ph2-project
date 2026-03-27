#!/bin/bash
# Webapp trajectory 데이터 → BC 학습용 numpy 변환
cd "$(dirname "$0")/.." || exit 1

python code/extract.py \
    --traj-dir ../webapp/data/trajectories \
    --out-dir data/bc \
    "$@"
