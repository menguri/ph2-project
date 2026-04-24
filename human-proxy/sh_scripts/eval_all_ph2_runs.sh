#!/bin/bash
# 모든 PH2 run 에 대해 human-proxy BC × RL 평가.
#
# Usage: bash eval_all_ph2_runs.sh <GPU> <BC_DIR> <OUT_ROOT> <NUM_EVAL_SEEDS>
# Default: GPU=6 BC_DIR=models_all_norecompute OUT_ROOT=results_all_ph2_runs SEEDS=5
#
# 중단 후 재시작 가능 (scores.csv 이미 있으면 skip).

set +e  # 개별 run 실패해도 전체는 계속 진행
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-6}
BC_DIR=${2:-models_all_norecompute}
OUT_ROOT=${3:-results_all_ph2_runs}
SEEDS=${4:-5}
PY=/home/mlic/mingukang/ph2-project/overcooked_v2/bin/python
PH2_RUNS=/home/mlic/mingukang/ph2-project/ph2/runs

LAYOUTS=(cramped_room asymm_advantages coord_ring counter_circuit forced_coord)
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"

echo "==== ALL PH2 runs × BC eval  GPU=$GPU  BC=$BC_DIR  $(date) ====" | tee -a "$MASTER"

total=0
for L in "${LAYOUTS[@]}"; do
    mkdir -p "$OUT_ROOT/$L"
    # layout 이 이름에 들어가고 ph2 이며 run_X 가 있는 디렉토리만
    for run_dir in "$PH2_RUNS"/*_${L}_e3t_ph2_*; do
        [[ ! -d "$run_dir" ]] && continue
        # run_* 서브디렉 최소 1개 존재 확인
        has_run=$(find "$run_dir" -maxdepth 1 -type d -name "run_*" | head -1)
        [[ -z "$has_run" ]] && continue
        # counter_circuit 이 cramped_room 에 매칭되지 않도록
        [[ "$L" == "cramped_room" && "$run_dir" == *counter_circuit* ]] && continue

        run_name=$(basename "$run_dir")
        out_dir="$OUT_ROOT/$L/$run_name"
        total=$((total+1))

        if [[ -f "$out_dir/scores.csv" ]]; then
            echo "[$total] [SKIP done] $L / $run_name" | tee -a "$MASTER"
            continue
        fi

        echo "[$total] $L / $run_name  start $(date +%H:%M:%S)" | tee -a "$MASTER"
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate.py \
            --algo-dir "$run_dir" \
            --layout "$L" \
            --bc-model-dir "$BC_DIR" \
            --num-eval-seeds "$SEEDS" \
            --output-dir "$out_dir" \
            --source ph2 > "$LOG_DIR/${L}__${run_name}.log" 2>&1
        rc=$?
        echo "[$total] $L / $run_name  done rc=$rc $(date +%H:%M:%S)" | tee -a "$MASTER"
    done
done

echo "==== 완료  총 $total runs  $(date) ====" | tee -a "$MASTER"
