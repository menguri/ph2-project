#!/bin/bash
# CEC 재실행 — 5 layouts
set +e
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-6}
SEEDS=${2:-3}  # 기존 forced_coord 3시간+ 걸려 seed 줄임
PY=/home/mlic/mingukang/ph2-project/overcooked_v2/bin/python

LOGDIR=results_final/logs
mkdir -p "$LOGDIR"

eval_cec() {
    local layout=$1
    local tag="cec_${layout}"
    local logf="$LOGDIR/${tag}.log"
    echo "[cec × $layout] start $(date)" | tee -a "$LOGDIR/cec_master.log"
    if [[ "$layout" == "forced_coord" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate_cec_final.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" \
            --output-dir "results_final/${tag}" \
            > "$logf" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate_cec.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" \
            --output-dir "results_final/${tag}" \
            > "$logf" 2>&1
        if [[ -f "results_final/${tag}/scores_cec.csv" ]]; then
            cp "results_final/${tag}/scores_cec.csv" "results_final/${tag}/scores.csv"
        fi
    fi
    local rc=$?
    echo "[cec × $layout] done rc=$rc $(date)" | tee -a "$LOGDIR/cec_master.log"
}

echo "======================================================" | tee -a "$LOGDIR/cec_master.log"
echo "CEC re-run (GPU=$GPU, seeds=$SEEDS) $(date)" | tee -a "$LOGDIR/cec_master.log"
echo "======================================================" | tee -a "$LOGDIR/cec_master.log"

# 기존 덮어쓰기 찌꺼기 제거
rm -rf results_final/cec_

for L in cramped_room asymm_advantages coord_ring counter_circuit forced_coord; do
    eval_cec "$L"
done

echo "======================================================" | tee -a "$LOGDIR/cec_master.log"
echo "CEC re-run 완료 $(date)" | tee -a "$LOGDIR/cec_master.log"
echo "======================================================" | tee -a "$LOGDIR/cec_master.log"
