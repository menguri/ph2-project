#!/bin/bash
# BC × RL cross-play 평가 — final/ 체크포인트 사용
# 5 layouts × 6 algos (sp, e3t, fcp, mep, gamma, ph2) = 30 cell
# 결과: results_final/{algo}_{layout}/scores.csv
# asymm_advantages 는 실패하면 스킵
set +e
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-6}
SEEDS=${2:-5}
PY=/home/mlic/mingukang/ph2-project/overcooked_v2/bin/python

F=../final

LOGDIR=results_final/logs
mkdir -p "$LOGDIR"

eval_one() {
    local algo=$1 layout=$2 run=$3 src=${4:-}
    if [[ ! -d "$run" ]]; then
        echo "  [SKIP] $algo × $layout: run 경로 없음: $run" | tee -a "$LOGDIR/master.log"
        return
    fi
    local tag="${algo}_${layout}"
    echo "  [$algo × $layout] $(basename $run)" | tee -a "$LOGDIR/master.log"
    CUDA_VISIBLE_DEVICES=$GPU $PY code/evaluate.py \
        --algo-dir "$run" --layout "$layout" \
        --bc-model-dir models --num-eval-seeds "$SEEDS" \
        --output-dir "results_final/${tag}" $src \
        > "$LOGDIR/${tag}.log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    [FAIL rc=$rc] $layout × $algo — log: $LOGDIR/${tag}.log" | tee -a "$LOGDIR/master.log"
    fi
}

echo "======================================================" | tee -a "$LOGDIR/master.log"
echo "Final eval (GPU=$GPU, seeds=$SEEDS) $(date)" | tee -a "$LOGDIR/master.log"
echo "======================================================" | tee -a "$LOGDIR/master.log"

# ── cramped_room ──
echo -e "\n──── cramped_room ────" | tee -a "$LOGDIR/master.log"
eval_one sp    cramped_room "$F/baseline/cramped_room/20260402-174158_d4adx8ic_cramped_room_sp_e30"
eval_one e3t   cramped_room "$F/baseline/cramped_room/20260407-052043_swhnnuls_cramped_room_e3t_h256"
eval_one fcp   cramped_room "$F/baseline/cramped_room/20260403-170228_qpooccn1_cramped_room_fcp"
eval_one mep   cramped_room "$F/baseline/cramped_room/20260408-050225_md4rvbma_cramped_room_mep_h64_pop4_ea0.1_e30"
eval_one gamma cramped_room "$F/baseline/cramped_room/20260403-085230_ubpmb358_cramped_room_gamma-vae_h64_pop5_z16_e100"
eval_one ph2   cramped_room "$F/ph2/cramped_room/20260402-134406_l2fth7ra_cramped_room_e3t_ph2_e0p2_o10_s2_k1_ct0" "--source ph2"

# ── asymm_advantages ── (실패 시 skip)
echo -e "\n──── asymm_advantages ────" | tee -a "$LOGDIR/master.log"
eval_one sp    asymm_advantages "$F/baseline/asymm_advantages/20260402-194856_efk6cujd_asymm_advantages_sp_e30"
eval_one e3t   asymm_advantages "$F/baseline/asymm_advantages/20260403-082745_mlec618c_asymm_advantages_e3t_e30"
eval_one fcp   asymm_advantages "$F/baseline/asymm_advantages/20260403-193352_6nlzt2da_asymm_advantages_fcp"
eval_one mep   asymm_advantages "$F/baseline/asymm_advantages/20260405-170609_yboqy292_asymm_advantages_mep_h64_pop5"
eval_one gamma asymm_advantages "$F/baseline/asymm_advantages/20260421-164939_j5nk0bob_asymm_advantages_gamma-vae_pop8_z16"
eval_one ph2   asymm_advantages "$F/ph2/asymm_advantages/20260405-171911_4iphli57_asymm_advantages_e3t_ph2_e0p2_o5_s3_k1_ct0" "--source ph2"

# ── coord_ring ──
echo -e "\n──── coord_ring ────" | tee -a "$LOGDIR/master.log"
eval_one sp    coord_ring "$F/baseline/coord_ring/20260402-231350_dm9zdemf_coord_ring_sp_e30"
eval_one e3t   coord_ring "$F/baseline/coord_ring/20260407-044458_c8zr5hld_coord_ring_e3t_h256"
eval_one fcp   coord_ring "$F/baseline/coord_ring/20260322-190630_m8jruk9n_coord_ring_fcp"
eval_one mep   coord_ring "$F/baseline/coord_ring/20260403-053239_4siekak8_coord_ring_mep_h64_pop5_e100"
eval_one gamma coord_ring "$F/baseline/coord_ring/20260403-111821_espoke7i_coord_ring_gamma-vae_h64_pop5_z16_e100"
eval_one ph2   coord_ring "$F/ph2/coord_ring/20260415-061513_azw2abpk_coord_ring_e3t_ph2_e0p2_o10_s2_k2_ct0" "--source ph2"

# ── counter_circuit ──
echo -e "\n──── counter_circuit ────" | tee -a "$LOGDIR/master.log"
eval_one sp    counter_circuit "$F/baseline/counter_circuit/20260403-040145_h8k9lr8n_counter_circuit_sp_e30"
eval_one e3t   counter_circuit "$F/baseline/counter_circuit/20260415-210152_mm2n4xiu_counter_circuit_e3t"
eval_one fcp   counter_circuit "$F/baseline/counter_circuit/20260404-045702_rh0mwdby_counter_circuit_fcp"
eval_one mep   counter_circuit "$F/baseline/counter_circuit/20260407-111547_2587p3u6_counter_circuit_mep_h64_pop4_ea0.1_e30"
eval_one gamma counter_circuit "$F/baseline/counter_circuit/20260416-054046_tooj17bj_counter_circuit_gamma-vae_pop8_z16"
eval_one ph2   counter_circuit "$F/ph2/counter_circuit/20260416-064554_pnafeifi_counter_circuit_e3t_ph2_e0p2_o10_s4_k1_ct0" "--source ph2"

# ── forced_coord ──
echo -e "\n──── forced_coord ────" | tee -a "$LOGDIR/master.log"
eval_one sp    forced_coord "$F/baseline/forced_coord/20260403-013836_vv9p1dxm_forced_coord_sp_e30"
eval_one e3t   forced_coord "$F/baseline/forced_coord/20260415-222321_zuxojqvn_forced_coord_e3t"
eval_one fcp   forced_coord "$F/baseline/forced_coord/20260404-020741_61kuulek_forced_coord_fcp"
eval_one mep   forced_coord "$F/baseline/forced_coord/20260416-162055_lz0c4ozg_forced_coord_mep_pop8"
eval_one gamma forced_coord "$F/baseline/forced_coord/20260403-135551_q9xd2b2f_forced_coord_gamma-vae_h64_pop5_z16_e100"
eval_one ph2   forced_coord "$F/ph2/forced_coord/20260407-080531_vfe65f4d_forced_coord_e3t_ph2_e0p2_o4_s8_k1_ct0" "--source ph2"

# ── CEC (모든 layout) ──
eval_cec() {
    local layout=$1
    local tag="cec_${layout}"
    echo "  [cec × $layout]" | tee -a "$LOGDIR/master.log"
    if [[ "$layout" == "forced_coord" ]]; then
        # forced_coord: cec_integration/ckpts/forced_coord_9 에서 8 seeds (11..16, 21, 22)
        CUDA_VISIBLE_DEVICES=$GPU $PY code/evaluate_cec_final.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" \
            --output-dir "results_final/${tag}" \
            > "$LOGDIR/${tag}.log" 2>&1
    else
        # 나머지 layout: webapp/models/{layout}/cec/run* 사용 (기존 evaluate_cec.py)
        CUDA_VISIBLE_DEVICES=$GPU $PY code/evaluate_cec.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" \
            --output-dir "results_final/${tag}" \
            > "$LOGDIR/${tag}.log" 2>&1
        # evaluate_cec.py 는 scores_cec.csv 로 저장 → scores.csv alias 생성
        if [[ -f "results_final/${tag}/scores_cec.csv" ]]; then
            cp "results_final/${tag}/scores_cec.csv" "results_final/${tag}/scores.csv"
        fi
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    [FAIL rc=$rc] cec × $layout — log: $LOGDIR/${tag}.log" | tee -a "$LOGDIR/master.log"
    fi
}

echo -e "\n──── CEC (모든 layout) ────" | tee -a "$LOGDIR/master.log"
for L in cramped_room asymm_advantages coord_ring counter_circuit forced_coord; do
    eval_cec "$L"
done

echo -e "\n======================================================" | tee -a "$LOGDIR/master.log"
echo "완료 $(date)" | tee -a "$LOGDIR/master.log"
echo "======================================================" | tee -a "$LOGDIR/master.log"
