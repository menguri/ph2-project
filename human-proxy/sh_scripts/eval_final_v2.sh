#!/bin/bash
# Fast path 기반 final eval (BC × RL + CEC × BC, 모두 JIT+vmap)
#
# 사용법: bash sh_scripts/eval_final_v2.sh [GPU=6] [SEEDS=5]
set +e
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-6}
SEEDS=${2:-5}
PY=/home/mlic/mingukang/ph2-project/overcooked_v2/bin/python

F=../final
LOGDIR=results_final/logs_v2
mkdir -p "$LOGDIR"

eval_one() {
    local algo=$1 layout=$2 run=$3 src=${4:-}
    if [[ ! -d "$run" ]]; then
        echo "  [SKIP] $algo × $layout: $run 없음" | tee -a "$LOGDIR/master.log"
        return
    fi
    local tag="${algo}_${layout}"
    echo "[$algo × $layout] start $(date +%H:%M:%S)" | tee -a "$LOGDIR/master.log"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate.py \
        --algo-dir "$run" --layout "$layout" \
        --bc-model-dir models --num-eval-seeds "$SEEDS" \
        --output-dir "results_final/${tag}" $src \
        > "$LOGDIR/${tag}.log" 2>&1
    local rc=$?
    echo "[$algo × $layout] done rc=$rc $(date +%H:%M:%S)" | tee -a "$LOGDIR/master.log"
}

eval_cec() {
    local layout=$1
    local tag="cec_${layout}"
    echo "[cec × $layout] start $(date +%H:%M:%S)" | tee -a "$LOGDIR/master.log"
    if [[ "$layout" == "forced_coord" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate_cec_final.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" --engine v1 \
            --output-dir "results_final/${tag}" \
            > "$LOGDIR/${tag}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY code/evaluate_cec.py \
            --layout "$layout" --num-eval-seeds "$SEEDS" --engine v1 \
            --output-dir "results_final/${tag}" \
            > "$LOGDIR/${tag}.log" 2>&1
        [[ -f "results_final/${tag}/scores_cec.csv" ]] && \
            cp "results_final/${tag}/scores_cec.csv" "results_final/${tag}/scores.csv"
    fi
    local rc=$?
    echo "[cec × $layout] done rc=$rc $(date +%H:%M:%S)" | tee -a "$LOGDIR/master.log"
}

echo "======================================================" | tee -a "$LOGDIR/master.log"
echo "Fast path eval (GPU=$GPU, seeds=$SEEDS) $(date)" | tee -a "$LOGDIR/master.log"
echo "======================================================" | tee -a "$LOGDIR/master.log"

# ── BC × RL (6 algos × 5 layouts = 30) ──
for L in cramped_room asymm_advantages coord_ring counter_circuit forced_coord; do
    echo -e "\n── $L ──" | tee -a "$LOGDIR/master.log"
    case $L in
        cramped_room)
            eval_one sp    $L "$F/baseline/$L/20260402-174158_d4adx8ic_cramped_room_sp_e30"
            eval_one e3t   $L "$F/baseline/$L/20260407-052043_swhnnuls_cramped_room_e3t_h256"
            eval_one fcp   $L "$F/baseline/$L/20260403-170228_qpooccn1_cramped_room_fcp"
            eval_one mep   $L "$F/baseline/$L/20260408-050225_md4rvbma_cramped_room_mep_h64_pop4_ea0.1_e30"
            eval_one gamma $L "$F/baseline/$L/20260403-085230_ubpmb358_cramped_room_gamma-vae_h64_pop5_z16_e100"
            eval_one ph2   $L "$F/ph2/$L/20260402-134406_l2fth7ra_cramped_room_e3t_ph2_e0p2_o10_s2_k1_ct0" "--source ph2"
            ;;
        asymm_advantages)
            eval_one sp    $L "$F/baseline/$L/20260402-194856_efk6cujd_asymm_advantages_sp_e30"
            eval_one e3t   $L "$F/baseline/$L/20260403-082745_mlec618c_asymm_advantages_e3t_e30"
            eval_one fcp   $L "$F/baseline/$L/20260403-193352_6nlzt2da_asymm_advantages_fcp"
            eval_one mep   $L "$F/baseline/$L/20260405-170609_yboqy292_asymm_advantages_mep_h64_pop5"
            eval_one gamma $L "$F/baseline/$L/20260421-164939_j5nk0bob_asymm_advantages_gamma-vae_pop8_z16"
            eval_one ph2   $L "$F/ph2/$L/20260405-171911_4iphli57_asymm_advantages_e3t_ph2_e0p2_o5_s3_k1_ct0" "--source ph2"
            ;;
        coord_ring)
            eval_one sp    $L "$F/baseline/$L/20260402-231350_dm9zdemf_coord_ring_sp_e30"
            eval_one e3t   $L "$F/baseline/$L/20260407-044458_c8zr5hld_coord_ring_e3t_h256"
            eval_one fcp   $L "$F/baseline/$L/20260322-190630_m8jruk9n_coord_ring_fcp"
            eval_one mep   $L "$F/baseline/$L/20260403-053239_4siekak8_coord_ring_mep_h64_pop5_e100"
            eval_one gamma $L "$F/baseline/$L/20260403-111821_espoke7i_coord_ring_gamma-vae_h64_pop5_z16_e100"
            eval_one ph2   $L "$F/ph2/$L/20260415-061513_azw2abpk_coord_ring_e3t_ph2_e0p2_o10_s2_k2_ct0" "--source ph2"
            ;;
        counter_circuit)
            eval_one sp    $L "$F/baseline/$L/20260403-040145_h8k9lr8n_counter_circuit_sp_e30"
            eval_one e3t   $L "$F/baseline/$L/20260415-210152_mm2n4xiu_counter_circuit_e3t"
            eval_one fcp   $L "$F/baseline/$L/20260404-045702_rh0mwdby_counter_circuit_fcp"
            eval_one mep   $L "$F/baseline/$L/20260407-111547_2587p3u6_counter_circuit_mep_h64_pop4_ea0.1_e30"
            eval_one gamma $L "$F/baseline/$L/20260416-054046_tooj17bj_counter_circuit_gamma-vae_pop8_z16"
            eval_one ph2   $L "$F/ph2/$L/20260416-064554_pnafeifi_counter_circuit_e3t_ph2_e0p2_o10_s4_k1_ct0" "--source ph2"
            ;;
        forced_coord)
            eval_one sp    $L "$F/baseline/$L/20260403-013836_vv9p1dxm_forced_coord_sp_e30"
            eval_one e3t   $L "$F/baseline/$L/20260415-222321_zuxojqvn_forced_coord_e3t"
            eval_one fcp   $L "$F/baseline/$L/20260404-020741_61kuulek_forced_coord_fcp"
            eval_one mep   $L "$F/baseline/$L/20260416-162055_lz0c4ozg_forced_coord_mep_pop8"
            eval_one gamma $L "$F/baseline/$L/20260420-140916_vuhbzyo4_forced_coord_gamma-vae_pop8_z16"
            eval_one ph2   $L "$F/ph2/$L/20260419-070914_8ixp5kyd_forced_coord_e3t_ph2_e0p3_o3_s8_k1_ct0" "--source ph2"
            ;;
    esac
done

# ── CEC (V1 engine + JIT, 5 layouts) ──
echo -e "\n── CEC (V1+jit) ──" | tee -a "$LOGDIR/master.log"
for L in cramped_room asymm_advantages coord_ring counter_circuit forced_coord; do
    eval_cec "$L"
done

echo -e "\n======================================================" | tee -a "$LOGDIR/master.log"
echo "Fast path eval 완료 $(date)" | tee -a "$LOGDIR/master.log"
echo "======================================================" | tee -a "$LOGDIR/master.log"
