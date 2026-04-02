#!/bin/bash
# 전 알고리즘 × 전 레이아웃 BC cross-play 평가 (순차 실행)
# 결과: results/{algo}_{layout}/scores.csv
#
# 사용법:
#   cd human-proxy && bash sh_scripts/eval_all.sh [GPU_IDX] [NUM_EVAL_SEEDS]
set -e
cd "$(dirname "$0")/.." || exit 1

GPU=${1:-0}
SEEDS=${2:-5}
B="../baseline/runs"
P="../ph2/runs"

echo "========================================"
echo "BC Cross-Play 전체 평가 (GPU=$GPU, seeds=$SEEDS)"
echo "========================================"

eval_one() {
    local algo=$1 layout=$2 run=$3 src=${4:-}
    echo "  [$algo × $layout] $(basename $run)"
    CUDA_VISIBLE_DEVICES=$GPU python code/evaluate.py \
        --algo-dir "$run" --layout "$layout" \
        --bc-model-dir models --num-eval-seeds "$SEEDS" \
        --output-dir "results/${algo}_${layout}" $src
}

# ── cramped_room ──
echo -e "\n──── cramped_room ────"
eval_one sp  cramped_room "$B/20260321-123250_z08brfvg_cramped_room_sp"
eval_one e3t cramped_room "$B/20260321-225509_3tkkkit1_cramped_room_e3t"
eval_one fcp cramped_room "$B/20260319-064201_0sx5beao_cramped_room_fcp"
eval_one mep cramped_room "$B/20260319-170303_sycdfdtk_cramped_room_m2"
eval_one ph2 cramped_room "$P/20260319-113704_6lre6zzu_cramped_room_e3t_ph2_e0p2_o10_s2" "--source ph2"

# ── asymm_advantages ──
echo -e "\n──── asymm_advantages ────"
eval_one sp  asymm_advantages "$B/20260321-074843_iw3up21x_asymm_advantages_sp"
eval_one e3t asymm_advantages "$B/20260322-050112_28vrfjj2_asymm_advantages_e3t"
eval_one fcp asymm_advantages "$B/20260322-172759_752jcfwa_asymm_advantages_fcp"
eval_one mep asymm_advantages "$B/20260319-173817_90zz6n9j_asymm_advantages_m2"
eval_one ph2 asymm_advantages "$P/20260319-071726_iqy7gow0_asymm_advantages_e3t_ph2_e0p2_o5_s3" "--source ph2"

# ── coord_ring ──
echo -e "\n──── coord_ring ────"
eval_one sp  coord_ring "$B/20260321-093737_kskzpq1m_coord_ring_sp"
eval_one e3t coord_ring "$B/20260322-002030_163hiri4_coord_ring_e3t"
eval_one fcp coord_ring "$B/20260319-040839_mkp6oq1s_coord_ring_fcp"
eval_one mep coord_ring "$B/20260319-183450_c5clczpn_coord_ring_m2"
eval_one ph2 coord_ring "$P/20260321-133010_ik6k22pk_coord_ring_e3t_ph2_e0p2_o10_s2" "--source ph2"

# ── counter_circuit ──
echo -e "\n──── counter_circuit ────"
eval_one sp  counter_circuit "$B/20260321-060547_ymmj0qum_counter_circuit_sp"
eval_one e3t counter_circuit "$B/20260322-031945_r33r0yd7_counter_circuit_e3t"
eval_one fcp counter_circuit "$B/20260319-085357_frqgnspy_counter_circuit_fcp"
eval_one mep counter_circuit "$B/20260320-081124_1bikfs29_counter_circuit_m2"
eval_one ph2 counter_circuit "$P/20260320-102644_go5w421o_counter_circuit_e3t_ph2_e0p2_o10_s2" "--source ph2"

# ── forced_coord ──
echo -e "\n──── forced_coord ────"
eval_one sp  forced_coord "$B/20260321-110623_80jj4aro_forced_coord_sp"
eval_one e3t forced_coord "$B/20260322-015131_jwlvixzr_forced_coord_e3t"
eval_one fcp forced_coord "$B/20260319-052532_p957gjsu_forced_coord_fcp"
eval_one mep forced_coord "$B/20260319-191438_rbykepcg_forced_coord_m2"
eval_one ph2 forced_coord "$P/20260324-173318_7b41jx1b_forced_coord_e3t_ph2_e0p2_o2_s3_k1_ct0" "--source ph2"

# ── BC × BC (baseline) ──
echo -e "\n──── BC × BC ────"
CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys; sys.path.insert(0, 'code')
from policy import setup_pythonpath, BCPolicy
setup_pythonpath('baseline')
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.eval.utils import make_eval_env
import jax, numpy as np, csv, os

layouts = ['cramped_room', 'asymm_advantages', 'coord_ring', 'counter_circuit', 'forced_coord']
num_seeds = $SEEDS

os.makedirs('results/bc_bc', exist_ok=True)
rows = []
for layout in layouts:
    bc0 = BCPolicy.from_pretrained(f'models/{layout}/pos_0/seed_0')
    bc1 = BCPolicy.from_pretrained(f'models/{layout}/pos_1/seed_0')
    pairing = PolicyPairing(bc0, bc1)
    env, _, _ = make_eval_env(layout, {})
    rewards = []
    for si in range(num_seeds):
        key = jax.random.PRNGKey(si)
        rollout = get_rollout(pairing, env, key, use_jit=True)
        rewards.append(float(rollout.total_reward))
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f'  BC×BC {layout}: {mean_r:.1f} ± {std_r:.1f}')
    rows.append([layout, f'{mean_r:.2f}', f'{std_r:.2f}'])

with open('results/bc_bc/scores.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layout', 'mean_reward', 'std_reward'])
    w.writerows(rows)
print('점수 저장: results/bc_bc/scores.csv')
"

echo -e "\n========================================"
echo "완료."
echo "========================================"
