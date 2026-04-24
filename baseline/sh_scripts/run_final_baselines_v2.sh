#!/usr/bin/env bash
# =============================================================================
# Final Baseline V2 — counter_circuit / forced_coord 전용 튜닝
# GAMMA, MEP 2개 알고리즘 (E3T는 주석 처리 유지)
#
# 변경점 (v1 대비):
#   - 좁은 레이아웃(narrow corridor, 강제 협력)에서 MEP S1 실패 & MEP/GAMMA S2 oscillation
#     문제 해결을 위해 알고리즘별 추가 파라미터 (*_EXTRA) 도입.
#   - PH2_OVERRIDES 17개 파라미터는 v1과 완전히 동일 (고정).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPUS="0,6"
HARD_LAYOUTS=(cramped_room coord_ring asymm_advantages)

# PH2 IPPO 공통 하이퍼파라미터 — ph2/config/model/rnn.yaml 기준으로 전 알고리즘 정렬
# 제외: TOTAL_TIMESTEPS(원래 설정 유지), OBS_ENCODER(GAMMA는 CNNGamma — 알고리즘별 세팅).
# NUM_ENVS=64, NUM_STEPS=256, NUM_MINIBATCHES=16 공통 (batch=16,384 / mb_size=1024).
PH2_OVERRIDES=(
  --extra "model.LR=2.5e-4"
  --extra "model.ANNEAL_LR=True"
  --extra "model.LR_WARMUP=0.05"
  --extra "model.NUM_ENVS=64"
  --extra "model.NUM_STEPS=256"
  --extra "model.NUM_MINIBATCHES=64"
  --extra "model.UPDATE_EPOCHS=4"
  --extra "model.GAMMA=0.99"
  --extra "model.GAE_LAMBDA=0.95"
  --extra "model.CLIP_EPS=0.2"
  --extra "model.ENT_COEF=0.01"
  --extra "model.VF_COEF=0.5"
  --extra "model.MAX_GRAD_NORM=0.25"
  --extra "model.GRU_HIDDEN_DIM=128"
  --extra "model.FC_DIM_SIZE=128"
  --extra "model.CNN_FEATURES=32"
  --extra "model.ACTIVATION=relu"
)


# =============================================================================
# 좁은 레이아웃 전용 추가 파라미터
# =============================================================================
# 진단:
#   1) MEP S1 실패: MEP_ENTROPY_ALPHA=0.1이 좁은 레이아웃에서 "올바른 행동에 페널티" 역효과.
#      좁은 통로에서 유효 행동이 1-2개뿐인데 -log π_pop(a|s) bonus가 "아무도 안 하는 행동"을 장려.
#   2) S2 oscillation: prioritized sampling α가 어려운 파트너를 과하게 선택 → best-response cycling.
#      또한 PH2의 NUM_ENVS=64가 S2 파트너 분포에 비해 배치 variance 큼.
#   3) GAMMA VAE: S1 population이 narrow하면 VAE 학습 데이터 품질 낮음 + z~N(0,I) 샘플링에서
#      out-of-distribution 파트너 생성 가능.

# ----- MEP 전용 추가 파라미터 -----
MEP_EXTRA=(
  # === 원본 MEP S1 재현 ===
  # 원본 MEP: (1) 100% self-play (round-robin per-episode, 같은 member 양쪽 slot)
  #         (2) MEP bonus α=0.01 (forced_coord 원본은 0.04) 로 pop 다양성
  # 우리 기존 cross-play + 조정된 α 는 원본과 구조적으로 다름.
  --extra "MEP_ENTROPY_ALPHA=0.01"
  --extra "+MEP_S1_SP_WARMUP_FRAC=1.0"    # 100% self-play (원본 동작)

  # PPO ENT_COEF 는 PH2_OVERRIDES 의 0.01 상수 그대로 (schedule 불사용)

  # REW_SHAPING_HORIZON=1e10: S1/S2 전 구간에서 shaping 끝까지 유지.
  # forced_coord 같은 좁은 레이아웃은 dense shaping이 사라지면 sparse 0에서 길을 잃음.
  --extra "model.REW_SHAPING_HORIZON=3e7"
  --extra "model.S2_REW_SHAPING_HORIZON=1e10"

  # 4 → 8: 다양한 성공 정책 발견 확률 증가, partner 다양성 확보
  --extra "MEP_POPULATION_SIZE=8"

  # S2 Prioritized sampling α 1.0 → 0.3: 어려운 파트너 과샘플링 완화 → best-response cycling 억제
  --extra "MEP_PRIORITIZED_ALPHA=0.7"

  # S2 배치 확대 (64 → 128): gradient variance 감소로 oscillation 완화
  # `+` prefix: rnn-mep.yaml model 섹션에 S2_NUM_ENVS 키가 없어 신규 추가가 필요 (Hydra strict mode)
  --extra "+model.S2_NUM_ENVS=64"
)

# ----- GAMMA 전용 추가 파라미터 -----
# GAMMA 가 narrow layout 에서 학습 안 되는 근본원인 조합:
#   (a) MAPPO centralized critic + ValueNorm + Huber → delivery(+20) 같은 희귀 보상
#       에 대한 value gradient 가 이중으로 완화됨 → S1 에서 value 학습이 너무 느림
#   (b) -log π_pop (MEP entropy bonus) 가 reward 에 직접 가산돼 sparse 0 구간에서
#       task reward 대신 entropy 쪽으로 policy 가 흐름
#   (c) PPO 자체 entropy coef (default START=0.5) 가 uniform policy 로 고정 → task signal 소실
#
# 아래 조정: MEP 에는 없고 GAMMA 에만 있는 "value 안정화" (USE_VALUENORM/HUBER) 를
# 꺼서 MEP 와 동일한 value loss 경로로 맞춘다 — 공정성은 유지 (MEP 쪽에 없는 것을 제거).
# 나머지는 모두 entropy 압력 축소 및 VAE 품질 보강.
GAMMA_EXTRA=(
  # === 원본 GAMMA/MEP S1 재현 ===
  # 원본 구조: (1) 100% self-play (round-robin per-episode, 양쪽 agent 모두 같은 member)
  #           (2) MEP bonus α=0.01 로 pop 다양성 부과 (−log π_pop(a|s) 를 reward 에 가산)
  # 우리 기존 cross-play + α=0 은 원본과 구조적으로 다른 변형이었음.
  # `+` prefix: rnn.yaml 에 없는 신규 키라 Hydra strict mode 회피.
  --extra "MEP_ENTROPY_ALPHA=0.01"         # 원본 값
  --extra "+GAMMA_S1_SP_WARMUP_FRAC=1.0"    # 100% self-play (원본 동작과 동일)

  # ValueNorm / Huber 비활성화 — GAMMA 기본값 True/10.0 이지만 MEP 는 둘 다 미사용.
  # narrow sparse 환경에서 delivery reward(+20) 가 ValueNorm running stats 에
  # 빠르게 반영 안 되고 Huber δ=10 으로 gradient clipping 돼 value 학습 정체.
  # MEP 와 동일한 MSE + 비정규화 value loss 로 통일하여 공정성 유지.
  --extra "model.USE_VALUENORM=False"
  --extra "model.HUBER_DELTA=200.0"

  # reward shaping S1/S2 끝까지 유지
  --extra "model.REW_SHAPING_HORIZON=3e7"
  --extra "model.S2_REW_SHAPING_HORIZON=1e10"

  # population 크기 — GAMMA VAE 는 pool 다양성이 S2 품질 직결. 8 → 16 으로 증가.
  --extra "MEP_POPULATION_SIZE=8"

  # prioritized sampling α 3.0(원본) → 0.7 로 완화 (어려운 파트너 과샘플링 억제)
  --extra "MEP_PRIORITIZED_ALPHA=0.7"

  # VAE 튜닝
  --extra "GAMMA_VAE_KL_PENALTY=0.7"      # KL loose (prior 에서 더 벗어나게, 정보 확보)
  --extra "GAMMA_VAE_KL_INIT=0.1"         # 학습 초기 prior 근처
  --extra "GAMMA_VAE_EPOCHS=1000"         # 500 → 1000 (narrow 는 VAE 가 쓸 데이터 자체가 적음)
  --extra "GAMMA_VAE_ROLLOUT_EPISODES=300" # 100 → 300
  --extra "GAMMA_VAE_CHUNK_LENGTH=200"     # 100 → 200 (400-step 에피의 절반 단위로 context)
  --extra "GAMMA_VAE_Z_CHANGE_PROB=0.1"    # mid-episode z resample: 에피 내 파트너 변화 허용

  # S2 배치 확대
  --extra "+model.S2_NUM_ENVS=64"
)

# =============================================================================
# 1. GAMMA — counter_circuit / forced_coord (MAPPO, S1=30M, S2=100M VAE)
# =============================================================================
echo "============================================================"
echo "  GAMMA (MAPPO, S1=30M, S2=100M VAE, 10 seeds, PH2 tuned + narrow-layout EXTRA)"
echo "============================================================"

for layout in "${HARD_LAYOUTS[@]}"; do
  echo "[GAMMA] ${layout}"
  ./run_user_wandb.sh \
    --exp rnn-gamma \
    --env "${layout}" \
    --gpus "${GPUS}" \
    --seeds 1 \
    --tags gamma,final_v2,ph2tuned,narrow_tuned \
    --extra "++GAMMA_S2_METHOD=vae" \
    --extra "model.TOTAL_TIMESTEPS=3e7" \
    --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
    --extra "S2_NUM_SEEDS=10" \
    --extra "model.OBS_ENCODER=CNNGamma" \
    "${PH2_OVERRIDES[@]}" \
    "${GAMMA_EXTRA[@]}"
done

# =============================================================================
# 2. MEP — counter_circuit / forced_coord (S1=30M, S2=100M)
# =============================================================================
# echo "============================================================"
# echo "  MEP (S1=30M, S2=100M, 10 seeds, PH2 tuned + narrow-layout EXTRA)"
# echo "============================================================"

# for layout in "${HARD_LAYOUTS[@]}"; do
#   echo "[MEP] ${layout}"
#   ./run_user_wandb.sh \
#     --exp rnn-mep \
#     --env "${layout}" \
#     --gpus "${GPUS}" \
#     --seeds 1 \
#     --tags mep,final_v2,ph2tuned,narrow_tuned \
#     --extra "model.TOTAL_TIMESTEPS=3e7" \
#     --extra "model.S2_TOTAL_TIMESTEPS=1e8" \
#     --extra "S2_NUM_SEEDS=10" \
#     --extra "model.OBS_ENCODER=CNN" \
#     "${PH2_OVERRIDES[@]}" \
#     "${MEP_EXTRA[@]}"
# done

# =============================================================================
# 3. E3T — 주석 처리 유지 (v1과 동일)
# =============================================================================
# echo "============================================================"
# echo "  E3T (30M, 10 seeds, PH2 tuned)"
# echo "============================================================"

# for layout in "${HARD_LAYOUTS[@]}"; do
#   echo "[E3T] ${layout}"
#   ./run_user_wandb.sh \
#     --exp rnn-e3t \
#     --env "${layout}" \
#     --gpus "${GPUS}" \
#     --seeds 10 \
#     --tags e3t,final,ph2tuned \
#     --extra "model.TOTAL_TIMESTEPS=3e7" \
#     --extra "model.OBS_ENCODER=CNN" \
#     "${PH2_OVERRIDES[@]}"
# done

echo "============================================================"
echo "  전체 완료"
echo "============================================================"
