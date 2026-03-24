#!/bin/bash
# =============================================================================
# ToyCoop (Dual Destination) — SP / E3T / FCP / MEP(S1→S2) 실행
# baseline/ 에서 실행
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR="$(pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_BASE="${PROJECT_DIR}/runs"

# --- 기본 설정 ---
GPUS="${1:-6,7}"
: "${RANDOM_RESET:=true}"     # true면 매 에피소드 랜덤 배치 (procedural generation)
ENV_DEVICE="cpu"
TOYCOOP_NENVS=512
TOYCOOP_NSTEPS=100
FCP_DEVICE="gpu"
FCP_SEEDS=10
MEP_S1_SEEDS=1
MEP_S2_SEEDS=10

# FCP population annealing
: "${POP_ANNEAL_ENABLE:=0}"
: "${POP_ANNEAL_HORIZON:=1000000}"   # 50M (100M의 절반)
: "${POP_ANNEAL_BEGIN:=0}"

# random_reset override 인자 (true일 때만 추가)
RANDOM_RESET_ARGS=""
if [[ "$RANDOM_RESET" == "true" ]]; then
  RANDOM_RESET_ARGS="--random-reset true"
fi

# annealing 인자
POP_ANNEAL_ARGS=""
if [[ "$POP_ANNEAL_ENABLE" == "1" ]]; then
  POP_ANNEAL_ARGS="--pop-anneal-horizon $POP_ANNEAL_HORIZON --pop-anneal-begin $POP_ANNEAL_BEGIN"
fi

echo "============================================="
echo "  ToyCoop Pipeline"
echo "  GPUs: $GPUS"
echo "  random_reset: $RANDOM_RESET"
echo "============================================="

# =============================================================================
# 1) SP (Self-Play)
# =============================================================================
echo ""
echo "[1/5] ====== ToyCoop SP ======"
./run_user_wandb.sh \
    --gpus "$GPUS" \
    --env "toy_coop" \
    --exp "rnn-sp-toycoop" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$TOYCOOP_NENVS" \
    --nsteps "$TOYCOOP_NSTEPS" \
    --tags "toycoop,sp" \
    $RANDOM_RESET_ARGS

echo "[1/5] ====== ToyCoop SP 완료 ======"

# # =============================================================================
# # 2) E3T
# # =============================================================================
echo ""
echo "[2/5] ====== ToyCoop E3T ======"
./run_user_wandb.sh \
    --gpus "$GPUS" \
    --env "toy_coop" \
    --exp "rnn-e3t-toycoop" \
    --env-device "$ENV_DEVICE" \
    --nenvs "$TOYCOOP_NENVS" \
    --nsteps "$TOYCOOP_NSTEPS" \
    --e3t-epsilon 0.2 \
    --use-partner-modeling True \
    --pred-loss-coef 1.0 \
    --tags "toycoop,e3t" \
    $RANDOM_RESET_ARGS

echo "[2/5] ====== ToyCoop E3T 완료 ======"

# # =============================================================================
# # 3) FCP — SP population 필요
# # =============================================================================
# echo ""
# echo "[3/5] ====== ToyCoop FCP ======"

# # SP 체크포인트에서 FCP population 경로 찾기
# FCP_POP_PATH="fcp_populations/toy_coop_sp"
# if [[ -d "$PROJECT_DIR/$FCP_POP_PATH" ]]; then
#     ./run_user_wandb.sh \
#         --gpus "$GPUS" \
#         --env "toy_coop" \
#         --exp "rnn-fcp-toycoop" \
#         --env-device "$ENV_DEVICE" \
#         --nenvs "$TOYCOOP_NENVS" \
#         --nsteps "$TOYCOOP_NSTEPS" \
#         --seeds "$FCP_SEEDS" \
#         --fcp "$FCP_POP_PATH" \
#         --fcp-device "$FCP_DEVICE" \
#         --tags "toycoop,fcp" \
#         $RANDOM_RESET_ARGS \
#         $POP_ANNEAL_ARGS
#     echo "[3/5] ====== ToyCoop FCP 완료 ======"
# else
#     echo "[3/5] ====== ToyCoop FCP 스킵: $FCP_POP_PATH 없음 ======"
#     echo "        먼저 SP를 돌리고 fcp_populations/toy_coop_sp 에 체크포인트를 배치하세요."
# fi

# =============================================================================
# 4) MEP S1 — Population 생성
# =============================================================================
echo ""
echo "[4/5] ====== ToyCoop MEP S1 ======"
./run_user_wandb.sh \
    --gpus "$GPUS" \
    --env "toy_coop" \
    --exp "rnn-mep-s1-toycoop" \
    --env-device "$ENV_DEVICE" \
    --seeds "$MEP_S1_SEEDS" \
    --tags "toycoop,mep_s1" \
    $RANDOM_RESET_ARGS

echo "[4/5] ====== ToyCoop MEP S1 완료 ======"

# =============================================================================
# 5) MEP S2 — Population으로 학습
# =============================================================================
echo ""
echo "[5/5] ====== ToyCoop MEP S2 ======"

# S1에서 생성된 가장 최근 mep_population 디렉토리 자동 탐색
MEP_POP_DIR=$(find "${RUNS_BASE}" -maxdepth 3 -type d -name "mep_population" \
    -path "*ToyCoop*" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}')

if [[ -n "${MEP_POP_DIR}" && -d "${MEP_POP_DIR}" ]]; then
    echo "  mep_population: ${MEP_POP_DIR}"
    ./run_user_wandb.sh \
        --gpus "$GPUS" \
        --env "toy_coop" \
        --exp "rnn-mep-s2-toycoop" \
        --env-device "$ENV_DEVICE" \
        --seeds "$MEP_S2_SEEDS" \
        --mep-pop-dir "${MEP_POP_DIR}" \
        --tags "toycoop,mep_s2" \
        $RANDOM_RESET_ARGS
    echo "[5/5] ====== ToyCoop MEP S2 완료 ======"
else
    echo "[5/5] ====== ToyCoop MEP S2 스킵: mep_population 디렉토리 없음 ======"
    echo "        MEP S1이 정상 완료되었는지 확인하세요."
fi

echo ""
echo "============================================="
echo "  ToyCoop Pipeline 완료"
echo "============================================="
