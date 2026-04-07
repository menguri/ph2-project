#!/usr/bin/env bash
# =============================================================================
# GridSpread Baseline 실험: SP, E3T, FCP, MEP
# 사용법:
#   bash run_factory_spread.sh                  # 기본 N=4
#   bash run_factory_spread.sh 6                # N=6
#   bash run_factory_spread.sh 2 4 6 8 10       # 여러 N 순차 실행
#
# FCP population 구성:
#   SP 학습 완료 후 build_fcp_population 함수가 자동으로
#   runs/ 에서 SP 체크포인트를 fcp_populations/gridspread_{N}a_sp/ 로 복사.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 5,6
GPUS="${GPUS:-6,7}"
ENV_DEVICE="${ENV_DEVICE:-cpu}"
NENVS="${NENVS:-128}"
NSTEPS="${NSTEPS:-512}"
TOTAL_TS="1e8"                # 100M timesteps (S2에도 적용)

# 에이전트 수: CLI 인자 또는 기본값 4
if [ $# -eq 0 ]; then
  AGENT_COUNTS=(4)
else
  AGENT_COUNTS=("$@")
fi

# rnn-cole.yaml의 NUM_MINIBATCHES=5는 16×128=2048을 나누지 못함 → 4로 override
# 각 override는 별도 --extra 플래그로 전달해야 Hydra가 올바르게 파싱함
TS_EXTRA="++model.TOTAL_TIMESTEPS=${TOTAL_TS}"
S2_TS_EXTRA="++model.S2_TOTAL_TIMESTEPS=${TOTAL_TS}"
MINIBATCH_EXTRA="++model.NUM_MINIBATCHES=4"
MLP_ENCODER_EXTRA="++model.OBS_ENCODER=MLP"
ENT_START_EXTRA="++model.ENT_COEF_START=0.1"
ENT_END_EXTRA="++model.ENT_COEF_END=0.01"
ENT_ANNEAL_EXTRA="++model.ENT_COEF_ANNEAL_STEPS=1e7"
# === DIST_SHAPING_EXPERIMENT === (0이면 off, 예: 0.01) > 최대 penalty: 0.1(0.05) 0.2(0.1) 0.5(0.25)
DIST_SHAPING_EXTRA="++env.ENV_KWARGS.dist_shaping_coef=0.0"
# all_covered 즉시 종료 모드
EARLY_TERM_EXTRA="++env.ENV_KWARGS.early_terminate=true"
# Eval 체크포인트: eval/ 서브폴더에 1M step 단위로 저장 (TOTAL_TS=1e8 → 101 ckpts: 0M..100M)
EVAL_CKPT_DIR_EXTRA="++SAVE_TO_EVAL_DIR=true"
EVAL_NUM_CKPT_EXTRA="++NUM_CHECKPOINTS=101"

# -----------------------------------------------------------------------------
# FCP population 빌드: SP runs → fcp_populations/gridspread_{N}a_sp/
# SP 학습 완료 직후 호출. 8 runs씩 fcp_N 하위 디렉토리로 그룹핑.
# -----------------------------------------------------------------------------
build_fcp_population() {
  local n_agents=$1
  # bash cwd=sh_scripts/ → ../fcp_populations/ = baseline/fcp_populations/
  local pop_dir="../fcp_populations/gridspread_${n_agents}a_sp"

  # runs/ 아래에서 가장 최근 GridSpread SP run 디렉토리 탐색
  local sp_run
  sp_run=$(find ../runs -maxdepth 1 -type d -iname "*gridspread_${n_agents}a*sp*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

  if [[ -z "$sp_run" ]]; then
    echo "[FCP-POP] SP run not found for N=${n_agents} — SKIP"
    return 1
  fi

  rm -rf "$pop_dir"
  mkdir -p "$pop_dir"

  local counter=0
  local subdir_counter=0
  for run in "$sp_run"/run_*; do
    [[ ! -d "$run" ]] && continue
    if (( counter % 8 == 0 )); then
      subdir_counter=$((subdir_counter + 1))
      mkdir -p "$pop_dir/fcp_${subdir_counter}"
    fi
    cp -r "$run" "$pop_dir/fcp_${subdir_counter}/"
    counter=$((counter + 1))
  done

  echo "[FCP-POP] N=${n_agents}: ${counter} runs → ${subdir_counter} groups → ${pop_dir}"
}

NAGENTS_EXTRA="++env.ENV_KWARGS.n_agents="   # N을 뒤에 붙여서 사용

for N in "${AGENT_COUNTS[@]}"; do
  ENV="gridspread"
  echo "================================================================"
  echo "  GridSpread N=${N}  (env=${ENV})"
  echo "================================================================"

  # # ===========================================================================
  # # 1. SP — 기존 rnn-sp-cole + 100M
  # # ===========================================================================
  # echo "[SP] N=${N}"
  # ./run_user_wandb.sh \
  #   --exp rnn-sp-cole \
  #   --env "${ENV}" \
  #   --gpus "${GPUS}" \
  #   --env-device "${ENV_DEVICE}" \
  #   --nenvs "${NENVS}" \
  #   --nsteps "${NSTEPS}" \
  #   --tags "sp,spread,N${N}" \
  #   --extra "${TS_EXTRA}" \
  #   --extra "${MINIBATCH_EXTRA}" \
  #   --extra "${MLP_ENCODER_EXTRA}" \
  #   --extra "${ENT_START_EXTRA}" \
  #   --extra "${ENT_END_EXTRA}" \
  #   --extra "${ENT_ANNEAL_EXTRA}" \
  #   --extra "${DIST_SHAPING_EXTRA}" \
  #   --extra "${EARLY_TERM_EXTRA}" \
  #   --extra "${NAGENTS_EXTRA}${N}"

  # # ===========================================================================
  # # 1.5. FCP population 빌드 (SP 완료 직후)
  # # ===========================================================================
  # build_fcp_population "${N}"

  # # ===========================================================================
  # # 3. FCP — SP population 필요
  # # ===========================================================================
  # # FCP_DIR_PY: Python cwd=baseline/ 기준 (run_user_wandb.sh --fcp 인자)
  # # FCP_DIR_BASH: bash cwd=sh_scripts/ 기준 ([ -d ] 체크용)
  # FCP_DIR_PY="fcp_populations/gridspread_${N}a_sp"
  # FCP_DIR_BASH="../fcp_populations/gridspread_${N}a_sp"
  # if [ -d "${FCP_DIR_BASH}" ]; then
  #   echo "[FCP] N=${N}"
  #   ./run_user_wandb.sh \
  #     --exp rnn-fcp-cole \
  #     --env "${ENV}" \
  #     --gpus "${GPUS}" \
  #     --env-device "${ENV_DEVICE}" \
  #     --nenvs "${NENVS}" \
  #     --nsteps "${NSTEPS}" \
  #     --fcp "${FCP_DIR_PY}" \
  #     --seeds 10 \
  #     --tags "fcp,spread,N${N}" \
  #     --extra "${TS_EXTRA}" \
  #     --extra "${MINIBATCH_EXTRA}" \
  #     --extra "${MLP_ENCODER_EXTRA}" \
  #     --extra "${ENT_COEF_EXTRA}" \
  #     --extra "${NAGENTS_EXTRA}${N}"
  # else
  #   echo "[FCP] SKIP N=${N} — population 없음: ${FCP_DIR_BASH}"
  # fi

  # ===========================================================================
  # 2. E3T — 기존 rnn-e3t + epsilon 0.2 + prediction OFF + 100M
  # ===========================================================================
  echo "[E3T] N=${N}"
  ./run_user_wandb.sh \
    --exp rnn-e3t \
    --env "${ENV}" \
    --gpus "${GPUS}" \
    --env-device "${ENV_DEVICE}" \
    --nenvs "${NENVS}" \
    --nsteps "${NSTEPS}" \
    --e3t-epsilon 0.2 \
    --tags "e3t,spread,N${N}" \
    --extra "${TS_EXTRA}" \
    --extra "${MINIBATCH_EXTRA}" \
    --extra "${MLP_ENCODER_EXTRA}" \
    --extra "++USE_PREDICTION=False" \
    --extra "${ENT_START_EXTRA}" \
    --extra "${ENT_END_EXTRA}" \
    --extra "${ENT_ANNEAL_EXTRA}" \
    --extra "${DIST_SHAPING_EXTRA}" \
    --extra "${EARLY_TERM_EXTRA}" \
    --extra "${EVAL_CKPT_DIR_EXTRA}" \
    --extra "${EVAL_NUM_CKPT_EXTRA}" \
    --extra "${NAGENTS_EXTRA}${N}"


  # # ===========================================================================
  # # 4. MEP — 기존 rnn-mep + S2 100M
  # # ===========================================================================
  # echo "[MEP] N=${N}"
  # ./run_user_wandb.sh \
  #   --exp rnn-mep \
  #   --env "${ENV}" \
  #   --gpus "${GPUS}" \
  #   --nenvs "${NENVS}" \
  #   --nsteps "${NSTEPS}" \
  #   --seeds 1 \
  #   --tags "mep,spread,N${N}" \
  #   --extra "++model.TOTAL_TIMESTEPS=5e7" \
  #   --extra "${S2_TS_EXTRA}" \
  #   --extra "${MINIBATCH_EXTRA}" \
  #   --extra "${MLP_ENCODER_EXTRA}" \
  #   --extra "${ENT_START_EXTRA}" \
  #   --extra "${ENT_END_EXTRA}" \
  #   --extra "${ENT_ANNEAL_EXTRA}" \
  #   --extra "${DIST_SHAPING_EXTRA}" \
  #   --extra "${EARLY_TERM_EXTRA}" \
  #   --extra "${NAGENTS_EXTRA}${N}"

done

echo "================================================================"
echo "  GridSpread 전체 완료"
echo "================================================================"
