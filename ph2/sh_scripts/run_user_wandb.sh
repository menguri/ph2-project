#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# run_user_wandb.sh — JAX-AHT OvercookedV2 PPO launcher (uv/conda 무관)
# ------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/overcooked_v2"
LEGACY_VENV_DIR="${REPO_ROOT}/overcookedv2"

# ==============================================================================
# 1) 기본값 설정 (환경변수로 덮어쓰기 가능)
# ==============================================================================

: "${CUDA_VISIBLE_DEVICES:=0}"                 # GPU 할당 (콤마 구분: 예 0,1)
: "${WANDB_PROJECT:=zsc-experiment}"                  # W&B 프로젝트명
: "${WANDB_ENTITY:=m-personal-experiment}"
: "${NUM_SEEDS:=5}"                           # 실험 시드 수
: "${SEED:=100}"                                 # 기본 RNG seed override (선택)
: "${NUM_ITERATIONS:=1}"

# 환경/실험 프리셋
: "${ENV_GROUP:=original}"                     # 예: original, grounded_coord_simple, test_time_wide
: "${LAYOUT:=cramped_room}"                   # ENV_GROUP=original 일 때 사용
: "${EXPERIMENT:=cnn}"                         # 예: cnn, rnn-op, rnn-sa, rnn-fcp

# E3T (Mixture Partner Policy) defaults
: "${ANCHOR_ENABLED:=0}"       # 1 => enable STL anchor

# JAX 메모리 설정
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"    # 메모리 선할당 방지
: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.7}"     # 0.0~1.0 비율 (너무 낮으면 성능 저하 가능)

# FCP_DEVICE 설정 (기본값: cpu)
: "${FCP_DEVICE:=cpu}"

: "${USE_PARTNER_MODELING:=True}"
: "${PRED_LOSS_COEF:=1.0}"

# PH1 defaults
: "${PH1_ENABLED:=}"
: "${PH1_BLOCK_TYPE:=}"
: "${PH1_BETA:=}"
: "${PH1_BETA_SCHEDULE_ENABLED:=}"
: "${PH1_BETA_START:=}"
: "${PH1_BETA_END:=}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=}"
: "${PH1_OMEGA:=}"
: "${PH1_SIGMA:=}"
: "${PH1_DISTANCE_THRESHOLD:=}"
: "${PH1_POOL_SIZE:=}"
: "${PH1_NORMAL_PROB:=}"
: "${PH1_MULTI_PENALTY_ENABLED:=}"
: "${PH1_MAX_PENALTY_COUNT:=}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=}"

: "${PH1_POP_DIR:=}"
: "${PH1_POP_NAME:=}"

# PH2 defaults
: "${PH2_RATIO_STAGE1:=}"
: "${PH2_RATIO_STAGE2:=}"
: "${PH2_RATIO_STAGE3:=}"
: "${PH2_FIXED_IND_PROB:=}"
: "${PH2_EPSILON:=}"
: "${ACTION_PREDICTION:=}"
: "${CYCLE_LOSS_ENABLED:=}"
: "${CYCLE_LOSS_COEF:=}"
: "${Z_PREDICTION_ENABLED:=}"
: "${Z_PRED_LOSS_COEF:=}"
: "${LATENT_MODE:=}"
: "${SHARED_PREDICTION:=}"
: "${SAVE_EVAL_CHECKPOINTS:=}"
# CycleTransformer (CT) 파라미터
: "${TRANSFORMER_ACTION:=}"
: "${TRANSFORMER_WINDOW_SIZE:=}"
: "${TRANSFORMER_D_C:=}"
: "${TRANSFORMER_N_HEADS:=}"
: "${TRANSFORMER_N_LAYERS:=}"
: "${TRANSFORMER_RECON_COEF:=}"
: "${TRANSFORMER_PRED_COEF:=}"
: "${TRANSFORMER_CYCLE_COEF:=}"
: "${TRANSFORMER_V2:=}"
: "${TRANSFORMER_V3:=}"

# XLA_FLAGS: 기본 CUDA data dir 설정
: "${XLA_FLAGS:=--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda-12.2}}"

# cuPTI 경로 (CUDA 12.2 기준)
if [ -d "/usr/local/cuda-12.2/extras/CUPTI/lib64" ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_FLAGS

# Host thread 폭증 방지 (학습 로직/보상에는 영향 없음)
# - pthread_create failed: Resource temporarily unavailable 완화 목적
# - 필요 시 외부에서 env로 override 가능
# - 기본값은 기존(1)보다 완화: host 코어 수 기반으로 최대 8 스레드 사용
HOST_CPU_COUNT=8
if command -v nproc >/dev/null 2>&1; then
  HOST_CPU_COUNT=$(nproc)
fi
if [[ -z "${HOST_CPU_COUNT}" || "${HOST_CPU_COUNT}" -lt 1 ]]; then
  HOST_CPU_COUNT=8
fi
DEFAULT_HOST_THREADS=$HOST_CPU_COUNT
if [[ "${DEFAULT_HOST_THREADS}" -gt 8 ]]; then
  DEFAULT_HOST_THREADS=8
fi

: "${OMP_NUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${OPENBLAS_NUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${MKL_NUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${VECLIB_MAXIMUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${NUMEXPR_NUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${TF_NUM_INTRAOP_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${TF_NUM_INTEROP_THREADS:=2}"
: "${JAX_NUM_THREADS:=${DEFAULT_HOST_THREADS}}"
: "${XLA_CPU_THREAD_LIMIT:=${DEFAULT_HOST_THREADS}}"

export OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS
export MKL_NUM_THREADS
export VECLIB_MAXIMUM_THREADS
export NUMEXPR_NUM_THREADS
export TF_NUM_INTRAOP_THREADS
export TF_NUM_INTEROP_THREADS
export JAX_NUM_THREADS
export XLA_CPU_THREAD_LIMIT

# 프로젝트 전용 Python 가상환경 bin 경로 우선
if [[ -d "${VENV_DIR}/bin" ]]; then
  export PATH="${VENV_DIR}/bin:$PATH"
elif [[ -d "${LEGACY_VENV_DIR}/bin" ]]; then
  export PATH="${LEGACY_VENV_DIR}/bin:$PATH"
else
  echo "[WARN] No venv bin directory found at ${VENV_DIR} or ${LEGACY_VENV_DIR}"
fi

# ==============================================================================
# 2) GPU / CUDA 환경 설정
# ==============================================================================

# CUDA_VISIBLE_DEVICES 기본값 재확인 (필요 시 사용자 지정 가능)
: "${CUDA_VISIBLE_DEVICES:=0}"

# JAX 플랫폼 설정: 비워두면 자동 감지, 강제 CPU는 --cpu 또는 JAX_PLATFORMS=cpu
: "${JAX_PLATFORMS:=}"

# CUDA 경로 설정 (필요 시 사용자 지정 가능)
: "${CUDA_HOME:=/usr/local/cuda-12.2}"

# PTX 경고 억제 토글 및 패턴
: "${SUPPRESS_PTX_WARN:=1}"

# '+ptx89 ... not a recognized feature for this target' 류의 소음 로그를 숨기기 위한 정규식
PTX_WARN_RE="(\+)?ptx[0-9]+.*not a recognized feature for this target|not a recognized feature for this target.*(\+)?ptx[0-9]+"

# PTX 경고 필터 함수: 토글에 따라 해당 라인을 제거하거나 그대로 통과
filter_ptx() {
  if [[ "${SUPPRESS_PTX_WARN}" == "1" ]]; then
    # PTX 경고 및 잔여 "(ignoring feature)" 라인을 제거
    grep -v -E "${PTX_WARN_RE}|\(ignoring feature\)\s*$" || true
  else
    cat
  fi
}

# 시스템 CUDA 라이브러리 사용 여부 (기본 0: jaxlib 내 번들 사용)
: "${USE_SYSTEM_CUDA_LIBS:=0}"
if [[ "${USE_SYSTEM_CUDA_LIBS}" == "1" ]]; then
  # LD_LIBRARY_PATH에 CUDA 경로 추가 (권장하지 않음: 충돌 가능)
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
  fi
  # LD_LIBRARY_PATH 정리 (중복 제거)
  CLEANED_LD_LIBRARY_PATH=$(
    echo "${LD_LIBRARY_PATH:-}" \
      | tr ':' '\n' \
      | awk 'NF && !seen[$0]++' \
      | tr '\n' ':'
  )
  export LD_LIBRARY_PATH="$CLEANED_LD_LIBRARY_PATH"
  echo "[INFO] Using system CUDA libs (USE_SYSTEM_CUDA_LIBS=1)"
  echo "[INFO] Cleaned LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
  echo "[INFO] Skipping system CUDA libs (USE_SYSTEM_CUDA_LIBS=0)."
  echo "       We'll unset LD_LIBRARY_PATH and XLA_FLAGS at Python launch to avoid lib conflicts."
fi

# 공통 환경 변수 export
export JAX_PLATFORMS
export CUDA_HOME
export LD_LIBRARY_PATH

# GPU 설정 확인 메시지
echo "[INFO] JAX 플랫폼: JAX_PLATFORMS=$JAX_PLATFORMS"
echo "[INFO] CUDA 경로: CUDA_HOME=$CUDA_HOME"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "[INFO] Thread limits: OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS TF_INTRA=$TF_NUM_INTRAOP_THREADS TF_INTER=$TF_NUM_INTEROP_THREADS JAX_NUM_THREADS=$JAX_NUM_THREADS"

NOTES=""
TAGS=""
SEEDS_EXPLICIT="0"
ITERATIONS_OVERRIDE=$NUM_ITERATIONS

FCP_DIR=""                    # FCP population 디렉토리(선택)
ENV_DEVICE=""                 # env를 CPU/GPU 어디에 둘지: cpu|gpu (기본: 자동)
CAST_OBS_BF16="0"             # 관측을 bf16으로 캐스팅하여 메모리 절감
MODEL_NUM_ENVS_OVERRIDE=""    # model.NUM_ENVS override
MODEL_NUM_STEPS_OVERRIDE=""   # model.NUM_STEPS override

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)       export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --seeds)      NUM_SEEDS="$2"; SEEDS_EXPLICIT="1"; shift 2;;
    --seed)       SEED="$2"; shift 2;;
    --env)        ENV_GROUP="$2"; shift 2;;
    --layout)     LAYOUT="$2"; shift 2;;
    --exp|--experiment) EXPERIMENT="$2"; shift 2;;
    --project)    WANDB_PROJECT="$2"; shift 2;;
    --entity)     WANDB_ENTITY="$2"; shift 2;;
    --notes)      NOTES="$2"; shift 2;;
    --tags)       TAGS="$2"; shift 2;;  # "a,b" 또는 "a b"
    --iters|--iterations) ITERATIONS_OVERRIDE="$2"; shift 2;;
    --fcp)        FCP_DIR="$2"; shift 2;;  # 예: --fcp runs/fcp_populations/grounded_coord_simple
    --cpu)        export JAX_PLATFORMS=cpu; shift 1;;
    --env-device) ENV_DEVICE="$2"; shift 2;;     # cpu|gpu
    --bf16-obs)   CAST_OBS_BF16="1"; shift 1;;
    --nenvs)      MODEL_NUM_ENVS_OVERRIDE="$2"; shift 2;;
  --nsteps)     MODEL_NUM_STEPS_OVERRIDE="$2"; shift 2;;
    --e3t-epsilon) E3T_EPSILON="$2"; shift 2;;
    --anchor)     ANCHOR_ENABLED="1"; shift 1;;
    --mem-frac)   XLA_PYTHON_CLIENT_MEM_FRACTION="$2"; shift 2;;
    --fcp-device) FCP_DEVICE="$2"; shift 2 ;;
    --use-partner-modeling) USE_PARTNER_MODELING="$2"; shift 2;;
    --pred-loss-coef) PRED_LOSS_COEF="$2"; shift 2;;
    --ph1-block-type) PH1_BLOCK_TYPE="$2"; shift 2;;
    --ph1-beta)       PH1_BETA="$2"; shift 2;;
    --ph1-beta-schedule-enabled) PH1_BETA_SCHEDULE_ENABLED="$2"; shift 2;;
    --ph1-beta-start) PH1_BETA_START="$2"; shift 2;;
    --ph1-beta-end) PH1_BETA_END="$2"; shift 2;;
    --ph1-beta-schedule-horizon-env-steps) PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS="$2"; shift 2;;
    --ph1-omega)      PH1_OMEGA="$2"; shift 2;;
    --ph1-sigma)      PH1_SIGMA="$2"; shift 2;;
    --ph1-dist)       PH1_DISTANCE_THRESHOLD="$2"; shift 2;;
    --ph1-pool-size)  PH1_POOL_SIZE="$2"; shift 2;;
    --ph1-normal-prob) PH1_NORMAL_PROB="$2"; shift 2;;
    --ph1-multi-penalty-enabled) PH1_MULTI_PENALTY_ENABLED="$2"; shift 2;;
    --ph1-max-penalty-count) PH1_MAX_PENALTY_COUNT="$2"; shift 2;;
    --ph1-multi-penalty-single-weight) PH1_MULTI_PENALTY_SINGLE_WEIGHT="$2"; shift 2;;
    --ph1-multi-penalty-other-weight) PH1_MULTI_PENALTY_OTHER_WEIGHT="$2"; shift 2;;
    --ph1-enabled)    PH1_ENABLED="$2"; shift 2;;
    --ph1-epsilon)    PH1_EPSILON="$2"; shift 2;;
    --ph1-warmup-steps) PH1_WARMUP_STEPS="$2"; shift 2;;
    --ph1-pop-dir)    PH1_POP_DIR="$2"; shift 2;;
    --ph1-pop-name)   PH1_POP_NAME="$2"; shift 2;;
    --ph2-ratio-stage1) PH2_RATIO_STAGE1="$2"; shift 2;;
    --ph2-ratio-stage2) PH2_RATIO_STAGE2="$2"; shift 2;;
    --ph2-ratio-stage3) PH2_RATIO_STAGE3="$2"; shift 2;;
    --ph2-fixed-ind-prob) PH2_FIXED_IND_PROB="$2"; shift 2;;
    --ph2-epsilon) PH2_EPSILON="$2"; shift 2;;
    --action-prediction) ACTION_PREDICTION="$2"; shift 2;;
    --cycle-loss-enabled) CYCLE_LOSS_ENABLED="$2"; shift 2;;
    --cycle-loss-coef) CYCLE_LOSS_COEF="$2"; shift 2;;
    --z-prediction-enabled) Z_PREDICTION_ENABLED="$2"; shift 2;;
    --z-pred-loss-coef) Z_PRED_LOSS_COEF="$2"; shift 2;;
    --latent-mode) LATENT_MODE="$2"; shift 2;;
    --shared-prediction) SHARED_PREDICTION="$2"; shift 2;;
    --save-eval-checkpoints) SAVE_EVAL_CHECKPOINTS="$2"; shift 2;;
    --transformer-action) TRANSFORMER_ACTION="$2"; shift 2;;
    --transformer-window-size) TRANSFORMER_WINDOW_SIZE="$2"; shift 2;;
    --transformer-d-c) TRANSFORMER_D_C="$2"; shift 2;;
    --transformer-n-heads) TRANSFORMER_N_HEADS="$2"; shift 2;;
    --transformer-n-layers) TRANSFORMER_N_LAYERS="$2"; shift 2;;
    --transformer-recon-coef) TRANSFORMER_RECON_COEF="$2"; shift 2;;
    --transformer-pred-coef) TRANSFORMER_PRED_COEF="$2"; shift 2;;
    --transformer-cycle-coef) TRANSFORMER_CYCLE_COEF="$2"; shift 2;;
    --transformer-v2) TRANSFORMER_V2="$2"; shift 2;;
    --transformer-v3) TRANSFORMER_V3="$2"; shift 2;;
    --ph2-*|--PH2-*)
      echo "[ERROR] Unsupported PH2 flag: $1" >&2
      echo "        Supported PH2 flags: --ph2-ratio-stage1/2/3, --ph2-fixed-ind-prob, --ph2-epsilon, --action-prediction" >&2
      exit 1
      ;;
    --random-reset) RANDOM_RESET_OVERRIDE="$2"; shift 2;;
    --extra) EXTRA_ARGS+=("$2"); shift 2;;
    *)            echo "[WARN] Unknown arg: $1"; shift 1;;
  esac
done

export WANDB_PROJECT
export WANDB_ENTITY

# ==============================================================================
# 4) 실행 정보 로깅
# ==============================================================================

RUN_NAME="${EXPERIMENT}"

echo "==============================================================="
echo "  Run Name     : $RUN_NAME"
echo "  GPUs         : $CUDA_VISIBLE_DEVICES"

if [[ "$EXPERIMENT" == "rnn-fcp" || -n "$FCP_DIR" ]]; then
  if [[ "$SEEDS_EXPLICIT" == "1" ]]; then
    echo "  Seeds        : $NUM_SEEDS (explicit)"
  else
    echo "  Seeds        : (cfg default for FCP, 1)"
  fi
else
  echo "  Seeds        : $NUM_SEEDS"
fi
[[ -n "$SEED" ]] && echo "  Base Seed    : $SEED (override)"

echo "  Env Group    : $ENV_GROUP"
echo "  Layout       : $LAYOUT"
echo "  Experiment   : $EXPERIMENT"
echo "  W&B Project  : $WANDB_PROJECT"
echo "  W&B Entity   : $WANDB_ENTITY"

if [[ -n "$ITERATIONS_OVERRIDE" ]]; then
  echo "  Iterations   : $ITERATIONS_OVERRIDE (override)"
else
  echo "  Iterations   : (cfg default if defined)"
fi

[[ -n "$NOTES" ]]                  && echo "  Notes        : $NOTES"
[[ -n "$TAGS"  ]]                  && echo "  Extra Tags   : $TAGS"
[[ -n "$FCP_DIR" ]]                && echo "  FCP Pop Dir  : $FCP_DIR"
[[ -n "$ENV_DEVICE" ]]             && echo "  Env Device   : $ENV_DEVICE"
[[ "$CAST_OBS_BF16" == "1" ]]      && echo "  Obs DType    : bfloat16 (CAST_OBS_BF16)"
[[ -n "$MODEL_NUM_ENVS_OVERRIDE" ]]   && echo "  NUM_ENVS     : $MODEL_NUM_ENVS_OVERRIDE (override)"
[[ -n "$MODEL_NUM_STEPS_OVERRIDE" ]] && echo "  NUM_STEPS    : $MODEL_NUM_STEPS_OVERRIDE (override)"
[[ "$USE_PARTNER_MODELING" == "True" ]] && echo "  Partner Mod  : Enabled (Coef: $PRED_LOSS_COEF)"

echo "==============================================================="

# ==============================================================================
# 5) GPU 메모리 상태 확인 및 진단
# ==============================================================================

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi (selected GPUs: $CUDA_VISIBLE_DEVICES)"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader \
    | awk -v sel="${CUDA_VISIBLE_DEVICES}" '
        BEGIN{ split(sel, a, ","); for(i in a) sel_idx[a[i]]=1 }
        {
          split($0, f, ", ");
          idx=f[1];
          if (idx in sel_idx)
            print "  GPU " idx ": " f[2] ", total=" f[3] ", used=" f[4] ", free=" f[5];
        }'

  FREE=$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | awk -v sel="${CUDA_VISIBLE_DEVICES}" '
          BEGIN{ split(sel, a, ","); for(i in a) sel_idx[a[i]]=1 }
          { if (NR-1 in sel_idx) print $1 }' \
      | head -n1
  )

  if [[ -n "$FREE" && "$FREE" -lt 512 ]]; then
    echo "[WARN] 선택된 GPU의 여유 메모리가 ${FREE} MiB 입니다 (< 512 MiB)."
    echo "       cuDNN 초기화 실패 가능성이 높습니다."
    echo "       다른 GPU를 지정하세요: --gpus <id> (예: --gpus 1)"
  fi
fi

# ==============================================================================
# 6) W&B 태그 구성 (실험/환경/레이아웃 + 사용자 태그)
# ==============================================================================

# 1) 기본 태그 리스트 구성
RAW_TAGS=("${EXPERIMENT}" "${ENV_GROUP}" "${LAYOUT}" "${ITERATIONS_OVERRIDE}")

# 2) --tags "a,b c" 같이 들어온 사용자 태그 파싱 (콤마/스페이스 모두 구분자)
if [[ -n "$TAGS" ]]; then
  IFS=', ' read -r -a EXTRA_TAGS <<< "$TAGS"
  for tag in "${EXTRA_TAGS[@]}"; do
    [[ -z "$tag" ]] && continue
    RAW_TAGS+=("$tag")
  done
fi

# 3) Hydra 문법용 문자열로 직렬화: ['t1','t2','t3']
serialize_tags() {
  local out="" sep=""
  for t in "$@"; do
    # 작은따옴표 이스케이프: a'b -> 'a'"'"'b'
    local esc=${t//\'/\'\"\'\"\'}
    out+="${sep}'${esc}'"
    sep=","
  done
  printf "%s" "$out"
}

TAGS_SERIALIZED=$(serialize_tags "${RAW_TAGS[@]}")
WANDB_TAGS_ARG="+wandb.tags=[${TAGS_SERIALIZED}]"

# ==============================================================================
# 7) 학습 실행 (Hydra 인자 구성 및 main 실행)
# ==============================================================================

PY_ARGS=(
  "+experiment=${EXPERIMENT}"
  "+env=${ENV_GROUP}"
  "+wandb.name=${RUN_NAME}"
  "+wandb.project=${WANDB_PROJECT}"
  "+wandb.entity=${WANDB_ENTITY}"
  "+wandb.notes=${NOTES}"
  "${WANDB_TAGS_ARG}"
)

# NUM_ITERATIONS override: 값이 존재하고 숫자이며, 1 초과인 경우만 적용
if [[ -n "${ITERATIONS_OVERRIDE:-}" && "${ITERATIONS_OVERRIDE}" =~ ^[0-9]+$ && "${ITERATIONS_OVERRIDE}" -gt 1 ]]; then
  echo "[INFO] Using NUM_ITERATIONS override from CLI: ${ITERATIONS_OVERRIDE}"
  PY_ARGS+=("NUM_ITERATIONS=${ITERATIONS_OVERRIDE}")
else
  echo "[INFO] NUM_ITERATIONS override not applied (must be integer > 1)."
fi

# NUM_SEEDS 처리: FCP 실험은 cfg 기본 1, --seeds로만 override
if [[ "$EXPERIMENT" == "rnn-fcp" || -n "$FCP_DIR" ]]; then
  if [[ "$SEEDS_EXPLICIT" == "1" ]]; then
    PY_ARGS+=("NUM_SEEDS=${NUM_SEEDS}")
  fi
else
  PY_ARGS+=("NUM_SEEDS=${NUM_SEEDS}")
fi

if [[ -n "$SEED" ]]; then
  PY_ARGS+=("SEED=${SEED}")
fi

# FCP population 디렉토리 override
if [[ -n "$FCP_DIR" ]]; then
  PY_ARGS+=("+FCP=${FCP_DIR}")
  PY_ARGS+=("+FCP_DEVICE=${FCP_DEVICE}")
fi

# env 그룹이 original일 때만 layout override
if [[ "${ENV_GROUP}" == "original" ]]; then
  PY_ARGS+=("env.ENV_KWARGS.layout=${LAYOUT}")
fi

# [PH1] OV2 환경에서만 partial view(agent_view_size=2) 강제
# - OV1(classic) 레이아웃은 full view 유지
is_ph1_run=false
if [[ "${EXPERIMENT}" == "rnn-ph1" || "${PH1_ENABLED}" == "True" || "${PH1_ENABLED}" == "true" || "${PH1_ENABLED}" == "1" ]]; then
  is_ph1_run=true
fi

is_ov2_env=false
case "${ENV_GROUP}" in
  grounded_*|demo_cook_*|test_time_*)
    is_ov2_env=true
    ;;
esac

if [[ "${is_ph1_run}" == "true" && "${is_ov2_env}" == "true" ]]; then
  PY_ARGS+=("env.ENV_KWARGS.agent_view_size=2")
fi

# 새 옵션 전달: ENV_DEVICE / CAST_OBS_BF16 / 배치 크기 오버라이드
if [[ -n "$ENV_DEVICE" ]]; then
  PY_ARGS+=("+ENV_DEVICE=${ENV_DEVICE}")
fi
if [[ "$CAST_OBS_BF16" == "1" ]]; then
  PY_ARGS+=("+CAST_OBS_BF16=True")
fi
if [[ -n "$MODEL_NUM_ENVS_OVERRIDE" ]]; then
  PY_ARGS+=("model.NUM_ENVS=${MODEL_NUM_ENVS_OVERRIDE}")
fi
if [[ -n "$MODEL_NUM_STEPS_OVERRIDE" ]]; then
  PY_ARGS+=("model.NUM_STEPS=${MODEL_NUM_STEPS_OVERRIDE}")
fi

# set -u 대응: 선택적 변수 기본값 초기화
: "${PRED_LOSS_COEF:=}"
: "${PH1_BLOCK_TYPE:=}"
: "${PH1_BETA:=}"
: "${PH1_BETA_SCHEDULE_ENABLED:=}"
: "${PH1_BETA_START:=}"
: "${PH1_BETA_END:=}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=}"
: "${PH1_OMEGA:=}"
: "${PH1_SIGMA:=}"
: "${PH1_DISTANCE_THRESHOLD:=}"
: "${PH1_POOL_SIZE:=}"
: "${PH1_NORMAL_PROB:=}"
: "${PH1_MULTI_PENALTY_ENABLED:=}"
: "${PH1_MAX_PENALTY_COUNT:=}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=}"
: "${PH1_ENABLED:=}"
: "${PH1_EPSILON:=}"
: "${PH1_WARMUP_STEPS:=}"
: "${PH2_RATIO_STAGE1:=}"
: "${PH2_RATIO_STAGE2:=}"
: "${PH2_RATIO_STAGE3:=}"
: "${PH2_FIXED_IND_PROB:=}"
: "${PH2_EPSILON:=}"
: "${ACTION_PREDICTION:=}"
: "${SAVE_EVAL_CHECKPOINTS:=}"
: "${TRANSFORMER_ACTION:=}"
: "${TRANSFORMER_WINDOW_SIZE:=}"
: "${TRANSFORMER_D_C:=}"
: "${TRANSFORMER_RECON_COEF:=}"
: "${TRANSFORMER_PRED_COEF:=}"
: "${TRANSFORMER_CYCLE_COEF:=}"
: "${TRANSFORMER_V2:=}"
: "${TRANSFORMER_V3:=}"
: "${SHARED_PREDICTION:=}"
: "${RANDOM_RESET_FLAG:=}"
if [[ -z "${EXTRA_ARGS+x}" ]]; then EXTRA_ARGS=(); fi

if [[ "$USE_PARTNER_MODELING" == "True" ]]; then
  PY_ARGS+=("USE_PARTNER_MODELING=True")
fi
if [[ -n "$PRED_LOSS_COEF" ]]; then
  PY_ARGS+=("PRED_LOSS_COEF=$PRED_LOSS_COEF")
fi

# PH1 Arguments (rnn-ph1.yaml에 이미 정의됨 -> + 제거)
if [[ -n "$PH1_BLOCK_TYPE" ]]; then
  PY_ARGS+=("BLOCK_TARGET_TYPE=$PH1_BLOCK_TYPE")
fi
if [[ -n "$PH1_BETA" ]]; then
  PY_ARGS+=("PH1_BETA=$PH1_BETA")
fi
if [[ -n "$PH1_BETA_SCHEDULE_ENABLED" ]]; then
  PY_ARGS+=("PH1_BETA_SCHEDULE_ENABLED=$PH1_BETA_SCHEDULE_ENABLED")
fi
if [[ -n "$PH1_BETA_START" ]]; then
  PY_ARGS+=("PH1_BETA_START=$PH1_BETA_START")
fi
if [[ -n "$PH1_BETA_END" ]]; then
  PY_ARGS+=("PH1_BETA_END=$PH1_BETA_END")
fi
if [[ -n "$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS" ]]; then
  PY_ARGS+=("PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS")
fi
if [[ -n "$PH1_OMEGA" ]]; then
  PY_ARGS+=("PH1_OMEGA=$PH1_OMEGA")
fi
if [[ -n "$PH1_SIGMA" ]]; then
  PY_ARGS+=("PH1_SIGMA=$PH1_SIGMA")
fi
if [[ -n "$PH1_DISTANCE_THRESHOLD" ]]; then
  PY_ARGS+=("PH1_DISTANCE_THRESHOLD=$PH1_DISTANCE_THRESHOLD")
fi
if [[ -n "$PH1_POOL_SIZE" ]]; then
  PY_ARGS+=("PH1_POOL_SIZE=$PH1_POOL_SIZE")
fi
if [[ -n "$PH1_NORMAL_PROB" ]]; then
  PY_ARGS+=("PH1_NORMAL_PROB=$PH1_NORMAL_PROB")
fi
if [[ -n "$PH1_MULTI_PENALTY_ENABLED" ]]; then
  PY_ARGS+=("PH1_MULTI_PENALTY_ENABLED=$PH1_MULTI_PENALTY_ENABLED")
fi
if [[ -n "$PH1_MAX_PENALTY_COUNT" ]]; then
  PY_ARGS+=("PH1_MAX_PENALTY_COUNT=$PH1_MAX_PENALTY_COUNT")
fi
if [[ -n "$PH1_MULTI_PENALTY_SINGLE_WEIGHT" ]]; then
  PY_ARGS+=("PH1_MULTI_PENALTY_SINGLE_WEIGHT=$PH1_MULTI_PENALTY_SINGLE_WEIGHT")
fi
if [[ -n "$PH1_MULTI_PENALTY_OTHER_WEIGHT" ]]; then
  PY_ARGS+=("PH1_MULTI_PENALTY_OTHER_WEIGHT=$PH1_MULTI_PENALTY_OTHER_WEIGHT")
fi
if [[ -n "$PH1_ENABLED" ]]; then
  PY_ARGS+=("PH1_ENABLED=$PH1_ENABLED")
fi
if [[ -n "$PH1_EPSILON" ]]; then
  PY_ARGS+=("PH1_EPSILON=$PH1_EPSILON")
fi
if [[ -n "$PH1_WARMUP_STEPS" ]]; then
  PY_ARGS+=("PH1_WARMUP_STEPS=$PH1_WARMUP_STEPS")
fi

# PH1 population arguments
if [[ -n "$PH1_POP_DIR" ]]; then
  PY_ARGS+=("PH1_POP_DIR=$PH1_POP_DIR")
fi
if [[ -n "$PH1_POP_NAME" ]]; then
  PY_ARGS+=("PH1_POP_NAME=$PH1_POP_NAME")
fi

# PH2 Arguments
if [[ -n "$PH2_RATIO_STAGE1" ]]; then
  PY_ARGS+=("PH2_RATIO_STAGE1=$PH2_RATIO_STAGE1")
fi
if [[ -n "$PH2_RATIO_STAGE2" ]]; then
  PY_ARGS+=("PH2_RATIO_STAGE2=$PH2_RATIO_STAGE2")
fi
if [[ -n "$PH2_RATIO_STAGE3" ]]; then
  PY_ARGS+=("PH2_RATIO_STAGE3=$PH2_RATIO_STAGE3")
fi
if [[ -n "$PH2_FIXED_IND_PROB" ]]; then
  PY_ARGS+=("PH2_FIXED_IND_PROB=$PH2_FIXED_IND_PROB")
fi
if [[ -n "$PH2_EPSILON" ]]; then
  PY_ARGS+=("PH2_EPSILON=$PH2_EPSILON")
fi
if [[ -n "$ACTION_PREDICTION" ]]; then
  PY_ARGS+=("ACTION_PREDICTION=$ACTION_PREDICTION")
fi
if [[ -n "$CYCLE_LOSS_ENABLED" ]]; then
  PY_ARGS+=("CYCLE_LOSS_ENABLED=$CYCLE_LOSS_ENABLED")
fi
if [[ -n "$CYCLE_LOSS_COEF" ]]; then
  PY_ARGS+=("CYCLE_LOSS_COEF=$CYCLE_LOSS_COEF")
fi
if [[ -n "$Z_PREDICTION_ENABLED" ]]; then
  PY_ARGS+=("Z_PREDICTION_ENABLED=$Z_PREDICTION_ENABLED")
fi
if [[ -n "$Z_PRED_LOSS_COEF" ]]; then
  PY_ARGS+=("Z_PRED_LOSS_COEF=$Z_PRED_LOSS_COEF")
fi
if [[ -n "$LATENT_MODE" ]]; then
  PY_ARGS+=("LATENT_MODE=$LATENT_MODE")
fi
if [[ -n "$SHARED_PREDICTION" ]]; then
  PY_ARGS+=("SHARED_PREDICTION=$SHARED_PREDICTION")
fi
if [[ -n "$SAVE_EVAL_CHECKPOINTS" ]]; then
  PY_ARGS+=("SAVE_EVAL_CHECKPOINTS=$SAVE_EVAL_CHECKPOINTS")
fi
if [[ -n "$TRANSFORMER_ACTION" ]]; then
  PY_ARGS+=("++TRANSFORMER_ACTION=$TRANSFORMER_ACTION")
fi
if [[ -n "$TRANSFORMER_WINDOW_SIZE" ]]; then
  PY_ARGS+=("++TRANSFORMER_WINDOW_SIZE=$TRANSFORMER_WINDOW_SIZE")
fi
if [[ -n "$TRANSFORMER_D_C" ]]; then
  PY_ARGS+=("++TRANSFORMER_D_C=$TRANSFORMER_D_C")
fi
if [[ -n "$TRANSFORMER_N_HEADS" ]]; then
  PY_ARGS+=("++TRANSFORMER_N_HEADS=$TRANSFORMER_N_HEADS")
fi
if [[ -n "$TRANSFORMER_N_LAYERS" ]]; then
  PY_ARGS+=("++TRANSFORMER_N_LAYERS=$TRANSFORMER_N_LAYERS")
fi
if [[ -n "$TRANSFORMER_RECON_COEF" ]]; then
  PY_ARGS+=("++TRANSFORMER_RECON_COEF=$TRANSFORMER_RECON_COEF")
fi
if [[ -n "$TRANSFORMER_PRED_COEF" ]]; then
  PY_ARGS+=("++TRANSFORMER_PRED_COEF=$TRANSFORMER_PRED_COEF")
fi
if [[ -n "$TRANSFORMER_CYCLE_COEF" ]]; then
  PY_ARGS+=("++TRANSFORMER_CYCLE_COEF=$TRANSFORMER_CYCLE_COEF")
fi
if [[ -n "$TRANSFORMER_V2" ]]; then
  PY_ARGS+=("++TRANSFORMER_V2=$TRANSFORMER_V2")
fi
if [[ -n "$TRANSFORMER_V3" ]]; then
  PY_ARGS+=("++TRANSFORMER_V3=$TRANSFORMER_V3")
fi

# random_reset override (ToyCoop procedural generation)
if [[ -n "${RANDOM_RESET_OVERRIDE:-}" ]]; then
  PY_ARGS+=("env.ENV_KWARGS.random_reset=${RANDOM_RESET_OVERRIDE}")
fi

# E3T epsilon override (rnn-ph1.yaml에 정의됨)
if [[ -v E3T_EPSILON && -n "$E3T_EPSILON" ]]; then
  PY_ARGS+=("E3T_EPSILON=${E3T_EPSILON}")
fi

# STL Anchor override
if [[ "$ANCHOR_ENABLED" == "1" ]]; then
  PY_ARGS+=("model.anchor=True")
fi

# ====================================================
# LD_LIBRARY_PATH / XLA_FLAGS를 해제하여 라이브러리 충돌 회피
cd "${REPO_ROOT}"

# 1) venv 활성화
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${LEGACY_VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${LEGACY_VENV_DIR}/bin/activate"
else
  echo "[ERROR] venv activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

# 2) WandB 로그인
if [[ -f "${REPO_ROOT}/wandb_info/wandb_api_key" ]]; then
  WANDB_API_KEY=$(cat "${REPO_ROOT}/wandb_info/wandb_api_key")
  export WANDB_API_KEY
  echo "[INFO] WandB API key loaded from ${REPO_ROOT}/wandb_info/wandb_api_key"
  wandb login "$WANDB_API_KEY"
else
  echo "[WARN] WandB API key file not found at ${REPO_ROOT}/wandb_info/wandb_api_key"
fi

# 3) 파이썬 진단 (가상환경 활성화 후)
RUNTIME_XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${XLA_CPU_THREAD_LIMIT}"
echo "[INFO] Runtime XLA_FLAGS: ${RUNTIME_XLA_FLAGS}"

env -u LD_LIBRARY_PATH XLA_FLAGS="${RUNTIME_XLA_FLAGS}" python - <<'PY' 2>&1 | filter_ptx
import os
import jax
import sys

print("[DEBUG] Python executable:", sys.executable)
print("[DEBUG] Python version:", sys.version)
print("[DEBUG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG] JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"))
print("[DEBUG] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

try:
    devices = jax.devices()
    print("[JAX] devices:", devices)
except Exception as e:
    print("[JAX] Error initializing devices:", e)
PY

# 3) 실험 실행
cd "${PROJECT_DIR}"

# import 경로: 현재 프로젝트 디렉토리 + JaxMARL
export PYTHONPATH="${PROJECT_DIR}:${REPO_ROOT}/JaxMARL"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"

# 추가 Hydra override (--extra 플래그)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_ARGS[@]}"; do
    PY_ARGS+=("${arg}")
  done
fi

env -u LD_LIBRARY_PATH XLA_FLAGS="${RUNTIME_XLA_FLAGS}" \
  python overcooked_v2_experiments/ppo/main.py "${PY_ARGS[@]}" 2>&1 | filter_ptx
