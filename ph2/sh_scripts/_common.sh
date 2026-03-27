#!/usr/bin/env bash
# 공통 함수 — 각 run_factory_*.sh에서 source "$(dirname "$0")/_common.sh"

# 호출자의 디렉토리를 기준으로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_BASE="${PROJECT_DIR}/runs"

run_experiment() {
  local gpus=$1
  local exp=$2
  local layout=$3
  local seeds=${4:-1}
  shift 4
  local extra_args=("$@")

  echo "===== [${exp}] gpus=${gpus} layout=${layout} seeds=${seeds} ====="
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ./run_user_wandb.sh \
      --exp "${exp}" \
      --env "${layout}" \
      --layout "${layout}" \
      --gpus "${gpus}" \
      --seeds "${seeds}" \
      "${extra_args[@]}"
}

find_latest_dir() {
  local name=$1
  find "${RUNS_BASE}" -maxdepth 3 -type d -name "${name}" \
    -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | awk '{print $2}'
}
