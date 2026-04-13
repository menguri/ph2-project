#!/usr/bin/env bash
# =============================================================================
# final/ → webapp/models/ 동기화
# final 폴더의 ckpt_final만 webapp으로 복사 (sp, e3t, fcp, mep, gamma, ph2)
# hsp 제거
# =============================================================================
set -uo pipefail

ROOT="/home/mlic/mingukang/ph2-project"
WEBAPP_MODELS="${ROOT}/webapp/models"
FINAL_BASELINE="${ROOT}/final/baseline"
FINAL_PH2="${ROOT}/final/ph2"

LAYOUTS=(cramped_room counter_circuit coord_ring asymm_advantages forced_coord)
BASELINE_ALGOS=(sp e3t fcp mep gamma)
MAX_RUNS=10  # webapp에 복사할 최대 run 수

echo "=== Final → Webapp 동기화 시작 ==="

for layout in "${LAYOUTS[@]}"; do
  echo ""
  echo "======== ${layout} ========"

  # --- hsp 제거 ---
  hsp_dir="${WEBAPP_MODELS}/${layout}/hsp"
  if [ -d "$hsp_dir" ]; then
    echo "  [삭제] hsp"
    rm -rf "$hsp_dir"
  fi

  # --- baseline 알고리즘 ---
  for algo in "${BASELINE_ALGOS[@]}"; do
    # final에서 해당 algo run 디렉토리 찾기
    fin_run=$(ls -d ${FINAL_BASELINE}/${layout}/*_${algo}_* ${FINAL_BASELINE}/${layout}/*_${algo} ${FINAL_BASELINE}/${layout}/*_${algo}-* 2>/dev/null | head -1)
    if [ -z "$fin_run" ]; then
      echo "  [경고] ${algo}: final에 없음"
      continue
    fi

    # webapp 대상 디렉토리
    wa_algo_dir="${WEBAPP_MODELS}/${layout}/${algo}"

    # 기존 webapp 디렉토리 삭제 후 새로 생성
    rm -rf "$wa_algo_dir"
    mkdir -p "$wa_algo_dir"

    # final의 run_N/ckpt_final → webapp의 runN/ckpt_final
    copied=0
    for i in $(seq 0 $((MAX_RUNS - 1))); do
      src="${fin_run}/run_${i}/ckpt_final"
      if [ -d "$src" ]; then
        dst="${wa_algo_dir}/run${i}/ckpt_final"
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        copied=$((copied + 1))
      fi
    done
    echo "  [복사] ${algo}: ${copied} runs (from $(basename $fin_run))"
  done

  # --- ph2 ---
  # final/ph2에서 가장 최신 run 디렉토리 사용
  fin_ph2=$(ls -dt ${FINAL_PH2}/${layout}/* 2>/dev/null | head -1)
  if [ -z "$fin_ph2" ]; then
    echo "  [경고] ph2: final에 없음"
    continue
  fi

  wa_ph2_dir="${WEBAPP_MODELS}/${layout}/ph2"
  rm -rf "$wa_ph2_dir"
  mkdir -p "$wa_ph2_dir"

  # ph2는 eval/run_N/ckpt_final 구조
  copied=0
  for i in $(seq 0 $((MAX_RUNS - 1))); do
    src="${fin_ph2}/eval/run_${i}/ckpt_final"
    if [ ! -d "$src" ]; then
      # eval 없으면 run_N/ckpt_final 시도
      src="${fin_ph2}/run_${i}/ckpt_final"
    fi
    if [ -d "$src" ]; then
      dst="${wa_ph2_dir}/run${i}/ckpt_final"
      mkdir -p "$(dirname "$dst")"
      cp -r "$src" "$dst"
      copied=$((copied + 1))
    fi
  done
  echo "  [복사] ph2: ${copied} runs (from $(basename $fin_ph2))"
done

echo ""
echo "=== 동기화 완료 ==="

# 검증
echo ""
echo "=== 검증: webapp 모델 현황 ==="
for layout in "${LAYOUTS[@]}"; do
  algos=""
  for algo in sp e3t fcp mep gamma ph2; do
    d="${WEBAPP_MODELS}/${layout}/${algo}"
    if [ -d "$d" ]; then
      n=$(ls -d ${d}/run*/ckpt_final 2>/dev/null | wc -l)
      algos="${algos} ${algo}(${n})"
    fi
  done
  echo "${layout}:${algos}"
done
