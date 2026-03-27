#!/usr/bin/env bash
# runs/ 디렉토리를 알고리즘·레이아웃별로 정리하여 출력.
# run_metadata.json이 있으면 파싱, 없으면 디렉토리 이름에서 추정.
#
# 사용: ./list_runs.sh [runs_dir]
#       ./list_runs.sh ../runs
#       ./list_runs.sh ../../ph2/runs

RUNS_DIR="${1:-../runs}"

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "Error: $RUNS_DIR not found" >&2
  exit 1
fi

printf "%-65s  %-12s  %-20s  %s\n" "RUN" "ALG" "LAYOUT" "PARAMS"
printf "%s\n" "$(printf '%.0s-' {1..120})"

for dir in "$RUNS_DIR"/*/; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  meta="$dir/run_metadata.json"

  if [[ -f "$meta" ]]; then
    # run_metadata.json에서 파싱
    alg=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('alg_name','?'))" 2>/dev/null)
    layout=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('layout','?'))" 2>/dev/null)
    gru=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('model',{}).get('GRU_HIDDEN_DIM',''))" 2>/dev/null)
    seeds=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('num_seeds',''))" 2>/dev/null)
    params="h=${gru} seeds=${seeds}"
  else
    # 디렉토리 이름에서 추정
    alg=$(echo "$name" | rev | cut -d_ -f1 | rev)
    layout=$(echo "$name" | cut -d_ -f3)
    params="(no metadata)"
  fi

  printf "%-65s  %-12s  %-20s  %s\n" "$name" "$alg" "$layout" "$params"
done
