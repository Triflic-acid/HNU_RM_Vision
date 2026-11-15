#!/usr/bin/env bash
# 跑一次并归档日志：outputs/history/<时间戳>_pred.csv / _det.csv
# 用法：./run_and_save.sh [附加参数...]
set -eo pipefail
REPO=${REPO:-$HOME/RMStage2/HNU_RM_Vision}
OUT="$REPO/simulation/outputs"
HIST="$OUT/history"
mkdir -p "$HIST"

STAMP=$(date +%Y%m%d_%H%M%S)
cd "$REPO"
./scripts/run_training.sh "$@" || true

for n in pred_log det_log; do
  if [ -f "$OUT/$n.csv" ]; then
    cp "$OUT/$n.csv" "$HIST/${STAMP}_${n}.csv"
    echo "已归档: $HIST/${STAMP}_${n}.csv ($(wc -l < "$OUT/$n.csv") 行)"
  fi
done
