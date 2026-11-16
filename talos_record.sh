#!/usr/bin/env bash
# 录一段仿真器真值到 history/，文件名带时间戳
set -euo pipefail
B=${REPO:-$HOME/RMStage2/HNU_RM_Vision}
TS=$(date +%Y%m%d_%H%M%S)
OUT="$B/simulation/outputs/history/${TS}_truth.csv"
mkdir -p "$(dirname "$OUT")"
echo "记录到 $OUT   （Ctrl-C 结束）"
exec python3 "$(dirname "$0")/talos_watch.py" --csv "$OUT" --hz 20
