#!/usr/bin/env bash
# 跑一次并归档日志：simulation/outputs/history/<时间戳>_pred_log.csv / _det_log.csv
#
# 位置：<仓库根>/scripts/run_and_save.sh
# 用法：
#   ./scripts/run_and_save.sh [训练程序参数...]
#   （在仓库根目录执行；脚本会自己切到仓库根）
#
# 全程相对路径：仓库根 = 本脚本所在目录的上一级，不依赖 $HOME 与当前工作目录。
set -eo pipefail

# 本脚本在 <仓库根>/scripts/ 下，所以仓库根 = scripts 的上一级
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO"

OUT="simulation/outputs"
HIST="$OUT/history"

if [ ! -x scripts/run_training.sh ]; then
  echo "错误：$REPO/scripts/run_training.sh 不存在或不可执行" >&2
  exit 1
fi

mkdir -p "$HIST"

STAMP="$(date +%Y%m%d_%H%M%S)"
./scripts/run_training.sh "$@" || true

for n in pred_log det_log; do
  if [ -f "$OUT/$n.csv" ]; then
    cp "$OUT/$n.csv" "$HIST/${STAMP}_${n}.csv"
    echo "已归档: $HIST/${STAMP}_${n}.csv ($(wc -l < "$OUT/$n.csv") 行)"
  fi
done
