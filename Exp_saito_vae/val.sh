#!/usr/bin/env bash

# 路径
PY=/baksv/CIGIT/GXN_Liuxy/CondaEnvs/envs/Timelib/bin/python
VAL_SCRIPT="/baksv/CIGIT/GXN_Liuxy/Time-Series-Library/vae/Exp_saito_vae/val.py"
WEIGHT_DIR="/baksv/CIGIT/GXN_Liuxy/Time-Series-Library/vae/Exp_saito_vae/weight"
LOG_FILE="eval_results.log"

VIDEOS=("video20220722" "video20220729" "video20220801" "video20221110" "video20230315")
MODES=("Surgery-out" "Senquence-out")

# 初始化总日志
{
  echo "=== Evaluation Log ==="
  echo "Start time: $(date)"
  echo "=========================="
} > "$LOG_FILE"

for v in "${VIDEOS[@]}"; do
  for mode in "${MODES[@]}"; do
    # 选择权重
    if [[ "$mode" == "Surgery-out" ]]; then
      MODEL_PATH="${WEIGHT_DIR}/vae_${v}_best.pth"
    else
      MODEL_PATH="${WEIGHT_DIR}/vae_Sequence-out_best.pth"
    fi

    INDIV_LOG="log_${v}_${mode}.txt"

    # 记录开始
    {
      echo -e "\n========== Start Evaluation: ${v} | Mode: ${mode} =========="
      echo "Start time: $(date)"
    } >> "$INDIV_LOG"

    # 顺序执行评估；只写 stdout ➜ log，stderr 仍输出到终端
    "${PY}" - <<PYCODE > "$INDIV_LOG"
import sys
sys.path.append("$(dirname "$VAL_SCRIPT")")
from val import evaluate_model
evaluate_model("${MODEL_PATH}", mode="${mode}", surgery="${v}")
PYCODE

    # 记录结束
    {
      echo "End time: $(date)"
      echo -e "========== End Evaluation: ${v} | Mode: ${mode} ==========\n"
    } >> "$INDIV_LOG"

  done
done

