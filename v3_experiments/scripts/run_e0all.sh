#!/bin/bash
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}"
source "${CODONBENCH_SERVER_HOME:-/path/to/your/server-home}/anaconda3/etc/profile.d/conda.sh"
conda activate base
setsid nohup python -u WB-e0-run-all.py > wb_e0_all.out 2>&1 < /dev/null &
sleep 2
echo "STARTED pid=$(ps -eo pid,cmd | grep '[W]B-e0-run-all' | awk '{print $1}')"
