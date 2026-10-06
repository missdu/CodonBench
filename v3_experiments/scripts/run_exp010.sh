#!/bin/bash
# EXP-010 后台启动器（纯 CPU，base 环境）
# 教训：ssh 前台运行会被 SIGTERM 断掉，且 `| tail` 会缓冲输出导致丢失。
# 必须：写文件 + setsid + nohup。
cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}"
source "${CODONBENCH_SERVER_HOME:-/path/to/your/server-home}/anaconda3/etc/profile.d/conda.sh"
conda activate base
setsid nohup python -u WB-exp010-reviewstatus.py > wb_exp010.out 2>&1 < /dev/null &
echo "STARTED pid=$!"
exit 0
