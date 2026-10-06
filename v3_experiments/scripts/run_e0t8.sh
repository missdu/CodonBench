#!/bin/bash
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}"
source "${CODONBENCH_SERVER_HOME:-/path/to/your/server-home}/anaconda3/etc/profile.d/conda.sh"
conda activate base
python -u WB-e0-t8-seqhash.py 2>&1 | tee wb_e0_t8.out
