#!/bin/bash
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
source "${CODONBENCH_SERVER_HOME:-/path/to/your/server-home}/anaconda3/etc/profile.d/conda.sh"
conda activate copra_h
cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}"
nohup python -u WB-mispath-multimodel.py > wb_mispath_multi.out 2>&1 &
echo "STARTED pid=$!"
exit 0
