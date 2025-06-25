#!/bin/bash

# 启动时间
cd "$(dirname "$0")"
echo "Start time: $(date)"

# 日志文件按时间命名
LOGFILE="main_$(date +%Y%m%d_%H%M%S).log"

# 启动 main.py
echo "Running: $(realpath main.py)"
nohup python -u main.py > $LOGFILE 2>&1 &
echo $! > main.pid
echo "main.py running in background. Log: $LOGFILE"
