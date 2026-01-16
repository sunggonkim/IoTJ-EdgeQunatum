#!/bin/bash
cd /home/jetson/skim/edgeQuantum-iotj
echo "Starting Unified Benchmark..."
nohup python3 -u code/unified_benchmark.py > data/unified_benchmark_post_reboot.log 2>&1 &
echo "Benchmark running in background. PID: $!"
echo "Tailing log (Ctrl+C to stop tailing, process will continue)..."
tail -f data/unified_benchmark_post_reboot.log
