#!/bin/bash
# run_cuquantum_bench.sh

echo "🔎 Locating cuQuantum libraries..."
CUTENSOR_LIB=$(find /home/jetson/.local -name "libcutensor.so.1" | head -n 1 | xargs dirname)

if [ -z "$CUTENSOR_LIB" ]; then
    # Fallback to finding .so.2 if .so.1 not found (symlink might be in a different place)
    CUTENSOR_LIB=$(find /home/jetson/.local -name "libcutensor.so*" | head -n 1 | xargs dirname)
fi

CUQUANTUM_LIB=$(find /home/jetson/.local -name "custatevec.so" | head -n 1 | xargs dirname) # custatevec is inside cuquantum/lib usually
if [ -z "$CUQUANTUM_LIB" ]; then
    CUQUANTUM_LIB="/home/jetson/.local/lib/python3.8/site-packages/cuquantum/lib"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUTENSOR_LIB:$CUQUANTUM_LIB
echo "✅ LD_LIBRARY_PATH set to:Str$LD_LIBRARY_PATH"

echo "🚀 Starting Native cuQuantum Benchmark (28, 30, 32, 33, 34 qubits)..."
nohup python3 -u code/cuquantum_sim.py 35 > data/cuquantum_35q.log 2>&1 &
echo "✅ Benchmark running in background. PID: $!"
echo "📄 Logs: data/cuquantum_35q.log"
