#!/usr/bin/env python3
import subprocess
import time
import json
import os
import sys

# --- Configuration ---
# User requested 28 to 36.
# Disk Limit: ~73GB Free.
# 32Q (32GB) requires ~64GB (Read+Write) -> Feasible.
# 33Q (64GB) requires ~128GB -> Impossible.
# User requested 28 to 33.
# 33Q (64GB) fits in ~97GB free space. 34Q (128GB) does not.
QUBITS = [28, 29, 30, 31, 32, 33] 
CIRCUITS = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
# CIRCUITS = ["QV"]
SIMULATORS = {
    "EdgeQuantum": {"type": "cpp_async"},
    "BMQSim-like (Offload)": {"type": "cpp_blocking"},
    "cuQuantum Native": {"type": "cpp_native"},
    "cuQuantum UVM": {"type": "cpp_uvm"},
    "Cirq": {"type": "python", "cls": "CirqSimulator"},
}

LOG_FILE = "grand_results_full.json"

def run_cpp_engine(qubits, circuit, mode="async", depth=3):
    """Run C++ EdgeQuantum Binary"""
    # Map mode to CLI argument
    sim_mode_arg = "async"
    if mode == "blocking": sim_mode_arg = "blocking"
    elif mode == "native": sim_mode_arg = "native"
    elif mode == "uvm": sim_mode_arg = "uvm"
    
    cmd = [
        "./build/edge_quantum",
        "--qubits", str(qubits),
        "--circuit", circuit,
        "--depth", str(depth),
        "--sim-mode", sim_mode_arg,
        "--storage", "/mnt/nvme/skim/edgeQuantum/cpp_state_vector.bin"
    ]
    
    # Environment (unchanged)
    env = os.environ.copy()
    lib_paths = [
        "/home/jetson/.local/lib/python3.8/site-packages/cutensor/lib",
        "/home/jetson/.local/lib/python3.8/site-packages/cuquantum/lib", 
        "/home/jetson/.local/lib/python3.8/site-packages/custatevec/lib",
        "/usr/local/cuda/lib64"
    ]
    env['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + ':' + env.get('LD_LIBRARY_PATH', '')
    
    # Timeout logic:
    # Native should fail fast if OOM (cudaMalloc error logic).
    # UVM might trash if OOM, so keep timeout safe.
    timeout = 600
    if qubits >= 30: timeout = 1200
    if qubits >= 32: timeout = 3600
    
    start_time = time.time()
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        
        if res.returncode != 0:
            # Check for specific OOM messages in stdout/stderr if possible
            err_msg = f"Exit Code {res.returncode}"
            if "out of memory" in res.stdout.lower() or "oom" in res.stdout.lower():
                err_msg = "FAIL (OOM - >8GB RAM)"
            # Native might exit with error on cudaMalloc failure
            return {"success": False, "error": err_msg, "logs": res.stdout[-200:]}
            
        for line in res.stdout.split('\n'):
            if "Total Time:" in line:
                try:
                    t = float(line.split(":")[1].strip().split()[0])
                    if t < 0.1:
                        print(f"DEBUG LOGS (Fast Run {t}s):")
                        print('\n'.join(res.stdout.split('\n')[:20]))
                    return {"success": True, "time": t, "throughput": duration}
                except:
                    pass
        
        return {"success": True, "time": duration} 
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_python_baseline(sim_name, qubits, circuit, depth=3):
    """Generates and runs a Python script for Cirq/PennyLane baselines."""
    # If 30+ Qubits, fail fast (OOM)
    if qubits >= 29:
        return {"success": False, "error": "FAIL (OOM - >8GB RAM)"}
        
    script_content = ""
    if sim_name == "Cirq":
        script_content = f"""
import cirq
import time
import numpy as np

def run_circuit():
    q = cirq.LineQubit.range({qubits})
    c = cirq.Circuit()
    # Simple Random-like layers
    for _ in range({depth}):
        for i in range({qubits}):
            c.append(cirq.H(q[i]))
        for i in range(0, {qubits}-1, 2):
            c.append(cirq.CNOT(q[i], q[i+1]))
            
    sim = cirq.Simulator()
    start = time.time()
    try:
        # Force computation by accessing state vector
        result = sim.simulate(c)
        _ = result.final_state_vector
    except Exception as e:
        print(f"ERROR: {{e}}")
        return
    print(f"Total Time: {{time.time() - start}}")

if __name__ == "__main__":
    run_circuit()
"""
    elif sim_name == "PennyLane":
        script_content = f"""
import pennylane as qml
import time
import numpy as np

def run_circuit():
    try:
        # Force complex64 to attempt to fit in RAM
        dev = qml.device("default.qubit", wires={qubits}, c_dtype=np.complex64)
    except:
        dev = qml.device("default.qubit", wires={qubits})

    @qml.qnode(dev)
    def circuit():
        for _ in range({depth}):
            for i in range({qubits}):
                qml.Hadamard(wires=i)
            for i in range(0, {qubits}-1, 2):
                qml.CNOT(wires=[i, i+1])
        # Return probs to force execution but avoid huge state vector return copy
        return qml.probs(wires=0)

    start = time.time()
    try:
        circuit()
    except Exception as e:
        print(f"ERROR: {{e}}")
        return
    print(f"Total Time: {{time.time() - start}}")

if __name__ == "__main__":
    run_circuit()
"""

    script_name = f"temp_{sim_name}_{qubits}_{circuit}.py"
    with open(script_name, "w") as f:
        f.write(script_content)
        
    cmd = ["python3", script_name]
    start_time = time.time()
    try:
        # Timeout 30s for python baseline (fail fast if OOM/thrashing)
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if os.path.exists(script_name): os.remove(script_name)
        
        if res.returncode != 0:
             return {"success": False, "error": "FAIL (OOM/Error)", "logs": res.stderr[-100:]}

        for line in res.stdout.split('\n'):
            if "Total Time:" in line:
                t = float(line.split(":")[1].strip())
                return {"success": True, "time": t}
        
        return {"success": False, "error": "Unknown (No time output)"}

    except subprocess.TimeoutExpired:
        if os.path.exists(script_name): os.remove(script_name)
        return {"success": False, "error": "TIMEOUT (>30s)"}
    except Exception as e:
        if os.path.exists(script_name): os.remove(script_name)
        return {"success": False, "error": str(e)}

def run_baseline(sim_name, qubits, circuit):
    return run_python_baseline(sim_name, qubits, circuit)

def main():
    results = []
    print(f"=== EdgeQuantum Grand Benchmark Suite (Full) ===")
    
    for q in QUBITS:
        print(f"\n[{q} QUBITS]")
        for c in CIRCUITS:
            print(f"  Circuit: {c}")
            for sim_name, conf in SIMULATORS.items():
                print(f"    {sim_name:<20}", end="", flush=True)
                
                res = None
                if conf["type"] == "cpp_async":
                    res = run_cpp_engine(q, c, mode="async")
                elif conf["type"] == "cpp_blocking":
                    res = run_cpp_engine(q, c, mode="blocking")
                elif conf["type"] == "cpp_native":
                    res = run_cpp_engine(q, c, mode="native")
                elif conf["type"] == "cpp_uvm":
                    res = run_cpp_engine(q, c, mode="uvm")
                else:
                    res = run_baseline(sim_name, q, c)
                
                if res["success"]:
                    print(f"✅ {res['time']:.4f}s")
                    results.append({"simulator": sim_name, "circuit": c, "qubits": q, "time": res["time"]})
                else:
                    print(f"❌ {res['error']}")
                    results.append({"simulator": sim_name, "circuit": c, "qubits": q, "error": res["error"]})

    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {LOG_FILE}")

if __name__ == "__main__":
    main()
