#!/usr/bin/env python3
"""
Comprehensive Benchmark for EdgeQuantum Paper
Compares: cuQuantum Native, cuQuantum UVM, BMQSim-like (Offload), Cirq, EdgeQuantum
Circuits: QV, VQC, QSVM, Random, GHZ, VQE
Qubits: 26-34
"""
import subprocess
import time
import json
import os
import sys
import datetime
import traceback

# === Configuration ===
QUBITS = [26, 27, 28, 29, 30, 31, 32, 33, 34]
CIRCUITS = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
DEPTH = 5
TIMEOUT_SEC = 3600  # 1 hour timeout per run

SCHEMES = [
    {"name": "cuQuantum Native", "type": "cpp", "mode": "native", "force_mode": False},
    {"name": "cuQuantum UVM", "type": "cpp", "mode": "uvm", "force_mode": False},
    {"name": "BMQSim-like (Offload)", "type": "cpp", "mode": "blocking", "force_mode": True},
    {"name": "EdgeQuantum", "type": "cpp", "mode": "async", "force_mode": True},
    {"name": "Cirq", "type": "cirq", "mode": None, "force_mode": False}
]

RESULTS_FILE = "comprehensive_results.json"
BINARY = "./build/edge_quantum"

# === Helper Functions ===

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def run_cpp_engine(qubits, circuit, depth, sim_mode, force_mode):
    """Run C++ EdgeQuantum Binary"""
    cmd = [BINARY, "--qubits", str(qubits), "--circuit", circuit, "--depth", str(depth)]
    
    if sim_mode and sim_mode != "async":
        cmd += ["--sim-mode", sim_mode]
    if force_mode:
        cmd += ["--force-mode"]
    
    log(f"  CMD: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
        wall_time = time.time() - start_time
        
        if res.returncode != 0:
            return {
                "success": False, 
                "error": f"Exit Code {res.returncode}", 
                "wall_time": wall_time,
                "logs": res.stdout[-2000:] + res.stderr[-2000:]
            }
        
        # Parse "Total Time: X s"
        sim_time = wall_time
        for line in res.stdout.split('\n'):
            if "Total Time:" in line:
                try:
                    sim_time = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
        
        return {"success": True, "sim_time": sim_time, "wall_time": wall_time}
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "TIMEOUT", "wall_time": TIMEOUT_SEC}
    except Exception as e:
        return {"success": False, "error": str(e), "wall_time": time.time() - start_time}

def run_cirq(qubits, circuit, depth):
    """Run Cirq Simulator"""
    try:
        import cirq
        import numpy as np
    except ImportError:
        return {"success": False, "error": "Cirq not installed"}
    
    log(f"  Cirq: Building circuit...")
    
    try:
        q = cirq.LineQubit.range(qubits)
        circ = cirq.Circuit()
        
        if circuit == "GHZ":
            circ.append(cirq.H(q[0]))
            for i in range(1, qubits):
                circ.append(cirq.CNOT(q[i-1], q[i]))
        
        elif circuit == "QV":
            import random
            for _ in range(depth):
                perm = list(range(qubits))
                random.shuffle(perm)
                for i in range(0, qubits - 1, 2):
                    circ.append(cirq.H(q[perm[i]]))
                    circ.append(cirq.CNOT(q[perm[i]], q[perm[i+1]]))
        
        elif circuit == "VQC":
            for _ in range(depth):
                for i in range(qubits):
                    circ.append(cirq.rz(np.random.uniform(0, 2*np.pi))(q[i]))
                    circ.append(cirq.H(q[i]))
                for i in range(qubits - 1):
                    circ.append(cirq.CNOT(q[i], q[i+1]))
        
        elif circuit == "QSVM":
            for _ in range(2):
                for i in range(qubits):
                    circ.append(cirq.H(q[i]))
                for i in range(qubits):
                    circ.append(cirq.rz(np.random.uniform(0, 2*np.pi))(q[i]))
        
        elif circuit == "VQE":
            for layer in range(depth):
                for i in range(qubits):
                    circ.append(cirq.ry(np.random.uniform(0, np.pi))(q[i]))
                    circ.append(cirq.rz(np.random.uniform(0, 2*np.pi))(q[i]))
                for i in range(qubits - 1):
                    circ.append(cirq.CNOT(q[i], q[i+1]))
        
        else:  # Random
            for _ in range(depth):
                for i in range(qubits):
                    gate = np.random.choice([cirq.H, cirq.X, cirq.Y, cirq.Z])
                    circ.append(gate(q[i]))
                for i in range(qubits - 1):
                    if np.random.random() > 0.5:
                        circ.append(cirq.CNOT(q[i], q[i+1]))
        
        log(f"  Cirq: Simulating {len(circ)} ops...")
        simulator = cirq.Simulator()
        
        start_time = time.time()
        result = simulator.simulate(circ)
        sim_time = time.time() - start_time
        
        return {"success": True, "sim_time": sim_time, "wall_time": sim_time}
        
    except MemoryError:
        return {"success": False, "error": "OOM"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"runs": [], "meta": {"start_time": datetime.datetime.now().isoformat()}}

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def already_run(results, scheme, circuit, qubits):
    for r in results["runs"]:
        if r["scheme"] == scheme and r["circuit"] == circuit and r["qubits"] == qubits:
            return True
    return False

# === Main ===

def main():
    log("=" * 60)
    log("EdgeQuantum Comprehensive Benchmark")
    log(f"Qubits: {QUBITS}")
    log(f"Circuits: {CIRCUITS}")
    log(f"Schemes: {[s['name'] for s in SCHEMES]}")
    log("=" * 60)
    
    # Check binary exists
    if not os.path.exists(BINARY):
        log(f"ERROR: Binary not found: {BINARY}")
        log("Run 'make -j' first!")
        sys.exit(1)
    
    # Apply jetson_clocks for max performance
    log("Applying jetson_clocks for maximum performance...")
    os.system("echo '1234qwer' | sudo -S /usr/bin/jetson_clocks 2>/dev/null")
    
    results = load_results()
    total_runs = len(QUBITS) * len(CIRCUITS) * len(SCHEMES)
    completed = len(results["runs"])
    
    log(f"Starting from run {completed + 1} / {total_runs}")
    
    for qubits in QUBITS:
        for circuit in CIRCUITS:
            for scheme in SCHEMES:
                scheme_name = scheme["name"]
                
                # Skip if already done
                if already_run(results, scheme_name, circuit, qubits):
                    log(f"[SKIP] {scheme_name} | {circuit} | {qubits}q (already done)")
                    continue
                
                log(f"[RUN] {scheme_name} | {circuit} | {qubits}q")
                
                # Run benchmark
                if scheme["type"] == "cirq":
                    result = run_cirq(qubits, circuit, DEPTH)
                else:
                    result = run_cpp_engine(qubits, circuit, DEPTH, scheme["mode"], scheme["force_mode"])
                
                # Record result
                run_record = {
                    "scheme": scheme_name,
                    "circuit": circuit,
                    "qubits": qubits,
                    "depth": DEPTH,
                    "timestamp": datetime.datetime.now().isoformat(),
                    **result
                }
                results["runs"].append(run_record)
                
                # Save after each run
                save_results(results)
                
                if result["success"]:
                    log(f"  ✓ Time: {result['sim_time']:.2f}s")
                else:
                    log(f"  ✗ Failed: {result.get('error', 'Unknown')}")
                
                # Cool-down between runs
                time.sleep(2)
    
    log("=" * 60)
    log("Benchmark Complete!")
    log(f"Results saved to: {RESULTS_FILE}")
    
    # Summary
    success_count = sum(1 for r in results["runs"] if r["success"])
    log(f"Success: {success_count} / {len(results['runs'])}")

if __name__ == "__main__":
    main()
