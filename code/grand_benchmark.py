#!/usr/bin/env python3
import subprocess
import time
import json
import os
import sys
import datetime

# --- Configuration ---
QUBITS = [28]
CIRCUITS = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
SCHEMES = [
    {"name": "cuQuantum Native", "type": "cpp", "mode": "native", "force_mode": False},
    {"name": "cuQuantum UVM", "type": "cpp", "mode": "uvm", "force_mode": False},
    {"name": "BMQSim-like (Offload)", "type": "cpp", "mode": "blocking", "force_mode": True},
    {"name": "EdgeQuantum", "type": "cpp", "mode": "async", "force_mode": True},
    {"name": "Cirq", "type": "cirq"}
]

LOG_FILE = "grand_results_28q.json"

# --- Helper Functions ---

def run_cpp_engine(qubits, circuit, depth=5, sim_mode="async", force_mode=False):
    """Run C++ EdgeQuantum Binary"""
    cmd = [
        "./build/edge_quantum",
        "--qubits", str(qubits),
        "--circuit", circuit,
        "--depth", str(depth)
    ]

    if sim_mode and sim_mode != "async":
        cmd += ["--sim-mode", sim_mode]
    if force_mode:
        cmd += ["--force-mode"]
    
    start_time = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200) # 20 mins
        duration = time.time() - start_time
        
        if res.returncode != 0:
            return {"success": False, "error": f"Exit Code {res.returncode}", "logs": res.stdout + res.stderr}
            
        # Parse output for "Total Time: X s"
        for line in res.stdout.split('\n'):
            if "Total Time:" in line:
                try:
                    t = float(line.split(":")[1].strip().split()[0])
                    return {"success": True, "time": t, "throughput": duration}
                except:
                    pass
        
        return {"success": True, "time": duration} # Fallback
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def build_cirq_circuit(name, n_qubits, depth):
    import cirq
    import numpy as np

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    rz = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex128)

    if name == "GHZ":
        circuit.append(cirq.MatrixGate(h)(qubits[0]))
        for i in range(1, n_qubits):
            circuit.append(cirq.MatrixGate(x)(qubits[i]))
        return circuit

    if name == "QV":
        for _ in range(depth):
            for i in range(0, n_qubits - 1, 2):
                circuit.append(cirq.MatrixGate(h)(qubits[i]))
                circuit.append(cirq.MatrixGate(h)(qubits[i + 1]))
        return circuit

    if name == "VQC":
        for _ in range(depth):
            for i in range(n_qubits):
                theta = np.random.uniform(0, 2 * np.pi)
                circuit.append(cirq.MatrixGate(rz(theta))(qubits[i]))
            for i in range(n_qubits):
                circuit.append(cirq.MatrixGate(h)(qubits[i]))
        return circuit

    if name == "QSVM":
        for _ in range(2):
            for i in range(n_qubits):
                circuit.append(cirq.MatrixGate(h)(qubits[i]))
            for i in range(n_qubits):
                phi = np.random.uniform(0, 2 * np.pi)
                circuit.append(cirq.MatrixGate(rz(phi))(qubits[i]))
        return circuit

    if name == "VQE":
        for _ in range(depth):
            for i in range(n_qubits):
                theta = np.random.uniform(0, np.pi)
                ry = np.array([
                    [np.cos(theta/2), -np.sin(theta/2)],
                    [np.sin(theta/2), np.cos(theta/2)]
                ], dtype=np.complex128)
                circuit.append(cirq.MatrixGate(ry)(qubits[i]))
            for i in range(n_qubits):
                phi = np.random.uniform(0, 2 * np.pi)
                circuit.append(cirq.MatrixGate(rz(phi))(qubits[i]))
        return circuit

    # Random
    gate_set = [h, x, rz(np.pi / 4)]
    for _ in range(depth):
        for i in range(n_qubits):
            gate = gate_set[np.random.randint(len(gate_set))]
            circuit.append(cirq.MatrixGate(gate)(qubits[i]))
    return circuit


def run_cirq(qubits, circuit, depth=5):
    try:
        import cirq
        import numpy as np
    except Exception as e:
        return {"success": False, "error": f"Cirq not available: {e}"}

    try:
        circ = build_cirq_circuit(circuit, qubits, depth)
        sim = cirq.Simulator()
        start_time = time.time()
        result = sim.simulate(circ)
        duration = time.time() - start_time

        state = getattr(result, "final_state_vector", None)
        if state is None:
            return {"success": False, "error": "No state vector"}

        norm = float(np.sum(np.abs(state) ** 2))
        ok = abs(norm - 1.0) < 1e-3
        return {"success": True, "time": duration, "norm": norm, "verify": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    results = []
    
    print(f"=== EdgeQuantum Grand Benchmark (28Q) ===")
    
    for c in CIRCUITS:
        print(f"\nCircuit: {c}")
        for scheme in SCHEMES:
            name = scheme["name"]
            print(f"  Simulator: {name:<22}", end="", flush=True)

            if scheme["type"] == "cpp":
                res = run_cpp_engine(QUBITS[0], c, depth=5, sim_mode=scheme["mode"], force_mode=scheme["force_mode"])
            else:
                res = run_cirq(QUBITS[0], c, depth=3)

            if res.get("success"):
                print(f"✅ {res['time']:.4f}s")
                results.append({"simulator": name, "circuit": c, "qubits": QUBITS[0], "time": res["time"], "verify": res.get("verify")})
            else:
                print(f"❌ {res.get('error', 'FAIL')}")
                results.append({"simulator": name, "circuit": c, "qubits": QUBITS[0], "error": res.get("error", "FAIL")})

    # Save
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {LOG_FILE}")

if __name__ == "__main__":
    main()
