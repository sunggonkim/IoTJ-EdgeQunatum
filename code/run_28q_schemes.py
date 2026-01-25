#!/usr/bin/env python3
"""
Run 28-qubit validation across schemes with timing + correctness checks.
Schemes: cuQuantum Native, cuQuantum UVM, BMQSim-like (Offload), Cirq, EdgeQuantum.
"""
import json
import os
import subprocess
import time
from typing import Dict, Any

import numpy as np

RESULTS_FILE = "results_28q_schemes.json"
QUBITS = 28
CIRCUITS = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
DEPTH = 3

EDGE_BIN = "./build/edge_quantum"

SCHEMES = [
    {"name": "cuQuantum Native", "mode": "native", "force_mode": False},
    {"name": "cuQuantum UVM", "mode": "uvm", "force_mode": False},
    {"name": "BMQSim-like (Offload)", "mode": "blocking", "force_mode": True},
    {"name": "EdgeQuantum", "mode": "async", "force_mode": True},
]


def get_mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return 0


def run_cpp_scheme(scheme: Dict[str, Any]) -> Dict[str, Any]:
    mode = scheme["mode"]
    force_mode = scheme["force_mode"]
    results = {"scheme": scheme["name"], "runs": []}

    for circuit in CIRCUITS:
        cmd = [EDGE_BIN, "--qubits", str(QUBITS), "--circuit", circuit, "--depth", str(DEPTH)]
        if mode != "async":
            cmd += ["--sim-mode", mode]
        if force_mode:
            cmd += ["--force-mode"]

        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            elapsed = time.time() - start
            if proc.returncode != 0:
                results["runs"].append({
                    "circuit": circuit,
                    "success": False,
                    "error": f"Exit {proc.returncode}",
                    "stderr": proc.stderr[-2000:],
                    "stdout": proc.stdout[-2000:],
                })
                continue

            parsed_time = None
            for line in proc.stdout.splitlines():
                if "Total Time:" in line:
                    try:
                        parsed_time = float(line.split(":")[1].strip().split()[0])
                    except Exception:
                        pass

            results["runs"].append({
                "circuit": circuit,
                "success": True,
                "time": parsed_time if parsed_time is not None else elapsed,
                "wall_time": elapsed,
            })
        except subprocess.TimeoutExpired:
            results["runs"].append({
                "circuit": circuit,
                "success": False,
                "error": "TIMEOUT",
            })

    # Correctness check (Hadamard validation)
    verify_cmd = [EDGE_BIN, "--qubits", str(QUBITS), "--verify"]
    if mode != "async":
        verify_cmd += ["--sim-mode", mode]
    if force_mode:
        verify_cmd += ["--force-mode"]

    try:
        proc = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=600)
        results["verify"] = {
            "success": proc.returncode == 0,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        results["verify"] = {"success": False, "error": "TIMEOUT"}

    return results


def build_cirq_circuit(name: str, n_qubits: int, depth: int):
    import cirq

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


def run_cirq_scheme() -> Dict[str, Any]:
    results = {"scheme": "Cirq", "runs": []}
    try:
        import cirq
    except Exception as e:
        results["error"] = f"Cirq not available: {e}"
        return results

    # Pre-check memory requirements for full state-vector simulation
    bytes_per_amp = 16  # complex128
    expected_bytes = (1 << QUBITS) * bytes_per_amp
    available_bytes = get_mem_available_bytes()
    if available_bytes and expected_bytes > int(available_bytes * 0.7):
        results["error"] = (
            f"Insufficient host memory for Cirq state vector: "
            f"need ~{expected_bytes / (1024**3):.2f} GiB, "
            f"available ~{available_bytes / (1024**3):.2f} GiB"
        )
        return results

    for circuit_name in CIRCUITS:
        try:
            circuit = build_cirq_circuit(circuit_name, QUBITS, DEPTH)
            sim = cirq.Simulator()
            start = time.time()
            result = sim.simulate(circuit)
            elapsed = time.time() - start

            state = getattr(result, "final_state_vector", None)
            if state is None:
                results["runs"].append({"circuit": circuit_name, "success": False, "error": "No state vector"})
                continue

            norm = float(np.sum(np.abs(state) ** 2))
            ok = abs(norm - 1.0) < 1e-3

            results["runs"].append({
                "circuit": circuit_name,
                "success": True,
                "time": elapsed,
                "norm": norm,
                "verify": ok,
            })
        except Exception as e:
            results["runs"].append({"circuit": circuit_name, "success": False, "error": str(e)})

    return results


def main():
    all_results = []

    for scheme in SCHEMES:
        all_results.append(run_cpp_scheme(scheme))

    all_results.append(run_cirq_scheme())

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
