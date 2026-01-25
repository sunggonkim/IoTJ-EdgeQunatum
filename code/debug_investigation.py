import subprocess
import time
import os
import sys

def debug_cpp():
    print("--- Debugging EdgeQuantum VQC 28Q (Async -> Smart Native) ---")
    cmd = [
        "./build/edge_quantum",
        "--qubits", "28",
        "--circuit", "VQC",
        "--depth", "3",
        "--sim-mode", "async",
        "--storage", "/mnt/nvme/skim/edgeQuantum/cpp_state_vector.bin"
    ]
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start
    print(f"Duration: {dur:.4f}s")
    print("Output Head:")
    print('\n'.join(res.stdout.split('\n')[:10]))
    if res.returncode != 0:
        print("STDERR:")
        print(res.stderr)

def debug_cirq():
    print("\n--- Debugging Cirq 28Q ---")
    try:
        import cirq
        print(f"Cirq Version: {cirq.__version__}")
        q = cirq.LineQubit.range(28)
        c = cirq.Circuit()
        for i in range(28): c.append(cirq.H(q[i]))
        sim = cirq.Simulator()
        start = time.time()
        print("Simulating...")
        result = sim.simulate(c)
        print("Accessing State Vector...")
        sv = result.final_state_vector
        print(f"State Vector Shape: {sv.shape}")
        print(f"Duration: {time.time() - start:.4f}s")
    except ImportError:
        print("Cirq not installed.")
    except Exception as e:
        print(f"Cirq Error: {e}")

def debug_pennylane():
    print("\n--- Debugging PennyLane 28Q ---")
    try:
        import pennylane as qml
        import numpy as np
        print(f"PennyLane Version: {qml.__version__}")
        
        dev_name = "default.qubit"
        # Force complex64 for default.qubit to save memory
        dev = qml.device(dev_name, wires=27, c_dtype=np.complex64)
        print(f"Using device: {dev_name} with complex64")
        
        @qml.qnode(dev)
        def circuit():
            for i in range(27): qml.Hadamard(wires=i)
            # Return probs of wire 0 to force execution but avoid 2GB state return copy
            return qml.probs(wires=0)

        start = time.time()
        print("Executing PennyLane...")
        res = circuit()
        print(f"Result: {res}")
        print(f"Duration: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"PennyLane Error: {e}")

if __name__ == "__main__":
    # debug_cpp()
    # debug_cirq()
    debug_pennylane()
