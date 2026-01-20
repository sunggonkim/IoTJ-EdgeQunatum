#!/usr/bin/env python3
"""
Cirq-only Benchmark - Verify if Cirq actually simulates at high qubit counts
"""
import cirq
import numpy as np
import time
import sys

print("="*60)
print("   CIRQ VERIFICATION BENCHMARK")
print("   Testing if Cirq actually simulates at high qubit counts")
print("="*60)

# Test qubit counts
qubit_counts = [20, 22, 24, 26, 28, 30, 32]

circuits = {
    "QV": lambda n: int(n * n / 2),  # depth=n, n/2 gates per layer
    "Random": lambda n: 10 * n,       # depth=10
    "GHZ": lambda n: n,               # n gates
}

for n_qubits in qubit_counts:
    print(f"\n--- {n_qubits} Qubits ---")
    
    for circuit_name, gate_count_fn in circuits.items():
        gate_count = gate_count_fn(n_qubits)
        
        try:
            # Create circuit
            qubits = cirq.LineQubit.range(n_qubits)
            circuit = cirq.Circuit()
            
            # Add random gates
            H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
            
            for i in range(gate_count):
                gate = cirq.MatrixGate(H)
                circuit.append(gate(qubits[i % n_qubits]))
            
            # Time the simulation
            sim = cirq.Simulator()
            
            t0 = time.time()
            result = sim.simulate(circuit)
            t1 = time.time()
            
            # Verify result has actual state vector
            state_size = len(result.final_state_vector) if hasattr(result, 'final_state_vector') else 0
            expected_size = 2 ** n_qubits
            
            if state_size == expected_size:
                status = "✅ VALID"
            else:
                status = f"⚠️ INVALID (state size: {state_size}, expected: {expected_size})"
            
            print(f"  {circuit_name}: {t1-t0:.4f}s ({gate_count} gates) {status}")
            
        except MemoryError:
            print(f"  {circuit_name}: ❌ OOM (MemoryError)")
        except Exception as e:
            print(f"  {circuit_name}: ❌ ERROR: {str(e)[:50]}")

print("\n" + "="*60)
print("   CIRQ VERIFICATION COMPLETE")
print("="*60)
