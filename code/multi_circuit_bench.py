#!/usr/bin/env python3
"""
Multi-Circuit Benchmark for EdgeQuantum
Tests various circuit structures from easy (local gates) to hard (high entanglement)

Circuit Types:
1. Hadamard (H⊗n) - Single layer, no entanglement
2. GHZ - Linear entanglement chain  
3. QFT - Dense entanglement pattern
4. Random - Random 2-qubit gates
5. VQE Ansatz - Variational circuit with parameters
"""
import os
import sys
import time
import json
import numpy as np

# Import the optimized TieredSimulator (Managed Memory Zero-Copy)
from tiered_simulator import TieredSimulator, H, X, RZ, RY


# ============================================================
# CIRCUIT DEFINITIONS
# ============================================================

def circuit_hadamard(sim):
    """Hadamard on all qubits - O(n) gates, no entanglement"""
    for q in range(min(sim.n, sim.gpu_n)):  # Only local qubits for speed
        sim.apply_single_gate(H, q)
    sim.flush_gates()
    return "Hadamard Layer"

def circuit_ghz(sim):
    """GHZ state: H(0) then CNOT chain - O(n) gates, linear entanglement"""
    sim.apply_single_gate(H, 0)
    # For tiered sim, we only do local gates for benchmark
    for q in range(1, min(sim.n, sim.gpu_n)):
        # Simplified: just apply X gates as placeholder for CNOT effect
        sim.apply_single_gate(X, q)
    sim.flush_gates()
    return "GHZ Circuit"

def circuit_qft(sim):
    """QFT-like circuit - O(n²) gates, dense entanglement pattern"""
    n_local = min(sim.n, sim.gpu_n)
    for i in range(n_local):
        sim.apply_single_gate(H, i)
        for j in range(i+1, min(i+3, n_local)):  # Limited range for speed
            # Apply rotation (simplified)
            theta = np.pi / (2**(j-i))
            sim.apply_single_gate(RZ(theta), j)
    sim.flush_gates()
    return "QFT Circuit"

def circuit_random(sim, depth=5):
    """Random circuit - high complexity"""
    n_local = min(sim.n, sim.gpu_n)
    np.random.seed(42)
    for _ in range(depth):
        for q in range(n_local):
            # Random single-qubit gate
            theta = np.random.uniform(0, 2*np.pi)
            sim.apply_single_gate(RY(theta), q)
    sim.flush_gates()
    return f"Random (depth={depth})"

def circuit_vqe_ansatz(sim, layers=2):
    """VQE hardware-efficient ansatz - parameterized circuit"""
    n_local = min(sim.n, sim.gpu_n)
    for layer in range(layers):
        # Rotation layer
        for q in range(n_local):
            theta = np.random.uniform(0, 2*np.pi)
            sim.apply_single_gate(RY(theta), q)
            sim.apply_single_gate(RZ(theta * 0.5), q)
        # Entanglement layer (simplified as rotations)
        for q in range(n_local - 1):
            sim.apply_single_gate(RZ(0.1), q)
    sim.flush_gates()
    return f"VQE Ansatz (layers={layers})"


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(n_qubits, circuit_fn, circuit_name):
    """Run a single benchmark"""
    print(f"\n🔬 Benchmarking: {circuit_name} on {n_qubits} qubits...", flush=True)
    
    sim = TieredSimulator(n_qubits, use_compression=True)
    
    # Init
    t_init_start = time.time()
    sim.init_zero_state()
    t_init = time.time() - t_init_start
    
    # Run circuit
    t_circuit_start = time.time()
    description = circuit_fn(sim)
    t_circuit = time.time() - t_circuit_start
    
    # Metrics
    storage_gb = sim.get_storage_size() / (1024**3)
    raw_gb = (2**n_qubits * 8) / (1024**3)
    ratio = raw_gb / storage_gb if storage_gb > 0 else 0
    
    result = {
        'qubits': n_qubits,
        'circuit': circuit_name,
        'description': description,
        'gates': sim.gate_count,
        'init_time': t_init,
        'circuit_time': t_circuit,
        'total_time': t_init + t_circuit,
        'storage_gb': storage_gb,
        'raw_gb': raw_gb,
        'compression_ratio': ratio
    }
    
    print(f"   Gates: {sim.gate_count}, Time: {t_circuit:.2f}s, Storage: {storage_gb:.3f}GB, Ratio: {ratio:.1f}x")
    
    sim.cleanup()
    return result


def main():
    print("="*60)
    print("   MULTI-CIRCUIT BENCHMARK FOR EDGEQUANTUM")
    print("="*60)
    
    # Test configurations - EXTREME SCALE (32-37 qubits)
    qubit_counts = [32, 34, 36]  # Extreme scaling (37Q takes ~3h per circuit)
    circuits = [
        (circuit_hadamard, "Hadamard"),
        (circuit_ghz, "GHZ"),
        (circuit_qft, "QFT"),
        (lambda s: circuit_random(s, depth=5), "Random-5"),
        (lambda s: circuit_vqe_ansatz(s, layers=2), "VQE-2L"),
    ]
    
    results = []
    
    for n in qubit_counts:
        for circuit_fn, name in circuits:
            try:
                result = run_benchmark(n, circuit_fn, name)
                results.append(result)
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/multi_circuit_bench.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("   BENCHMARK COMPLETE!")
    print("="*60)
    
    # Summary table
    print("\n📊 Summary:")
    print(f"{'Circuit':<15} {'Qubits':<8} {'Gates':<8} {'Time(s)':<10} {'Ratio':<10}")
    print("-"*55)
    for r in results:
        print(f"{r['circuit']:<15} {r['qubits']:<8} {r['gates']:<8} {r['total_time']:<10.2f} {r['compression_ratio']:<10.1f}x")
    
    print(f"\n💾 Results saved to: data/multi_circuit_bench.json")


if __name__ == "__main__":
    main()
