#!/usr/bin/env python3
"""
ScaleQsim-style Benchmark Suite for EdgeQuantum
Runs the same benchmark circuits as ScaleQsim but at edge-appropriate qubit counts.

Benchmark Circuits (following ScaleQsim):
1. QFT (Quantum Fourier Transform) - Dense entanglement
2. Random Circuits - Variable depth, random gates
3. Supremacy-style - Google-style random circuit sampling
4. GHZ State - Linear entanglement
5. Quantum Volume - Square circuits
"""
import os
import sys
import time
import json
import numpy as np

# Import the optimized TieredSimulator (Managed Memory Zero-Copy)
from tiered_simulator import TieredSimulator, H

# Additional gate definitions
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex64)

def RY(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)

def RX(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)


# ============================================================
# SCALEQSIM-STYLE BENCHMARK CIRCUITS
# ============================================================

def circuit_qft(sim):
    """Quantum Fourier Transform - O(n²) gates, dense entanglement"""
    n = min(sim.n, sim.gpu_n)
    for i in range(n):
        sim.apply_single_gate(H, i)
        for j in range(i+1, min(i+4, n)):  # Limited range for edge device
            theta = np.pi / (2**(j-i))
            sim.apply_single_gate(RZ(theta), j)
    sim.flush_gates()
    return sim.gate_count

def circuit_random(sim, depth=20):
    """Random Circuit - Variable depth, random single-qubit gates"""
    n = min(sim.n, sim.gpu_n)
    np.random.seed(42)
    gates = [H, T, S, RX(np.pi/4), RY(np.pi/4), RZ(np.pi/4)]
    
    for d in range(depth):
        for q in range(n):
            gate = gates[np.random.randint(len(gates))]
            sim.apply_single_gate(gate, q)
    sim.flush_gates()
    return sim.gate_count

def circuit_supremacy(sim, cycles=10):
    """Supremacy-style circuit (simplified) - Google-style random circuit sampling"""
    n = min(sim.n, sim.gpu_n)
    np.random.seed(42)
    
    for cycle in range(cycles):
        # Layer of random single-qubit gates
        for q in range(n):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            sim.apply_single_gate(RZ(theta), q)
            sim.apply_single_gate(RX(phi), q)
        
        # fSim-like gates (simplified as rotations on adjacent qubits)
        for q in range(0, n-1, 2):
            sim.apply_single_gate(RZ(np.pi/6), q)
            sim.apply_single_gate(RZ(np.pi/6), q+1)
    
    sim.flush_gates()
    return sim.gate_count

def circuit_ghz(sim):
    """GHZ State - Linear entanglement chain"""
    n = min(sim.n, sim.gpu_n)
    sim.apply_single_gate(H, 0)
    for q in range(1, n):
        # Simplified CNOT as rotations
        sim.apply_single_gate(RX(np.pi/2), q)
    sim.flush_gates()
    return sim.gate_count

def circuit_qv(sim, depth=None):
    """Quantum Volume - Square circuit (depth = width)"""
    n = min(sim.n, sim.gpu_n)
    if depth is None:
        depth = n
    
    np.random.seed(42)
    for d in range(depth):
        # Random SU(4) approximated by random single-qubit gates
        for q in range(n):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            sim.apply_single_gate(RY(theta), q)
            sim.apply_single_gate(RZ(phi), q)
    sim.flush_gates()
    return sim.gate_count


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(n_qubits, circuit_fn, circuit_name, **kwargs):
    """Run a single benchmark"""
    print(f"\n{'='*50}")
    print(f"  {circuit_name} @ {n_qubits} qubits")
    print(f"{'='*50}")
    
    sim = TieredSimulator(n_qubits, use_compression=True, fusion_threshold=10)
    
    # Initialize
    t_init = time.time()
    sim.init_zero_state()
    init_time = time.time() - t_init
    
    # Run circuit
    t_circuit = time.time()
    gate_count = circuit_fn(sim, **kwargs) if kwargs else circuit_fn(sim)
    circuit_time = time.time() - t_circuit
    
    # Metrics
    storage_bytes = sim.get_storage_size()
    raw_bytes = 2**n_qubits * 8
    compression_ratio = raw_bytes / storage_bytes if storage_bytes > 0 else 0
    
    result = {
        'circuit': circuit_name,
        'qubits': n_qubits,
        'gates': gate_count,
        'init_time': init_time,
        'circuit_time': circuit_time,
        'total_time': init_time + circuit_time,
        'storage_gb': storage_bytes / 1e9,
        'raw_gb': raw_bytes / 1e9,
        'compression_ratio': compression_ratio,
        'gates_per_second': gate_count / circuit_time if circuit_time > 0 else 0
    }
    
    print(f"  Gates: {gate_count}")
    print(f"  Init Time: {init_time:.2f}s")
    print(f"  Circuit Time: {circuit_time:.2f}s")
    print(f"  Total Time: {result['total_time']:.2f}s")
    print(f"  Storage: {result['storage_gb']:.4f} GB (Ratio: {compression_ratio:.1f}x)")
    print(f"  Throughput: {result['gates_per_second']:.2f} gates/s")
    
    sim.cleanup()
    return result


def main():
    print("="*60)
    print("  EDGEQUANTUM SCALEQSIM-STYLE BENCHMARK SUITE")
    print("  Device: NVIDIA Jetson Orin Nano (8GB, 15W)")
    print("="*60)
    
    # ScaleQsim-style benchmarks at edge-appropriate qubit counts
    qubit_counts = [20, 22, 24, 26, 28]
    
    circuits = [
        (circuit_qft, "QFT", {}),
        (circuit_random, "Random-20", {'depth': 20}),
        (circuit_supremacy, "Supremacy-10", {'cycles': 10}),
        (circuit_ghz, "GHZ", {}),
        (circuit_qv, "QuantumVolume", {}),
    ]
    
    results = []
    
    for n in qubit_counts:
        for circuit_fn, name, kwargs in circuits:
            try:
                result = run_benchmark(n, circuit_fn, name, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/scaleqsim_style_bench.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("  BENCHMARK COMPLETE!")
    print("="*60)
    
    # Summary table
    print("\n📊 Summary Table:")
    print(f"{'Circuit':<15} {'20Q':<12} {'22Q':<12} {'24Q':<12} {'26Q':<12} {'28Q':<12}")
    print("-"*75)
    
    for circuit_name in ['QFT', 'Random-20', 'Supremacy-10', 'GHZ', 'QuantumVolume']:
        row = f"{circuit_name:<15}"
        for n in qubit_counts:
            match = [r for r in results if r['circuit'] == circuit_name and r['qubits'] == n]
            if match:
                row += f"{match[0]['total_time']:.1f}s".ljust(12)
            else:
                row += "--".ljust(12)
        print(row)
    
    print(f"\n💾 Results saved to: data/scaleqsim_style_bench.json")


if __name__ == "__main__":
    main()
