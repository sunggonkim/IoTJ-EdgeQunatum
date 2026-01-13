#!/usr/bin/env python3
"""
EdgeQuantum IoT Benchmark - cuQuantum State Vector Simulation on Jetson
Measures quantum circuit simulation performance on edge IoT devices.
"""

import time
import numpy as np
import cupy as cp
from cuquantum import custatevec as cusv
import subprocess

def get_gpu_info():
    """Get GPU memory and utilization info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def benchmark_statevector_simulation(n_qubits_list, n_gates=100, n_trials=3):
    """
    Benchmark cuQuantum state vector simulation for different qubit counts.
    
    Args:
        n_qubits_list: List of qubit counts to test
        n_gates: Number of random gates to apply
        n_trials: Number of trials for averaging
    
    Returns:
        dict with benchmark results
    """
    results = {}
    
    print("=" * 60)
    print("cuQuantum EdgeIoT Benchmark - Jetson Device")
    print("=" * 60)
    print(f"GPU Info: {get_gpu_info()}")
    print(f"Testing {len(n_qubits_list)} qubit configurations with {n_gates} gates each")
    print("-" * 60)
    
    # Initialize cuStateVec handle
    handle = cusv.create()
    
    for n_qubits in n_qubits_list:
        dim = 2 ** n_qubits
        print(f"\n[{n_qubits} qubits] State vector size: {dim:,} ({dim * 16 / 1024 / 1024:.2f} MB complex128)")
        
        trial_times = []
        
        for trial in range(n_trials):
            # Initialize state vector |0...0>
            sv = cp.zeros(dim, dtype=cp.complex128)
            sv[0] = 1.0 + 0.0j
            
            # Define some common quantum gates
            # Hadamard gate
            H = cp.array([[1, 1], [1, -1]], dtype=cp.complex128) / np.sqrt(2)
            # Pauli-X gate
            X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
            # Phase gate
            S = cp.array([[1, 0], [0, 1j]], dtype=cp.complex128)
            # T gate
            T = cp.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=cp.complex128)
            
            gates = [H, X, S, T]
            
            # Warm up GPU
            cp.cuda.Device().synchronize()
            
            start_time = time.perf_counter()
            
            # Apply random single-qubit gates
            for gate_idx in range(n_gates):
                target_qubit = gate_idx % n_qubits
                gate = gates[gate_idx % len(gates)]
                
                # Use cuStateVec to apply the gate
                cusv.apply_matrix(
                    handle,
                    sv.data.ptr,
                    cusv.cudaDataType.CUDA_C_64F,
                    n_qubits,
                    gate.data.ptr,
                    cusv.cudaDataType.CUDA_C_64F,
                    cusv.MatrixLayout.ROW,
                    0,  # adjoint = False
                    [target_qubit],
                    None,  # control_bits
                    0,  # n_control_bits
                    cusv.ComputeType.COMPUTE_64F,
                    0  # workspace
                )
            
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start_time
            trial_times.append(elapsed)
            
            print(f"  Trial {trial + 1}: {elapsed * 1000:.2f} ms ({n_gates / elapsed:.0f} gates/sec)")
        
        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        gates_per_sec = n_gates / avg_time
        
        results[n_qubits] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'gates_per_sec': gates_per_sec,
            'memory_mb': dim * 16 / 1024 / 1024
        }
        
        print(f"  Average: {avg_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
        print(f"  Throughput: {gates_per_sec:,.0f} gates/sec")
    
    cusv.destroy(handle)
    
    return results

def benchmark_tensor_network():
    """
    Benchmark cuTensorNet for tensor network contraction.
    This simulates quantum circuits using tensor network methods.
    """
    from cuquantum import cutensornet as cutn
    
    print("\n" + "=" * 60)
    print("cuTensorNet Benchmark - Tensor Network Contraction")
    print("=" * 60)
    
    # Simple tensor contraction benchmark
    handle = cutn.create()
    
    # Test different tensor sizes
    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    for size in sizes:
        m, n, k = size
        
        # Create random tensors
        A = cp.random.random((m, k), dtype=cp.float64) + 1j * cp.random.random((m, k), dtype=cp.float64)
        B = cp.random.random((k, n), dtype=cp.float64) + 1j * cp.random.random((k, n), dtype=cp.float64)
        
        cp.cuda.Device().synchronize()
        
        # Time the contraction (simple matmul for now)
        start = time.perf_counter()
        for _ in range(10):
            C = cp.matmul(A, B)
        cp.cuda.Device().synchronize()
        elapsed = (time.perf_counter() - start) / 10
        
        flops = 2 * m * n * k * 8  # complex matmul
        gflops = flops / elapsed / 1e9
        
        print(f"Tensor size {size}: {elapsed * 1000:.3f} ms ({gflops:.1f} GFLOP/s)")
    
    cutn.destroy(handle)

def main():
    print("\n🚀 EdgeQuantum IoT - cuQuantum Benchmark Suite")
    print("Testing quantum simulation capabilities on Jetson Edge Device\n")
    
    # Test with varying qubit counts
    # Jetson typically has limited memory, so we start small
    qubit_configs = [4, 6, 8, 10, 12, 14]
    
    try:
        # State vector simulation benchmark
        sv_results = benchmark_statevector_simulation(qubit_configs, n_gates=100, n_trials=3)
        
        # Tensor network benchmark
        benchmark_tensor_network()
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"{'Qubits':<8} {'Dim':<12} {'Memory (MB)':<12} {'Time (ms)':<12} {'Gates/sec':<12}")
        print("-" * 60)
        for qubits, data in sv_results.items():
            dim = 2 ** qubits
            print(f"{qubits:<8} {dim:<12,} {data['memory_mb']:<12.2f} {data['avg_time_ms']:<12.2f} {data['gates_per_sec']:<12,.0f}")
        
        print("\n✅ Benchmark completed successfully!")
        print("This demonstrates cuQuantum quantum simulation capability on IoT edge devices.")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
