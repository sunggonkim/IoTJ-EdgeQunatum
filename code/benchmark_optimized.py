#!/usr/bin/env python3
"""
EdgeQuantum vs cuQuantum Direct Comparison (Optimized)
"""
import time
import numpy as np
import sys
sys.path.insert(0, '/home/jetson/skim/edgeQuantum-iotj/code')

from tiered_simulator import TieredSimulator, H

# Also test raw cuQuantum for comparison
import cupy as cp
import cuquantum as cq
from cuquantum import custatevec as cusv

def benchmark_cuquantum_native(n_qubits, n_gates):
    """Raw cuQuantum Native benchmark"""
    handle = cusv.create()
    sv = cp.zeros(2**n_qubits, dtype=cp.complex64)
    sv[0] = 1.0
    
    # Pre-allocate
    matrix_gpu = cp.asarray(H, dtype=cp.complex64)
    targets = np.array([0], dtype=np.int32)
    
    t0 = time.time()
    for i in range(n_gates):
        targets[0] = i % n_qubits
        cusv.apply_matrix(
            handle=handle, sv=sv.data.ptr,
            sv_data_type=cq.cudaDataType.CUDA_C_32F, n_index_bits=n_qubits,
            matrix=matrix_gpu.data.ptr, matrix_data_type=cq.cudaDataType.CUDA_C_32F,
            layout=cusv.MatrixLayout.ROW, adjoint=0,
            targets=targets.ctypes.data, n_targets=1,
            controls=0, control_bit_values=0, n_controls=0,
            compute_type=cq.ComputeType.COMPUTE_32F, workspace=0, workspace_size=0
        )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0
    
    del sv
    cusv.destroy(handle)
    return elapsed

def benchmark_edgequantum(n_qubits, n_gates):
    """EdgeQuantum (Optimized) benchmark"""
    sim = TieredSimulator(n_qubits, use_compression=True, fusion_threshold=100)
    sim.init_zero_state()
    
    t0 = time.time()
    for i in range(n_gates):
        sim.apply_single_gate(H, i % n_qubits)
    sim.flush_gates()
    elapsed = time.time() - t0
    
    sim.cleanup()
    return elapsed

print("="*60)
print("   EdgeQuantum vs cuQuantum Native (OPTIMIZED)")
print("="*60)

for n_qubits in [20, 22, 24, 26]:
    n_gates = 100
    
    # Warm up
    benchmark_cuquantum_native(n_qubits, 10)
    benchmark_edgequantum(n_qubits, 10)
    
    # Actual benchmark
    cu_time = benchmark_cuquantum_native(n_qubits, n_gates)
    eq_time = benchmark_edgequantum(n_qubits, n_gates)
    
    speedup = cu_time / eq_time if eq_time > 0 else 0
    winner = "EdgeQuantum ✓" if eq_time <= cu_time else "cuQuantum ✓"
    
    print(f"\n{n_qubits}Q ({n_gates} gates):")
    print(f"  cuQuantum Native: {cu_time:.4f}s")
    print(f"  EdgeQuantum:      {eq_time:.4f}s")
    print(f"  Ratio:            {speedup:.2f}x ({winner})")

print("\n" + "="*60)
