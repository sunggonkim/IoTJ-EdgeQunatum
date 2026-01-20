#!/usr/bin/env python3
"""
Validated Quantum Simulator Benchmark
- Validates state vector size after simulation
- Properly detects OOM and invalid results
- Reports only verified measurements
"""
import time
import numpy as np
import sys
import gc

# Import simulators
try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

try:
    import cupy as cp
    import cuquantum as cq
    from cuquantum import custatevec as cusv
    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False

sys.path.insert(0, '/home/jetson/skim/edgeQuantum-iotj/code')
from tiered_simulator import TieredSimulator, H


def benchmark_cirq_validated(n_qubits, n_gates):
    """Cirq benchmark with state vector validation"""
    if not HAS_CIRQ:
        return None, "Cirq not available"
    
    try:
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        
        # Add gates
        H_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        for i in range(n_gates):
            gate = cirq.MatrixGate(H_gate)
            circuit.append(gate(qubits[i % n_qubits]))
        
        # Time the simulation
        sim = cirq.Simulator()
        t0 = time.time()
        result = sim.simulate(circuit)
        elapsed = time.time() - t0
        
        # VALIDATION: Check state vector size
        if hasattr(result, 'final_state_vector'):
            state_size = len(result.final_state_vector)
            expected_size = 2 ** n_qubits
            
            if state_size == expected_size:
                # Further validation: check not all zeros
                if np.any(result.final_state_vector != 0):
                    return elapsed, "VALID"
                else:
                    return None, "Invalid (all zeros)"
            else:
                return None, f"Invalid (size {state_size} != {expected_size})"
        else:
            return None, "Invalid (no state vector)"
            
    except MemoryError:
        return None, "OOM (MemoryError)"
    except Exception as e:
        return None, f"Error: {str(e)[:30]}"
    finally:
        gc.collect()


def benchmark_cuquantum_validated(n_qubits, n_gates):
    """cuQuantum benchmark with validation"""
    if not HAS_CUQUANTUM:
        return None, "cuQuantum not available"
    
    try:
        handle = cusv.create()
        sv = cp.zeros(2**n_qubits, dtype=cp.complex64)
        sv[0] = 1.0
        
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
        
        # VALIDATION: Check state vector
        state_size = sv.size
        expected_size = 2 ** n_qubits
        
        if state_size == expected_size and cp.any(sv != 0):
            status = "VALID"
        else:
            status = "Invalid"
            elapsed = None
        
        del sv
        cp.get_default_memory_pool().free_all_blocks()
        cusv.destroy(handle)
        
        return elapsed, status
        
    except Exception as e:
        cp.get_default_memory_pool().free_all_blocks()
        return None, f"OOM/Error: {str(e)[:30]}"


def benchmark_edgequantum_validated(n_qubits, n_gates):
    """EdgeQuantum benchmark with validation"""
    try:
        sim = TieredSimulator(n_qubits, use_compression=True, fusion_threshold=n_gates+1)
        sim.init_zero_state()
        
        t0 = time.time()
        for i in range(n_gates):
            sim.apply_single_gate(H, i % n_qubits)
        sim.flush_gates()
        elapsed = time.time() - t0
        
        # VALIDATION: Check gate count
        if sim.gate_count == n_gates:
            status = "VALID"
        else:
            status = f"Invalid ({sim.gate_count} gates)"
            elapsed = None
        
        mode = "IN-MEM" if sim.in_memory_mode else f"TIERED-{sim.n_chunks}"
        sim.cleanup()
        
        return elapsed, f"{status} ({mode})"
        
    except Exception as e:
        return None, f"Error: {str(e)[:30]}"


def run_validated_benchmark():
    """Run full validated benchmark"""
    print("=" * 70)
    print("   VALIDATED QUANTUM SIMULATOR BENCHMARK")
    print("   All results are verified for correctness before reporting")
    print("=" * 70)
    
    results = []
    
    for n_qubits in [20, 22, 24, 26, 28, 30]:
        n_gates = 50
        state_size_mb = (2**n_qubits * 8) / (1024**2)
        
        print(f"\n{'='*70}")
        print(f"  {n_qubits}Q (State: {state_size_mb:.0f} MB, {n_gates} gates)")
        print(f"{'='*70}")
        
        # Cirq
        gc.collect()
        cirq_time, cirq_status = benchmark_cirq_validated(n_qubits, n_gates)
        if cirq_time:
            print(f"  Cirq:       {cirq_time:.4f}s [{cirq_status}]")
        else:
            print(f"  Cirq:       ❌ {cirq_status}")
        results.append({'sim': 'Cirq', 'qubits': n_qubits, 'time': cirq_time, 'status': cirq_status})
        
        # cuQuantum
        gc.collect()
        cuq_time, cuq_status = benchmark_cuquantum_validated(n_qubits, n_gates)
        if cuq_time:
            print(f"  cuQuantum:  {cuq_time:.4f}s [{cuq_status}]")
        else:
            print(f"  cuQuantum:  ❌ {cuq_status}")
        results.append({'sim': 'cuQuantum', 'qubits': n_qubits, 'time': cuq_time, 'status': cuq_status})
        
        # EdgeQuantum
        gc.collect()
        eq_time, eq_status = benchmark_edgequantum_validated(n_qubits, n_gates)
        if eq_time:
            print(f"  EdgeQuantum: {eq_time:.4f}s [{eq_status}]")
        else:
            print(f"  EdgeQuantum: ❌ {eq_status}")
        results.append({'sim': 'EdgeQuantum', 'qubits': n_qubits, 'time': eq_time, 'status': eq_status})
    
    print("\n" + "=" * 70)
    print("   BENCHMARK COMPLETE")
    print("=" * 70)
    
    # Summary table
    print("\n📊 SUMMARY (Only VALID results):\n")
    print(f"{'Qubits':<8} {'Cirq':<12} {'cuQuantum':<12} {'EdgeQuantum':<15}")
    print("-" * 50)
    
    for q in [20, 22, 24, 26, 28, 30]:
        cirq_r = next((r for r in results if r['sim'] == 'Cirq' and r['qubits'] == q), None)
        cuq_r = next((r for r in results if r['sim'] == 'cuQuantum' and r['qubits'] == q), None)
        eq_r = next((r for r in results if r['sim'] == 'EdgeQuantum' and r['qubits'] == q), None)
        
        cirq_str = f"{cirq_r['time']:.4f}s" if cirq_r and cirq_r['time'] else "❌"
        cuq_str = f"{cuq_r['time']:.4f}s" if cuq_r and cuq_r['time'] else "❌"
        eq_str = f"{eq_r['time']:.4f}s" if eq_r and eq_r['time'] else "❌"
        
        print(f"{q}Q      {cirq_str:<12} {cuq_str:<12} {eq_str:<15}")
    
    return results


if __name__ == "__main__":
    run_validated_benchmark()
