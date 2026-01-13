#!/usr/bin/env python3
"""
Extreme Qubit Scaling Test - Jetson Memory Limit Discovery
Pushes quantum simulation to VRAM/DRAM exhaustion.
"""

import time
import numpy as np
import cupy as cp
import psutil
import gc
import subprocess
import json
from datetime import datetime

def get_memory_status():
    """Get comprehensive memory status."""
    # CPU memory
    mem = psutil.virtual_memory()
    cpu_used = mem.used / 1024**3
    cpu_total = mem.total / 1024**3
    cpu_avail = mem.available / 1024**3
    
    # GPU memory
    try:
        mempool = cp.get_default_memory_pool()
        gpu_used = mempool.used_bytes() / 1024**3
        gpu_total = mempool.total_bytes() / 1024**3
    except:
        gpu_used, gpu_total = 0, 0
    
    # nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(', ')
        nv_used = int(parts[0]) / 1024
        nv_total = int(parts[1]) / 1024
    except:
        nv_used, nv_total = gpu_used, gpu_total
    
    return {
        'cpu_used_gb': cpu_used,
        'cpu_total_gb': cpu_total,
        'cpu_avail_gb': cpu_avail,
        'gpu_used_gb': nv_used,
        'gpu_total_gb': nv_total
    }

def extreme_statevector_test(start_qubits=20, max_qubits=32):
    """
    Push state vector simulation to memory limits.
    """
    print("=" * 70)
    print("🚀 EXTREME QUBIT SCALING TEST - MEMORY LIMIT DISCOVERY")
    print("=" * 70)
    print(f"Starting from {start_qubits} qubits, pushing to {max_qubits} or OOM")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    mem_init = get_memory_status()
    print(f"\n📊 Initial Memory Status:")
    print(f"   CPU: {mem_init['cpu_used_gb']:.2f} / {mem_init['cpu_total_gb']:.2f} GB")
    print(f"   GPU: {mem_init['gpu_used_gb']:.2f} / {mem_init['gpu_total_gb']:.2f} GB")
    print("-" * 70)
    
    results = []
    max_achieved = 0
    
    for n_qubits in range(start_qubits, max_qubits + 1):
        dim = 2 ** n_qubits
        mem_required_gb = dim * 8 / 1024**3  # complex64 = 8 bytes
        
        print(f"\n🔬 [{n_qubits} qubits]")
        print(f"   Hilbert dim: {dim:,}")
        print(f"   Memory required: {mem_required_gb:.2f} GB (state vector only)")
        
        # Force garbage collection
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        mem_before = get_memory_status()
        
        try:
            # Allocate state vector on GPU
            print(f"   Allocating on GPU...")
            start_time = time.perf_counter()
            
            sv = cp.zeros(dim, dtype=cp.complex64)
            sv[0] = 1.0 + 0.0j
            cp.cuda.Device().synchronize()
            
            alloc_time = time.perf_counter() - start_time
            print(f"   ✓ GPU allocation: {alloc_time:.2f}s")
            
            # Apply a few gates to verify it works
            print(f"   Applying test gates...")
            H = cp.array([[1, 1], [1, -1]], dtype=cp.complex64) / np.sqrt(2)
            
            # Apply H to qubit 0 using reshape method
            shape = [2] * n_qubits
            sv_reshaped = sv.reshape(shape)
            
            # Apply to first qubit only (for speed)
            sv_flat = sv_reshaped.reshape(2, -1)
            sv_flat = cp.matmul(H, sv_flat)
            sv = sv_flat.flatten()
            
            cp.cuda.Device().synchronize()
            gate_time = time.perf_counter() - start_time - alloc_time
            
            # Compute norm to verify
            norm = float(cp.abs(cp.vdot(sv, sv)))
            
            mem_after = get_memory_status()
            
            result = {
                'qubits': n_qubits,
                'hilbert_dim': dim,
                'mem_required_gb': mem_required_gb,
                'alloc_time_s': alloc_time,
                'gate_time_s': gate_time,
                'cpu_used_gb': mem_after['cpu_used_gb'],
                'gpu_used_gb': mem_after['gpu_used_gb'],
                'norm': norm,
                'success': True
            }
            results.append(result)
            max_achieved = n_qubits
            
            print(f"   ✓ Gate applied in {gate_time:.2f}s")
            print(f"   ✓ Norm: {norm:.6f}")
            print(f"   📊 Memory: CPU={mem_after['cpu_used_gb']:.2f}GB, GPU={mem_after['gpu_used_gb']:.2f}GB")
            
            # Clean up for next iteration
            del sv, sv_reshaped, sv_flat
            
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"   ❌ GPU OUT OF MEMORY!")
            print(f"   Error: {e}")
            results.append({
                'qubits': n_qubits,
                'hilbert_dim': dim,
                'mem_required_gb': mem_required_gb,
                'error': 'GPU_OOM',
                'success': False
            })
            break
            
        except MemoryError as e:
            print(f"   ❌ CPU OUT OF MEMORY!")
            print(f"   Error: {e}")
            results.append({
                'qubits': n_qubits,
                'hilbert_dim': dim,
                'mem_required_gb': mem_required_gb,
                'error': 'CPU_OOM',
                'success': False
            })
            break
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'qubits': n_qubits,
                'hilbert_dim': dim,
                'mem_required_gb': mem_required_gb,
                'error': str(e),
                'success': False
            })
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXTREME SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':<8} {'Dim':<16} {'Mem(GB)':<12} {'Alloc(s)':<10} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        status = "✓ OK" if r['success'] else f"✗ {r.get('error', 'FAIL')}"
        alloc = f"{r.get('alloc_time_s', 0):.2f}" if r['success'] else "-"
        print(f"{r['qubits']:<8} {r['hilbert_dim']:<16,} {r['mem_required_gb']:<12.2f} {alloc:<10} {status:<10}")
    
    print("-" * 70)
    print(f"🏆 MAXIMUM ACHIEVED: {max_achieved} qubits")
    print(f"   Hilbert dimension: {2**max_achieved:,}")
    print(f"   State vector size: {2**max_achieved * 8 / 1024**3:.2f} GB")
    
    return results, max_achieved

def main():
    print("\n" + "🔥" * 35)
    print("  JETSON ORIN NANO - QUANTUM SIMULATION STRESS TEST")
    print("🔥" * 35 + "\n")
    
    # Start from 20 qubits (known to work)
    results, max_q = extreme_statevector_test(start_qubits=20, max_qubits=30)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': 'Jetson Orin Nano',
        'test': 'extreme_qubit_scaling',
        'max_qubits_achieved': max_q,
        'max_hilbert_dim': 2 ** max_q,
        'max_memory_gb': 2 ** max_q * 8 / 1024**3,
        'results': results
    }
    
    results_path = '/home/jetson/skim/edgeQuantum-iotj/data/extreme_scaling_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n" + "=" * 70)
    print("✅ STRESS TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
