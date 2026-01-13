#!/usr/bin/env python3
"""
EdgeQuantum Tiered Memory Simulator
AURORA-Q inspired DRAM offloading for >26 qubit simulation on Jetson.

Key idea:
- Split state vector into chunks that fit in GPU memory
- Keep active chunks on GPU, rest in DRAM (numpy)
- Prefetch next chunk asynchronously while computing current
"""

import numpy as np
import cupy as cp
import time
import gc
import json
from datetime import datetime
import psutil

class TieredStateVector:
    """
    Tiered memory state vector: GPU VRAM + CPU DRAM.
    Enables simulation beyond GPU memory limits.
    """
    
    def __init__(self, n_qubits, gpu_chunk_qubits=24):
        """
        Args:
            n_qubits: Total qubits to simulate
            gpu_chunk_qubits: Qubits per GPU chunk (24 = 128MB)
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.gpu_chunk_qubits = min(gpu_chunk_qubits, n_qubits)
        self.chunk_size = 2 ** self.gpu_chunk_qubits
        self.n_chunks = self.dim // self.chunk_size
        
        print(f"[TieredSV] {n_qubits} qubits = {self.dim:,} amplitudes")
        print(f"[TieredSV] Chunks: {self.n_chunks} x {self.chunk_size:,} = {self.chunk_size * 8 / 1024**2:.1f}MB each")
        print(f"[TieredSV] Total memory: {self.dim * 8 / 1024**3:.2f} GB (DRAM)")
        
        # State stored in DRAM as numpy arrays
        self.dram_chunks = []
        for i in range(self.n_chunks):
            chunk = np.zeros(self.chunk_size, dtype=np.complex64)
            if i == 0:
                chunk[0] = 1.0 + 0.0j  # |0...0> state
            self.dram_chunks.append(chunk)
        
        # GPU cache (single chunk at a time)
        self.gpu_chunk = None
        self.gpu_chunk_idx = -1
    
    def _load_chunk_to_gpu(self, chunk_idx):
        """Load a chunk from DRAM to GPU."""
        if self.gpu_chunk_idx == chunk_idx:
            return  # Already loaded
        
        # Save current GPU chunk back to DRAM if modified
        if self.gpu_chunk is not None and self.gpu_chunk_idx >= 0:
            self.dram_chunks[self.gpu_chunk_idx] = cp.asnumpy(self.gpu_chunk)
        
        # Load new chunk to GPU
        self.gpu_chunk = cp.asarray(self.dram_chunks[chunk_idx])
        self.gpu_chunk_idx = chunk_idx
    
    def _sync_gpu_to_dram(self):
        """Sync current GPU chunk back to DRAM."""
        if self.gpu_chunk is not None and self.gpu_chunk_idx >= 0:
            self.dram_chunks[self.gpu_chunk_idx] = cp.asnumpy(self.gpu_chunk)
    
    def apply_single_qubit_gate(self, gate, target_qubit):
        """
        Apply single-qubit gate using tiered memory.
        
        If target qubit is in the "chunk" qubits (lower bits), 
        we can apply gate within each chunk independently.
        If in "global" qubits (upper bits), need cross-chunk ops.
        """
        gate_gpu = cp.asarray(gate) if isinstance(gate, np.ndarray) else gate
        
        # Determine if gate is local to chunk or global
        chunk_qubits = self.gpu_chunk_qubits  # Lower qubits within chunk
        global_qubits = self.n_qubits - chunk_qubits  # Upper qubits for chunk index
        
        if target_qubit < chunk_qubits:
            # LOCAL GATE: Apply independently to each chunk
            for chunk_idx in range(self.n_chunks):
                self._load_chunk_to_gpu(chunk_idx)
                self._apply_gate_to_gpu_chunk(gate_gpu, target_qubit)
            self._sync_gpu_to_dram()
        else:
            # GLOBAL GATE: Requires cross-chunk communication
            # Pair up chunks that differ only in target qubit bit
            global_target = target_qubit - chunk_qubits
            
            for base_chunk in range(self.n_chunks):
                # Check if this chunk's global_target bit is 0
                if (base_chunk >> global_target) & 1 == 0:
                    partner_chunk = base_chunk | (1 << global_target)
                    if partner_chunk < self.n_chunks:
                        self._apply_global_gate(gate_gpu, base_chunk, partner_chunk)
    
    def _apply_gate_to_gpu_chunk(self, gate, target_qubit):
        """Apply gate to currently loaded GPU chunk."""
        shape = [2] * self.gpu_chunk_qubits
        sv_reshaped = self.gpu_chunk.reshape(shape)
        
        axes = list(range(self.gpu_chunk_qubits))
        axes.remove(target_qubit)
        axes.insert(0, target_qubit)
        sv_t = cp.transpose(sv_reshaped, axes)
        
        sv_flat = sv_t.reshape(2, -1)
        sv_flat = cp.matmul(gate, sv_flat)
        
        sv_t = sv_flat.reshape([2] + [2] * (self.gpu_chunk_qubits - 1))
        inv_axes = [0] * self.gpu_chunk_qubits
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        sv_reshaped = cp.transpose(sv_t, inv_axes)
        self.gpu_chunk = sv_reshaped.flatten()
    
    def _apply_global_gate(self, gate, chunk0_idx, chunk1_idx):
        """Apply gate across two chunks (for global qubits)."""
        # Load both chunks (one on GPU, one stays in RAM temporarily)
        chunk0 = cp.asarray(self.dram_chunks[chunk0_idx])
        chunk1 = cp.asarray(self.dram_chunks[chunk1_idx])
        
        # Stack and apply gate
        stacked = cp.stack([chunk0, chunk1], axis=0)  # (2, chunk_size)
        result = cp.matmul(gate, stacked)  # (2, chunk_size)
        
        # Save back
        self.dram_chunks[chunk0_idx] = cp.asnumpy(result[0])
        self.dram_chunks[chunk1_idx] = cp.asnumpy(result[1])
        
        del chunk0, chunk1, stacked, result
        cp.get_default_memory_pool().free_all_blocks()
    
    def get_norm(self):
        """Compute norm (should be 1.0)."""
        self._sync_gpu_to_dram()
        total = 0.0
        for chunk in self.dram_chunks:
            total += np.sum(np.abs(chunk) ** 2)
        return float(np.sqrt(total))
    
    def get_memory_usage(self):
        """Get memory usage stats."""
        dram_usage = sum(chunk.nbytes for chunk in self.dram_chunks) / 1024**3
        gpu_usage = self.gpu_chunk.nbytes / 1024**3 if self.gpu_chunk is not None else 0
        return {'dram_gb': dram_usage, 'gpu_gb': gpu_usage}


def H_gate():
    return np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)

def X_gate():
    return np.array([[0, 1], [1, 0]], dtype=np.complex64)


def run_tiered_scaling_test(qubit_configs, gpu_chunk_qubits=24):
    """Test tiered memory simulator at various qubit counts."""
    
    print("=" * 70)
    print("🚀 TIERED MEMORY QUANTUM SIMULATION (AURORA-Q Style)")
    print("=" * 70)
    print(f"GPU chunk size: {gpu_chunk_qubits} qubits = {2**gpu_chunk_qubits * 8 / 1024**2:.0f} MB")
    
    mem = psutil.virtual_memory()
    print(f"System DRAM: {mem.available / 1024**3:.1f} GB available / {mem.total / 1024**3:.1f} GB total")
    print("-" * 70)
    
    results = []
    
    for n_qubits in qubit_configs:
        print(f"\n🔬 [{n_qubits} qubits]")
        
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        required_dram = 2**n_qubits * 8 / 1024**3
        available = psutil.virtual_memory().available / 1024**3
        
        if required_dram > available * 0.9:
            print(f"   ⚠️ Required {required_dram:.2f}GB > Available {available:.2f}GB - SKIP")
            results.append({'qubits': n_qubits, 'error': 'DRAM insufficient'})
            break
        
        try:
            start = time.time()
            sv = TieredStateVector(n_qubits, gpu_chunk_qubits)
            init_time = time.time() - start
            print(f"   ✓ Initialized in {init_time:.2f}s")
            
            # Apply H gate to qubit 0 (local gate test)
            start = time.time()
            sv.apply_single_qubit_gate(H_gate(), 0)
            cp.cuda.Device().synchronize()
            local_gate_time = time.time() - start
            print(f"   ✓ Local gate (q0): {local_gate_time:.2f}s")
            
            # Apply H gate to highest qubit (global gate test)
            if n_qubits > gpu_chunk_qubits:
                start = time.time()
                sv.apply_single_qubit_gate(H_gate(), n_qubits - 1)
                cp.cuda.Device().synchronize()
                global_gate_time = time.time() - start
                print(f"   ✓ Global gate (q{n_qubits-1}): {global_gate_time:.2f}s")
            else:
                global_gate_time = 0
            
            norm = sv.get_norm()
            mem_usage = sv.get_memory_usage()
            
            print(f"   ✓ Norm: {norm:.6f}")
            print(f"   📊 Memory: DRAM={mem_usage['dram_gb']:.2f}GB, GPU={mem_usage['gpu_gb']:.3f}GB")
            
            results.append({
                'qubits': n_qubits,
                'hilbert_dim': 2**n_qubits,
                'init_time_s': init_time,
                'local_gate_time_s': local_gate_time,
                'global_gate_time_s': global_gate_time,
                'dram_gb': mem_usage['dram_gb'],
                'gpu_gb': mem_usage['gpu_gb'],
                'norm': norm,
                'success': True
            })
            
            del sv
            
        except MemoryError as e:
            print(f"   ❌ MEMORY ERROR: {e}")
            results.append({'qubits': n_qubits, 'error': 'MemoryError'})
            break
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'qubits': n_qubits, 'error': str(e)})
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TIERED MEMORY SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':<8} {'Dim':<16} {'DRAM(GB)':<10} {'Init(s)':<10} {'Gate(s)':<10} {'Status':<10}")
    print("-" * 70)
    
    max_qubits = 0
    for r in results:
        if r.get('success'):
            max_qubits = r['qubits']
            print(f"{r['qubits']:<8} {r['hilbert_dim']:<16,} {r['dram_gb']:<10.2f} "
                  f"{r['init_time_s']:<10.2f} {r['local_gate_time_s']:<10.2f} ✓")
        else:
            print(f"{r['qubits']:<8} {'--':<16} {'--':<10} {'--':<10} {'--':<10} ✗ {r.get('error', '')}")
    
    print("-" * 70)
    print(f"🏆 MAXIMUM ACHIEVED: {max_qubits} qubits with DRAM offloading")
    
    return results, max_qubits


def main():
    print("\n" + "🔥" * 35)
    print("  AURORA-Q STYLE TIERED MEMORY SIMULATION")
    print("🔥" * 35 + "\n")
    
    # Test from 26 to 32 qubits (beyond GPU-only limit)
    qubit_configs = [26, 27, 28, 29, 30]
    
    results, max_q = run_tiered_scaling_test(qubit_configs, gpu_chunk_qubits=24)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': 'Jetson Orin Nano',
        'method': 'AURORA-Q style DRAM offloading',
        'gpu_chunk_qubits': 24,
        'max_qubits': max_q,
        'results': results
    }
    
    results_path = '/home/jetson/skim/edgeQuantum-iotj/data/tiered_memory_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
