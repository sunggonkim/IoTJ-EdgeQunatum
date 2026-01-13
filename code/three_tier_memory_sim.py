#!/usr/bin/env python3
"""
EdgeQuantum 3-Tier Memory Simulator
AURORA-Q style: GPU VRAM → DRAM → Storage (NVMe/SSD)

Enables 29+ qubit simulation by offloading to disk.
"""

import numpy as np
import cupy as cp
import time
import gc
import json
import os
import tempfile
import shutil
from datetime import datetime
import psutil

class ThreeTierStateVector:
    """
    3-Tier Memory State Vector: GPU → DRAM → Storage
    
    Memory hierarchy:
    - GPU VRAM: Active computation (single chunk)
    - DRAM: Hot cache (limited chunks)
    - Storage: Cold storage (all chunks)
    """
    
    def __init__(self, n_qubits, gpu_chunk_qubits=22, dram_cache_chunks=4, 
                 storage_dir=None):
        """
        Args:
            n_qubits: Total qubits
            gpu_chunk_qubits: Qubits per GPU chunk (22 = 32MB)
            dram_cache_chunks: Max chunks in DRAM cache
            storage_dir: Directory for chunk files
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.gpu_chunk_qubits = min(gpu_chunk_qubits, n_qubits)
        self.chunk_size = 2 ** self.gpu_chunk_qubits
        self.n_chunks = self.dim // self.chunk_size
        self.dram_cache_max = min(dram_cache_chunks, self.n_chunks)
        
        # Storage directory
        if storage_dir is None:
            self.storage_dir = tempfile.mkdtemp(prefix='edgequantum_')
        else:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
        
        chunk_mem_mb = self.chunk_size * 8 / 1024**2
        total_mem_gb = self.dim * 8 / 1024**3
        dram_cache_gb = self.dram_cache_max * self.chunk_size * 8 / 1024**3
        
        print(f"[3-Tier] {n_qubits} qubits = {self.dim:,} amplitudes")
        print(f"[3-Tier] Chunks: {self.n_chunks} x {self.chunk_size:,} = {chunk_mem_mb:.1f}MB each")
        print(f"[3-Tier] Total: {total_mem_gb:.2f} GB (on storage)")
        print(f"[3-Tier] DRAM cache: {self.dram_cache_max} chunks = {dram_cache_gb:.2f} GB")
        print(f"[3-Tier] Storage: {self.storage_dir}")
        
        # Stats - initialize BEFORE writing chunks
        self.storage_reads = 0
        self.storage_writes = 0
        
        # DRAM cache: LRU cache
        self.dram_cache = {}
        self.lru_order = []
        
        # GPU: single chunk
        self.gpu_chunk = None
        self.gpu_chunk_idx = -1
        
        # Initialize state vector |0...0>
        # Write all chunks to storage
        print("[3-Tier] Initializing state vector to storage...")
        for i in range(self.n_chunks):
            chunk = np.zeros(self.chunk_size, dtype=np.complex64)
            if i == 0:
                chunk[0] = 1.0 + 0.0j
            self._write_chunk_to_storage(i, chunk)
    
    def _get_chunk_path(self, chunk_idx):
        return os.path.join(self.storage_dir, f"chunk_{chunk_idx:06d}.bin")
    
    def _write_chunk_to_storage(self, chunk_idx, data):
        """Write chunk to storage."""
        path = self._get_chunk_path(chunk_idx)
        data.tofile(path)
        self.storage_writes += 1
    
    def _read_chunk_from_storage(self, chunk_idx):
        """Read chunk from storage."""
        path = self._get_chunk_path(chunk_idx)
        data = np.fromfile(path, dtype=np.complex64)
        self.storage_reads += 1
        return data
    
    def _load_to_dram_cache(self, chunk_idx):
        """Load chunk to DRAM cache (LRU eviction)."""
        if chunk_idx in self.dram_cache:
            # Move to front of LRU
            self.lru_order.remove(chunk_idx)
            self.lru_order.insert(0, chunk_idx)
            return
        
        # Evict if cache full
        while len(self.dram_cache) >= self.dram_cache_max:
            # Evict LRU (last in list)
            evict_idx = self.lru_order.pop()
            evict_data = self.dram_cache.pop(evict_idx)
            self._write_chunk_to_storage(evict_idx, evict_data)
        
        # Load from storage
        self.dram_cache[chunk_idx] = self._read_chunk_from_storage(chunk_idx)
        self.lru_order.insert(0, chunk_idx)
    
    def _load_chunk_to_gpu(self, chunk_idx):
        """Load chunk to GPU (via DRAM cache)."""
        if self.gpu_chunk_idx == chunk_idx:
            return
        
        # Save current GPU chunk back to DRAM
        if self.gpu_chunk is not None and self.gpu_chunk_idx >= 0:
            if self.gpu_chunk_idx in self.dram_cache:
                self.dram_cache[self.gpu_chunk_idx] = cp.asnumpy(self.gpu_chunk)
            else:
                self._write_chunk_to_storage(self.gpu_chunk_idx, cp.asnumpy(self.gpu_chunk))
        
        # Load to DRAM cache first
        self._load_to_dram_cache(chunk_idx)
        
        # Then to GPU
        self.gpu_chunk = cp.asarray(self.dram_cache[chunk_idx])
        self.gpu_chunk_idx = chunk_idx
    
    def _sync_all_to_storage(self):
        """Sync GPU and DRAM cache to storage."""
        # Sync GPU
        if self.gpu_chunk is not None and self.gpu_chunk_idx >= 0:
            if self.gpu_chunk_idx in self.dram_cache:
                self.dram_cache[self.gpu_chunk_idx] = cp.asnumpy(self.gpu_chunk)
            else:
                self._write_chunk_to_storage(self.gpu_chunk_idx, cp.asnumpy(self.gpu_chunk))
        
        # Sync DRAM cache
        for chunk_idx, data in self.dram_cache.items():
            self._write_chunk_to_storage(chunk_idx, data)
        
        self.dram_cache.clear()
        self.lru_order.clear()
        self.gpu_chunk = None
        self.gpu_chunk_idx = -1
    
    def apply_single_qubit_gate(self, gate, target_qubit):
        """Apply single-qubit gate using 3-tier memory."""
        gate_gpu = cp.asarray(gate) if isinstance(gate, np.ndarray) else gate
        
        chunk_qubits = self.gpu_chunk_qubits
        
        if target_qubit < chunk_qubits:
            # LOCAL GATE: process each chunk independently
            for chunk_idx in range(self.n_chunks):
                self._load_chunk_to_gpu(chunk_idx)
                self._apply_gate_to_gpu_chunk(gate_gpu, target_qubit)
        else:
            # GLOBAL GATE: cross-chunk operations
            global_target = target_qubit - chunk_qubits
            
            for base_chunk in range(self.n_chunks):
                if (base_chunk >> global_target) & 1 == 0:
                    partner_chunk = base_chunk | (1 << global_target)
                    if partner_chunk < self.n_chunks:
                        self._apply_global_gate(gate_gpu, base_chunk, partner_chunk)
    
    def _apply_gate_to_gpu_chunk(self, gate, target_qubit):
        """Apply gate to GPU chunk."""
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
        """Apply gate across two chunks."""
        # Load both to DRAM
        self._load_to_dram_cache(chunk0_idx)
        self._load_to_dram_cache(chunk1_idx)
        
        # Load to GPU and apply
        chunk0 = cp.asarray(self.dram_cache[chunk0_idx])
        chunk1 = cp.asarray(self.dram_cache[chunk1_idx])
        
        stacked = cp.stack([chunk0, chunk1], axis=0)
        result = cp.matmul(gate, stacked)
        
        self.dram_cache[chunk0_idx] = cp.asnumpy(result[0])
        self.dram_cache[chunk1_idx] = cp.asnumpy(result[1])
        
        del chunk0, chunk1, stacked, result
        cp.get_default_memory_pool().free_all_blocks()
    
    def get_norm(self):
        """Compute norm (should be 1.0)."""
        self._sync_all_to_storage()
        total = 0.0
        for i in range(self.n_chunks):
            chunk = self._read_chunk_from_storage(i)
            total += np.sum(np.abs(chunk) ** 2)
        return float(np.sqrt(total))
    
    def get_stats(self):
        """Get I/O statistics."""
        chunk_mb = self.chunk_size * 8 / 1024**2
        return {
            'storage_reads': self.storage_reads,
            'storage_writes': self.storage_writes,
            'total_io_gb': (self.storage_reads + self.storage_writes) * chunk_mb / 1024
        }
    
    def cleanup(self):
        """Remove storage directory."""
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)


def H_gate():
    return np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


def run_3tier_scaling_test(qubit_configs, gpu_chunk_qubits=22, dram_cache_chunks=4):
    """Test 3-tier memory simulator."""
    
    print("=" * 70)
    print("🚀 3-TIER MEMORY SIMULATION: GPU → DRAM → STORAGE")
    print("=" * 70)
    print(f"GPU chunk: {gpu_chunk_qubits} qubits = {2**gpu_chunk_qubits * 8 / 1024**2:.0f} MB")
    print(f"DRAM cache: {dram_cache_chunks} chunks")
    
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    print(f"DRAM: {mem.available / 1024**3:.1f} GB available")
    print(f"Disk: {disk.free / 1024**3:.1f} GB available")
    print("-" * 70)
    
    results = []
    storage_base = '/tmp/edgequantum_3tier'
    
    for n_qubits in qubit_configs:
        print(f"\n🔬 [{n_qubits} qubits]")
        
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        required_disk = 2**n_qubits * 8 / 1024**3
        available = psutil.disk_usage('/').free / 1024**3
        
        if required_disk > available * 0.8:
            print(f"   ⚠️ Required {required_disk:.2f}GB > Available {available:.2f}GB - SKIP")
            results.append({'qubits': n_qubits, 'error': 'Disk insufficient'})
            break
        
        storage_dir = f"{storage_base}_{n_qubits}q"
        
        try:
            start = time.time()
            sv = ThreeTierStateVector(
                n_qubits, 
                gpu_chunk_qubits=gpu_chunk_qubits,
                dram_cache_chunks=dram_cache_chunks,
                storage_dir=storage_dir
            )
            init_time = time.time() - start
            print(f"   ✓ Initialized in {init_time:.2f}s")
            
            # Local gate
            start = time.time()
            sv.apply_single_qubit_gate(H_gate(), 0)
            cp.cuda.Device().synchronize()
            local_time = time.time() - start
            print(f"   ✓ Local gate (q0): {local_time:.2f}s")
            
            # Global gate
            if n_qubits > gpu_chunk_qubits:
                start = time.time()
                sv.apply_single_qubit_gate(H_gate(), n_qubits - 1)
                cp.cuda.Device().synchronize()
                global_time = time.time() - start
                print(f"   ✓ Global gate (q{n_qubits-1}): {global_time:.2f}s")
            else:
                global_time = 0
            
            norm = sv.get_norm()
            stats = sv.get_stats()
            
            print(f"   ✓ Norm: {norm:.6f}")
            print(f"   📊 I/O: {stats['storage_reads']} reads, {stats['storage_writes']} writes = {stats['total_io_gb']:.2f} GB")
            
            results.append({
                'qubits': n_qubits,
                'hilbert_dim': 2**n_qubits,
                'storage_gb': required_disk,
                'init_time_s': init_time,
                'local_gate_time_s': local_time,
                'global_gate_time_s': global_time,
                'io_reads': stats['storage_reads'],
                'io_writes': stats['storage_writes'],
                'io_total_gb': stats['total_io_gb'],
                'norm': norm,
                'success': True
            })
            
            sv.cleanup()
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'qubits': n_qubits, 'error': str(e)})
            # Cleanup
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 3-TIER SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':<8} {'Dim':<16} {'Disk(GB)':<10} {'Init(s)':<10} {'Gate(s)':<10} {'I/O(GB)':<10}")
    print("-" * 70)
    
    max_qubits = 0
    for r in results:
        if r.get('success'):
            max_qubits = r['qubits']
            print(f"{r['qubits']:<8} {r['hilbert_dim']:<16,} {r['storage_gb']:<10.2f} "
                  f"{r['init_time_s']:<10.2f} {r['local_gate_time_s']:<10.2f} {r['io_total_gb']:<10.2f}")
        else:
            print(f"{r['qubits']:<8} {'--':<16} {'--':<10} {'--':<10} {'--':<10} ✗ {r.get('error', '')}")
    
    print("-" * 70)
    print(f"🏆 MAXIMUM ACHIEVED: {max_qubits} qubits with 3-tier memory")
    
    return results, max_qubits


def main():
    print("\n" + "🔥" * 35)
    print("  3-TIER MEMORY: GPU → DRAM → STORAGE")
    print("🔥" * 35 + "\n")
    
    # Test 29-32 qubits (beyond DRAM-only limit)
    qubit_configs = [28, 29, 30, 31]
    
    results, max_q = run_3tier_scaling_test(
        qubit_configs, 
        gpu_chunk_qubits=22,  # 32MB chunks
        dram_cache_chunks=8   # 256MB DRAM cache
    )
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': 'Jetson Orin Nano',
        'method': '3-tier (GPU + DRAM + Storage)',
        'gpu_chunk_qubits': 22,
        'dram_cache_chunks': 8,
        'max_qubits': max_q,
        'results': results
    }
    
    results_path = '/home/jetson/skim/edgeQuantum-iotj/data/3tier_memory_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
