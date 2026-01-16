#!/usr/bin/env python3
"""
EdgeQuantum TieredSimulator - True Zero-Copy Quantum Simulator for IoT Edge Devices

Key Features:
1. True Double Buffering: ThreadPoolExecutor for async I/O prefetching
2. CUDA Managed Memory: True Zero-Copy via cudaMallocManaged (custatevec compatible)
3. Gate Fusion: Batches multiple gates before I/O to minimize disk thrashing
4. Async Write-Back Queue: Overlaps disk writes with next chunk processing
5. Race Condition Protection: Safe double buffering with write future tracking

Author: EdgeQuantum Team (Sunggon Kim)
Target: Jetson Orin Nano (Unified Memory Architecture)
"""
import os
import sys
import time
import json
import tempfile
import numpy as np
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

# Check for cuQuantum
try:
    import cupy as cp
    import cupy.cuda.runtime as cuda_runtime  # Low-level CUDA Runtime API
    import cuquantum as cq
    from cuquantum import custatevec as cusv
    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False
    print("⚠️ cuQuantum not available")

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Gate definitions
H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
RZ = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex64)
RY = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex64)


# ============================================================
# MANAGED MEMORY BUFFER (Unified Virtual Memory - custatevec Compatible)
# ============================================================

class ManagedMemoryBuffer:
    """
    Wrapper for CUDA Managed Memory (Unified Virtual Memory).
    
    KEY INSIGHT:
    - cudaMallocManaged memory is tagged as "Device/Managed" type
    - custatevec accepts Managed Memory (passes internal validation)
    - On Jetson (Tegra), Managed Memory physically resides in system RAM
    - Result: TRUE ZERO-COPY that custatevec accepts!
    
    This solves the Mapped Memory rejection issue while achieving
    the same zero-copy performance on Jetson's unified memory architecture.
    """
    
    # cudaMemAttachGlobal = 1 (accessible from all GPUs and CPU)
    cudaMemAttachGlobal = 1
    
    def __init__(self, size_bytes):
        self.size = size_bytes
        
        # Allocate Managed Memory via CuPy runtime API (cudaMallocManaged)
        # This returns a device pointer that is accessible from both CPU and GPU
        self.ptr = cp.cuda.runtime.mallocManaged(size_bytes, self.cudaMemAttachGlobal)
    
    def get_numpy_view(self, dtype):
        """CPU access view (Zero-Copy on Jetson)
        
        Creates a NumPy array that directly accesses the managed memory.
        On Jetson, no data migration occurs - CPU and GPU share the same RAM.
        """
        num_elements = self.size // np.dtype(dtype).itemsize
        # Create ctypes buffer from the managed pointer
        buffer_type = ctypes.c_char * self.size
        c_buf = buffer_type.from_address(self.ptr)
        # Create numpy array wrapper (zero-copy view)
        arr = np.ndarray((num_elements,), dtype=dtype, buffer=c_buf)
        arr.flags.writeable = True
        return arr
    
    def get_cupy_view(self, shape, dtype):
        """GPU access view (Zero-Copy on Jetson)
        
        Creates a CuPy array directly on the managed memory.
        custatevec sees this as valid GPU memory.
        """
        # Create UnownedMemory wrapping the managed pointer
        mem = cp.cuda.UnownedMemory(self.ptr, self.size, owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(shape, dtype=dtype, memptr=memptr)
    
    def prefetch_to_device(self, device_id=0, stream=None):
        """Optional: Hint driver to optimize for GPU access"""
        pass  # Jetson handles this via hardware cache coherence
    
    def prefetch_to_host(self, stream=None):
        """Optional: Hint driver to optimize for CPU access"""
        pass  # Jetson handles this via hardware cache coherence
    
    def free(self):
        """Release managed memory"""
        if self.ptr:
            try:
                cp.cuda.runtime.free(self.ptr)
            except:
                pass
            self.ptr = 0
    
    def __del__(self):
        self.free()


class TieredSimulator:
    """
    IoT-Optimized Tiered Memory Quantum Simulator
    
    Key Features:
    - True Async Prefetching: CPU reads Chunk N+1 while GPU computes Chunk N
    - Managed Memory: Zero-copy GPU access via cudaMallocManaged
    - Write-Back Queue: Async disk writes overlap with computation
    - Gate Fusion: Batches gates to minimize I/O operations
    """
    
    def __init__(self, n_qubits, gpu_chunk_qubits=26, use_compression=True, 
                 fusion_threshold=10, use_managed_memory=True):
        self.n = n_qubits
        self.gpu_n = min(gpu_chunk_qubits, n_qubits)
        self.chunk_size = 2**self.gpu_n
        self.n_chunks = 2**(n_qubits - self.gpu_n)
        self.use_compression = use_compression and HAS_LZ4
        self.storage_dir = tempfile.mkdtemp(prefix=f'eq_{n_qubits}q_')
        self.gate_count = 0
        self.total_gate_time = 0
        self.fusion_threshold = fusion_threshold
        self.use_managed_memory = use_managed_memory
        
        # IN-MEMORY MODE: When state fits in single chunk, skip file I/O entirely
        self.in_memory_mode = (self.n_chunks == 1)
        self.in_memory_state = None  # GPU state vector for in-memory mode
        
        # Gate queue for fusion
        self.gate_queue = deque()
        
        # Async I/O infrastructure (only needed for multi-chunk mode)
        if not self.in_memory_mode:
            self.io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='eq_io')
            self.write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eq_write')
            self.pending_writes = []
        else:
            self.io_executor = None
            self.write_executor = None
            self.pending_writes = []
        
        # Performance metrics
        self.metrics = {
            'io_wait_time': 0,
            'compute_time': 0,
            'write_time': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0
        }
        
        # Keep references to raw memory objects to prevent GC
        self._raw_mem_a = None
        self._raw_mem_b = None
        
        if HAS_CUQUANTUM:
            self.handle = cusv.create()
            self.stream_compute = cp.cuda.Stream(non_blocking=True)
            
            if self.in_memory_mode:
                # IN-MEMORY: Allocate state directly in GPU memory (like cuQuantum Native)
                self.in_memory_state = cp.zeros(2**n_qubits, dtype=cp.complex64)
                print(f"[EdgeQuantum] {n_qubits}Q: IN-MEMORY MODE (cuQuantum-speed)")
            else:
                # TIERED: Allocate double buffers for async I/O
                if self.use_managed_memory:
                    self._init_managed_buffers()
                else:
                    self._init_pinned_buffers()
                print(f"[EdgeQuantum] {n_qubits}Q: {self.n_chunks} chunks × {self.chunk_size*8/1e6:.1f}MB")
                mode = "Managed (Zero-Copy)" if self.use_managed_memory else "Pinned"
                print(f"[EdgeQuantum] Memory: {mode}, Async Prefetch: ON, Gate Fusion: {fusion_threshold}")
    
    def _init_managed_buffers(self):
        """Initialize Managed Memory (Unified Virtual Memory)
        
        TRUE ZERO-COPY with custatevec compatibility:
        - cudaMallocManaged memory is tagged as Device/Managed type
        - custatevec accepts this as valid GPU memory
        - On Jetson, physically resides in system RAM (no migration)
        """
        bytes_per_chunk = self.chunk_size * 8  # complex64 = 8 bytes
        
        try:
            # Use ManagedMemoryBuffer (cudaMallocManaged internally)
            self._raw_mem_a = ManagedMemoryBuffer(bytes_per_chunk)
            self._raw_mem_b = ManagedMemoryBuffer(bytes_per_chunk)
            
            # CPU Views (for Disk I/O) - same physical memory
            self.buffer_a = self._raw_mem_a.get_numpy_view(np.complex64)
            self.buffer_b = self._raw_mem_b.get_numpy_view(np.complex64)
            
            # GPU Views (for custatevec) - same physical memory, custatevec accepts this!
            self.gpu_view_a = self._raw_mem_a.get_cupy_view((self.chunk_size,), np.complex64)
            self.gpu_view_b = self._raw_mem_b.get_cupy_view((self.chunk_size,), np.complex64)
            
            self._managed_mode = True
            print("[EdgeQuantum] ✅ TRUE ZERO-COPY: Managed Memory (custatevec compatible)")
            
        except Exception as e:
            # Fallback to pinned memory if managed allocation fails
            print(f"[EdgeQuantum] ⚠️ Managed memory failed: {e}")
            print("[EdgeQuantum] Falling back to pinned memory")
            self._init_pinned_buffers()
            self._managed_mode = False
    
    def _init_pinned_buffers(self):
        """Initialize pinned memory buffers (for PCIe systems or fallback)"""
        bytes_per_chunk = self.chunk_size * 8
        
        self.pinned_buffer_a = cp.cuda.alloc_pinned_memory(bytes_per_chunk)
        self.pinned_buffer_b = cp.cuda.alloc_pinned_memory(bytes_per_chunk)
        
        self.buffer_a = np.frombuffer(self.pinned_buffer_a, dtype=np.complex64)
        self.buffer_b = np.frombuffer(self.pinned_buffer_b, dtype=np.complex64)
        
        self._managed_mode = False
    
    def _get_buffer(self, which):
        """Get buffer by name ('a' or 'b')"""
        return self.buffer_a if which == 'a' else self.buffer_b
    
    def _get_gpu_array(self, which):
        """Get GPU array from buffer
        
        TRUE ZERO-COPY: Returns CuPy view of Managed Memory.
        custatevec operates directly on this - no memcpy needed!
        GPU writes go directly to RAM on Jetson.
        """
        if self._managed_mode:
            # Return cached GPU view (True Zero-Copy via Managed Memory)
            return self.gpu_view_a if which == 'a' else self.gpu_view_b
        else:
            # Fallback: Standard copy from pinned to GPU
            buf = self._get_buffer(which)
            return cp.asarray(buf)
    
    def init_zero_state(self):
        """Initialize |0⟩^n state"""
        if self.in_memory_mode:
            # IN-MEMORY: Direct GPU initialization (no file I/O)
            self.in_memory_state[:] = 0
            self.in_memory_state[0] = 1.0
            cp.cuda.Device().synchronize()
        else:
            # TIERED: Write chunks to disk
            for i in range(self.n_chunks):
                chunk = np.zeros(self.chunk_size, dtype=np.complex64)
                if i == 0:
                    chunk[0] = 1.0
                self._write_chunk_sync(i, chunk)
    
    def _write_chunk_sync(self, idx, data):
        """Synchronous chunk write (for initialization)"""
        path = os.path.join(self.storage_dir, f"{idx}.bin")
        if self.use_compression:
            compressed = lz4.compress(data.tobytes(), compression_level=1)
            with open(path, 'wb') as f:
                f.write(compressed)
        else:
            with open(path, 'wb') as f:
                f.write(data.tobytes())
    
    def _write_chunk_async(self, idx, data):
        """Async chunk write (non-blocking)"""
        def _write():
            t0 = time.time()
            self._write_chunk_sync(idx, data)
            return time.time() - t0
        
        future = self.write_executor.submit(_write)
        self.pending_writes.append(future)
        return future
    
    def _read_chunk_to_buffer(self, idx, buffer_name):
        """Read chunk into specified buffer (for prefetching)"""
        path = os.path.join(self.storage_dir, f"{idx}.bin")
        with open(path, 'rb') as f:
            data = f.read()
        if self.use_compression:
            data = lz4.decompress(data)
        
        buf = self._get_buffer(buffer_name)
        np.copyto(buf, np.frombuffer(data, dtype=np.complex64))
        return buffer_name
    
    def queue_gate(self, gate_matrix, target_qubit):
        """Queue a gate for fusion (deferred execution)"""
        self.gate_queue.append((gate_matrix.copy(), target_qubit))
        self.gate_count += 1
        
        if len(self.gate_queue) >= self.fusion_threshold:
            self.flush_gates()
    
    def flush_gates(self):
        """Execute all queued gates with optimized async I/O"""
        if not self.gate_queue:
            return
        
        t0 = time.time()
        
        if self.in_memory_mode:
            # IN-MEMORY MODE: Direct GPU execution (cuQuantum-speed)
            self._apply_gates_in_memory()
        else:
            # TIERED MODE: Async I/O with prefetching
            # Separate local and global gates
            local_gates = [(g, t) for g, t in self.gate_queue if t < self.gpu_n]
            global_gates = [(g, t) for g, t in self.gate_queue if t >= self.gpu_n]
            
            # Process local gates with true async prefetching
            if local_gates:
                self._apply_fused_local_gates_async(local_gates)
            
            # Process global gates
            for gate, target in global_gates:
                self._apply_global_gate(gate, target)
        
        self.gate_queue.clear()
        self.total_gate_time += time.time() - t0
    
    def _apply_gates_in_memory(self):
        """Apply all queued gates directly on GPU memory (IN-MEMORY MODE)"""
        t_compute_start = time.time()
        
        with self.stream_compute:
            for gate_matrix, target_qubit in self.gate_queue:
                matrix_gpu = cp.asarray(gate_matrix)
                targets = np.array([target_qubit], dtype=np.int32)
                cusv.apply_matrix(
                    handle=self.handle,
                    sv=self.in_memory_state.data.ptr,
                    sv_data_type=cq.cudaDataType.CUDA_C_32F,
                    n_index_bits=self.n,
                    matrix=matrix_gpu.data.ptr,
                    matrix_data_type=cq.cudaDataType.CUDA_C_32F,
                    layout=cusv.MatrixLayout.ROW,
                    adjoint=0,
                    targets=targets.ctypes.data,
                    n_targets=1,
                    controls=0,
                    control_bit_values=0,
                    n_controls=0,
                    compute_type=cq.ComputeType.COMPUTE_32F,
                    workspace=0,
                    workspace_size=0
                )
            
            self.stream_compute.synchronize()
        
        self.metrics['compute_time'] += time.time() - t_compute_start
    
    def _apply_fused_local_gates_async(self, gates):
        """
        True Double Buffering with Async Prefetching + Race Condition Protection
        
        Timeline:
        - CPU Thread: Read Chunk N+1 → Read Chunk N+2 → ...
        - GPU Stream: Compute Chunk N → Compute Chunk N+1 → ...
        - Write Thread: Write Chunk N-1 → Write Chunk N → ...
        
        Safety: buffer_write_futures ensures read doesn't overwrite pending writes
        """
        # Prefetch first chunk (prime the pump)
        current_buffer = 'a'
        next_buffer = 'b'
        
        # [CRITICAL FIX] Track pending writes per buffer to prevent race condition
        buffer_write_futures = {'a': None, 'b': None}
        
        # Start prefetching chunk 0
        future_read = self.io_executor.submit(
            self._read_chunk_to_buffer, 0, current_buffer
        )
        
        for i in range(self.n_chunks):
            # 1. Wait for current chunk to be ready
            t_wait_start = time.time()
            future_read.result()  # Block until read completes
            self.metrics['io_wait_time'] += time.time() - t_wait_start
            
            # 2. Start prefetching NEXT chunk (if exists) - WITH SAFETY CHECK!
            if i + 1 < self.n_chunks:
                # [CRITICAL FIX] Wait for any pending write to next_buffer BEFORE overwriting
                if buffer_write_futures[next_buffer] is not None:
                    buffer_write_futures[next_buffer].result()  # Block until write completes
                
                future_read = self.io_executor.submit(
                    self._read_chunk_to_buffer, i + 1, next_buffer
                )
            
            # 3. GPU compute on current chunk (while CPU reads next)
            t_compute_start = time.time()
            
            chunk_gpu = self._get_gpu_array(current_buffer)
            
            with self.stream_compute:
                for gate_matrix, target_qubit in gates:
                    if HAS_CUQUANTUM:
                        matrix_gpu = cp.asarray(gate_matrix)
                        targets = np.array([target_qubit], dtype=np.int32)
                        cusv.apply_matrix(
                            handle=self.handle,
                            sv=chunk_gpu.data.ptr,
                            sv_data_type=cq.cudaDataType.CUDA_C_32F,
                            n_index_bits=self.gpu_n,
                            matrix=matrix_gpu.data.ptr,
                            matrix_data_type=cq.cudaDataType.CUDA_C_32F,
                            layout=cusv.MatrixLayout.ROW,
                            adjoint=0,
                            targets=targets.ctypes.data,
                            n_targets=1,
                            controls=0,
                            control_bit_values=0,
                            n_controls=0,
                            compute_type=cq.ComputeType.COMPUTE_32F,
                            workspace=0,
                            workspace_size=0
                        )
                
                self.stream_compute.synchronize()
            
            self.metrics['compute_time'] += time.time() - t_compute_start
            
            # 4. Copy result back to host buffer (if not mapped, result is already there)
            if self._managed_mode:
                # For mapped memory, GPU wrote directly to host buffer
                result = self._get_buffer(current_buffer).copy()
            else:
                result = cp.asnumpy(chunk_gpu)
            
            # 5. Async write-back (overlaps with next iteration)
            # [CRITICAL FIX] Track this write future for race condition protection
            write_future = self._write_chunk_async(i, result)
            buffer_write_futures[current_buffer] = write_future
            
            # 6. Swap buffers for next iteration
            current_buffer, next_buffer = next_buffer, current_buffer
        
        # Wait for all pending writes to complete
        for future in self.pending_writes:
            try:
                self.metrics['write_time'] += future.result()
            except Exception as e:
                print(f"Write error: {e}")
        self.pending_writes.clear()
    
    def _apply_global_gate(self, gate_matrix, target_qubit):
        """Apply global gate (requires chunk pairing) - inherited from V2"""
        global_bit = target_qubit - self.gpu_n
        for i in range(self.n_chunks):
            partner = i ^ (1 << global_bit)
            if partner > i:
                path_i = os.path.join(self.storage_dir, f"{i}.bin")
                path_j = os.path.join(self.storage_dir, f"{partner}.bin")
                
                with open(path_i, 'rb') as f:
                    data_i = f.read()
                with open(path_j, 'rb') as f:
                    data_j = f.read()
                
                if self.use_compression:
                    data_i = lz4.decompress(data_i)
                    data_j = lz4.decompress(data_j)
                
                chunk_i = np.frombuffer(data_i, dtype=np.complex64).copy()
                chunk_j = np.frombuffer(data_j, dtype=np.complex64).copy()
                
                new_i = gate_matrix[0,0] * chunk_i + gate_matrix[0,1] * chunk_j
                new_j = gate_matrix[1,0] * chunk_i + gate_matrix[1,1] * chunk_j
                
                self._write_chunk_sync(i, new_i)
                self._write_chunk_sync(partner, new_j)
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        """API compatibility: queue gate for fusion"""
        self.queue_gate(gate_matrix, target_qubit)
    
    def get_storage_size(self):
        """Get total storage size in bytes"""
        total = 0
        for i in range(self.n_chunks):
            path = os.path.join(self.storage_dir, f"{i}.bin")
            if os.path.exists(path):
                total += os.path.getsize(path)
        return total
    
    def get_metrics(self):
        """Return performance metrics"""
        return {
            'io_wait_time': self.metrics['io_wait_time'],
            'compute_time': self.metrics['compute_time'],
            'write_time': self.metrics['write_time'],
            'gate_count': self.gate_count,
            'total_gate_time': self.total_gate_time,
            'io_compute_overlap': max(0, 1 - self.metrics['io_wait_time'] / 
                                      (self.metrics['compute_time'] + 0.001))
        }
    
    def cleanup(self):
        """Remove temporary files and free resources"""
        self.flush_gates()
        
        # Wait for pending writes
        for future in self.pending_writes:
            future.result()
        
        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.write_executor.shutdown(wait=True)
        
        import shutil
        shutil.rmtree(self.storage_dir, ignore_errors=True)
        
        if HAS_CUQUANTUM:
            cusv.destroy(self.handle)
        
        # Free ZeroCopyMemory (if using mapped mode)
        if self._managed_mode and self._raw_mem_a:
            self._raw_mem_a.free()
            self._raw_mem_b.free()


# ============================================================
# SIMPLE BENCHMARK
# ============================================================

def run_benchmark(n_qubits=26, n_gates=50):
    """Run a simple benchmark"""
    print("=" * 60)
    print(f"   EdgeQuantum Benchmark: {n_qubits}Q, {n_gates} gates")
    print("=" * 60)
    
    sim = TieredSimulator(n_qubits, use_compression=True, fusion_threshold=10)
    
    t0 = time.time()
    sim.init_zero_state()
    t_init = time.time() - t0
    
    t0 = time.time()
    for i in range(n_gates):
        sim.apply_single_gate(H, i % 22)
    sim.flush_gates()
    t_gate = time.time() - t0
    
    metrics = sim.get_metrics()
    sim.cleanup()
    
    print(f"\n🚀 Results:")
    print(f"   Init: {t_init:.2f}s")
    print(f"   Gates: {t_gate:.2f}s")
    print(f"   Compute: {metrics['compute_time']:.2f}s")
    print(f"   I/O Wait: {metrics['io_wait_time']:.2f}s")
    print("=" * 60)
    
    return {'init_time': t_init, 'gate_time': t_gate, 'metrics': metrics}


if __name__ == "__main__":
    run_benchmark()
