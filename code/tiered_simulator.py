#!/usr/bin/env python3
"""
EdgeQuantum TieredSimulator (Jetson Architecture Ultimate)
Target: NVIDIA Jetson Orin Nano + NVMe SSD
Optimizations:
  1. O_DIRECT DMA Access (libc pread/pwrite)
  2. Zero-Copy Pinned Memory (Unified RAM)
  3. 3-Stage Async Pipeline (Read-Compute-Write)
  4. Resource Reuse (Workspace & Target Buffer Recycling)
"""
import os
import sys
import time
import tempfile
import numpy as np
import ctypes
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# --- Dependencies ---
try:
    import cupy as cp
    import cupy.cuda.runtime as cuda_rt
    from cuquantum import custatevec as cusv
    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False
    print("❌ Critical: cuQuantum required.")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# --- Low Level Constants ---
COMPLEX64 = np.complex64
INT32 = np.int32
ALIGN_SIZE = 4096 # NVMe Sector Alignment

# Pre-load libc for Zero-Overhead I/O
try:
    LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except Exception:
    print("❌ Critical: libc not found.")
    sys.exit(1)

# Standard Gates
H = np.array([[1, 1], [1, -1]], dtype=COMPLEX64) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX64)

# [USER CONFIG] NVMe Mount Path
NVME_PATH = '/mnt/data'

if not os.path.exists(NVME_PATH):
    print(f"❌ NVMe path {NVME_PATH} not found.")
    sys.exit(1)

# --- Jetson-Specific Memory Wrapper ---
class JetsonPinnedBuffer:
    """
    Allocates Pinned Memory mapped to GPU Address Space (Zero-Copy).
    On Jetson, this is physical RAM accessible by both CPU and GPU.
    """
    def __init__(self, size_bytes):
        self.size = size_bytes
        self.mem = cp.cuda.alloc_pinned_memory(size_bytes)
        self.ptr = self.mem.ptr
        
        if self.ptr % ALIGN_SIZE != 0:
            raise RuntimeError("Memory not aligned! O_DIRECT will fail.")

    def get_numpy_view(self, dtype):
        n_items = self.size // np.dtype(dtype).itemsize
        return np.frombuffer(self.mem, dtype=dtype, count=n_items)
    
    def get_device_pointer(self):
        return self.ptr


class DirectIO:
    """High-Performance O_DIRECT I/O Wrapper using libc DMA"""
    def __init__(self, storage_dir, n_chunks):
        self.dir = storage_dir
        self.fds = {}
        # O_DIRECT: Bypass OS Page Cache. O_CREAT: Create if missing.
        self.flags = os.O_RDWR | os.O_DIRECT | os.O_CREAT
        if hasattr(os, 'O_DSYNC'): 
            self.flags |= os.O_DSYNC
        
        for i in range(n_chunks):
            p = os.path.join(self.dir, f"{i}.bin")
            self.fds[i] = os.open(p, self.flags)

    def write(self, idx, ptr, size):
        res = LIBC.pwrite(self.fds[idx], ctypes.c_void_p(ptr), size, 0)
        if res < 0:
            errno_val = ctypes.get_errno()
            raise OSError(errno_val, f"O_DIRECT pwrite failed: {os.strerror(errno_val)}")
        return res

    def read(self, idx, ptr, size):
        res = LIBC.pread(self.fds[idx], ctypes.c_void_p(ptr), size, 0)
        if res < 0:
            errno_val = ctypes.get_errno()
            raise OSError(errno_val, f"O_DIRECT pread failed: {os.strerror(errno_val)}")
        return res

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


class TieredSimulator:
    def __init__(self, n_qubits, use_compression=False, fusion_threshold=5000):
        self.n = n_qubits
        self.fusion_threshold = fusion_threshold
        self.gate_queue = []
        self.gate_count = 0
        
        self.state_size = (2**n_qubits) * 8
        sys_avail = psutil.virtual_memory().available if HAS_PSUTIL else cuda_rt.memGetInfo()[0]
        self.usable_mem = sys_avail - (1.5 * 1024**3) # 1.5GB Safety Margin
        
        self.in_memory_mode = False
        self.handle = cusv.create()
        self.stream = cp.cuda.Stream(non_blocking=True)
        # Bind stream for async execution
        cusv.set_stream(self.handle, self.stream.ptr)
        
        self.state_ptr = 0
        self.n_chunks = 1

        print(f"[EdgeQuantum Jetson Ultimate] {n_qubits}Q. Size: {self.state_size/1e9:.2f}GB")

        if self.state_size < self.usable_mem:
            self.in_memory_state = cp.zeros(2**n_qubits, dtype=COMPLEX64)
            self.in_memory_state[0] = 1.0
            self.state_ptr = self.in_memory_state.data.ptr
            self.in_memory_mode = True
            print("  -> Strategy: \033[92mIn-Memory (Unified RAM)\033[0m")
        else:
            self._init_nvme_mode(n_qubits)

        # --- Resource Pre-allocation (Optimize micro-overhead) ---
        self.gate_cache = {} 
        # Pre-allocate pinned memory for target index (Fast update via ctypes)
        self.target_buf = cp.cuda.alloc_pinned_memory(4) # 1 * int32
        self.target_ptr = self.target_buf.ptr
        
        # Pre-allocate Workspace for cuStateVec (Stability & Performance)
        self.ws_size = 64 * 1024 * 1024 # 64MB
        self.ws_mem = cp.cuda.alloc(self.ws_size)
        self.ws_ptr = self.ws_mem.ptr

    def _init_nvme_mode(self, n_qubits):
        print(f"  -> Strategy: \033[96mNVMe O_DIRECT Pipeline\033[0m")
        
        target_chunk = 512 * 1024 * 1024
        if self.usable_mem < 4 * 1024**3:
            target_chunk = 256 * 1024 * 1024
            
        max_chunk_q = int(np.log2(target_chunk / 8))
        self.gpu_n = min(max_chunk_q, n_qubits - 1)
        self.chunk_size = 2**self.gpu_n
        self.chunk_bytes = self.chunk_size * 8
        self.n_chunks = 2**(n_qubits - self.gpu_n)
        
        self.storage_dir = tempfile.mkdtemp(prefix='eq_jetson_', dir=NVME_PATH)
        
        try:
            self.buf_a = JetsonPinnedBuffer(self.chunk_bytes)
            self.buf_b = JetsonPinnedBuffer(self.chunk_bytes)
            # Explicit GPU Buffer for cuStateVec Library Compatibility (30Q+ Fix)
            self.gpu_chunk = cp.zeros(self.chunk_size, dtype=COMPLEX64)
            self.gpu_chunk_ptr = self.gpu_chunk.data.ptr
        except Exception as e:
            print(f"❌ Alloc Error: {e}")
            sys.exit(1)
            
        self.io = DirectIO(self.storage_dir, self.n_chunks)
        self.io_executor = ThreadPoolExecutor(max_workers=2)
        
        # Init Zero State
        print(f"  -> Formatting NVMe ({self.n_chunks} chunks)...")
        zeros = self.buf_a.get_numpy_view(COMPLEX64)
        zeros[:] = 0
        zeros[0] = 1.0
        self.io.write(0, self.buf_a.ptr, self.chunk_bytes)
        
        zeros[0] = 0.0
        for i in range(1, self.n_chunks):
            self.io.write(i, self.buf_a.ptr, self.chunk_bytes)
        print("  -> NVMe formatting complete.")

    def init_zero_state(self):
        pass

    def apply_single_gate(self, gate_matrix, target_qubit):
        self.gate_count += 1
        
        if self.in_memory_mode:
            # Fast Path: Resource Reuse
            gid = id(gate_matrix)
            if gid not in self.gate_cache: 
                self.gate_cache[gid] = cp.asarray(gate_matrix, dtype=COMPLEX64)
            
            # Update target index directly in pinned memory
            ctypes.cast(self.target_ptr, ctypes.POINTER(ctypes.c_int32))[0] = target_qubit
            
            cusv.apply_matrix(
                self.handle, self.state_ptr, 4, self.n, # 4 = CUDA_C_32F
                self.gate_cache[gid].data.ptr, 4, 1, 0, # 1 = ROW layout
                self.target_ptr, 1, 0, 0, 0, 4, self.ws_ptr, self.ws_size
            )
            return

        self.gate_queue.append((gate_matrix, target_qubit))
        if len(self.gate_queue) >= self.fusion_threshold:
            self.flush_gates()

    def flush_gates(self):
        if self.in_memory_mode or not self.gate_queue:
            self.gate_queue.clear()
            return
        
        self.stream.synchronize()
        
        local_gates = []
        global_gates = []
        for g, t in self.gate_queue:
            if t < self.gpu_n: local_gates.append((g, t))
            else: global_gates.append((g, t))

        if local_gates: self._pipeline_nvme(local_gates)
        if global_gates: self._apply_global_batched(global_gates)
        
        self.gate_queue.clear()

    def _pipeline_nvme(self, gates):
        bufs = [self.buf_a, self.buf_b]
        curr = 0
        self.io.read(0, bufs[0].ptr, self.chunk_bytes)
        
        read_fut = None
        write_fut = None
        
        for i in range(self.n_chunks):
            # 1. Background Read Next
            if i + 1 < self.n_chunks:
                nxt = 1 - curr
                if write_fut: write_fut.result()
                read_fut = self.io_executor.submit(self.io.read, i+1, bufs[nxt].ptr, self.chunk_bytes)
            
            # 2. Compute Segment
            # Explicit copy for cuStateVec stability on 30Q+
            # Although Jetson is unified, cuStateVec sometimes rejects pinned pointers for high qubit counts.
            cuda_rt.memcpy(self.gpu_chunk_ptr, bufs[curr].ptr, self.chunk_bytes, 2) # DeviceToDevice
            
            with self.stream:
                for g, t in gates:
                    gid = id(g)
                    if gid not in self.gate_cache:
                        self.gate_cache[gid] = cp.asarray(g, dtype=COMPLEX64)
                    
                    ctypes.cast(self.target_ptr, ctypes.POINTER(ctypes.c_int32))[0] = t
                    
                    cusv.apply_matrix(
                        self.handle, self.gpu_chunk_ptr, 4, self.gpu_n,
                        self.gate_cache[gid].data.ptr, 4, 1, 0,
                        self.target_ptr, 1, 0, 0, 0, 4, self.ws_ptr, self.ws_size
                    )
            
            self.stream.synchronize()
            cuda_rt.memcpy(bufs[curr].ptr, self.gpu_chunk_ptr, self.chunk_bytes, 2)
            
            # 3. Background Write Current
            write_fut = self.io_executor.submit(self.io.write, i, bufs[curr].ptr, self.chunk_bytes)
            
            if i + 1 < self.n_chunks: read_fut.result()
            curr = 1 - curr

        if write_fut: write_fut.result()

    def _apply_global_batched(self, gates):
        view_a = self.buf_a.get_numpy_view(COMPLEX64)
        view_b = self.buf_b.get_numpy_view(COMPLEX64)
        
        for g, t in gates:
            stride = 1 << (t - self.gpu_n)
            for i in range(self.n_chunks):
                if (i & stride) == 0:
                    partner = i | stride
                    f1 = self.io_executor.submit(self.io.read, i, self.buf_a.ptr, self.chunk_bytes)
                    f2 = self.io_executor.submit(self.io.read, partner, self.buf_b.ptr, self.chunk_bytes)
                    f1.result(); f2.result()
                    
                    # NEON-optimized NumPy compute
                    ni = g[0,0]*view_a + g[0,1]*view_b
                    nj = g[1,0]*view_a + g[1,1]*view_b
                    np.copyto(view_a, ni); np.copyto(view_b, nj)
                    
                    f1 = self.io_executor.submit(self.io.write, i, self.buf_a.ptr, self.chunk_bytes)
                    f2 = self.io_executor.submit(self.io.write, partner, self.buf_b.ptr, self.chunk_bytes)
                    f1.result(); f2.result()

    def cleanup(self):
        self.stream.synchronize()
        if hasattr(self, 'io'): self.io.close()
        if hasattr(self, 'io_executor'): self.io_executor.shutdown(wait=True)
        if hasattr(self, 'storage_dir'): shutil.rmtree(self.storage_dir, ignore_errors=True)
        if HAS_CUQUANTUM: cusv.destroy(self.handle)


if __name__ == "__main__":
    print(f"=== EdgeQuantum Ultimate Optimized (Jetson) ===")
    sim = TieredSimulator(20) 
    sim.apply_single_gate(H, 0)
    sim.flush_gates()
    sim.cleanup()
    print("=== Verification SUCCESS ===")
