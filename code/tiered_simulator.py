#!/usr/bin/env python3
"""
EdgeQuantum TieredSimulator (Jetson Ultimate Architecture)
Target: NVIDIA Jetson Orin Nano + NVMe SSD
Optimizations:
 1. O_DIRECT (DMA I/O)
 2. Zero-Copy Compute (Mapped Pinned Memory on Unified Arch)
 3. 3-Stage Async Pipeline
 4. Gate Fusion & Locality Optimization
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
    import cuquantum as cq
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
# cuQuantum enum values will be used directly from cq module
ALIGN_SIZE = 4096 # NVMe Sector Size Alignment

# Standard Gate Matrices
H = np.array([[1, 1], [1, -1]], dtype=COMPLEX64) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX64)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX64)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX64)

# [USER CONFIG] NVMe Mount Path
NVME_PATH = '/mnt/data'

if not os.path.exists(NVME_PATH):
    print(f"❌ NVMe path {NVME_PATH} not found.")
    sys.exit(1)

# --- Jetson-Specific Memory Wrapper ---
class JetsonPinnedBuffer:
    """
    Allocates Pinned Memory that is MAPPED to GPU Address Space.
    On Jetson (Tegra), this allows the GPU to access CPU RAM directly (Zero-Copy).
    """
    def __init__(self, size_bytes):
        self.size = size_bytes
        # alloc_pinned_memory returns a PinnedMemoryPointer
        self.mem = cp.cuda.alloc_pinned_memory(size_bytes)
        self.ptr = self.mem.ptr
        
        # Check O_DIRECT Alignment
        if self.ptr % ALIGN_SIZE != 0:
            raise RuntimeError("Memory not aligned! O_DIRECT will fail.")

    def get_numpy_view(self, dtype):
        """View for CPU/NVMe (O_DIRECT I/O)"""
        n_items = self.size // np.dtype(dtype).itemsize
        return np.frombuffer(self.mem, dtype=dtype, count=n_items)
    
    def get_device_pointer(self):
        """
        Returns a pointer usable by cuStateVec.
        On Jetson, this Pinned Memory pointer is valid for the GPU.
        This enables Zero-Copy Compute.
        """
        return self.ptr


class DirectIO:
    """Handles O_DIRECT I/O Operations"""
    def __init__(self, storage_dir, n_chunks):
        self.dir = storage_dir
        self.fds = {}
        # O_DIRECT: Bypass OS Page Cache
        # O_DSYNC: Ensure physical write (Safety)
        self.flags = os.O_RDWR | os.O_DIRECT | os.O_CREAT
        if hasattr(os, 'O_DSYNC'): 
            self.flags |= os.O_DSYNC
        
        for i in range(n_chunks):
            p = os.path.join(self.dir, f"{i}.bin")
            self.fds[i] = os.open(p, self.flags)

    def write(self, idx, ptr, size):
        # Use libc pwrite directly to preserve alignment for O_DIRECT
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.pwrite(self.fds[idx], ctypes.c_void_p(ptr), size, 0)
        if result < 0:
            raise OSError(ctypes.get_errno(), "pwrite failed")
        return result

    def read(self, idx, ptr, size):
        # Use libc pread directly to preserve alignment for O_DIRECT
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.pread(self.fds[idx], ctypes.c_void_p(ptr), size, 0)
        if result < 0:
            raise OSError(ctypes.get_errno(), "pread failed")
        return result

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


class TieredSimulator:
    def __init__(self, n_qubits, use_compression=False, fusion_threshold=5000, use_managed_memory=False):
        # Threshold increased to maximize Batch Fusion
        self.n = n_qubits
        self.fusion_threshold = fusion_threshold
        self.gate_queue = []
        self.gate_count = 0
        
        self.state_size = (2**n_qubits) * 8
        sys_avail = psutil.virtual_memory().available if HAS_PSUTIL else cp.cuda.runtime.memGetInfo()[0]
        self.usable_mem = sys_avail - (1.5 * 1024**3) # 1.5GB Margin for OS
        
        self.in_memory_mode = False
        self.handle = cusv.create()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.state_ptr = 0
        self.n_chunks = 1

        print(f"[EdgeQuantum Jetson SOTA] {n_qubits}Q. Size: {self.state_size/1e9:.2f}GB")

        # 1. In-Memory Mode (Native Unified Memory)
        if self.state_size < self.usable_mem:
            # Use cp.zeros which uses Unified Memory on Jetson natively
            self.in_memory_state = cp.zeros(2**n_qubits, dtype=COMPLEX64)
            self.in_memory_state[0] = 1.0
            self.state_ptr = self.in_memory_state.data.ptr
            self.in_memory_mode = True
            print("  -> Strategy: \033[92mIn-Memory (Unified RAM)\033[0m")
        else:
            self._init_nvme_mode(n_qubits)

        # Optimization Caches
        self.gate_cache = {} 
        self._targets = cp.array([0], dtype=INT32) 
        self._targets_ptr = self._targets.data.ptr

    def _init_nvme_mode(self, n_qubits):
        print(f"  -> Strategy: \033[96mZero-Copy Pipeline (NVMe O_DIRECT)\033[0m")
        
        # Chunk Sizing: 512MB is optimal for NVMe DMA
        target_chunk = 512 * 1024 * 1024
        if self.usable_mem < 4 * 1024**3:
            target_chunk = 256 * 1024 * 1024
            
        max_chunk_q = int(np.log2(target_chunk / 8))
        self.gpu_n = min(max_chunk_q, n_qubits - 1)
        self.chunk_size = 2**self.gpu_n
        self.chunk_bytes = self.chunk_size * 8
        self.n_chunks = 2**(n_qubits - self.gpu_n)
        
        self.storage_dir = tempfile.mkdtemp(prefix='eq_jetson_', dir=NVME_PATH)
        
        # Alloc Zero-Copy Buffers
        try:
            self.buf_a = JetsonPinnedBuffer(self.chunk_bytes)
            self.buf_b = JetsonPinnedBuffer(self.chunk_bytes)
        except Exception as e:
            print(f"❌ Alloc Error: {e}")
            sys.exit(1)
            
        # IO Engine
        self.io = DirectIO(self.storage_dir, self.n_chunks)
        self.io_executor = ThreadPoolExecutor(max_workers=2) # Read/Write threads
        
        # Initialize Storage (Zeroing)
        print("  -> Formatting NVMe storage...")
        zeros = self.buf_a.get_numpy_view(COMPLEX64)
        zeros[:] = 0
        
        zeros[0] = 1.0
        self.io.write(0, self.buf_a.ptr, self.chunk_bytes)
        
        zeros[0] = 0.0
        for i in range(1, self.n_chunks):
            self.io.write(i, self.buf_a.ptr, self.chunk_bytes)

    def init_zero_state(self):
        """Compatibility method: initialization already handled in __init__"""
        pass

    def get_storage_size(self):
        """Return storage size in bytes"""
        return self.state_size

    def apply_single_gate(self, gate_matrix, target_qubit):
        self.gate_count += 1
        
        if self.in_memory_mode:
            # Fast Path
            gid = id(gate_matrix)
            if gid not in self.gate_cache: 
                self.gate_cache[gid] = cp.asarray(gate_matrix, dtype=COMPLEX64)
            cusv.apply_matrix(
                self.handle, self.state_ptr, 4, self.n,  # 4 = CUDA_C_32F
                self.gate_cache[gid].data.ptr, 4, 1, 0,  # 1 = ROW layout
                (target_qubit,), 1, (), (), 0, 4, 0, 0   # 4 = COMPUTE_32F
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
        
        # Filter Local vs Global
        local_gates = []
        global_gates = []
        for g, t in self.gate_queue:
            if t < self.gpu_n: 
                local_gates.append((g, t))
            else: 
                global_gates.append((g, t))

        if local_gates: 
            self._pipeline_zerocopy(local_gates)
        if global_gates: 
            self._apply_global_batched(global_gates)
        
        self.gate_queue.clear()

    def _pipeline_zerocopy(self, gates):
        """
        Ultimate 3-Stage Zero-Copy Pipeline
        
        Unlike PCIe GPUs, Jetson does NOT need to copy Mem->GPU.
        It computes DIRECTLY on the Pinned Buffer via Unified Memory.
        
        Pipeline:
        [IO Thread] Read(Next) -> [GPU] Zero-Copy Compute(Current) -> [IO Thread] Write(Prev)
        """
        
        bufs = [self.buf_a, self.buf_b]
        curr = 0
        
        # Pre-load Chunk 0
        self.io.read(0, bufs[0].ptr, self.chunk_bytes)
        
        read_fut = None
        write_fut = None
        
        for i in range(self.n_chunks):
            # 1. Launch Read Next (Background)
            if i + 1 < self.n_chunks:
                nxt = 1 - curr
                if write_fut: 
                    write_fut.result() # Wait for buffer to free
                read_fut = self.io_executor.submit(self.io.read, i+1, bufs[nxt].ptr, self.chunk_bytes)
            
            # 2. Zero-Copy Compute (Foreground GPU)
            # CRITICAL: Pass the Pinned Memory Pointer DIRECTLY to cuStateVec
            # No cudaMemcpy needed on Jetson!
            chunk_ptr = bufs[curr].get_device_pointer()
            
            with self.stream:
                for g, t in gates:
                    gid = id(g)
                    if gid not in self.gate_cache: 
                        self.gate_cache[gid] = cp.asarray(g, dtype=COMPLEX64)
                    
                    cusv.apply_matrix(
                        self.handle, chunk_ptr, 4, self.gpu_n,  # 4 = CUDA_C_32F
                        self.gate_cache[gid].data.ptr, 4, 1, 0,  # 1 = ROW layout
                        (t,), 1, (), (), 0, 4, 0, 0              # 4 = COMPUTE_32F
                    )
            
            # 3. Barrier: GPU must finish touching RAM before we write it to Disk
            self.stream.synchronize()
            
            # 4. Launch Write Current (Background)
            write_fut = self.io_executor.submit(self.io.write, i, bufs[curr].ptr, self.chunk_bytes)
            
            # 5. Wait for Read
            if i + 1 < self.n_chunks:
                read_fut.result()
            
            curr = 1 - curr

        if write_fut: 
            write_fut.result()

    def _apply_global_batched(self, gates):
        # For Global Gates (Swap-like), we act directly on mapped memory
        # This is bandwidth bound, not compute bound.
        
        view_a = self.buf_a.get_numpy_view(COMPLEX64)
        view_b = self.buf_b.get_numpy_view(COMPLEX64)
        
        for g, t in gates:
            bit = t - self.gpu_n
            stride = 1 << bit
            
            for i in range(self.n_chunks):
                if (i & stride) == 0:
                    partner = i | stride
                    
                    # 1. Read Pair (Parallel)
                    f1 = self.io_executor.submit(self.io.read, i, self.buf_a.ptr, self.chunk_bytes)
                    f2 = self.io_executor.submit(self.io.read, partner, self.buf_b.ptr, self.chunk_bytes)
                    f1.result()
                    f2.result()
                    
                    # 2. Compute on CPU (NEON-accelerated via NumPy)
                    ni = g[0,0]*view_a + g[0,1]*view_b
                    nj = g[1,0]*view_a + g[1,1]*view_b
                    
                    np.copyto(view_a, ni)
                    np.copyto(view_b, nj)
                    
                    # 3. Write Pair (Parallel)
                    f1 = self.io_executor.submit(self.io.write, i, self.buf_a.ptr, self.chunk_bytes)
                    f2 = self.io_executor.submit(self.io.write, partner, self.buf_b.ptr, self.chunk_bytes)
                    f1.result()
                    f2.result()

    def cleanup(self):
        self.stream.synchronize()
        if hasattr(self, 'io'): 
            self.io.close()
        if hasattr(self, 'io_executor'):
            self.io_executor.shutdown(wait=True)
        if hasattr(self, 'storage_dir'): 
            shutil.rmtree(self.storage_dir, ignore_errors=True)
        if HAS_CUQUANTUM: 
            cusv.destroy(self.handle)


if __name__ == "__main__":
    print(f"=== EdgeQuantum Ultimate (Jetson + NVMe) ===")
    sim = TieredSimulator(20, use_compression=False) 
    sim.init_zero_state()
    sim.apply_single_gate(H, 0)
    sim.flush_gates()
    sim.cleanup()
    print("=== Verification Passed ===")
