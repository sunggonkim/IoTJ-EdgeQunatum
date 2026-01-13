#!/usr/bin/env python3
"""
EdgeQuantum "True" cuQuantum Simulator
Uses NVIDIA cuQuantum SDK (custatevec) for gate operations.
Optimized for Tiered Memory + LZ4 Compression.
"""

import numpy as np
import cupy as cp
import cuquantum as cq
from cuquantum import custatevec as cusv
import time
import os
import tempfile
import shutil
import psutil
import json
import sys

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

class CuQuantumTieredSim:
    def __init__(self, n_qubits, gpu_chunk_qubits=22, use_compression=True):
        self.n = n_qubits
        self.gpu_n = min(gpu_chunk_qubits, n_qubits)
        self.chunk_size = 2**self.gpu_n
        self.n_chunks = 2**(n_qubits - self.gpu_n)
        self.use_compression = use_compression and HAS_LZ4
        self.storage_dir = tempfile.mkdtemp(prefix='eq_cusv_')
        
        # cuQuantum Handle & Workspace
        self.handle = cusv.create()
        self.workspace = None
        self.workspace_size = 0
        
        print(f"⚡ [cuQuantum] {n_qubits} qubits (custatevec enabled)")
        print(f"⚡ [cuQuantum] Tiered Memory: {self.n_chunks} chunks, Compression: {'ON' if self.use_compression else 'OFF'}")

    def _get_workspace_size(self, sv_dtype, adj_map, matrix, targets, controls, compute_type):
        # minimal workspace check
        return 1024 * 1024 * 32 # Reserve 32MB workspace
        
    def _apply_gate_custatevec(self, vector_ptr, gate_matrix, targets, controls=[]):
        """Apply gate using cuStream + cuStateVec"""
        targets = np.array(targets, dtype=np.int32)
        controls = np.array(controls, dtype=np.int32)
        n_controls = len(controls)
        
        # Need to cast matrix to complex64 (cuFloatComplex)
        matrix = matrix.astype(np.complex64)
        
        # Workspace allocation (lazy)
        if self.workspace is None:
             self.workspace = cp.cuda.alloc(1024*1024*64) # 64MB
             self.workspace_size = 1024*1024*64

        # Apply gate
        cusv.apply_matrix(
            self.handle,
            vector_ptr,
            self.chunk_size, # This is 'nIndexBits' for apply_matrix? No, it's cudaDataType
            cq.cudaDataType.CUDA_C_32F, 
            self.gpu_n, # nIndexBits (local)
            matrix.data.ptr, 
            cq.cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            0, # adjoint
            targets.ctypes.data, 
            len(targets),
            controls.ctypes.data, 
            0, # controlTypes
            n_controls,
            compute_type=cusv.ComputeType.COMPUTE_32F,
            workspace=self.workspace.ptr, 
            workspace_size=self.workspace_size
        )

    def _init_state(self):
        print("  Running Init...", flush=True)
        t0 = time.time()
        zeros = np.zeros(self.chunk_size, dtype=np.complex64)
        if self.use_compression:
             zeros_comp = lz4.compress(zeros.tobytes(), compression_level=1)
        
        for i in range(self.n_chunks):
            if i % 128 == 0:
                 print(f"  Init: {i}/{self.n_chunks} ({i*100//self.n_chunks}%)", flush=True)
            
            path = os.path.join(self.storage_dir, f"{i}.bin")
            if i == 0:
                chunk = np.zeros(self.chunk_size, dtype=np.complex64)
                chunk[0] = 1.0
                with open(path, 'wb') as f:
                    f.write(lz4.compress(chunk.tobytes()) if self.use_compression else chunk.tobytes())
            else:
                with open(path, 'wb') as f:
                    f.write(zeros_comp if self.use_compression else zeros.tobytes())
        print(f"  Init Done: {time.time()-t0:.2f}s")
        return time.time()-t0

    def apply_hadamard_q0(self):
        """Apply H gate to qubit 0 (Local Gate) using custatevec"""
        print("  Applying H(q0) with custatevec...", flush=True)
        
        H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        H_gpu = cp.asarray(H)
        
        t0 = time.time()
        
        for i in range(self.n_chunks):
            # 1. Read & Decompress
            path = os.path.join(self.storage_dir, f"{i}.bin")
            with open(path, 'rb') as f:
                data = f.read()
            if self.use_compression:
                data = lz4.decompress(data)
            
            # 2. Host -> Device
            chunk_gpu = cp.frombuffer(data, dtype=np.complex64) # Zero-copy if pinned? Allocates GPU mem
            
            # 3. cuStateVec Apply
            self._apply_gate_custatevec(chunk_gpu.data.ptr, H_gpu, [0])
            
            # 4. Device -> Host
            chunk_host = cp.asnumpy(chunk_gpu)
            
            # 5. Compress & Write
            if self.use_compression:
                cw = lz4.compress(chunk_host.tobytes(), compression_level=1)
                with open(path, 'wb') as f:
                    f.write(cw)
            else:
                 with open(path, 'wb') as f:
                    f.write(chunk_host.tobytes())
            
            if i % 128 == 0:
                 print(f"  Gate: {i}/{self.n_chunks} ({i*100//self.n_chunks}%)", flush=True)

        cp.cuda.Device().synchronize()
        print(f"  Gate Done: {time.time()-t0:.2f}s")
        return time.time()-t0

    def cleanup(self):
        cusv.destroy(self.handle)
        shutil.rmtree(self.storage_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        qubits = int(sys.argv[1])
    else:
        qubits = 28 # Default test
    
    sim = CuQuantumTieredSim(qubits, use_compression=True)
    try:
        t_init = sim._init_state()
        t_gate = sim.apply_hadamard_q0()
        print(f"\n✅ [Success] {qubits}q cuQuantum Tiered Simulation")
        print(f"Init: {t_init:.2f}s")
        print(f"Gate: {t_gate:.2f}s")
        print(f"Total: {t_init+t_gate:.2f}s")
    finally:
        sim.cleanup()
