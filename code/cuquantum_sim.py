#!/usr/bin/env python3
"""
EdgeQuantum "Native" cuQuantum Benchmark Simulator
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
from datetime import datetime

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

class CuQuantumTieredSim:
    def __init__(self, n_qubits, gpu_chunk_qubits=22, use_compression=True):
        self.n = n_qubits
        self.gpu_n = min(gpu_chunk_qubits, n_qubits)
        self.chunk_size = 2**self.gpu_n # number of amplitudes in chunk
        self.n_chunks = 2**(n_qubits - self.gpu_n)
        self.use_compression = use_compression and HAS_LZ4
        self.storage_dir = tempfile.mkdtemp(prefix=f'eq_cusv_{n_qubits}q_')
        self.bytes_written = 0
        
        # cuQuantum Handle & Workspace
        self.handle = cusv.create()
        self.workspace = None
        self.workspace_size = 0
        
        print(f"\n⚡ [cuQuantum] {n_qubits} qubits (custatevec enabled)")
        print(f"   - Chunks: {self.n_chunks}, Chunk Size: {self.chunk_size*8/1024**2:.1f}MB")
        print(f"   - Compression: {'ON (LZ4)' if self.use_compression else 'OFF'}")

    def _apply_gate_custatevec(self, vector_ptr, gate_matrix, targets, controls=[]):
        """Apply gate using cuStream + cuStateVec"""
        targets = np.array(targets, dtype=np.int32)
        controls = np.array(controls, dtype=np.int32)
        n_controls = len(controls)
        
        # Need to cast matrix to complex64 (cuFloatComplex)
        matrix = gate_matrix.astype(np.complex64)
        
        # Workspace allocation (lazy)
        if self.workspace is None:
             # Allocate 64MB workspace (adjust if needed)
             self.workspace_size = 1024*1024*64 
             self.workspace = cp.cuda.alloc(self.workspace_size)

        # Apply gate with keyword arguments to avoid positional errors
        cusv.apply_matrix(
            handle=self.handle,
            sv=vector_ptr,
            sv_data_type=cq.cudaDataType.CUDA_C_32F,
            n_index_bits=self.gpu_n,
            matrix=matrix.data.ptr,
            matrix_data_type=cq.cudaDataType.CUDA_C_32F,
            layout=cusv.MatrixLayout.ROW,
            adjoint=0,
            targets=targets.ctypes.data,
            n_targets=len(targets),
            controls=controls.ctypes.data,
            control_bit_values=0,
            n_controls=n_controls,
            compute_type=cq.ComputeType.COMPUTE_32F,
            workspace=self.workspace.ptr, 
            workspace_size=self.workspace_size
        )

    def _init_state(self):
        print("  Running Init...", flush=True)
        t0 = time.time()
        self.bytes_written = 0
        
        # Prepare Zero Chunk
        zeros = np.zeros(self.chunk_size, dtype=np.complex64)
        if self.use_compression:
             zeros_comp = lz4.compress(zeros.tobytes(), compression_level=1)
        
        for i in range(self.n_chunks):
            path = os.path.join(self.storage_dir, f"{i}.bin")
            
            if i == 0:
                chunk = np.zeros(self.chunk_size, dtype=np.complex64)
                chunk[0] = 1.0 + 0.0j
                data = lz4.compress(chunk.tobytes()) if self.use_compression else chunk.tobytes()
            else:
                data = zeros_comp if self.use_compression else zeros.tobytes()
            
            with open(path, 'wb') as f:
                f.write(data)
            self.bytes_written += len(data)
            
            if self.n_chunks >= 10 and i % (self.n_chunks//10) == 0:
                 print(f"  Init: {i}/{self.n_chunks} ({i*100//self.n_chunks}%)", flush=True)
                 
        print(f"  Init Done: {time.time()-t0:.2f}s")
        return time.time()-t0

    def apply_hadamard_q0(self):
        """Apply H gate to qubit 0 (Local Gate) using custatevec"""
        print("  Applying H(q0) with custatevec...", flush=True)
        
        H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        H_gpu = cp.asarray(H)
        
        t0 = time.time()
        new_bytes_written = 0
        
        for i in range(self.n_chunks):
            # 1. Read & Decompress
            path = os.path.join(self.storage_dir, f"{i}.bin")
            with open(path, 'rb') as f:
                data = f.read()
            if self.use_compression:
                data = lz4.decompress(data)
            
            # 2. Host -> Device
            chunk_gpu = cp.frombuffer(data, dtype=np.complex64)
            
            # 3. cuStateVec Apply
            self._apply_gate_custatevec(chunk_gpu.data.ptr, H_gpu, [0])
            
            # 4. Device -> Host
            chunk_host = cp.asnumpy(chunk_gpu)
            
            # 5. Compress & Write
            if self.use_compression:
                cw = lz4.compress(chunk_host.tobytes(), compression_level=1)
                with open(path, 'wb') as f:
                    f.write(cw)
                new_bytes_written += len(cw)
            else:
                 with open(path, 'wb') as f:
                    f.write(chunk_host.tobytes())
                 new_bytes_written += len(chunk_host.tobytes())
            
            if self.n_chunks >= 10 and i % (self.n_chunks//10) == 0:
                 print(f"  Gate: {i}/{self.n_chunks} ({i*100//self.n_chunks}%)", flush=True)

        cp.cuda.Device().synchronize()
        self.bytes_written = new_bytes_written # Update to current size
        print(f"  Gate Done: {time.time()-t0:.2f}s")
        return time.time()-t0

    def cleanup(self):
        try:
            cusv.destroy(self.handle)
            shutil.rmtree(self.storage_dir)
        except:
            pass
        
    def get_storage_gb(self):
        return self.bytes_written / (1024**3)

def run_benchmark():
    qubits_list = [28, 30, 32, 33, 34]
    results = []
    
    print("========================================")
    print("   NATIVE CUQUANTUM BENCHMARK START")
    print("========================================")
    
    for q in qubits_list:
        sim = None
        try:
            sim = CuQuantumTieredSim(q, use_compression=True)
            
            # Init
            t_init = sim._init_state()
            
            # Gate
            t_gate = sim.apply_hadamard_q0()
            
            # Stats
            storage_gb = sim.get_storage_gb()
            raw_size_gb = (2**q * 8) / (1024**3)
            comp_ratio = raw_size_gb / max(storage_gb, 1e-9)
            
            res = {
                "qubits": q,
                "init_time_s": t_init,
                "gate_time_s": t_gate,
                "total_time_s": t_init + t_gate,
                "storage_gb": storage_gb,
                "raw_gb": raw_size_gb,
                "compression_ratio": comp_ratio,
                "success": True
            }
            results.append(res)
            
            print(f"✅ [PASS] {q}Q: Init={t_init:.2f}s, Gate={t_gate:.2f}s, Storage={storage_gb:.3f}GB (ratio={comp_ratio:.1f}x)")
            
        except Exception as e:
            print(f"❌ [FAIL] {q}Q: {e}")
            results.append({"qubits": q, "success": False, "error": str(e)})
        finally:
            if sim: sim.cleanup()
            
        # Save intermediate results
        with open('data/cuquantum_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific qubits from command line
        qubits_to_run = [int(x) for x in sys.argv[1:]]
        # Custom mini-benchmark runner
        results = []
        print(f"🚀 Running Native cuQuantum for: {qubits_to_run}")
        
        for q in qubits_to_run:
            sim = None
            try:
                sim = CuQuantumTieredSim(q, use_compression=True)
                t_init = sim._init_state()
                t_gate = sim.apply_hadamard_q0()
                storage_gb = sim.get_storage_gb()
                raw_size_gb = (2**q * 8) / (1024**3)
                comp_ratio = raw_size_gb / max(storage_gb, 1e-9)
                
                print(f"✅ [PASS] {q}Q: Init={t_init:.2f}s, Gate={t_gate:.2f}s, Storage={storage_gb:.3f}GB (ratio={comp_ratio:.1f}x)")
                
                # Append to existing json if needed or print
                res = {
                    "qubits": q,
                    "init_time_s": t_init,
                    "gate_time_s": t_gate,
                    "total_time_s": t_init + t_gate,
                    "storage_gb": storage_gb,
                    "raw_gb": raw_size_gb,
                    "compression_ratio": comp_ratio,
                    "success": True
                }
                
                # Append to file
                try:
                    with open('data/cuquantum_benchmark.json', 'r') as f:
                        data = json.load(f)
                except:
                    data = []
                data.append(res)
                with open('data/cuquantum_benchmark.json', 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                print(f"❌ [FAIL] {q}Q: {e}")
            finally:
                if sim: sim.cleanup()
    else:
        # Run default suite
        run_benchmark()
