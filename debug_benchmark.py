#!/usr/bin/env python3
"""
EdgeQuantum Unified Benchmark Suite
Compares EdgeQuantum with baseline simulators:
1. cuQuantum (Native) - GPU in-memory, limited by VRAM
2. NumPy (CPU) - CPU in-memory simulation
3. EdgeQuantum - Tiered memory with Zero-Copy

Author: Sunggon Kim
Target: Jetson Orin Nano (8GB DRAM, 15W)
"""
import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Check dependencies
try:
    import cupy as cp
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

# Import EdgeQuantum
from tiered_simulator import TieredSimulator

# Gate definitions
H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex64)

def RY(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


# ============================================================
# BASELINE 2: cuQuantum UVM (Lazy Paging)
# ============================================================

class CuQuantumUVM:
    """Unified Virtual Memory with Lazy Paging (Driver handling)"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.state_ptr = None
        self.handle = None
        self.gate_count = 0
        
        if not HAS_CUQUANTUM:
            raise RuntimeError("cuQuantum not available")
        
        self.handle = cusv.create()
    
    def init_zero_state(self):
        """Initialize using cudaMallocManaged (Oversubscription)"""
        size_bytes = 2**self.n * 8
        try:
            # Allocate Managed Memory (UVM)
            self.state_ptr = cp.cuda.runtime.mallocManaged(size_bytes, 1) # cudaMemAttachGlobal
            
            # Initialize to |0...0> (First element 1, rest 0)
            # We use a trick to avoiding full memset if possible, but for safety:
            # Creating a large array on UVM might trigger page faults immediately
            # Here we just set the first element
            
            # Create a CuPy wrapper to access the memory
            mem = cp.cuda.UnownedMemory(self.state_ptr, size_bytes, owner=self)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            self.state = cp.ndarray((2**self.n,), dtype=cp.complex64, memptr=memptr)
            
            self.state[:] = 0
            self.state[0] = 1.0
            cp.cuda.Device().synchronize()
            
        except Exception as e:
            raise MemoryError(f"UVM Allocation failed: {e}")
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        """Apply gate - GPU Driver handles page faults"""
        matrix_gpu = cp.asarray(gate_matrix)
        targets = np.array([target_qubit], dtype=np.int32)
        
        # This call will trigger page faults as GPU accesses pages
        cusv.apply_matrix(
            handle=self.handle,
            sv=self.state.data.ptr,
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
        # Synchronization is important to ensure page faults are resolved
        cp.cuda.Device().synchronize() 
        self.gate_count += 1
    
    def flush_gates(self):
        cp.cuda.Device().synchronize()
    
    def get_storage_size(self):
        return 2**self.n * 8
    
    def cleanup(self):
        if self.state_ptr:
            cp.cuda.runtime.free(self.state_ptr)
        if self.handle:
            cusv.destroy(self.handle)


# ============================================================
# BASELINE 3: BMQSim-like (Offloading + Compression)
# ============================================================

from tiered_simulator import TieredSimulator

class BMQSimBaseline(TieredSimulator):
    """
    BMQSim-like: Offloading + Compression WITHOUT async prefetch
    
    Key differences from EdgeQuantum:
    - NO async prefetching (blocking I/O)
    - NO zero-copy (explicit memcpy)
    - Same gate batching (fair comparison)
    - Same LZ4 compression
    """
    def __init__(self, n_qubits):
        # Use same fusion threshold as EdgeQuantum for fair gate batching
        super().__init__(n_qubits, use_compression=True, fusion_threshold=10, use_managed_memory=False)
    
    def flush_gates(self):
        """Execute gates with BLOCKING I/O (no prefetch overlap)"""
        if not self.gate_queue:
            return
        
        # IN-MEMORY MODE: Same as parent
        if self.in_memory_mode:
            super()._apply_gates_in_memory()
            self.gate_queue.clear()
            return
        
        t0 = time.time()
        
        # Collect all local gates (fair: same batching as EdgeQuantum)
        local_gates = [(g, t) for g, t in self.gate_queue if t < self.gpu_n]
        
        if local_gates:
            # BLOCKING I/O: No overlap between chunks (key difference from EdgeQuantum)
            for i in range(self.n_chunks):
                # 1. Blocking read
                chunk = self._read_chunk_sync(i)
                
                # 2. Copy to GPU (explicit, not zero-copy)
                chunk_gpu = cp.asarray(chunk.copy())
                
                # 3. Apply ALL batched gates
                for gate, target in local_gates:
                    self._apply_gate_custatevec(chunk_gpu, gate, target)
                
                # 4. Sync and copy back
                cp.cuda.Device().synchronize()
                result = cp.asnumpy(chunk_gpu)
                
                # 5. Blocking write
                self._write_chunk_sync(i, result)
        
        self.gate_queue.clear()
        self.total_gate_time += time.time() - t0
    
    def _read_chunk_sync(self, idx):
        """Synchronous blocking read"""
        path = os.path.join(self.storage_dir, f"{idx}.bin")
        with open(path, 'rb') as f:
            data = f.read()
        if self.use_compression:
            data = lz4.decompress(data)
        return np.frombuffer(data, dtype=np.complex64).copy()
    
    def _write_chunk_sync(self, idx, data):
        """Synchronous blocking write"""
        path = os.path.join(self.storage_dir, f"{idx}.bin")
        if self.use_compression:
            compressed = lz4.compress(data.tobytes(), compression_level=1)
            with open(path, 'wb') as f:
                f.write(compressed)
        else:
            with open(path, 'wb') as f:
                f.write(data.tobytes())
    
    def _apply_gate_custatevec(self, chunk_gpu, gate, target):
        """Apply gate using custatevec"""
        matrix_gpu = cp.asarray(gate)
        targets = np.array([target], dtype=np.int32)
        cusv.apply_matrix(
            handle=self.handle, sv=chunk_gpu.data.ptr,
            sv_data_type=cq.cudaDataType.CUDA_C_32F, n_index_bits=self.gpu_n,
            matrix=matrix_gpu.data.ptr, matrix_data_type=cq.cudaDataType.CUDA_C_32F,
            layout=cusv.MatrixLayout.ROW, adjoint=0,
            targets=targets.ctypes.data, n_targets=1,
            controls=0, control_bit_values=0, n_controls=0,
            compute_type=cq.ComputeType.COMPUTE_32F, workspace=0, workspace_size=0
        )
        if self.use_compression:
            data = lz4.decompress(data)
        return np.frombuffer(data, dtype=np.complex64)

    def _write_chunk(self, idx, data):
        # Direct synchronous write
        path = os.path.join(self.storage_dir, f"{idx}.bin")
        if self.use_compression:
            compressed = lz4.compress(data.tobytes(), compression_level=1)
            with open(path, 'wb') as f:
                f.write(compressed)
        else:
            with open(path, 'wb') as f:
                f.write(data.tobytes())

    def _apply_gate_gpu(self, chunk_gpu, gate, target):
        # Helper for GPU application
        matrix_gpu = cp.asarray(gate)
        targets = np.array([target], dtype=np.int32)
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


class CuQuantumNative:
    """Native cuQuantum simulator - all state in GPU memory"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.state = None
        self.handle = None
        self.gate_count = 0
        
        if not HAS_CUQUANTUM:
            raise RuntimeError("cuQuantum not available")
        
        self.handle = cusv.create()
    
    def init_zero_state(self):
        """Initialize |0⟩^n state in GPU memory"""
        try:
            self.state = cp.zeros(2**self.n, dtype=cp.complex64)
            self.state[0] = 1.0
        except cp.cuda.memory.OutOfMemoryError:
            raise MemoryError(f"Out of GPU memory for {self.n} qubits")
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        """Apply gate using custatevec"""
        matrix_gpu = cp.asarray(gate_matrix)
        targets = np.array([target_qubit], dtype=np.int32)
        
        cusv.apply_matrix(
            handle=self.handle,
            sv=self.state.data.ptr,
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
        self.gate_count += 1
    
    def flush_gates(self):
        """No-op for in-memory simulator"""
        cp.cuda.Device().synchronize()
    
    def get_storage_size(self):
        return 2**self.n * 8  # complex64 = 8 bytes
    
    def cleanup(self):
        del self.state
        if self.handle:
            cusv.destroy(self.handle)


# ============================================================
# BASELINE 2: NumPy CPU Simulator
# ============================================================

class NumpySimulator:
    """Simple NumPy-based CPU simulator"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.state = None
        self.gate_count = 0
    
    def init_zero_state(self):
        """Initialize |0⟩^n state in CPU memory"""
        try:
            self.state = np.zeros(2**self.n, dtype=np.complex64)
            self.state[0] = 1.0
        except MemoryError:
            raise MemoryError(f"Out of CPU memory for {self.n} qubits")
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        """Apply gate using tensor operations"""
        n = len(self.state)
        step = 1 << target_qubit
        
        for i in range(0, n, 2*step):
            for j in range(step):
                idx0 = i + j
                idx1 = i + j + step
                a, b = self.state[idx0], self.state[idx1]
                self.state[idx0] = gate_matrix[0,0]*a + gate_matrix[0,1]*b
                self.state[idx1] = gate_matrix[1,0]*a + gate_matrix[1,1]*b
        
        self.gate_count += 1
    
    def flush_gates(self):
        """No-op for in-memory simulator"""
        pass
    
    def get_storage_size(self):
        return 2**self.n * 8
    
    def cleanup(self):
        del self.state


# ============================================================
# CIRCUIT DEFINITIONS
# ============================================================

def circuit_hadamard(sim, n_gates=None):
    """Hadamard on all qubits"""
    n = min(sim.gpu_n, 22)  # Cap for speed
    if n_gates:
        n = min(n, n_gates)
    for q in range(n):
        sim.apply_single_gate(H, q)
    sim.flush_gates()
    return sim.gate_count

def circuit_random(sim, depth=10):
    """Random circuit"""
    n = min(sim.gpu_n, 22)
    np.random.seed(42)
    gates = [H, T, RY(np.pi/4), RZ(np.pi/4)]
    
    for d in range(depth):
        for q in range(n):
            gate = gates[np.random.randint(len(gates))]
            sim.apply_single_gate(gate, q)
    sim.flush_gates()
    return sim.gate_count

def circuit_qft(sim):
    """QFT-like circuit"""
    n = min(sim.gpu_n, 22)
    for i in range(n):
        sim.apply_single_gate(H, i)
        for j in range(i+1, min(i+4, n)):
            theta = np.pi / (2**(j-i))
            sim.apply_single_gate(RZ(theta), j)
    sim.flush_gates()
    return sim.gate_count



# ============================================================
# BASELINE 5: Google Cirq (CPU)
# ============================================================

class CirqSimulator:
    """Cirq Simulator (CPU)"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.gate_count = 0
        try:
            import cirq
            self.cirq = cirq
            self.sim = cirq.Simulator()
            self.qubits = cirq.LineQubit.range(n_qubits)
            self.circuit = cirq.Circuit()
        except ImportError:
            raise RuntimeError("Cirq not installed")
            
    def init_zero_state(self):
        self.circuit = self.cirq.Circuit()
        
    def apply_single_gate(self, gate_matrix, target_qubit):
        # Create custom gate from matrix
        class MatrixGate(self.cirq.Gate):
            def __init__(self, matrix):
                self.matrix = matrix
            def _num_qubits_(self):
                return 1
            def _unitary_(self):
                return self.matrix
            def _circuit_diagram_info_(self, args):
                return "M"
                
        
        gate = MatrixGate(gate_matrix)
        self.circuit.append(gate(self.qubits[target_qubit]))
        self.gate_count += 1
        
    def flush_gates(self):
        # Trigger actual simulation
        self.sim.simulate(self.circuit)
        # Clear circuit to avoid re-running previous gates if called multiple times (though benchmark calls once per circuit)
        # self.circuit = self.cirq.Circuit() # Do not clear, we want cumulative result if checked
        pass
        
    def get_storage_size(self):
        return 2**self.n * 16 # complex128 by default in Cirq
        
    def cleanup(self):
        del self.sim
        del self.circuit


# ============================================================
# BASELINE 6: PennyLane (Lightning)
# ============================================================

class PennyLaneSimulator:
    """PennyLane with Lightning backend (C++ optimized)"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.gate_count = 0
        try:
            import pennylane as qml
            self.qml = qml
            # Try lightning.qubit first, fall back to default.qubit
            try:
                self.dev = qml.device("lightning.qubit", wires=n_qubits)
            except:
                print("Warning: lightning.qubit not found, using default.qubit")
                self.dev = qml.device("default.qubit", wires=n_qubits)
                
            self.ops = []
        except ImportError:
            raise RuntimeError("PennyLane not installed")
            
    def init_zero_state(self):
        self.ops = []
        
    def apply_single_gate(self, gate_matrix, target_qubit):
        # We store operations to define QNode later
        self.ops.append((gate_matrix, target_qubit))
        self.gate_count += 1
        
    def flush_gates(self):
        # Execute the QNode
        @self.qml.qnode(self.dev)
        def circuit():
            for mat, wire in self.ops:
                self.qml.QubitUnitary(mat, wires=wire)
            return self.qml.state()
            
        circuit()
        
    def run_final(self):
        # Helper to actually run
        @self.qml.qnode(self.dev)
        def circuit():
            for mat, wire in self.ops:
                self.qml.QubitUnitary(mat, wires=wire)
            return self.qml.state()
        return circuit()

    def get_storage_size(self):
        return 2**self.n * 16 # complex128 default
        
    def cleanup(self):
        del self.dev
        self.ops = []


# ============================================================
# BASELINE 7: Intel-QS (MPI/OpenMP)
# ============================================================

class IntelQSSimulator:
    """Intel Quantum Simulator (qHiPSTER) via PyBind11"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.gate_count = 0
        try:
            import sys
            # Add Intel-QS build path
            sys.path.append("/home/jetson/skim/intel-qs/build/lib")
            import intelqs_py as iqs
            self.iqs = iqs
            self.qr = iqs.QubitRegister(n_qubits, "base", 0, 0)
        except ImportError:
            raise RuntimeError("Intel-QS not installed or module not found")
            
    def init_zero_state(self):
        # Re-initialize
        # Intel-QS QubitRegister doesn't have a simple 'reset', usually we create new
        # But for overhead reasons, we can just zero and set index 0.
        # Ideally, we recreate.
        self.qr.Initialize("base", 0)
        
    def apply_single_gate(self, gate_matrix, target_qubit):
        # Intel-QS supports custom gates via Apply1QubitGate(qubit, matrix)
        # The signature expects a numpy array (complex128)
        self.qr.Apply1QubitGate(target_qubit, gate_matrix.astype(np.complex128))
        self.gate_count += 1
    
    def flush_gates(self):
        pass
        
    def get_storage_size(self):
        return 2**self.n * 16 # complex128
        
    def cleanup(self):
        del self.qr


class QiskitAerGPU:
    """Qiskit Aer with GPU backend and Max Memory limit (Swapping)"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gpu_n = n_qubits
        self.gate_count = 0
        if self.n > 30: # Safe guard for heavy simulations
             pass 
        try:
            from qiskit_aer import AerSimulator
            from qiskit import QuantumCircuit
            self.qc = QuantumCircuit(n_qubits)
            # Set memory limit to force swapping (e.g., 4GB)
            # Jetson Orin Nano has 8GB shared, so 4GB limit leaves room for OS
            try:
                self.sim = AerSimulator(method='statevector', device='GPU', max_memory_mb=4096)
                # Test GPU availability
                self.sim.run(self.qc, shots=1).result()
            except Exception as e:
                print(f"Warning: Qiskit GPU failed ({e}), falling back to CPU")
                self.sim = AerSimulator(method='statevector', device='CPU', max_memory_mb=4096)
        except ImportError:
            raise RuntimeError("Qiskit Aer not installed")
            
    def init_zero_state(self):
        # Qiskit initializes to zero by default
        self.qc.reset(range(self.n))
        pass
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        # Convert matrix to unitary gate
        from qiskit.circuit.library import UnitaryGate
        gate = UnitaryGate(gate_matrix, label="Gate")
        self.qc.append(gate, [target_qubit])
        self.gate_count += 1
    
    def flush_gates(self):
        # Run simulation
        # Note: Qiskit compiles and runs the whole circuit at once usually.
        # But for gate-by-gate benchmark fairness, we might want to run simplified
        # However, Qiskit is a circuit runner. 
        # To match the API, we accumulate gates in self.qc and run() here.
        
        # Transpile is overhead, so we try to run directly if supported
        # AerSimulator can run circuits directly
        result = self.sim.run(self.qc, shots=1).result()
        # Verify success (optional)
        
    def get_storage_size(self):
        return 2**self.n * 8
    
    def cleanup(self):
        del self.qc
        del self.sim


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_single_benchmark(sim_class, sim_name, n_qubits, circuit_fn, circuit_name, **kwargs):
    """Run single benchmark and return result"""
    result = {
        'simulator': sim_name,
        'circuit': circuit_name,
        'qubits': n_qubits,
        'success': False,
        'error': None
    }
    
    try:
        # Create simulator
        if sim_name == "EdgeQuantum (Ours)":
            sim = sim_class(n_qubits, use_compression=True, fusion_threshold=10)
        elif sim_name == "BMQSim-like (Swap)":
            sim = sim_class(n_qubits) # Uses default TieredSim init but overridden methods
        else:
            sim = sim_class(n_qubits)
        
        # SAFETY CHECK: Skip purely CPU-based simulators for > 26 qubits (prevent freeze)
        is_cpu_sim = sim_name in ["Google Cirq", "PennyLane (Ltn)", "Intel-QS (MPI)", "Qiskit Aer (GPU)"]
        if is_cpu_sim and n_qubits > 26:
             result['error'] = "SKIPPED (>26Q CPU Limit)"
             if hasattr(sim, 'cleanup'):
                 sim.cleanup()
             return result
        
        # Initialize
        t0 = time.time()
        sim.init_zero_state()
        result['init_time'] = time.time() - t0
        
        # Run circuit
        t0 = time.time()
        # Clean gate count for Qiskit which accumulates
        if hasattr(sim, 'qc'):
            from qiskit import QuantumCircuit
            sim.qc = QuantumCircuit(n_qubits)
            
        gate_count = circuit_fn(sim, **kwargs) if kwargs else circuit_fn(sim)
        result['circuit_time'] = time.time() - t0
        
        result['gates'] = gate_count
        result['total_time'] = result['init_time'] + result['circuit_time']
        result['storage_gb'] = sim.get_storage_size() / (1024**3)
        result['success'] = True
        
        sim.cleanup()
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_benchmark_suite():
    """Run full benchmark suite"""
    print("=" * 80)
    print("   EDGEQUANTUM UNIFIED BENCHMARK SUITE")
    print("   Device: NVIDIA Jetson Orin Nano (8GB, 15W)")
    print("=" * 80)
    
    # Simulators to test
    simulators = [
        (CirqSimulator, "Google Cirq"),
    ]
    
    # Test configurations
    # Test configurations
    qubit_counts = [20]
    
    circuits = [
        (circuit_hadamard, "Hadamard", {}),
        (circuit_random, "Random-10", {'depth': 10}),
        (circuit_qft, "QFT", {}),
    ]
    
    results = []
    
    for n_qubits in qubit_counts:
        print(f"\n{'='*50}")
        print(f"  Testing {n_qubits} Qubits")
        print(f"{'='*50}")
        
        for circuit_fn, circuit_name, kwargs in circuits:
            print(f"\n  Circuit: {circuit_name}")
            
            for sim_class, sim_name in simulators:
                print(f"    {sim_name}... ", end="", flush=True)
                
                result = run_single_benchmark(
                    sim_class, sim_name, n_qubits, 
                    circuit_fn, circuit_name, **kwargs
                )
                results.append(result)
                
                if result['success']:
                    print(f"✅ {result['circuit_time']:.2f}s ({result['gates']} gates)")
                else:
                    print(f"❌ {result['error']}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'data/unified_benchmark_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("   RESULTS SUMMARY")
    print("=" * 70)
    
    # Group by circuit
    for circuit_name in ['Hadamard', 'Random-10', 'QFT']:
        print(f"\n📊 {circuit_name}:")
        print(f"{'Qubits':<8} {'Native':<11} {'UVM':<11} {'BMQSim':<11} {'Aer(CPU)':<11} {'Cirq':<11} {'PennyL':<11} {'IntelQS':<11} {'EdgeQ':<11}")
        print("-" * 120)
        
        for n in qubit_counts:
            row = f"{n:<8}"
            for sim_name in ['cuQuantum (Native)', 'cuQuantum (UVM)', 'BMQSim-like (Swap)', 'Qiskit Aer (GPU)', 'Google Cirq', 'PennyLane (Ltn)', 'Intel-QS (MPI)', 'EdgeQuantum (Ours)']:
                match = [r for r in results 
                         if r['qubits'] == n and r['circuit'] == circuit_name and r['simulator'] == sim_name]
                if match and match[0]['success']:
                    row += f"{match[0]['circuit_time']:.2f}s".ljust(11)
                else:
                    row += "FAIL".ljust(11)
            print(row)
    
    print(f"\n💾 Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_benchmark_suite()
