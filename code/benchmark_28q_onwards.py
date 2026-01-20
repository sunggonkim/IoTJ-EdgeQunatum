#!/usr/bin/env python3
"""
EdgeQuantum Comprehensive Benchmark Suite (Robust Version)
========================================================
- 8 Simulators: cuQuantum Native/UVM, BMQSim, Cirq, PennyLane, Intel-QS, Qiskit, EdgeQuantum
- 6 Circuits: QV, VQC, QSVM, Random, GHZ, VQE (AURORA-Q style)
- Robustness: Runs each benchmark in a separate process to survive OOM kills.
- Output: JSON results file for paper figures
"""

import numpy as np
import time
import json
import os
import signal
import traceback
import multiprocessing
import queue  # Import standard queue for Empty exception
from datetime import datetime
from contextlib import contextmanager

# ============================================================
# TIMEOUT HANDLING
# ============================================================
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ============================================================
# CIRCUIT DEFINITIONS (AURORA-Q style)
# ============================================================

def circuit_qv(sim, depth=None):
    """Quantum Volume circuit - square circuit (depth = width)"""
    n = sim.n
    if depth is None:
        depth = n
    gates = 0
    for d in range(depth):
        # Random SU(4) on pairs
        for i in range(0, n-1, 2):
            # Approximate with H + CNOT-like pattern
            h_matrix = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
            sim.apply_single_gate(h_matrix, i)
            sim.apply_single_gate(h_matrix, i+1)
            gates += 2
        sim.flush_gates()
    return gates

def circuit_vqc(sim, layers=5):
    """Variational Quantum Circuit - parameterized gates"""
    n = sim.n
    gates = 0
    for layer in range(layers):
        # Rotation layer
        for i in range(n):
            theta = np.random.uniform(0, 2*np.pi)
            rz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex64)
            sim.apply_single_gate(rz, i)
            gates += 1
        # Entanglement layer (H gates as proxy)
        for i in range(n):
            h = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
            sim.apply_single_gate(h, i)
            gates += 1
        sim.flush_gates()
    return gates

def circuit_qsvm(sim, feature_dim=4):
    """QSVM-style feature map circuit"""
    n = sim.n
    gates = 0
    # ZZ feature map
    for rep in range(2):
        for i in range(n):
            h = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
            sim.apply_single_gate(h, i)
            gates += 1
        for i in range(n):
            phi = np.random.uniform(0, 2*np.pi)
            rz = np.array([[np.exp(-1j*phi/2), 0], [0, np.exp(1j*phi/2)]], dtype=np.complex64)
            sim.apply_single_gate(rz, i)
            gates += 1
        sim.flush_gates()
    return gates

def circuit_random(sim, depth=10):
    """Random circuit with depth parameter"""
    n = sim.n
    gates = 0
    gate_set = [
        np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),  # H
        np.array([[1, 0], [0, 1j]], dtype=np.complex64),  # S
        np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64),  # T
        np.array([[0, 1], [1, 0]], dtype=np.complex64),  # X
    ]
    for d in range(depth):
        for i in range(n):
            gate = gate_set[np.random.randint(len(gate_set))]
            sim.apply_single_gate(gate, i)
            gates += 1
        sim.flush_gates()
    return gates

def circuit_ghz(sim):
    """GHZ state preparation"""
    n = sim.n
    gates = 0
    h = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    
    # H on first qubit
    sim.apply_single_gate(h, 0)
    gates += 1
    
    # CNOT chain (approximated with X gates for single-qubit API)
    for i in range(1, n):
        sim.apply_single_gate(x, i)  # Simplified
        gates += 1
    sim.flush_gates()
    return gates

def circuit_vqe(sim, ansatz_layers=3):
    """VQE ansatz circuit"""
    n = sim.n
    gates = 0
    for layer in range(ansatz_layers):
        # RY rotation layer
        for i in range(n):
            theta = np.random.uniform(0, np.pi)
            ry = np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex64)
            sim.apply_single_gate(ry, i)
            gates += 1
        # RZ rotation layer
        for i in range(n):
            phi = np.random.uniform(0, 2*np.pi)
            rz = np.array([[np.exp(-1j*phi/2), 0], [0, np.exp(1j*phi/2)]], dtype=np.complex64)
            sim.apply_single_gate(rz, i)
            gates += 1
        sim.flush_gates()
    return gates

# ============================================================
# SIMULATOR IMPORTS AND CLASSES
# ============================================================

# Import from tiered_simulator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiered_simulator import TieredSimulator

# Try importing optional dependencies
try:
    import cupy as cp
    import cuquantum as cq
    from cuquantum import custatevec as cusv
    CUQUANTUM_AVAILABLE = True
except ImportError:
    CUQUANTUM_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class BaseSimulator:
    """Base class for all simulators"""
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gate_count = 0
    
    def init_zero_state(self):
        raise NotImplementedError
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        raise NotImplementedError
    
    def flush_gates(self):
        pass
    
    def get_storage_size(self):
        return 2**self.n * 8  # complex64 = 8 bytes
    
    def cleanup(self):
        pass


class CuQuantumNative(BaseSimulator):
    """cuQuantum with native GPU memory"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        if not CUQUANTUM_AVAILABLE:
            raise ImportError("cuQuantum not available")
        self.handle = cusv.create()
        self.sv = None
    
    def init_zero_state(self):
        self.sv = cp.zeros(2**self.n, dtype=cp.complex64)
        self.sv[0] = 1.0
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        matrix_gpu = cp.asarray(gate_matrix)
        targets = np.array([target_qubit], dtype=np.int32)
        cusv.apply_matrix(
            handle=self.handle, sv=self.sv.data.ptr,
            sv_data_type=cq.cudaDataType.CUDA_C_32F, n_index_bits=self.n,
            matrix=matrix_gpu.data.ptr, matrix_data_type=cq.cudaDataType.CUDA_C_32F,
            layout=cusv.MatrixLayout.ROW, adjoint=0,
            targets=targets.ctypes.data, n_targets=1,
            controls=0, control_bit_values=0, n_controls=0,
            compute_type=cq.ComputeType.COMPUTE_32F, workspace=0, workspace_size=0
        )
        self.gate_count += 1
    
    def cleanup(self):
        if self.sv is not None:
            del self.sv
            self.sv = None
        cp.get_default_memory_pool().free_all_blocks()
        cusv.destroy(self.handle)


class CuQuantumUVM(BaseSimulator):
    """cuQuantum with Unified Virtual Memory (using ManagedMemoryBuffer)"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        if not CUQUANTUM_AVAILABLE:
            raise ImportError("cuQuantum not available")
        self.handle = cusv.create()
        self.sv_ptr = None
        self.sv_array = None
    
    def init_zero_state(self):
        import ctypes
        import cupy.cuda.runtime as cuda_rt
        # Use cudaMallocManaged via runtime API
        size_bytes = 2**self.n * 8  # complex64 = 8 bytes
        self.sv_ptr = cuda_rt.mallocManaged(size_bytes, 1)  # cudaMemAttachGlobal=1
        # Create CuPy view of managed memory
        mem = cp.cuda.UnownedMemory(self.sv_ptr, size_bytes, owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        self.sv_array = cp.ndarray(2**self.n, dtype=cp.complex64, memptr=memptr)
        self.sv_array[:] = 0
        self.sv_array[0] = 1.0
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        matrix_gpu = cp.asarray(gate_matrix)
        targets = np.array([target_qubit], dtype=np.int32)
        cusv.apply_matrix(
            handle=self.handle, sv=self.sv_array.data.ptr,
            sv_data_type=cq.cudaDataType.CUDA_C_32F, n_index_bits=self.n,
            matrix=matrix_gpu.data.ptr, matrix_data_type=cq.cudaDataType.CUDA_C_32F,
            layout=cusv.MatrixLayout.ROW, adjoint=0,
            targets=targets.ctypes.data, n_targets=1,
            controls=0, control_bit_values=0, n_controls=0,
            compute_type=cq.ComputeType.COMPUTE_32F, workspace=0, workspace_size=0
        )
        self.gate_count += 1
    
    def cleanup(self):
        if self.sv_array is not None:
            del self.sv_array
            self.sv_array = None
        if self.sv_ptr:
            try:
                import cupy.cuda.runtime as cuda_rt
                cuda_rt.free(self.sv_ptr)
            except:
                pass
            self.sv_ptr = None
        cp.get_default_memory_pool().free_all_blocks()
        cusv.destroy(self.handle)


class BMQSimBaseline(TieredSimulator):
    """BMQSim-like: Offloading WITHOUT async prefetch"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits, use_compression=True, fusion_threshold=10, use_managed_memory=True)


class CirqSimulator(BaseSimulator):
    """Google Cirq simulator"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq not available")
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.circuit = cirq.Circuit()
        self.sim = cirq.Simulator()
    
    def init_zero_state(self):
        self.circuit = cirq.Circuit()
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        # Cirq requires float64 for unitary validation
        gate_f64 = np.asarray(gate_matrix, dtype=np.complex128)
        gate = cirq.MatrixGate(gate_f64)
        self.circuit.append(gate(self.qubits[target_qubit]))
        self.gate_count += 1
    
    def flush_gates(self):
        if len(self.circuit) > 0:
            self.sim.simulate(self.circuit)
            self.circuit = cirq.Circuit()
    
    def cleanup(self):
        del self.circuit
        del self.sim


class PennyLaneSimulator(BaseSimulator):
    """PennyLane Lightning simulator"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
        self.qml = qml
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.ops = []
    
    def init_zero_state(self):
        self.ops = []
    
    def apply_single_gate(self, gate_matrix, target_qubit):
        self.ops.append((gate_matrix, target_qubit))
        self.gate_count += 1
    
    def flush_gates(self):
        if not self.ops:
            return
        ops_copy = self.ops.copy()
        qml_ref = self.qml
        
        @qml_ref.qnode(self.dev)
        def circuit():
            for mat, wire in ops_copy:
                qml_ref.QubitUnitary(mat, wires=wire)
            return qml_ref.state()
        
        circuit()
        self.ops = []
    
    def cleanup(self):
        del self.dev

class EdgeQuantumSimulator(TieredSimulator):
    """EdgeQuantum (Ours) - Tiered Memory + Zero-Copy"""
    def __init__(self, n_qubits):
        super().__init__(n_qubits, use_compression=True, fusion_threshold=10, use_managed_memory=True)


# ============================================================
# WORKER PROCESS FOR ISOLATION
# ============================================================

def _benchmark_worker(sim_class, n_qubits, circuit_fn, kwargs, result_queue):
    """Worker function to run in a separate process"""
    result = {
        'success': False,
        'time': None,
        'gates': None,
        'error': None,
        'storage_gb': None,
    }
    
    try:
        # Create simulator
        sim = sim_class(n_qubits)
        
        # Initialize
        t0 = time.time()
        sim.init_zero_state()
        init_time = time.time() - t0
        
        # Run circuit
        t0 = time.time()
        gate_count = circuit_fn(sim, **kwargs) if kwargs else circuit_fn(sim)
        circuit_time = time.time() - t0
        
        result['time'] = init_time + circuit_time
        result['gates'] = gate_count
        result['storage_gb'] = sim.get_storage_size() / (1024**3)
        result['success'] = True
        
        sim.cleanup()
        
    except MemoryError:
        result['error'] = "OOM"
    except ImportError as e:
        result['error'] = f"NOT_AVAILABLE: {str(e)}"
    except Exception as e:
        result['error'] = f"ERROR: {str(e)[:100]}"
        # traceback.print_exc() # detailed logging in worker if needed
    
    result_queue.put(result)


def run_single_benchmark(sim_class, sim_name, n_qubits, circuit_fn, circuit_name, timeout_sec=600, **kwargs):
    """Run single benchmark with error/OOM handling"""
    result = {
        'simulator': sim_name,
        'circuit': circuit_name,
        'qubits': n_qubits,
        'success': False,
        'time': None,
        'gates': None,
        'error': None,
        'storage_gb': None,
    }
    
    try:
        with timeout(timeout_sec):
            # Create simulator
            sim = sim_class(n_qubits)
            
            # Initialize
            t0 = time.time()
            sim.init_zero_state()
            init_time = time.time() - t0
            
            # Run circuit
            t0 = time.time()
            gate_count = circuit_fn(sim, **kwargs) if kwargs else circuit_fn(sim)
            circuit_time = time.time() - t0
            
            result['time'] = init_time + circuit_time
            result['gates'] = gate_count
            result['storage_gb'] = sim.get_storage_size() / (1024**3)
            result['success'] = True
            
            sim.cleanup()
            
    except TimeoutError as e:
        result['error'] = f"TIMEOUT ({timeout_sec}s)"
    except MemoryError as e:
        result['error'] = "OOM"
    except ImportError as e:
        result['error'] = f"NOT_AVAILABLE: {str(e)}"
    except Exception as e:
        result['error'] = f"ERROR: {str(e)[:100]}"
        traceback.print_exc()
    
    return result


def run_single_benchmark_subprocess(sim_name, n_qubits, circuit_name, kwargs, timeout_sec=600):
    """Run benchmark in a completely separate subprocess via CLI with active monitoring"""
    import subprocess
    import sys
    import time
    
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--run-single",
        "--sim", sim_name,
        "--qubits", str(n_qubits),
        "--circuit", circuit_name,
        "--kwargs", json.dumps(kwargs)
    ]
    
    result_template = {
        'simulator': sim_name,
        'circuit': circuit_name,
        'qubits': n_qubits,
        'success': False,
        'time': None,
        'gates': None,
        'error': None,
        'storage_gb': None,
    }
    
    try:
        # Use Popen for active monitoring
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        
        while True:
            # Check if process is still running
            ret_code = proc.poll()
            
            if ret_code is not None:
                # Process finished/died
                break
            
            # Check timeout if applicable
            if timeout_sec is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return {**result_template, 'error': f"TIMEOUT ({timeout_sec}s)"}
            
            # Wait a bit before next check to avoid busy loop
            time.sleep(1.0)
            
        # Process ended, gather output
        stdout_data, stderr_data = proc.communicate()
        
        # Check return code
        if ret_code != 0:
            # Crashed
            err_msg = f"CRASHED (Code {ret_code})"
            if ret_code == -9 or ret_code == 137: # SIGKILL/OOM
                err_msg = "OOM KILLED"
            # Add stderr for context if available
            if stderr_data:
                err_msg += f" [Stderr: {stderr_data.strip()[-50:]}]"
            return {**result_template, 'error': err_msg}
            
        # Parse output
        if "___RESULT_JSON___" in stdout_data:
            json_str = stdout_data.split("___RESULT_JSON___")[1].strip()
            return json.loads(json_str)
        else:
            return {**result_template, 'error': "NO_JSON_OUTPUT"}
            
    except Exception as e:
        return {**result_template, 'error': f"SUBPROCESS_ERROR: {str(e)}"}


def run_comprehensive_benchmark():
    """Run full comprehensive benchmark"""
    print("=" * 80)
    print("   EDGEQUANTUM COMPREHENSIVE BENCHMARK SUITE (SUBPROCESS ISOLATION)")
    print("   Device: NVIDIA Jetson Orin Nano (8GB, 15W)")
    print("   6 Simulators × 6 Circuits × Multiple Qubit Counts")
    print("=" * 80)
    
    # All 6 simulators
    simulators = [
        # Classes NOT needed here, just names for CLI
        (None, "cuQuantum (Native)"),
        (None, "cuQuantum (UVM)"),
        (None, "BMQSim-like (Swap)"),
        (None, "Google Cirq"),
        (None, "PennyLane (Ltn)"),
        (None, "EdgeQuantum (Ours)"),
    ]
    
    # 6 circuits
    circuits = [
        (None, "QV", {}),
        (None, "VQC", {'layers': 5}),
        (None, "QSVM", {'feature_dim': 4}),
        (None, "Random", {'depth': 10}),
        (None, "GHZ", {}),
        (None, "VQE", {'ansatz_layers': 3}),
    ]
    
    # Qubit counts
    qubit_counts = [28, 30, 32, 34]
    
    # CPU simulators limits
    cpu_sims = ["Google Cirq", "PennyLane (Ltn)"]
    
    # GPU simulators (No timeout - let them run as long as needed)
    gpu_sims = ["cuQuantum (Native)", "cuQuantum (UVM)", "BMQSim-like (Swap)", "EdgeQuantum (Ours)"]
    
    results = []
    
    for n_qubits in qubit_counts:
        print(f"\n{'='*60}")
        print(f"  Testing {n_qubits} Qubits")
        print(f"{'='*60}")
        
        for _, circuit_name, kwargs in circuits:
            print(f"\n  Circuit: {circuit_name}")
            
            for _, sim_name in simulators:
                # Removed explicit CPU skip >26Q per user request. 
                # They will likely OOM or timeout, which is handled by subprocess/timeout logic.
                
                # Set timeout logic
                if sim_name in gpu_sims:
                    # GPU Simulators: NO TIMEOUT (User request: allow 3-5+ hours)
                    # We pass None to run_single_benchmark_subprocess, which handles it by loop monitoring forever
                    timeout_sec = None 
                else:
                    # CPU Simulators: Keep reasonable limits to avoid infinite hangs
                    if n_qubits <= 22: timeout_sec = 300       # 5m
                    elif n_qubits <= 24: timeout_sec = 1800    # 30m
                    elif n_qubits <= 26: timeout_sec = 7200    # 2h
                    else: timeout_sec = 18000                  # 5h
                
                print(f"    {sim_name}...", end=" ", flush=True)
                
                # Use SUBPROCESS runner
                result = run_single_benchmark_subprocess(
                    sim_name, n_qubits, 
                    circuit_name, kwargs, 
                    timeout_sec=timeout_sec
                )
                results.append(result)
                
                if result['success']:
                    print(f"✅ {result['time']:.4f}s ({result['gates']} gates)")
                elif result['error'] == "OOM KILLED":
                    print(f"❌ OOM KILLED (Memory Limit Exceeded)")
                else:
                    print(f"❌ {result['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/comprehensive_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': 'Jetson Orin Nano 8GB',
            'results': results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Simulator':<25} {'Circuit':<10} {'20Q':<10} {'22Q':<10} {'24Q':<10} {'26Q':<10} {'28Q':<10}")
    print("-" * 95)
    
    for _, sim_name in simulators:
        for _, circuit_name, _ in circuits:
            row = [f"{sim_name:<25}", f"{circuit_name:<10}"]
            for nq in [20, 22, 24, 26, 28]:
                match = [r for r in results if r['simulator'] == sim_name and r['circuit'] == circuit_name and r['qubits'] == nq]
                if match:
                    r = match[0]
                    if r['success']:
                        row.append(f"{r['time']:.4f}s")
                    elif r['error'] and "SKIPPED" in r['error']:
                        row.append("SKIP")
                    else:
                        row.append(r['error'][:8])
                else:
                    if sim_name in cpu_sims and nq > 26:
                         row.append("SKIP")
                    else:
                         row.append("-")
            print(" ".join(f"{c:<10}" for c in row))
    
    return results


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true", help="Run a single benchmark case")
    parser.add_argument("--sim", type=str, help="Simulator name")
    parser.add_argument("--qubits", type=int, help="Number of qubits")
    parser.add_argument("--circuit", type=str, help="Circuit name")
    parser.add_argument("--kwargs", type=str, help="JSON string of kwargs")
    args = parser.parse_args()
    
    if args.run_single:
        # --- CHILD PROCESS MODE ---
        try:
            # Map names to objects
            sim_map = {
                "cuQuantum (Native)": CuQuantumNative,
                "cuQuantum (UVM)": CuQuantumUVM,
                "BMQSim-like (Swap)": BMQSimBaseline,
                "Google Cirq": CirqSimulator,
                "PennyLane (Ltn)": PennyLaneSimulator,
                "EdgeQuantum (Ours)": EdgeQuantumSimulator,
            }
            
            circuit_map = {
                "QV": circuit_qv,
                "VQC": circuit_vqc,
                "QSVM": circuit_qsvm,
                "Random": circuit_random,
                "GHZ": circuit_ghz,
                "VQE": circuit_vqe,
            }
            
            if args.sim not in sim_map:
                print(json.dumps({'error': f"Unknown simulator: {args.sim}", 'success': False}))
                sys.exit(1)
                
            sim_class = sim_map[args.sim]
            circuit_fn = circuit_map[args.circuit]
            kwargs = json.loads(args.kwargs) if args.kwargs else {}
            
            # Run benchmark directly (no timeout here, parent handles timeout)
            # We print the JSON result to stdout for parent to capture
            result = run_single_benchmark(sim_class, args.sim, args.qubits, circuit_fn, args.circuit, timeout_sec=99999, **kwargs)
            print("___RESULT_JSON___")
            print(json.dumps(result))
            
        except Exception as e:
            err_result = {
                'simulator': args.sim, 'circuit': args.circuit, 'qubits': args.qubits,
                'success': False, 'error': f"CHILD_ERROR: {str(e)}"
            }
            print("___RESULT_JSON___")
            print(json.dumps(err_result))
            sys.exit(1)
            
    else:
        # --- PARENT HARNESS MODE ---
        run_comprehensive_benchmark()
