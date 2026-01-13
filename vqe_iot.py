#!/usr/bin/env python3
"""
VQE (Variational Quantum Eigensolver) on Jetson IoT Edge
Implements VQE for finding ground state energies of molecular Hamiltonians.
Scalable from 2 to 20+ qubits on edge GPU using CuPy.
"""

import time
import numpy as np
import cupy as cp
from scipy.optimize import minimize
import json
from datetime import datetime
import subprocess
import psutil
import gc

# ============================================================
# Quantum Gates (CuPy GPU-accelerated)
# ============================================================

def H_gate():
    """Hadamard gate"""
    return cp.array([[1, 1], [1, -1]], dtype=cp.complex64) / np.sqrt(2)

def X_gate():
    """Pauli-X gate"""
    return cp.array([[0, 1], [1, 0]], dtype=cp.complex64)

def Y_gate():
    """Pauli-Y gate"""
    return cp.array([[0, -1j], [1j, 0]], dtype=cp.complex64)

def Z_gate():
    """Pauli-Z gate"""
    return cp.array([[1, 0], [0, -1]], dtype=cp.complex64)

def I_gate():
    """Identity gate"""
    return cp.array([[1, 0], [0, 1]], dtype=cp.complex64)

def RX_gate(theta):
    """Rotation around X-axis"""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return cp.array([[c, -1j*s], [-1j*s, c]], dtype=cp.complex64)

def RY_gate(theta):
    """Rotation around Y-axis"""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return cp.array([[c, -s], [s, c]], dtype=cp.complex64)

def RZ_gate(theta):
    """Rotation around Z-axis"""
    return cp.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=cp.complex64)

def CNOT_matrix():
    """CNOT gate (4x4)"""
    return cp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=cp.complex64)

# ============================================================
# State Vector Simulator
# ============================================================

class GPUStateVector:
    """GPU-accelerated state vector simulator for variational circuits."""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.reset()
    
    def reset(self):
        """Reset to |0...0> state"""
        self.sv = cp.zeros(self.dim, dtype=cp.complex64)
        self.sv[0] = 1.0 + 0.0j
    
    def apply_single_gate(self, gate, target):
        """Apply single-qubit gate using tensor reshape method."""
        shape = [2] * self.n_qubits
        sv_reshaped = self.sv.reshape(shape)
        
        # Transpose to bring target qubit to front
        axes = list(range(self.n_qubits))
        axes.remove(target)
        axes.insert(0, target)
        sv_t = cp.transpose(sv_reshaped, axes)
        
        # Apply gate
        sv_flat = sv_t.reshape(2, -1)
        sv_flat = cp.matmul(gate, sv_flat)
        
        # Transpose back
        sv_t = sv_flat.reshape([2] + [2] * (self.n_qubits - 1))
        inv_axes = [0] * self.n_qubits
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        sv_reshaped = cp.transpose(sv_t, inv_axes)
        self.sv = sv_reshaped.flatten()
    
    def apply_cnot(self, control, target):
        """Apply CNOT gate between control and target qubits."""
        # Use conditional application approach
        shape = [2] * self.n_qubits
        sv_reshaped = self.sv.reshape(shape)
        
        # Extract slices where control qubit = 1
        # and apply X gate to target
        slices_ctrl_1 = [slice(None)] * self.n_qubits
        slices_ctrl_1[control] = 1
        
        # Get the subspace where control=1
        sub = sv_reshaped[tuple(slices_ctrl_1)].copy()
        
        # Apply X gate to target qubit in this subspace
        target_rel = target if target < control else target - 1
        sub_shape = list(sub.shape)
        
        # Swap |0> and |1> on target qubit
        if sub.ndim > 0:
            axes = list(range(sub.ndim))
            # Move target to front
            if target < control:
                axes.remove(target)
                axes.insert(0, target)
            else:
                axes.remove(target - 1)
                axes.insert(0, target - 1)
            
            sub_t = cp.transpose(sub, axes)
            # Swap first axis (target qubit)
            sub_t = cp.flip(sub_t, axis=0)
            # Transpose back
            inv_axes = [0] * len(axes)
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            sub = cp.transpose(sub_t, inv_axes)
        
        sv_reshaped[tuple(slices_ctrl_1)] = sub
        self.sv = sv_reshaped.flatten()
    
    def get_state(self):
        """Return current state vector (on GPU)"""
        return self.sv
    
    def get_probabilities(self):
        """Return measurement probabilities"""
        return cp.abs(self.sv) ** 2
    
    def expectation_Z(self, qubit):
        """Compute <Z> expectation value for a single qubit"""
        probs = self.get_probabilities()
        
        # Z eigenvalue is +1 for |0>, -1 for |1>
        shape = [2] * self.n_qubits
        probs_reshaped = probs.reshape(shape)
        
        # Sum probabilities where qubit=0 (eigenvalue +1)
        # and qubit=1 (eigenvalue -1)
        idx_0 = [slice(None)] * self.n_qubits
        idx_0[qubit] = 0
        idx_1 = [slice(None)] * self.n_qubits
        idx_1[qubit] = 1
        
        p0 = float(cp.sum(probs_reshaped[tuple(idx_0)]))
        p1 = float(cp.sum(probs_reshaped[tuple(idx_1)]))
        
        return p0 - p1  # <Z> = P(0) - P(1)
    
    def expectation_ZZ(self, qubit1, qubit2):
        """Compute <Z⊗Z> expectation value for two qubits"""
        probs = self.get_probabilities()
        shape = [2] * self.n_qubits
        probs_reshaped = probs.reshape(shape)
        
        # Z⊗Z eigenvalues: |00>→+1, |01>→-1, |10>→-1, |11>→+1
        total = 0.0
        for b1 in [0, 1]:
            for b2 in [0, 1]:
                idx = [slice(None)] * self.n_qubits
                idx[qubit1] = b1
                idx[qubit2] = b2
                eigenvalue = 1 if b1 == b2 else -1
                total += eigenvalue * float(cp.sum(probs_reshaped[tuple(idx)]))
        
        return total

# ============================================================
# VQE Ansätze (Variational Circuits)
# ============================================================

class HardwareEfficientAnsatz:
    """Hardware-efficient ansatz with RY-RZ rotations and CNOT entanglement."""
    
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Parameters: 2 rotations per qubit per layer (RY, RZ)
        self.n_params = 2 * n_qubits * n_layers
    
    def get_n_params(self):
        return self.n_params
    
    def apply(self, sim: GPUStateVector, params):
        """Apply the ansatz circuit with given parameters."""
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                sim.apply_single_gate(RY_gate(params[param_idx]), q)
                param_idx += 1
                sim.apply_single_gate(RZ_gate(params[param_idx]), q)
                param_idx += 1
            
            # Entangling layer: linear CNOT chain
            for q in range(self.n_qubits - 1):
                sim.apply_cnot(q, q + 1)

class UCC_SD_Ansatz:
    """
    Simplified UCC-SD (Unitary Coupled Cluster Singles-Doubles) ansatz.
    For H2 molecule simulation.
    """
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        # For simplicity: one parameter per qubit pair
        self.n_params = max(1, n_qubits // 2)
    
    def get_n_params(self):
        return self.n_params
    
    def apply(self, sim: GPUStateVector, params):
        """Apply UCCSD-inspired circuit."""
        # Initialize in HF state (half-filled)
        n_electrons = self.n_qubits // 2
        for q in range(n_electrons):
            sim.apply_single_gate(X_gate(), q)
        
        # Apply excitation operators (simplified)
        param_idx = 0
        for i in range(0, self.n_qubits - 1, 2):
            if param_idx < len(params):
                theta = params[param_idx]
                # Single excitation-like rotation
                sim.apply_single_gate(RY_gate(theta), i)
                if i + 1 < self.n_qubits:
                    sim.apply_cnot(i, i + 1)
                    sim.apply_single_gate(RY_gate(-theta/2), i + 1)
                    sim.apply_cnot(i, i + 1)
                param_idx += 1

# ============================================================
# Hamiltonians
# ============================================================

class H2_Hamiltonian:
    """
    H2 molecule Hamiltonian in STO-3G basis (2-4 qubits).
    Jordan-Wigner encoding.
    """
    
    def __init__(self, bond_length=0.735):
        self.bond_length = bond_length
        # Coefficients for H2 at equilibrium (simplified)
        # H = c0*I + c1*Z0 + c2*Z1 + c3*Z0Z1 + c4*X0X1 + c5*Y0Y1
        self.coeffs = {
            'I': -1.0523,
            'Z0': 0.3979,
            'Z1': -0.3979,
            'Z0Z1': -0.0112,
            'X0X1': 0.1809,
            'Y0Y1': 0.1809
        }
        self.n_qubits = 2
        self.exact_energy = -1.137  # Exact ground state energy
    
    def compute_expectation(self, sim: GPUStateVector, params, ansatz):
        """Compute <H> expectation value."""
        # Reset and apply ansatz
        sim.reset()
        ansatz.apply(sim, params)
        
        energy = self.coeffs['I']  # Identity term
        energy += self.coeffs['Z0'] * sim.expectation_Z(0)
        energy += self.coeffs['Z1'] * sim.expectation_Z(1)
        energy += self.coeffs['Z0Z1'] * sim.expectation_ZZ(0, 1)
        
        # X0X1 and Y0Y1 require basis rotation
        # <X0X1> = apply H to both qubits, then measure ZZ
        sim.reset()
        ansatz.apply(sim, params)
        sim.apply_single_gate(H_gate(), 0)
        sim.apply_single_gate(H_gate(), 1)
        xx_exp = sim.expectation_ZZ(0, 1)
        energy += self.coeffs['X0X1'] * xx_exp
        
        # <Y0Y1> = apply S†H to both qubits, then measure ZZ
        # S†H = (1/√2) [[1, -i], [1, i]]
        SdagH = cp.array([[1, -1j], [1, 1j]], dtype=cp.complex64) / np.sqrt(2)
        sim.reset()
        ansatz.apply(sim, params)
        sim.apply_single_gate(SdagH, 0)
        sim.apply_single_gate(SdagH, 1)
        yy_exp = sim.expectation_ZZ(0, 1)
        energy += self.coeffs['Y0Y1'] * yy_exp
        
        return float(energy)

class IsingHamiltonian:
    """
    1D Transverse Field Ising Model: H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ
    Scalable to any number of qubits.
    """
    
    def __init__(self, n_qubits, J=1.0, h=0.5):
        self.n_qubits = n_qubits
        self.J = J
        self.h = h
    
    def compute_expectation(self, sim: GPUStateVector, params, ansatz):
        """Compute <H> expectation value."""
        # Reset and apply ansatz
        sim.reset()
        ansatz.apply(sim, params)
        cp.cuda.Device().synchronize()
        
        energy = 0.0
        
        # ZZ terms
        for i in range(self.n_qubits - 1):
            energy -= self.J * sim.expectation_ZZ(i, i + 1)
        
        # X terms (requires basis change)
        for i in range(self.n_qubits):
            sim.reset()
            ansatz.apply(sim, params)
            sim.apply_single_gate(H_gate(), i)  # H transforms X basis to Z
            energy -= self.h * sim.expectation_Z(i)
        
        return float(energy)

# ============================================================
# VQE Optimizer
# ============================================================

class VQE:
    """Variational Quantum Eigensolver with GPU acceleration."""
    
    def __init__(self, hamiltonian, ansatz, n_qubits):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.sim = GPUStateVector(n_qubits)
        self.history = []
        self.iteration = 0
    
    def cost_function(self, params):
        """VQE cost function: expectation value of Hamiltonian."""
        energy = self.hamiltonian.compute_expectation(self.sim, params, self.ansatz)
        self.history.append(energy)
        self.iteration += 1
        
        if self.iteration % 10 == 0:
            print(f"  Iteration {self.iteration}: E = {energy:.6f}")
        
        return energy
    
    def optimize(self, init_params=None, method='COBYLA', maxiter=100):
        """Run VQE optimization."""
        if init_params is None:
            init_params = np.random.uniform(-np.pi, np.pi, self.ansatz.get_n_params())
        
        self.history = []
        self.iteration = 0
        
        print(f"Starting VQE optimization with {len(init_params)} parameters...")
        start_time = time.perf_counter()
        
        result = minimize(
            self.cost_function,
            init_params,
            method=method,
            options={'maxiter': maxiter, 'rhobeg': 0.5}
        )
        
        elapsed = time.perf_counter() - start_time
        
        return {
            'optimal_energy': result.fun,
            'optimal_params': result.x.tolist(),
            'n_iterations': self.iteration,
            'history': self.history,
            'time_seconds': elapsed,
            'success': result.success
        }

# ============================================================
# Main Benchmark
# ============================================================

def get_memory_info():
    """Get current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    try:
        mempool = cp.get_default_memory_pool()
        gpu_used = mempool.used_bytes() / 1024 / 1024
        gpu_total = mempool.total_bytes() / 1024 / 1024
    except:
        gpu_used, gpu_total = 0, 0
    
    return {
        'cpu_rss_mb': mem_info.rss / 1024 / 1024,
        'gpu_used_mb': gpu_used,
        'gpu_pool_mb': gpu_total
    }

def run_vqe_benchmark(n_qubits_list, n_layers=2, maxiter=50):
    """Run VQE benchmark across different qubit counts."""
    
    print("=" * 70)
    print("🔬 VQE Benchmark on Jetson IoT Edge")
    print("=" * 70)
    print(f"Testing {len(n_qubits_list)} qubit configurations")
    print(f"Ansatz layers: {n_layers}, Max iterations: {maxiter}")
    print("-" * 70)
    
    results = {}
    
    for n_qubits in n_qubits_list:
        print(f"\n[{n_qubits} qubits]")
        
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        try:
            # Use Ising Hamiltonian for scalability
            hamiltonian = IsingHamiltonian(n_qubits, J=1.0, h=0.5)
            ansatz = HardwareEfficientAnsatz(n_qubits, n_layers)
            
            print(f"  Parameters: {ansatz.get_n_params()}")
            print(f"  Hilbert dim: {2**n_qubits:,}")
            
            mem_before = get_memory_info()
            
            vqe = VQE(hamiltonian, ansatz, n_qubits)
            result = vqe.optimize(maxiter=maxiter)
            
            mem_after = get_memory_info()
            
            results[n_qubits] = {
                'n_params': ansatz.get_n_params(),
                'hilbert_dim': 2 ** n_qubits,
                'optimal_energy': result['optimal_energy'],
                'n_iterations': result['n_iterations'],
                'time_seconds': result['time_seconds'],
                'time_per_iter_ms': result['time_seconds'] * 1000 / result['n_iterations'],
                'memory_cpu_mb': mem_after['cpu_rss_mb'],
                'memory_gpu_mb': mem_after['gpu_used_mb'],
                'converged': result['success']
            }
            
            print(f"  ✓ Energy: {result['optimal_energy']:.6f}")
            print(f"  ✓ Time: {result['time_seconds']:.2f}s ({result['time_seconds']*1000/result['n_iterations']:.1f} ms/iter)")
            print(f"  ✓ Memory: CPU={mem_after['cpu_rss_mb']:.1f}MB, GPU={mem_after['gpu_used_mb']:.1f}MB")
            
        except cp.cuda.memory.OutOfMemoryError:
            print(f"  ✗ OUT OF GPU MEMORY")
            results[n_qubits] = {'error': 'GPU OOM'}
            break
        except MemoryError:
            print(f"  ✗ OUT OF CPU MEMORY")
            results[n_qubits] = {'error': 'CPU OOM'}
            break
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[n_qubits] = {'error': str(e)}
    
    return results

def main():
    print("\n🚀 EdgeQuantum IoT - VQE Variational Algorithm")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Test H2 molecule first (2 qubits, well-known result)
    print("=" * 70)
    print("📊 Test 1: H2 Molecule Ground State (2 qubits)")
    print("=" * 70)
    
    h2_ham = H2_Hamiltonian()
    ansatz = UCC_SD_Ansatz(2)
    vqe = VQE(h2_ham, ansatz, 2)
    h2_result = vqe.optimize(maxiter=100)
    
    print(f"\n✓ H2 Ground State Energy: {h2_result['optimal_energy']:.6f} Ha")
    print(f"  Exact value: {h2_ham.exact_energy:.6f} Ha")
    print(f"  Error: {abs(h2_result['optimal_energy'] - h2_ham.exact_energy):.6f} Ha")
    
    # Scale up with Ising model
    print("\n")
    qubit_configs = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    
    ising_results = run_vqe_benchmark(qubit_configs, n_layers=2, maxiter=30)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VQE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':<8} {'Params':<8} {'Dim':<12} {'Energy':<12} {'Time(s)':<10} {'ms/iter':<10}")
    print("-" * 70)
    
    for qubits in sorted(ising_results.keys()):
        data = ising_results[qubits]
        if 'error' not in data:
            print(f"{qubits:<8} {data['n_params']:<8} {data['hilbert_dim']:<12,} "
                  f"{data['optimal_energy']:<12.4f} {data['time_seconds']:<10.2f} "
                  f"{data['time_per_iter_ms']:<10.1f}")
        else:
            print(f"{qubits:<8} {'ERROR: ' + data['error']}")
    
    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': 'Jetson Orin Nano',
        'algorithm': 'VQE',
        'h2_result': {
            'optimal_energy': h2_result['optimal_energy'],
            'exact_energy': h2_ham.exact_energy,
            'error': abs(h2_result['optimal_energy'] - h2_ham.exact_energy)
        },
        'ising_benchmark': ising_results
    }
    
    results_path = '/home/jetson/skim/edgeQuantum-iotj/vqe_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ VQE Benchmark completed!")

if __name__ == "__main__":
    main()
