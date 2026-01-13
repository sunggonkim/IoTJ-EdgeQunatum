#!/usr/bin/env python3
"""
QAOA (Quantum Approximate Optimization Algorithm) on Jetson IoT Edge
Implements QAOA for MaxCut problem on random graphs.
Scalable from 4 to 20+ qubits on edge GPU using CuPy.
"""

import time
import numpy as np
import cupy as cp
from scipy.optimize import minimize
import json
from datetime import datetime
import psutil
import gc
import random

# ============================================================
# Quantum Gates
# ============================================================

def H_gate():
    return cp.array([[1, 1], [1, -1]], dtype=cp.complex64) / np.sqrt(2)

def RX_gate(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return cp.array([[c, -1j*s], [-1j*s, c]], dtype=cp.complex64)

def RZ_gate(theta):
    return cp.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=cp.complex64)

def RZZ_diag(theta, n_qubits, i, j):
    """
    Create diagonal of exp(-i * theta/2 * Z_i Z_j).
    Returns diagonal elements for efficient application.
    """
    dim = 2 ** n_qubits
    diag = cp.ones(dim, dtype=cp.complex64)
    
    for k in range(dim):
        # Get bit i and bit j
        bi = (k >> (n_qubits - 1 - i)) & 1
        bj = (k >> (n_qubits - 1 - j)) & 1
        # ZZ eigenvalue: +1 if same, -1 if different
        zz = 1 if bi == bj else -1
        diag[k] = cp.exp(-1j * theta / 2 * zz)
    
    return diag

# ============================================================
# State Vector Simulator (reused from VQE)
# ============================================================

class GPUStateVector:
    """GPU-accelerated state vector simulator."""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.reset()
    
    def reset(self):
        self.sv = cp.zeros(self.dim, dtype=cp.complex64)
        self.sv[0] = 1.0 + 0.0j
    
    def init_superposition(self):
        """Initialize to |+⟩^n state (equal superposition)"""
        self.sv = cp.ones(self.dim, dtype=cp.complex64) / np.sqrt(self.dim)
    
    def apply_single_gate(self, gate, target):
        shape = [2] * self.n_qubits
        sv_reshaped = self.sv.reshape(shape)
        
        axes = list(range(self.n_qubits))
        axes.remove(target)
        axes.insert(0, target)
        sv_t = cp.transpose(sv_reshaped, axes)
        
        sv_flat = sv_t.reshape(2, -1)
        sv_flat = cp.matmul(gate, sv_flat)
        
        sv_t = sv_flat.reshape([2] + [2] * (self.n_qubits - 1))
        inv_axes = [0] * self.n_qubits
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        sv_reshaped = cp.transpose(sv_t, inv_axes)
        self.sv = sv_reshaped.flatten()
    
    def apply_diagonal(self, diag):
        """Apply diagonal operator efficiently."""
        self.sv = self.sv * diag
    
    def apply_rzz(self, theta, i, j):
        """Apply RZZ(theta) = exp(-i * theta/2 * Z_i Z_j)"""
        diag = RZZ_diag(theta, self.n_qubits, i, j)
        self.apply_diagonal(diag)
    
    def get_probabilities(self):
        return cp.abs(self.sv) ** 2
    
    def sample(self, n_shots=1000):
        """Sample measurement outcomes."""
        probs = cp.asnumpy(self.get_probabilities())
        samples = np.random.choice(self.dim, size=n_shots, p=probs)
        return samples

# ============================================================
# MaxCut Problem
# ============================================================

class MaxCutGraph:
    """Random graph for MaxCut problem."""
    
    def __init__(self, n_nodes, edge_prob=0.5, seed=42):
        self.n_nodes = n_nodes
        self.edges = []
        
        random.seed(seed)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < edge_prob:
                    self.edges.append((i, j))
        
        # Ensure at least some edges
        if len(self.edges) == 0:
            self.edges = [(0, 1)]
    
    def compute_cut(self, bitstring):
        """Compute cut value for a given bitstring."""
        cut = 0
        for i, j in self.edges:
            bi = (bitstring >> (self.n_nodes - 1 - i)) & 1
            bj = (bitstring >> (self.n_nodes - 1 - j)) & 1
            if bi != bj:
                cut += 1
        return cut
    
    def max_cut_brute_force(self):
        """Find exact MaxCut by brute force (for small graphs)."""
        if self.n_nodes > 20:
            return None, None
        
        best_cut = 0
        best_config = 0
        for config in range(2 ** self.n_nodes):
            cut = self.compute_cut(config)
            if cut > best_cut:
                best_cut = cut
                best_config = config
        
        return best_cut, best_config

# ============================================================
# QAOA Circuit
# ============================================================

class QAOA:
    """QAOA for MaxCut problem."""
    
    def __init__(self, graph: MaxCutGraph, p_layers: int):
        self.graph = graph
        self.n_qubits = graph.n_nodes
        self.p = p_layers
        self.n_params = 2 * p_layers  # gamma and beta for each layer
        self.sim = GPUStateVector(self.n_qubits)
        self.history = []
        self.iteration = 0
    
    def apply_cost_layer(self, gamma):
        """Apply cost Hamiltonian evolution: exp(-i * gamma * C)
        C = Σ_{(i,j) ∈ E} (1 - Z_i Z_j) / 2
        """
        for i, j in self.graph.edges:
            # exp(-i * gamma * (1 - ZiZj)/2) = exp(-i*gamma/2) * exp(i*gamma/2 * ZiZj)
            # We can ignore global phase, so just apply RZZ
            self.sim.apply_rzz(-gamma, i, j)
    
    def apply_mixer_layer(self, beta):
        """Apply mixer Hamiltonian evolution: exp(-i * beta * B)
        B = Σ_i X_i
        """
        for q in range(self.n_qubits):
            self.sim.apply_single_gate(RX_gate(2 * beta), q)
    
    def run_circuit(self, params):
        """Run QAOA circuit with given parameters."""
        gammas = params[:self.p]
        betas = params[self.p:]
        
        # Initialize to |+⟩^n
        self.sim.init_superposition()
        
        # Apply p layers
        for layer in range(self.p):
            self.apply_cost_layer(gammas[layer])
            self.apply_mixer_layer(betas[layer])
    
    def compute_cost(self, params, n_shots=500):
        """Compute expected cost (negative of cut value for minimization)."""
        self.run_circuit(params)
        cp.cuda.Device().synchronize()
        
        # Sample outcomes
        samples = self.sim.sample(n_shots)
        
        # Compute average cut value
        total_cut = 0
        for sample in samples:
            total_cut += self.graph.compute_cut(sample)
        
        avg_cut = total_cut / n_shots
        
        self.history.append(-avg_cut)  # Store negative for minimization
        self.iteration += 1
        
        if self.iteration % 10 == 0:
            print(f"  Iteration {self.iteration}: avg cut = {avg_cut:.2f}")
        
        return -avg_cut  # Negative because we want to maximize cut
    
    def optimize(self, init_params=None, method='COBYLA', maxiter=100, n_shots=500):
        """Run QAOA optimization."""
        if init_params is None:
            init_params = np.random.uniform(0, np.pi, self.n_params)
        
        self.history = []
        self.iteration = 0
        
        print(f"Starting QAOA optimization (p={self.p}, {self.n_params} params)...")
        start_time = time.perf_counter()
        
        result = minimize(
            lambda p: self.compute_cost(p, n_shots),
            init_params,
            method=method,
            options={'maxiter': maxiter, 'rhobeg': 0.5}
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Get best cut from final circuit
        self.run_circuit(result.x)
        samples = self.sim.sample(1000)
        cuts = [self.graph.compute_cut(s) for s in samples]
        best_sampled_cut = max(cuts)
        
        return {
            'best_cut': best_sampled_cut,
            'avg_cut': -result.fun,
            'optimal_params': result.x.tolist(),
            'n_iterations': self.iteration,
            'time_seconds': elapsed,
            'success': result.success
        }

# ============================================================
# QAOA Benchmark
# ============================================================

def get_memory_info():
    process = psutil.Process()
    mem_info = process.memory_info()
    try:
        mempool = cp.get_default_memory_pool()
        gpu_used = mempool.used_bytes() / 1024 / 1024
    except:
        gpu_used = 0
    return {
        'cpu_rss_mb': mem_info.rss / 1024 / 1024,
        'gpu_used_mb': gpu_used
    }

def run_qaoa_benchmark(n_qubits_list, p_layers=1, maxiter=30):
    """Run QAOA benchmark across different qubit counts."""
    
    print("=" * 70)
    print("🔬 QAOA MaxCut Benchmark on Jetson IoT Edge")
    print("=" * 70)
    print(f"Testing {len(n_qubits_list)} qubit configurations")
    print(f"QAOA layers (p): {p_layers}, Max iterations: {maxiter}")
    print("-" * 70)
    
    results = {}
    
    for n_qubits in n_qubits_list:
        print(f"\n[{n_qubits} qubits/nodes]")
        
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        try:
            # Create random graph
            graph = MaxCutGraph(n_qubits, edge_prob=0.4, seed=42)
            print(f"  Graph edges: {len(graph.edges)}")
            
            # Get exact solution for small graphs
            if n_qubits <= 18:
                exact_cut, exact_config = graph.max_cut_brute_force()
                print(f"  Exact MaxCut: {exact_cut}")
            else:
                exact_cut = None
            
            mem_before = get_memory_info()
            
            qaoa = QAOA(graph, p_layers)
            result = qaoa.optimize(maxiter=maxiter, n_shots=500)
            
            mem_after = get_memory_info()
            
            approx_ratio = result['best_cut'] / exact_cut if exact_cut else None
            
            results[n_qubits] = {
                'n_edges': len(graph.edges),
                'hilbert_dim': 2 ** n_qubits,
                'best_cut': result['best_cut'],
                'avg_cut': result['avg_cut'],
                'exact_cut': exact_cut,
                'approx_ratio': approx_ratio,
                'n_iterations': result['n_iterations'],
                'time_seconds': result['time_seconds'],
                'memory_cpu_mb': mem_after['cpu_rss_mb'],
                'memory_gpu_mb': mem_after['gpu_used_mb']
            }
            
            print(f"  ✓ Best Cut: {result['best_cut']} (exact: {exact_cut})")
            if approx_ratio:
                print(f"  ✓ Approximation Ratio: {approx_ratio:.2%}")
            print(f"  ✓ Time: {result['time_seconds']:.2f}s")
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
            import traceback
            traceback.print_exc()
            results[n_qubits] = {'error': str(e)}
    
    return results

def main():
    print("\n🚀 EdgeQuantum IoT - QAOA Variational Algorithm")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Scale up MaxCut problem
    qubit_configs = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    
    # Test with p=1 (single layer) first for speed
    results_p1 = run_qaoa_benchmark(qubit_configs, p_layers=1, maxiter=30)
    
    # Test with p=2 for comparison on smaller instances
    print("\n\n" + "=" * 70)
    print("Testing p=2 layers on smaller instances...")
    results_p2 = run_qaoa_benchmark([4, 6, 8, 10], p_layers=2, maxiter=30)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 QAOA BENCHMARK SUMMARY (p=1)")
    print("=" * 70)
    print(f"{'Qubits':<8} {'Edges':<8} {'BestCut':<10} {'Exact':<8} {'Ratio':<10} {'Time(s)':<10}")
    print("-" * 70)
    
    for qubits in sorted(results_p1.keys()):
        data = results_p1[qubits]
        if 'error' not in data:
            ratio_str = f"{data['approx_ratio']:.2%}" if data['approx_ratio'] else "N/A"
            exact_str = str(data['exact_cut']) if data['exact_cut'] else "N/A"
            print(f"{qubits:<8} {data['n_edges']:<8} {data['best_cut']:<10} "
                  f"{exact_str:<8} {ratio_str:<10} {data['time_seconds']:<10.2f}")
        else:
            print(f"{qubits:<8} {'ERROR: ' + data['error']}")
    
    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': 'Jetson Orin Nano',
        'algorithm': 'QAOA',
        'qaoa_p1': results_p1,
        'qaoa_p2': results_p2
    }
    
    results_path = '/home/jetson/skim/edgeQuantum-iotj/qaoa_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ QAOA Benchmark completed!")

if __name__ == "__main__":
    main()
