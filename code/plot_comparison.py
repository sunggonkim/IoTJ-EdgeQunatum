#!/usr/bin/env python3
"""Generate comparison figure for EdgeQuantum paper"""
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark
qubits = [20, 22, 24, 26]
cuquantum_native = [0.10, 0.04, 0.03, 0.04]
cuquantum_uvm = [0.09, 0.09, 0.09, 0.07]
bmqsim = [8.56, 25.22, 62.66, 223.92]
edgequantum = [1.02, 2.15, 4.77, 14.81]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Execution time comparison (log scale)
ax1.semilogy(qubits, bmqsim, 'o-', label='BMQSim-like', linewidth=2, markersize=8, color='#e74c3c')
ax1.semilogy(qubits, edgequantum, 's-', label='EdgeQuantum (Ours)', linewidth=2, markersize=8, color='#2ecc71')
ax1.semilogy(qubits, cuquantum_native, '^--', label='cuQuantum (Native)', linewidth=2, markersize=8, color='#3498db')
ax1.semilogy(qubits, cuquantum_uvm, 'v--', label='cuQuantum (UVM)', linewidth=2, markersize=8, color='#9b59b6')

ax1.set_xlabel('Number of Qubits', fontsize=12)
ax1.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
ax1.set_title('Random-10 Circuit Performance', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(qubits)

# Right: Speedup over BMQSim
speedups = [b/e for b, e in zip(bmqsim, edgequantum)]
bars = ax2.bar(qubits, speedups, color='#2ecc71', edgecolor='black', linewidth=1.5)
ax2.axhline(y=1, color='#e74c3c', linestyle='--', linewidth=2, label='BMQSim baseline')

# Add value labels on bars
for bar, speedup in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{speedup:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Number of Qubits', fontsize=12)
ax2.set_ylabel('Speedup over BMQSim-like', fontsize=12)
ax2.set_title('EdgeQuantum Speedup', fontsize=13, fontweight='bold')
ax2.set_xticks(qubits)
ax2.set_ylim(0, max(speedups) * 1.2)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../paper/figures/fig_baseline_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../paper/figures/fig_baseline_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: figures/fig_baseline_comparison.pdf")
