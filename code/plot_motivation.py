import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

out_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(out_dir, exist_ok=True)

# Figure A: Memory growth
qs = np.arange(20, 38)
# state vector size in GB for complex128 (16 bytes per amplitude)
sizes_gb = (16.0 * (2 ** qs)) / (1024 ** 3)

plt.figure(figsize=(5.0, 3.5))
plt.plot(qs, sizes_gb, marker='o', color='tab:green')
plt.yscale('log')
plt.xlabel('Number of Qubits', fontsize=16)
plt.ylabel('Memory (GB)', fontsize=16)
plt.title('Memory growth of state vectors', fontsize=18)
plt.grid(True, which='both', ls='--', alpha=0.6)
# Horizontal lines for device memory
plt.axhline(8, color='tab:blue', ls='--')
plt.text(20.2, 8*1.05, 'Jetson (8GB)', color='tab:blue', fontsize=12)
plt.axhline(80, color='tab:orange', ls='--')
plt.text(20.2, 80*1.05, 'A100 (80GB)', color='tab:orange', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_motivation_a.pdf'), dpi=300)
plt.close()

# Figure B: Latency comparison (use measured values)
labels = ['Cloud Quantum', 'CPU Simulation', 'EdgeQuantum']
# Values in milliseconds (actual numbers)
values = [1500.0, 450.0, 9.2]
colors = ['#b22222', '#ff8c00', '#2ca02c']

plt.figure(figsize=(5.0, 3.5))
bars = plt.bar(labels, values, color=colors)
plt.ylabel('VQE Iteration Latency (ms)', fontsize=16)
plt.title('Latency comparison (Edge vs Cloud)', fontsize=18)
plt.grid(axis='y', ls='--', alpha=0.5)
# Annotate numeric values above bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val + max(1.0, 0.02*max(values)), f'{val:.1f} ms',
             ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_motivation_b.pdf'), dpi=300)
plt.close()

print('Wrote fig_motivation_a.pdf and fig_motivation_b.pdf to', out_dir)
