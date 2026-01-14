import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
data_path = 'data/cuquantum_benchmark.json'
with open(data_path, 'r') as f:
    data = json.load(f)

# Extract metrics
qubits = [d['qubits'] for d in data]
raw_gb = [d['raw_gb'] for d in data]
storage_gb = [d['storage_gb'] for d in data]
times = [d['total_time_s'] for d in data]

# Setup style
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 1. Storage Efficiency Plot
fig, ax1 = plt.subplots(figsize=(8, 5))

x = np.arange(len(qubits))
width = 0.35

rects1 = ax1.bar(x - width/2, raw_gb, width, label='Raw Memory (Required)', color='#e74c3c', alpha=0.8)
rects2 = ax1.bar(x + width/2, storage_gb, width, label='Compressed Storage (Used)', color='#2ecc71', alpha=0.9)

ax1.set_ylabel('Memory / Storage (GB)')
ax1.set_xlabel('Qubits')
ax1.set_title('Storage Efficiency: Raw vs Compressed State')
ax1.set_xticks(x)
ax1.set_xticklabels(qubits)
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Annotate compression ratio
for i, (r, s) in enumerate(zip(raw_gb, storage_gb)):
    ratio = r / s
    ax1.text(i + width/2, s * 1.5, f'{int(ratio)}x', ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/fig_storage_efficiency.pdf')
plt.close()

# 2. Runtime Scaling Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(qubits, times, marker='o', linestyle='-', linewidth=2, color='#3498db', label='Native cuQuantum')

ax.set_xlabel('Qubits')
ax.set_ylabel('Total Simulation Time (s)')
ax.set_title('Runtime Scaling (Init + Gate)')
ax.set_yscale('log')
ax.grid(True, which="both", ls="-", alpha=0.3)

# Annotate values
for q, t in zip(qubits, times):
    time_str = f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m"
    ax.annotate(time_str, (q, t), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('paper/figures/fig_runtime_scaling.pdf')
plt.close()

print("Figures generated in paper/figures/")
