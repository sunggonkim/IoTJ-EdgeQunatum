import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'comprehensive_results.json')
out_dir = os.path.normpath(os.path.join(script_dir, '..', 'paper', 'figures'))
os.makedirs(out_dir, exist_ok=True)

with open(data_path) as f:
    j = json.load(f)
runs = j.get('runs', j if isinstance(j, list) else [])

df = pd.DataFrame(runs)
# Normalize columns
for col in ['wall_time','sim_time','time']:
    if col in df.columns:
        df['time_val'] = df[col]
        break
if 'time_val' not in df.columns:
    # try other names
    df['time_val'] = df.get('wall_time')

df['simulator'] = df.get('scheme')
if df['simulator'].isna().any() and 'simulator' in df.columns:
    df['simulator'] = df['simulator'].fillna(df['simulator'])
if df['simulator'].isna().all() and 'simulator' in df.columns:
    df['simulator'] = df['simulator']

# Aggregate: mean and median time per simulator per qubits
agg = df.groupby(['qubits','simulator'])['time_val'].agg(['mean','median','count']).reset_index()

# Line plot: mean time vs qubits per simulator
plt.figure(figsize=(8,4.5))
simulators = sorted(agg['simulator'].unique())
for sim in simulators:
    a = agg[agg['simulator']==sim].sort_values('qubits')
    plt.plot(a['qubits'], a['mean'], marker='o', label=sim)
plt.xlabel('Qubits', fontsize=14)
plt.ylabel('Mean wall time (s)', fontsize=14)
plt.title('Mean wall time vs Qubits (per simulator)', fontsize=16)
plt.grid(True, ls='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'summary_mean_time_vs_qubits.pdf'), dpi=200)
plt.close()

# Boxplot: distribution per simulator (all qubits)
plt.figure(figsize=(8,4))
data = [df[df['simulator']==s]['time_val'].dropna() for s in simulators]
plt.boxplot(data, labels=simulators, showfliers=False)
plt.yscale('symlog')
plt.ylabel('Wall time (s) (symlog)', fontsize=14)
plt.title('Wall time distribution per simulator (all qubits)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'summary_boxplot_simulators.pdf'), dpi=200)
plt.close()

print('Wrote summary plots to', out_dir)
