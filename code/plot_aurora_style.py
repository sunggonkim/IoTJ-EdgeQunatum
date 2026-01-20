import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Create figures directory
os.makedirs('paper/figures', exist_ok=True)

# Load Data
try:
    with open('data/benchmark_results_recovered.json', 'r') as f:
        data = json.load(f)
        results = data['results']
except FileNotFoundError:
    print("Error: data/benchmark_results_recovered.json not found.")
    exit(1)

# Convert to DataFrame for easier handling
df = pd.DataFrame(results)

# Filter out failures (for plotting lines)
df_success = df[df['success'] == True].copy()
df_success['qubits'] = df_success['qubits'].astype(int)
df_success['time'] = df_success['time'].astype(float)

# Simulators & Colors
# EdgeQuantum (Our Work) -> Green
# cuQuantum Native -> Red
# cuQuantum UVM -> Orange
# BMQSim -> Blue
# Cirq -> Purple
# PennyLane -> Cyan
colors = {
    "EdgeQuantum (Ours)": "#2E7D32",   # Dark Green
    "cuQuantum (Native)": "#C62828",   # Red
    "cuQuantum (UVM)": "#EF6C00",      # Orange
    "BMQSim-like (Swap)": "#1565C0",   # Blue
    "Google Cirq": "#6A1B9A",          # Purple
    "PennyLane (Ltn)": "#00838F"       # Cyan
}

markers = {
    "EdgeQuantum (Ours)": "s",
    "cuQuantum (Native)": "^",
    "cuQuantum (UVM)": "v",
    "BMQSim-like (Swap)": "D",
    "Google Cirq": "o",
    "PennyLane (Ltn)": "x"
}

linestyles = {
    "EdgeQuantum (Ours)": "-",
    "cuQuantum (Native)": "--",
    "cuQuantum (UVM)": "--",
    "BMQSim-like (Swap)": "-.",
    "Google Cirq": ":",
    "PennyLane (Ltn)": ":"
}

# Circuits order
circuits = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
titles = ["(a) QV", "(b) VQC", "(c) QSVM", "(d) Random", "(e) GHZ", "(f) VQE"]

# Plotting
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
    'font.family': 'serif' # Matches LaTeX style often
})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, circuit in enumerate(circuits):
    ax = axes[i]
    
    # Get data for this circuit
    circuit_df = df_success[df_success['circuit'] == circuit]
    
    # Max qubits reached by baseline (non-EdgeQuantum)
    baseline_df = circuit_df[circuit_df['simulator'] != "EdgeQuantum (Ours)"]
    max_baseline_q = baseline_df['qubits'].max() if not baseline_df.empty else 0
    
    # Max qubits reached by EdgeQuantum
    eq_df = circuit_df[circuit_df['simulator'] == "EdgeQuantum (Ours)"]
    max_eq_q = eq_df['qubits'].max() if not eq_df.empty else 0
    
    # Plot lines
    for sim_name in colors.keys():
        sim_data = circuit_df[circuit_df['simulator'] == sim_name].sort_values('qubits')
        if not sim_data.empty:
            ax.semilogy(sim_data['qubits'], sim_data['time'], 
                       label=sim_name, 
                       color=colors[sim_name],
                       marker=markers[sim_name],
                       linestyle=linestyles[sim_name],
                       markersize=6)
    
    # Shade Region "Only executable (ORION)" if Applicable
    # Using 'ORION' or 'EdgeQuantum' as the internal name? Paper says EdgeQuantum/CITADEL? 
    # Let's stick to "EdgeQuantum" label in plot for now or "Proposed"
    
    if max_eq_q > max_baseline_q and max_baseline_q > 0:
        ax.axvspan(max_baseline_q + 0.5, max_eq_q + 0.5, color='#2E7D32', alpha=0.1)
        # Add Text if enough space
        if max_eq_q - max_baseline_q >= 2:
            ax.text((max_baseline_q + max_eq_q)/2 + 0.5, ax.get_ylim()[1]*0.5, 
                   "Only Executable\n(EdgeQuantum)", 
                   ha='center', va='top', color='#2E7D32', fontweight='bold', fontsize=10)
            ax.axvline(x=max_baseline_q + 0.5, color='gray', linestyle='--', alpha=0.5)

    # Styling
    ax.set_title(titles[i], fontsize=14, fontweight='bold', y=-0.2) # Title at bottom like LaTeX subcaption? Or top? 
    # User image has titles at BOTTOM: "(a) qv.", "(b) vqc." 
    # But usually matplotlib titles are top. Let's put them at bottom using text or set_title with y position.
    # Actually standard is top, user image is a screenshot of specific paper style.
    # Let's put title at TOP for readability, but use the (a) convention.
    ax.set_title(titles[i], y=-0.25, fontsize=14) 
    
    ax.set_xlabel('# of Qubits', fontsize=12, fontweight='bold')
    if i % 3 == 0:
        ax.set_ylabel('Simulation Time (s)', fontsize=12, fontweight='bold')
    
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if i == 0: # Legend in first plot or separate?
        # Put legend outside or in the "empty" space of first plot if possible
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15) # Make space for bottom titles
plt.savefig('paper/figures/fig_aurora_style.pdf')
plt.savefig('paper/figures/fig_aurora_style.png') # For preview if needed
print("Generated paper/figures/fig_aurora_style.pdf and .png")
