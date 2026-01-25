import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

LOG_FILE = "grand_results_full.json"
OUTPUT_DIR = "."

def plot_results():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        return

    with open(LOG_FILE, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    # Filter for successes to plot times
    df_success = df[df.get("time").notna()]
    
    # Plot 1: Execution Time by Qubits & Simulator (Grouped Bar)
    # Pivot: Index=Circuit+Qubits, Columns=Simulator, Values=Time
    
    # Separate plots for each Qubit count? Or combined?
    # Let's do a facet grid or just separate plots per Qubit count for clarity.
    
    qubits = sorted(df['qubits'].unique())
    
    for q in qubits:
        df_q = df[df['qubits'] == q]
        
        plt.figure(figsize=(12, 6))
        
        circuits = sorted(df_q['circuit'].unique())
        simulators = sorted(df_q['simulator'].unique())
        
        # Position of bars
        bar_width = 0.15
        r = np.arange(len(circuits))
        
        for i, sim in enumerate(simulators):
            subset = df_q[df_q['simulator'] == sim]
            # Align subset with circuits
            times = []
            for c in circuits:
                row = subset[subset['circuit'] == c]
                if not row.empty and "time" in row.iloc[0] and not pd.isna(row.iloc[0]["time"]):
                    times.append(row.iloc[0]["time"])
                else:
                    times.append(0) # 0 for failure/missing
            
            # If all 0, check if error exists to label
            
            pos = [x + i * bar_width for x in r]
            plt.bar(pos, times, width=bar_width, edgecolor='white', label=sim)
            
            # Annotate OOM
            for j, c in enumerate(circuits):
                row = subset[subset['circuit'] == c]
                if not row.empty and "error" in row.iloc[0] and pd.notna(row.iloc[0]["error"]):
                    # error_msg = row.iloc[0]["error"]
                    plt.text(pos[j], 1, "X", ha='center', va='bottom', color='red', fontweight='bold')

        plt.xlabel('Circuit', fontweight='bold')
        plt.xticks([r + bar_width * (len(simulators)-1)/2 for r in range(len(circuits))], circuits)
        plt.ylabel('Execution Time (s)')
        plt.title(f'Benchmark Results: {q} Qubits')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"benchmark_{q}q.png")
        print(f"Saved benchmark_{q}q.png")

if __name__ == "__main__":
    plot_results()
