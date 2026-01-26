import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Prefer the grand results file if present, fall back to comprehensive results
grand_path = os.path.join(script_dir, "grand_results_full.json")
comp_path = os.path.join(script_dir, "comprehensive_results.json")
if os.path.exists(grand_path):
    LOG_FILE = grand_path
elif os.path.exists(comp_path):
    LOG_FILE = comp_path
else:
    LOG_FILE = os.path.join(script_dir, "grand_results_full.json")
OUTPUT_DIR = "."

# Plot styling
FONT_TITLE = 18
FONT_LABEL = 14
FONT_TICKS = 12
FONT_LEGEND = 12
FIGSIZE = (12, 6)

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': FONT_TICKS,
    'axes.titlesize': FONT_TITLE,
    'axes.labelsize': FONT_LABEL,
    'xtick.labelsize': FONT_TICKS,
    'ytick.labelsize': FONT_TICKS,
    'legend.fontsize': FONT_LEGEND,
})

def plot_results():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        return

    with open(LOG_FILE, 'r') as f:
        data = json.load(f)

    # Support JSON with top-level key 'runs' or a flat list
    if isinstance(data, dict) and 'runs' in data:
        runs = data['runs']
    else:
        runs = data

    # Normalize records: map scheme->simulator, sim_time/wall_time -> time
    norm = []
    for r in runs:
        rec = {}
        rec['simulator'] = r.get('scheme') or r.get('simulator')
        rec['circuit'] = r.get('circuit')
        rec['qubits'] = r.get('qubits')
        # prefer wall_time if present, else sim_time, else time
        rec['time'] = r.get('wall_time') if r.get('wall_time') is not None else r.get('sim_time') if r.get('sim_time') is not None else r.get('time')
        rec['error'] = None if r.get('success', True) else r.get('error', 'failed')
        norm.append(rec)

    df = pd.DataFrame(norm)
    df_success = df[df['time'].notna()]
    
    # Plot 1: Execution Time by Qubits & Simulator (Grouped Bar)
    # Pivot: Index=Circuit+Qubits, Columns=Simulator, Values=Time
    
    # Separate plots for each Qubit count? Or combined?
    # Let's do a facet grid or just separate plots per Qubit count for clarity.
    
    qubits = sorted(df['qubits'].unique())
    
    for q in qubits:
        df_q = df[df['qubits'] == q]
        
        plt.figure(figsize=FIGSIZE)
        
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
            bars = plt.bar(pos, times, width=bar_width, edgecolor='white', label=sim)

            # Annotate numeric values above bars (actual values only)
            for j, bar in enumerate(bars):
                h = bar.get_height()
                if h > 0:
                    # Format: show ms if <1s, else seconds with 2 decimals
                    if h < 1.0:
                        txt = f"{h*1000:.0f} ms"
                    else:
                        txt = f"{h:.2f} s"
                    plt.text(bar.get_x() + bar.get_width()/2, h + max(0.005, 0.01*max(1, h)), txt,
                             ha='center', va='bottom', fontsize=FONT_TICKS, rotation=0)

            # Annotate errors (OOM) with small red marker
            for j, c in enumerate(circuits):
                row = subset[subset['circuit'] == c]
                if not row.empty and "error" in row.iloc[0] and pd.notna(row.iloc[0]["error"]):
                    plt.text(pos[j], 0.02 * max(1, max(times) if len(times)>0 else 1), "OOM",
                             ha='center', va='bottom', color='red', fontweight='bold', fontsize=FONT_TICKS)

        plt.xlabel('Circuit', fontweight='bold')
        plt.xticks([r + bar_width * (len(simulators)-1)/2 for r in range(len(circuits))], circuits)
        plt.ylabel('Execution Time (s)')
        plt.title(f'Benchmark Results: {q} Qubits')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        out_png = f"benchmark_{q}q.png"
        out_pdf = f"benchmark_{q}q.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf, dpi=200)
        # Also copy to paper/figures if available
        paper_fig_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
        try:
            paper_fig_dir = os.path.normpath(paper_fig_dir)
            if os.path.exists(paper_fig_dir):
                dst_png = os.path.join(paper_fig_dir, out_png)
                dst_pdf = os.path.join(paper_fig_dir, out_pdf)
                plt.savefig(dst_png, dpi=200)
                plt.savefig(dst_pdf, dpi=200)
        except Exception:
            pass

        print(f"Saved {out_png} and {out_pdf}")

if __name__ == "__main__":
    plot_results()
