#!/usr/bin/env python3
"""
EdgeQuantum Paper Figures with UNIFIED COLOR SCHEME
Using REAL cuQuantum benchmark data from Jetson Orin Nano (8GB).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)

# ============================================================
# UNIFIED COLOR SCHEME (Consistent across all figures)
# ============================================================
COLORS = {
    'primary': '#2E7D32',      # Green (EdgeQuantum brand)
    'secondary': '#1565C0',    # Blue
    'accent': '#E65100',       # Orange
    'highlight': '#C62828',    # Red (for limits/warnings)
    'neutral': '#37474F',      # Dark gray
    'light': '#90A4AE',        # Light gray
    'success': '#43A047',      # Light green
    'bar1': '#66BB6A',         # Bar chart green
    'bar2': '#42A5F5',         # Bar chart blue
    'bar3': '#FFA726',         # Bar chart orange
}

# Professional style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# REAL CUQUANTUM BENCHMARK DATA
# ============================================================
REAL_DATA = {
    28: {'init': 0.11, 'gate': 6.85, 'total': 6.96, 'storage_gb': 0.008},
    30: {'init': 0.13, 'gate': 25.68, 'total': 25.81, 'storage_gb': 0.033},
    32: {'init': 0.42, 'gate': 102.99, 'total': 103.41, 'storage_gb': 0.132},
    33: {'init': 0.79, 'gate': 200.65, 'total': 201.44, 'storage_gb': 0.264},
    34: {'init': 2.25, 'gate': 384.82, 'total': 387.07, 'storage_gb': 0.527},
    35: {'init': 45.25, 'gate': 761.16, 'total': 806.41, 'storage_gb': 1.055},
    36: {'init': 147.32, 'gate': 5693.03, 'total': 5840.35, 'storage_gb': 2.110},
    37: {'init': 454.53, 'gate': 11576.87, 'total': 12031.40, 'storage_gb': 4.219},  # NEW RECORD!
}

def raw_size_gb(qubits):
    return (2**qubits * 8) / (1024**3)

# ============================================================
# FIGURE 1a: Memory Wall
# ============================================================
def plot_motivation_a():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    qubits = list(range(20, 40, 2))
    mem_req = [raw_size_gb(q) for q in qubits]
    
    ax.semilogy(qubits, mem_req, color=COLORS['primary'], linewidth=2.5, 
                marker='o', markersize=6, label='State Vector Size')
    ax.axhline(y=8, color=COLORS['secondary'], linestyle='--', linewidth=2, 
               label='Jetson (8GB)')
    ax.axhline(y=80, color=COLORS['accent'], linestyle='--', linewidth=2, 
               label='A100 (80GB)')
    
    ax.axvspan(27, 38, alpha=0.15, color=COLORS['highlight'])
    ax.annotate('Memory Gap\n(Tiered Memory)', xy=(32, 20), fontsize=9, 
                ha='center', color=COLORS['highlight'], fontweight='bold')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Memory (GB)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(20, 38)
    ax.set_ylim(0.001, 1000)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_motivation_a.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_motivation_a.png', bbox_inches='tight')
    plt.close()
    print("✅ fig_motivation_a.pdf")

# ============================================================
# FIGURE 1b: Edge Advantage
# ============================================================
def plot_motivation_b():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    categories = ['Cloud\nQuantum', 'CPU\nSimulation', 'EdgeQuantum\n(Ours)']
    latencies = [1500, 450, 9.15]
    colors = [COLORS['highlight'], COLORS['accent'], COLORS['primary']]
    
    bars = ax.bar(categories, latencies, color=colors, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 40,
                f'{val:.0f}ms' if val > 10 else f'{val:.1f}ms', 
                ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('VQE Iteration Latency (ms)')
    ax.set_ylim(0, 1800)
    
    # Speedup annotation
    ax.annotate('164×\nfaster', xy=(2, 150), fontsize=10, ha='center', 
                color=COLORS['primary'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_motivation_b.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_motivation_b.png', bbox_inches='tight')
    plt.close()
    print("✅ fig_motivation_b.pdf")

# ============================================================
# FIGURE 2: Scaling Performance (MAIN RESULT)
# ============================================================
def plot_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    
    qubits = sorted(REAL_DATA.keys())
    total_times = [REAL_DATA[q]['total'] for q in qubits]
    init_times = [REAL_DATA[q]['init'] for q in qubits]
    gate_times = [REAL_DATA[q]['gate'] for q in qubits]
    storage = [REAL_DATA[q]['storage_gb'] for q in qubits]
    raw_sizes = [raw_size_gb(q) for q in qubits]
    
    # Left: Time scaling
    ax1.semilogy(qubits, total_times, color=COLORS['primary'], linewidth=2.5, 
                 marker='o', markersize=7, label='Total Time')
    ax1.semilogy(qubits, gate_times, color=COLORS['secondary'], linewidth=2, 
                 marker='^', markersize=5, alpha=0.8, label='Gate Exec')
    ax1.semilogy(qubits, init_times, color=COLORS['accent'], linewidth=2, 
                 marker='s', markersize=5, alpha=0.8, label='Init')
    
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('(a) Simulation Time')
    ax1.legend(loc='upper left')
    ax1.set_xticks(qubits)
    
    # Annotate 36Q
    ax1.annotate(f'36Q: {total_times[-1]/60:.0f}min', xy=(36, total_times[-1]),
                xytext=(34.5, total_times[-1]*2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, fontweight='bold', color=COLORS['primary'])
    
    # Right: Compression
    ax2.semilogy(qubits, raw_sizes, color=COLORS['highlight'], linewidth=2, 
                 linestyle='--', marker='x', markersize=6, label='Raw State')
    ax2.semilogy(qubits, storage, color=COLORS['primary'], linewidth=2.5, 
                 marker='o', markersize=7, label='Compressed (LZ4)')
    
    ax2.fill_between(qubits, storage, raw_sizes, alpha=0.15, color=COLORS['primary'])
    ax2.annotate('242.7× Compression', xy=(33, 0.8), fontsize=10, 
                 fontweight='bold', color=COLORS['primary'])
    
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Storage (GB)')
    ax2.set_title('(b) Compression Efficiency')
    ax2.legend(loc='upper left')
    ax2.set_xticks(qubits)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_scaling.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_scaling.png', bbox_inches='tight')
    plt.close()
    print("✅ fig_scaling.pdf (REAL DATA)")

# ============================================================
# FIGURE 3: Architecture Diagram (Simple)
# ============================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Layers
    import matplotlib.patches as patches
    
    # GPU VRAM
    rect1 = patches.FancyBboxPatch((5, 35), 25, 12, boxstyle="round,pad=0.03",
                                    facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(17.5, 41, 'GPU VRAM\n(cuStateVec)', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # CPU DRAM
    rect2 = patches.FancyBboxPatch((35, 35), 25, 12, boxstyle="round,pad=0.03",
                                    facecolor=COLORS['secondary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(47.5, 41, 'CPU DRAM\n(LZ4 Cache)', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # NVMe SSD
    rect3 = patches.FancyBboxPatch((65, 35), 25, 12, boxstyle="round,pad=0.03",
                                    facecolor=COLORS['accent'], edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(77.5, 41, 'NVMe SSD\n(Compressed)', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(35, 41), xytext=(30, 41), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.annotate('', xy=(65, 41), xytext=(60, 41), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    
    # State Vector
    rect4 = patches.FancyBboxPatch((25, 10), 50, 15, boxstyle="round,pad=0.03",
                                    facecolor=COLORS['light'], edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(50, 17, 'State Vector Chunks\n(32MB each, LZ4 compressed)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flow arrows
    ax.annotate('', xy=(50, 25), xytext=(50, 35), 
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_architecture.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_architecture.png', bbox_inches='tight')
    plt.close()
    print("✅ fig_architecture.pdf")

# ============================================================
# FIGURE 4: Time Breakdown
# ============================================================
def plot_timebreakdown():
    fig, ax = plt.subplots(figsize=(7, 4))
    
    qubits = sorted(REAL_DATA.keys())
    init_times = [REAL_DATA[q]['init'] for q in qubits]
    gate_times = [REAL_DATA[q]['gate'] for q in qubits]
    
    x = np.arange(len(qubits))
    width = 0.6
    
    ax.bar(x, init_times, width, label='Initialization', color=COLORS['accent'])
    ax.bar(x, gate_times, width, bottom=init_times, label='Gate Execution', color=COLORS['primary'])
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Simulation Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qubits])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('paper/figures/timebreakdown.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/timebreakdown.png', bbox_inches='tight')
    plt.close()
    print("✅ timebreakdown.pdf")


# ============================================================
# FIGURE 5: Throughput Analysis
# ============================================================
def plot_throughput():
    """Plot throughput degradation vs chunks (IO bottleneck)"""
    qubits = [20, 22, 24, 26, 28]
    chunks = [1, 1, 4, 16, 64]
    
    # Real data from ScaleQsim benchmark (Random-20 throughput)
    # Gates: 20*n
    # 20Q: 400 gates / 30.0s = 13.33
    # 22Q: 440 gates / 110.4s = 3.98
    # 24Q: 480 gates / 422.2s = 1.14
    # 26Q: 520 gates / 1895.8s = 0.27
    # 28Q: 560 gates / 2779.7s = 0.20
    
    throughput = [13.33, 3.98, 1.14, 0.27, 0.20]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = COLORS['secondary']
    ax1.set_xlabel('Qubits (State Vector Size)', fontsize=12)
    ax1.set_ylabel('Throughput (Gates/s)', color=color, fontsize=12)
    ax1.plot(qubits, throughput, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.set_yticks([0.1, 1, 10, 100])
    ax1.set_yticklabels(['0.1', '1', '10', '100'])
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Secondary x-axis for chunks
    def quit2chunk(x):
        return 2**(x-20) if x >= 20 else 1
        
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(qubits)
    ax2.set_xticklabels([f"{q}Q\n({2**(q-22) if q>22 else 1} Chunks)" for q in qubits])
    ax2.set_xlabel('Chunks (I/O Complexity)', fontsize=12)
    
    # Annotation
    ax1.annotate('In-Memory\n(High Speed)', xy=(21, 8), xytext=(22, 20),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    ax1.annotate('I/O Bound\n(Capacity)', xy=(26, 0.3), xytext=(26, 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('paper/figures/fig_throughput.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_throughput.png', bbox_inches='tight')
    plt.close()
    print("✅ fig_throughput.pdf")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  UNIFIED COLOR SCHEME FIGURES")
    print("="*50 + "\n")
    
    plot_motivation_a()
    plot_motivation_b()
    plot_scaling()
    plot_architecture()
    plot_timebreakdown()
    plot_throughput() # Added
    
    print("\n" + "="*50)
    print("  ALL FIGURES GENERATED!")
    print("="*50)
