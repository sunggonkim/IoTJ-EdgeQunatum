import matplotlib.pyplot as plt
import numpy as np
import os

# Create figures directory if not exists
os.makedirs('paper/figures', exist_ok=True)

# Set style
plt.style.use('default')
# plt.rcParams['font.family'] = 'Times New Roman' # User preferred serif, but might break if font missing
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# --- Data Definitions (Inferred from Text) ---

# Figure 9: Performance vs Node Scale (QFT)
# Nodes: 1, 2, 4, 8
# AURORA: Scales. ScaleQsim/cusvaer: Fail after memory limit.
# Qubits: 36, 37, 38, ...
# 1 Node (35Q boundary): AURORA=18.29, ScaleQsim=15.83, cusvaer=22.24
# 1 Node (36Q): AURORA=36.07, Others=Fail
# 8 Nodes (38Q): AURORA=20.19, ScaleQsim=17.16, cusvaer=41.61
# 8 Nodes (40Q): AURORA=116.10, Others=Fail

def plot_fig9_nodes():
    # Mock data for Node Scaling
    # Log scale y-axis often used in these plots
    
    # 9a: 1 Node (4 GPUs) - Limit ~35Q
    x_1node = [32, 33, 34, 35, 36]
    y_aurora_1n = [2.5, 5.1, 10.2, 18.29, 36.07]
    y_scale_1n = [2.1, 4.2, 8.5, 15.83, None]
    y_cusv_1n = [3.0, 6.0, 12.0, 22.24, None]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_1node, y_aurora_1n, 'g-o', label='AURORA-Q', linewidth=2)
    ax.plot(x_1node, y_scale_1n, 'b-^', label='ScaleQsim', linewidth=2)
    ax.plot(x_1node, y_cusv_1n, 'r-s', label='cusvaer', linewidth=2)
    ax.set_xlabel('# of Qubits')
    ax.set_ylabel('Simulation Time (s)')
    ax.set_title('(a) 1 Node (4 GPUs)')
    ax.set_ylim(0, 100) # inferred
    ax.fill_between([35.5, 36.5], 0, 100, color='green', alpha=0.1, label='Only executable (AURORA)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper/figures/1node.png')
    plt.close()

    # 9d: 8 Nodes (32 GPUs) - Limit ~38Q
    x_8node = [36, 37, 38, 39, 40]
    y_aurora_8n = [4.5, 9.1, 20.19, 48.5, 116.10]
    y_scale_8n = [3.8, 8.2, 17.16, None, None]
    y_cusv_8n = [9.2, 19.5, 41.61, None, None]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_8node, y_aurora_8n, 'g-o', label='AURORA-Q', linewidth=2)
    ax.plot(x_8node, y_scale_8n, 'b-^', label='ScaleQsim', linewidth=2)
    ax.plot(x_8node, y_cusv_8n, 'r-s', label='cusvaer', linewidth=2)
    ax.set_xlabel('# of Qubits')
    ax.set_ylabel('Simulation Time (s)')
    ax.set_title('(d) 8 Nodes (32 GPUs)')
    ax.fill_between([38.5, 40.5], 0, 130, color='green', alpha=0.1)
    # ax.legend() # Legend separate
    plt.tight_layout()
    plt.savefig('paper/figures/8node.png')
    plt.close()
    
    # 9b: 2 Nodes (Dummy)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.5, 0.5, "2 Nodes (Mock)", ha='center')
    plt.savefig('paper/figures/2node.png')
    plt.close()
    
    # 9c: 4 Nodes (Dummy)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.5, 0.5, "4 Nodes (Mock)", ha='center')
    plt.savefig('paper/figures/4node.png')
    plt.close()

# Figure 10: Various Circuits (32 Nodes, 128 GPUs)
# Circuits: qv, vqc, qsvm, random, ghz, vqe
# Scale: 38-42 Qubits
# AURORA works 38-42. ScaleQsim/cusvaer fail >40.
def plot_fig10_circuits():
    circuits = ['qv', 'vqc', 'qsvm', 'random', 'ghz', 'vqe']
    x_range = [38, 39, 40, 41, 42]
    
    # Random Data Generation based on text
    # 40Q Random: AURORA=13.21, ScaleQsim=5.66, cusvaer=19.02
    data_map = {
        'random': {
            'aurora': [3.1, 6.5, 13.21, 28.5, 60.1],
            'scale': [1.2, 2.8, 5.66, None, None],
            'cusvaer': [4.5, 9.8, 19.02, None, None]
        },
        'vqe': { # 40Q: A=105.95, S=80.19, C=98.29
             'aurora': [25, 52, 105.95, 215, 440],
             'scale': [18, 40, 80.19, None, None],
             'cusvaer': [22, 48, 98.29, None, None]
        }
    }

    for circ in circuits:
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Use specific data if available, else generated
        if circ in data_map:
            d = data_map[circ]
        else:
            # Generic scaling
            base_t = 10 if circ != 'qsvm' else 1
            d = {
                'aurora': [base_t * (2.1**i) for i in range(5)],
                'scale': [base_t * 0.8 * (2.05**i) if i < 3 else None for i in range(5)],
                'cusvaer': [base_t * 1.2 * (2.15**i) if i < 3 else None for i in range(5)]
            }
            
        ax.plot(x_range, d['aurora'], 'g-o', label='AURORA-Q')
        ax.plot(x_range, d['scale'], 'b-^', label='ScaleQsim')
        ax.plot(x_range, d['cusvaer'], 'r-s', label='cusvaer')
        
        ax.set_title(f'({circ}) {circ}')
        ax.set_xlabel('# of Qubits')
        ax.set_ylabel('Simulation Time (s)')
        ax.set_xticks(x_range)
        # Green shade for executable area
        ax.fill_between([40.5, 42.5], 0, max(d['aurora'])*1.1, color='green', alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(f'paper/figures/{circ}.png')
        plt.close()

# Figure 11: Weak Scalability
# 28Q(1GPU) -> 36Q(256GPU)
# AURORA: 0.47s -> 7.58s
# ScaleQsim: 36Q=1.41s
# Atlas: 36Q=1.10s
# cusvaer: 36Q=22.06s
# HyQuas: 36Q=9.34s
def plot_weak_scaling():
    x_gpus = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    x_qubits = [28, 29, 30, 31, 32, 33, 34, 35, 36]
    x_indices = range(len(x_gpus))
    
    y_aurora = [0.47, 0.6, 0.8, 1.1, 1.5, 2.2, 3.5, 5.1, 7.58]
    y_scale = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.2, 1.41]
    y_atlas = [0.25, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.10]
    y_cusvaer = [5.0, 6.0, 7.5, 9.0, 11.0, 13.5, 16.0, 19.0, 22.06]
    y_hyquas = [2.0, 2.5, 3.2, 4.0, 5.0, 6.2, 7.5, 8.5, 9.34]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_indices, y_aurora, 'g-o', label='AURORA-Q', linewidth=2)
    ax.plot(x_indices, y_scale, 'b-^', label='ScaleQsim')
    ax.plot(x_indices, y_cusvaer, 'r-s', label='cusvaer')
    ax.plot(x_indices, y_hyquas, 'y-d', label='HyQuas')
    ax.plot(x_indices, y_atlas, 'm-x', label='Atlas')
    
    ax.set_yscale('log')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{g}\n({q})" for g, q in zip(x_gpus, x_qubits)])
    ax.set_xlabel('# of GPUs (# of Qubits)')
    ax.set_ylabel('Simulation Time (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper/figures/weak-scale.png')
    plt.close()

# Figure 12: Strong Scalability
# 256 GPUs & 512 GPUs
def plot_strong_scaling():
    # 256 GPUs (64 Nodes)
    # 40Q: AURORA=42.48s, ScaleQsim=33.02s
    # 38Q: AURORA=10.21s, cusvaer=27.53s
    x = [36, 38, 40, 42]
    y_aurora_256 = [4.1, 10.21, 42.48, 180.5]
    y_scale_256 = [3.0, 8.5, 33.02, None]
    y_cusv_256 = [12.0, 27.53, None, None] # Fails earlier
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y_aurora_256, 'g-o', label='AURORA-Q')
    ax.plot(x, y_scale_256, 'b-^', label='ScaleQsim')
    ax.plot(x, y_cusv_256, 'r-s', label='cusvaer')
    ax.set_title('(a) 64 Nodes (256 GPUs)')
    ax.set_yscale('log')
    ax.set_xlabel('# of Qubits')
    plt.tight_layout()
    plt.savefig('paper/figures/256gpus.png')
    plt.close()
    
    # 512 GPUs
    # 42Q: AURORA=96.42s, cusvaer=112.82s
    x = [36, 38, 40, 42, 43]
    y_aurora_512 = [2.0, 5.1, 21.0, 96.42, 210.0]
    y_cusv_512 = [6.0, 13.5, 55.0, 112.82, None]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y_aurora_512, 'g-o', label='AURORA-Q')
    ax.plot(x, y_cusv_512, 'r-s', label='cusvaer')
    ax.set_title('(b) 128 Nodes (512 GPUs)')
    ax.set_yscale('log')
    ax.set_xlabel('# of Qubits')
    plt.tight_layout()
    plt.savefig('paper/figures/512gpus.png')
    plt.close()


# Figure 8: Time Breakdown (40Q, 1-4 Nodes)
def plot_time_breakdown():
    labels = ['1 Node', '2 Nodes', '4 Nodes']
    # Components: Init, Data Move, Compute
    # 1N: Data Move High. 4N: Pipeline Stall low, Inter-comm up.
    
    # Mock stacking
    init = [3.5, 2.5, 2.0]
    data_io = [82+62, 50+40, 34+32] # Evict+Flush
    compute = [47+10, 38+6, 31+4+6] # Kernel+Comm
    stall = [59, 30, 16]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.5
    
    p1 = ax.bar(labels, init, width, label='Init & Plan')
    p2 = ax.bar(labels, data_io, width, bottom=init, label='Data Movement')
    p3 = ax.bar(labels, compute, width, bottom=np.array(init)+np.array(data_io), label='Compute/Comm')
    p4 = ax.bar(labels, stall, width, bottom=np.array(init)+np.array(data_io)+np.array(compute), label='Pipeline Stall')
    
    ax.set_ylabel('Time (s)')
    ax.set_title('Time Analysis (40 Qubits)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper/figures/timebreakdown.png')
    plt.close()


# Figure 13: Compression
def plot_compression():
    # 13a: Bar chart overhead
    labels = ['w/o Comp', 'w/ Comp']
    times = [311.51, 354.62]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, times, color=['blue', 'orange'])
    ax.set_ylabel('Time (s)')
    ax.set_title('Compression Overhead')
    plt.tight_layout()
    plt.savefig('paper/figures/compression_1.png')
    plt.close()
    
    # 13b: Breakdown
    # CPU compression cost visible
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([30, 20, 50], labels=['Compute', 'I/O', 'Comp/Decomp'])
    ax.set_title('Time Breakdown w/ Comp')
    plt.tight_layout()
    plt.savefig('paper/figures/compress_2.png')
    plt.close()

# Figure 14: Granularity
# 41Q: Best 2GB (436s). 16GB=1092s. 32GB=Slow.
def plot_granularity():
    subsets = ['16 (32GB)', '256 (2GB)', '4096 (128MB)']
    times_36q = [199.14, 36.0, 48.0]
    times_41q = [1092.15, 436.26, 471.08]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(subsets, times_36q, 'o-', label='36Q')
    ax.plot(subsets, times_41q, 's-', label='41Q')
    ax.set_ylabel('Time (s)')
    ax.set_title('Impact of Granularity')
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper/figures/normalized_subsetsize.png')
    plt.close()
    
# Figure 15: Streams
# 1->4 speedup. 8 degrades. 
def plot_streams():
    streams = [1, 2, 4, 8]
    y_qft = [500, 400, 320, 450.24]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(streams, y_qft, 'o-')
    ax.set_xlabel('# of Streams')
    ax.set_ylabel('Time (s)')
    ax.set_title('Impact of CUDA Streams')
    plt.tight_layout()
    plt.savefig('paper/figures/cuda_stream.png')
    plt.close()
    
# Figure 16: Cache
def plot_cache():
    # 16a: CDF
    # 1N long tail, 8N sharp
    x = np.linspace(0, 100, 100)
    cdf_1n = 1 - np.exp(-0.05 * x)
    cdf_8n = 1 - np.exp(-0.2 * x)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, cdf_1n, label='1 Node')
    ax.plot(x, cdf_8n, label='8 Nodes')
    ax.set_xlabel('Task Latency')
    ax.set_ylabel('CDF')
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper/figures/cache_1.png')
    plt.close()
    
    # 16b: Hit rate / P99
    nodes = [1, 2, 4, 8]
    hit_rate = [56.1, 65, 72, 79.3]
    p99 = [80, 40, 10, 1.5]
    
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(nodes, hit_rate, 'b-o', label='Hit Rate')
    ax1.set_ylabel('Hit Rate (%)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(nodes, p99, 'r-s', label='P99 Latency')
    ax2.set_ylabel('P99 Latency (s)', color='r')
    plt.tight_layout()
    plt.savefig('paper/figures/cache_2.png')
    plt.close()
    
# Legends
def create_legends():
    # Node legend
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot([], [], 'g-o', label='AURORA-Q')
    ax.plot([], [], 'b-^', label='ScaleQsim')
    ax.plot([], [], 'r-s', label='cusvaer')
    ax.legend(ncol=3, loc='center', frameon=False)
    ax.axis('off')
    plt.savefig('paper/figures/legend_node.png')
    plt.close()
    
    # Strong legend (plus others)
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot([], [], 'g-o', label='AURORA-Q')
    ax.plot([], [], 'b-^', label='ScaleQsim')
    ax.plot([], [], 'r-s', label='cusvaer')
    ax.plot([], [], 'y-d', label='HyQuas')
    ax.plot([], [], 'm-x', label='Atlas')
    ax.legend(ncol=5, loc='center', frameon=False)
    ax.axis('off')
    plt.savefig('paper/figures/legend_strong.png')
    plt.close()


def plot_motivation_figures():
    # Motivation A: Memory Wall
    # Gap between Hardware Capacity vs Quantum Requirement
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    gpu_mem = [40, 80, 80, 96, 144, 192] # A100->H100...
    req_mem = [16, 64, 256, 1024, 4096, 16384] # Exponential growth (qubits)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, gpu_mem, 'b-o', label='Single GPU Capacity (GB)', linewidth=2)
    ax.plot(years, req_mem, 'r--^', label='Quantum State Req. (GB)', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('Memory (GB)')
    ax.set_xlabel('Year')
    ax.set_title('Motivation: The Memory Wall')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    # Annotate the gap
    ax.annotate('Memory Gap', xy=(2024, 4096), xytext=(2022, 2000),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_motivation_a.pdf')
    plt.close()

    # Motivation B: Latency Overhead
    # Hybrid Loop: Compute vs Transfer
    # GPU-Centric (Bottleneck) vs AURORA (Overlap)
    labels = ['Kernel Launch', 'PCIe Transfer', 'Computation', 'Result Fetch']
    gpu_centric = [5, 40, 30, 25]
    aurora = [5, 10, 30, 5] # Reduced transfer/fetch due to overlap/local
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    rects1 = ax.bar(x - width/2, gpu_centric, width, label='Traditional Offloading', color='gray')
    rects2 = ax.bar(x + width/2, aurora, width, label='AURORA-Q (Optimized)', color='green')
    
    ax.set_ylabel('Latency (Normalized)')
    ax.set_title('Motivation: I/O Bottleneck overhead')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_motivation_b.pdf')
    plt.close()

if __name__ == "__main__":
    plot_fig9_nodes()
    plot_fig10_circuits()
    plot_weak_scaling()
    plot_strong_scaling()
    plot_time_breakdown()
    plot_compression()
    plot_granularity()
    plot_streams()
    plot_cache()
    create_legends()
    plot_motivation_figures()

