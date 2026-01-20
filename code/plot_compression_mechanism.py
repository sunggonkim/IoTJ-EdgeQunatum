import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_compression_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Draw Storage Layer (Disk)
    rect_disk = patches.Rectangle((5, 5), 90, 15, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect_disk)
    ax.text(50, 8, "Storage Layer (NVMe SSD)", ha='center', fontsize=12, fontweight='bold')

    # Draw Compressed Chunks
    for i in range(10):
        # Small compressed blocks
        rect = patches.Rectangle((10 + i*8, 12), 6, 6, linewidth=1, edgecolor='blue', facecolor='#a0cfff')
        ax.add_patch(rect)
        ax.text(13 + i*8, 15, "LZ4", ha='center', va='center', fontsize=8, color='white')

    # Draw RAM/GPU Layer (Active)
    rect_gpu = patches.Rectangle((35, 35), 30, 20, linewidth=2, edgecolor='green', facecolor='#e6fffa')
    ax.add_patch(rect_gpu)
    ax.text(50, 51, "GPU / RAM (Active Execution)", ha='center', fontsize=12, fontweight='bold')

    # Decompressed Chunk
    rect_active = patches.Rectangle((40, 38), 20, 10, linewidth=2, edgecolor='red', facecolor='#ffcccb')
    ax.add_patch(rect_active)
    ax.text(50, 43, "Raw State Vector\n(Decompressed)", ha='center', va='center', fontsize=10)

    # Arrows
    # Up (Read & Decompress)
    ax.annotate("", xy=(45, 35), xytext=(45, 20), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.text(43, 27, "Read &\nDecompress", ha='right', fontsize=9)

    # Down (Compress & Write)
    ax.annotate("", xy=(55, 20), xytext=(55, 35), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.text(57, 27, "Compress &\nWrite Back", ha='left', fontsize=9)

    # Explanation Text
    ax.text(50, 2, "Mechanism: Only active chunks consume full memory. Inactive chunks are compressed on disk.", ha='center', fontsize=10, style='italic')
    ax.text(80, 45, "Compression Ratio:\nHigh for sparse/structured states\n(e.g., Init, Early Gates)", ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray"))

    plt.tight_layout()
    plt.savefig('paper/figures/fig_compression_mechanism.pdf')
    plt.close()

if __name__ == "__main__":
    draw_compression_diagram()
