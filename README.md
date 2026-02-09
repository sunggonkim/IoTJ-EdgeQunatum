# EdgeQuantum

> **Tiered-Memory Quantum Circuit Simulator for NVIDIA Jetson Edge Devices**

Simulate quantum circuits with state vectors **far exceeding physical memory** on resource-constrained edge hardware. EdgeQuantum achieves up to **39-qubit simulation (4 TB state vector)** on a device with only **8 GB RAM** and a **256 GB NVMe SSD**.

```
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║                     EdgeQuantum Architecture                         ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║                                                                       ║
  ║   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ║
  ║   │   GPU Core   │    │  UVM Buffers (4x) │    │   NVMe Storage   │    ║
  ║   │  Ampere SM87 │◄──►│    256 MB each    │◄──►│  Samsung PM9A1   │    ║
  ║   │  cuStateVec  │    │  AttachHost/Global│    │  O_DIRECT + LZ4  │    ║
  ║   └─────────────┘    └──────────────────┘    └──────────────────┘    ║
  ║         220ms/chunk         ▲    ▲                 100ms R/W          ║
  ║                             │    │                                     ║
  ║              ┌──────────────┘    └──────────────┐                     ║
  ║              │  4-Buffer Async Pipeline          │                     ║
  ║              │  Read→GPU overlap Write→GPU       │                     ║
  ║              │  NO R+W overlap (NVMe contention) │                     ║
  ║              └───────────────────────────────────┘                     ║
  ║                                                                       ║
  ╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Key Results

### Simulation Time Scaling (EdgeQuantum, depth=5)

```
  Time (s)
  40000 ┤
        │                                                          ● VQC/VQE
  35000 ┤                                                         ╱
        │                                                        ╱
  30000 ┤                                                  ● Random
        │                                                 ╱
  25000 ┤                                           ● QV ╱
        │                                          ╱    ╱
  20000 ┤                                    ●────╱── VQC/VQE
        │                                   ╱   ╱
  15000 ┤                              ●───╱── Random   ● QSVM
        │                             ╱   ╱
  10000 ┤                        ●───╱── QV
        │                       ╱   ╱
   5000 ┤              ●───────╱── All circuits
        │            ╱╱╱╱     ╱
      0 ┤──●──●─●──╱╱╱╱─────────────────────────────────
        └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬
          29  30 31 32 33 34 35 36 37 38 39   Qubits
         4GB 8GB ──── 128GB ─── 256GB ──── 4TB   State Size
```

### Speedup vs BMQSim-like Blocking I/O (depth=5, 29–34 qubits)

```
  Speedup
   1.50x ┤
         │            ●──── QSVM (1.48x peak)
   1.45x ┤           ╱ ╲        ● VQC
         │      ●───╱   ●──────● VQE
   1.40x ┤     ╱   ╱
         │    ╱   ╱    ●─────── Random
   1.35x ┤   ╱  ╱    ╱
         │  ╱  ╱    ╱
   1.30x ┤ ╱  ╱   ╱
         │╱  ╱   ╱     ●─────── QV
   1.25x ┤  ╱  ╱╱
         │ ╱  ╱╱
   1.20x ┤╱  ●──────●──●──●── GHZ (consistent ~1.21x)
         │  ╱
   1.15x ┤ ╱
         │╱
   1.10x ┤
         └──┬─────┬─────┬─────┬─────┬──
           29     30    31    32    33    34   Qubits
```

**Average speedup: 1.28x** | **Peak: 1.48x** (QSVM @ 33 qubits)

---

### Compression-Enabled Extreme Scaling (35–39 qubits, EdgeQuantum only)

State vectors from **256 GB to 4 TB** — far exceeding both RAM (8 GB) and SSD (256 GB).

```
  Time (hours)  EdgeQuantum depth=1 scaling
  2.5h ┤
       │
  2.0h ┤                              ● VQC (2.2h)
       │                              ● VQE (2.2h)
  1.5h ┤                        ● Random (1.7h)
       │                        ● QV (1.5h)
       │
  1.0h ┤                  ● GHZ (1.3h @ 39q)
       │            ●────╱── VQC/VQE (1.0h)
       │           ╱    ╱
  0.5h ┤     ●────╱────╱
       │    ╱    ╱    ╱
       │   ╱    ╱    ╱
     0 ┤──●───●────●──────────────────
       └──┬───┬────┬────┬────┬──
         35   36   37   38   39   Qubits
        256GB 512GB 1TB  2TB  4TB  State Size
```

### LZ4 Compression Effectiveness

```
  Compression Ratio
  100% ┤ █████████████████████████████████████  Uncompressed (raw)
       │
  10%  ┤ ███████  Entangled states (3–10%)
       │
   1%  ┤ █  Initial sparse state (~0.39%)
       │
   0%  ┤
       └──────────────────────────────────────
         39q: 4096 GB raw → ~16 GB on disk (0.39% initial)
         LZ4 enables 4 TB simulations on 256 GB SSD!
```

---

## Performance Data Summary

### EdgeQuantum vs BMQSim-like (depth=5, seconds)

| Circuit | 29q (4GB) | 30q (8GB) | 31q (16GB) | 32q (32GB) | 33q (64GB) | 34q (128GB) |
|---------|-----------|-----------|------------|------------|------------|-------------|
| **BMQSim-like** | | | | | | |
| Random  | 33.5 | 67.0 | 233.9 | 484.7 | 974.9 | 1963.5 |
| QV      | 29.0 | 57.7 | 207.6 | 450.4 | 920.8 | 1853.8 |
| VQC     | 42.4 | 95.2 | 254.7 | 514.8 | 1017.9 | 2069.0 |
| QSVM    | 16.9 | 33.9 | 102.2 | 203.3 | 430.7 | 849.7 |
| GHZ     | 5.2 | 10.4 | 40.9 | 85.4 | 173.5 | 351.2 |
| VQE     | 42.4 | 84.7 | 259.0 | 509.5 | 1027.9 | 2050.3 |
| **EdgeQuantum** | | | | | | |
| Random  | 29.0 (**1.15x**) | 57.8 (**1.16x**) | 169.9 (**1.38x**) | 357.2 (**1.36x**) | 722.7 (**1.35x**) | 1463.9 (**1.34x**) |
| QV      | 24.6 (**1.18x**) | 48.5 (**1.19x**) | 171.3 (**1.21x**) | 352.9 (**1.28x**) | 727.1 (**1.27x**) | 1462.6 (**1.27x**) |
| VQC     | 38.4 (**1.10x**) | 76.9 (**1.24x**) | 175.6 (**1.45x**) | 361.7 (**1.42x**) | 756.9 (**1.34x**) | 1483.9 (**1.39x**) |
| QSVM    | 15.5 (**1.09x**) | 30.8 (**1.10x**) | 70.4 (**1.45x**) | 144.7 (**1.40x**) | 291.4 (**1.48x**) | 593.1 (**1.43x**) |
| GHZ     | 4.3 (**1.20x**) | 8.6 (**1.20x**) | 33.5 (**1.22x**) | 70.4 (**1.21x**) | 143.5 (**1.21x**) | 290.2 (**1.21x**) |
| VQE     | 38.3 (**1.11x**) | 76.9 (**1.10x**) | 176.9 (**1.46x**) | 361.0 (**1.41x**) | 738.1 (**1.39x**) | 1473.9 (**1.39x**) |

### Extended Scalability with LZ4 Compression (EdgeQuantum, depth=5)

| Circuit | 35q (256GB) | 36q (512GB) | 37q (1TB) | 38q (2TB) | 39q (4TB) |
|---------|-------------|-------------|-----------|-----------|-----------|
| Random  | 31.7 min | 63.9 min | 127.4 min | 254.2 min | 508.9 min |
| QV      | 27.1 min | 54.8 min | 127.4 min | 217.0 min | 434.1 min |
| VQC     | 41.2 min | 82.4 min | 127.4 min | 330.2 min | 661.9 min |
| QSVM    | 16.5 min | 33.1 min | 127.4 min | 132.0 min | 264.8 min |
| VQE     | 41.1 min | 82.3 min | 127.4 min | 330.3 min | — (running) |

```
  ┌──────────────────────────────────────────────────────────────────┐
  │              State Vector Size vs Physical Resources             │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  39 qubits ████████████████████████████████████████  4096 GB     │
  │  38 qubits ████████████████████                      2048 GB     │
  │  37 qubits ██████████                                1024 GB     │
  │  36 qubits █████                                      512 GB     │
  │  35 qubits ███                                        256 GB     │
  │  34 qubits ██                                         128 GB     │
  │  33 qubits █                                           64 GB     │
  │  32 qubits ▌                                           32 GB     │
  │  ─ ─ ─ ─ ─ SSD capacity: 256 GB ─ ─ ─ ─ ─ ─ ─ ─ ─           │
  │  ─ ─ ─ ─ ─ RAM: 8 GB ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─            │
  │  29 qubits ▏                                            4 GB     │
  │                                                                  │
  │  EdgeQuantum simulates 512x beyond RAM capacity!                 │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
edgeQuantum/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── code/
│   ├── src/                      # C++ simulator core
│   │   ├── main.cpp              # Entry point, CLI, mode selection
│   │   ├── simulator.cpp/hpp     # Pipeline engine (4-buffer async)
│   │   ├── chunk_manager.cpp/hpp # UVM buffer allocation
│   │   ├── compression.hpp       # LZ4 variable-size ping-pong I/O
│   │   ├── io_backend.hpp        # O_DIRECT NVMe I/O
│   │   └── utils.hpp             # Utilities
│   ├── Makefile / CMakeLists.txt # Build system
│   ├── comprehensive_benchmark.py# Full grid benchmark runner
│   ├── comprehensive_results.json# All experiment data (151 runs)
│   └── grand_benchmark.py        # Multi-circuit benchmark
├── paper/                        # LaTeX manuscript (IEEE IoT Journal)
│   ├── main.tex                  # Root document
│   ├── 1_Introduction.tex … 6_Conclusion.tex
│   └── Figures/                  # Generated figures
└── third_party/                  # cuQuantum SDK (not tracked)
```

---

## How It Works

### 4-Buffer Asynchronous Pipeline

```
  Pipeline Visualization (per gate layer, 32 chunks @ 34 qubits):

  Time ──────────────────────────────────────────────────────►

  Buf[0]: ║ Read[0] ║ GPU[0]  ║         ║         ║ Read[4] ║ GPU[4]  ║
  Buf[1]: ║         ║ Read[1] ║ GPU[1]  ║         ║         ║ Read[5] ║
  Buf[2]: ║         ║         ║ Read[2] ║ GPU[2]  ║         ║         ║
  Buf[3]: ║         ║         ║         ║ Read[3] ║ GPU[3]  ║         ║
  Write:  ║         ║         ║ Write[0]║ Write[1]║ Write[2]║ Write[3]║
                                ▲─────── overlaps with GPU ───────▲

  Key: Read and Write are NEVER concurrent (NVMe contention)
       Write overlaps with GPU compute (different hardware)
       4 buffers give Write 3 iterations to complete → no stalls
```

### Auto Mode Selection

```
  ┌──────────────────────────────────────────────┐
  │           State Size Decision Tree            │
  │                                                │
  │   state_size < 80% free RAM?                   │
  │      ├── YES → Native Mode (cudaMalloc)        │
  │      │         Fast, no I/O overhead           │
  │      └── NO  → Tiered Mode (EdgeQuantum)       │
  │                state_size > SSD capacity?       │
  │                   ├── YES → Enable LZ4         │
  │                   └── NO  → Raw I/O            │
  └──────────────────────────────────────────────┘
```

### NVMe I/O Strategy

```
  ┌────────────────────────────────────────────────────┐
  │             I/O Performance Profile                 │
  │                                                      │
  │   Operation      │ Time (256MB) │ Bandwidth          │
  │   ───────────────┼──────────────┼──────────────────  │
  │   Read (pread)   │    100 ms    │  ~2600 MB/s        │
  │   Write (pwrite) │ 100-450 ms   │  Varies (SLC/TLC)  │
  │   GPU compute    │    220 ms    │  N/A                │
  │   R+W concurrent │  208 ms EACH │  50% penalty ✗     │
  │                                                      │
  │   → Serialize R/W, overlap W+GPU = optimal           │
  └────────────────────────────────────────────────────┘
```

---

## Requirements

| Component | Specification |
|-----------|--------------|
| **Platform** | NVIDIA Jetson Orin (SM 8.7) |
| **OS** | Ubuntu 20.04 (JetPack 5.x) |
| **CUDA** | 11.4 (nvcc 11.4) |
| **cuQuantum** | SBSA 22.03 archive (SM 8.7 compatible) |
| **RAM** | 8 GB LPDDR5 (shared CPU/GPU) |
| **Storage** | NVMe SSD (256 GB+, PCIe Gen3 x4) |
| **Libraries** | liblz4-dev, build-essential, python3-pip |

## Installation

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential liblz4-dev python3-pip
```

### 2. cuQuantum SDK

```bash
mkdir -p third_party && cd third_party
wget -O cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz \
  https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz
tar -xvf cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz
```

### 3. Python dependencies

```bash
pip3 install -r requirements.txt
```

## Build

```bash
cd code
make clean && make CUQUANTUM_ROOT=../third_party/cuquantum-linux-sbsa-22.03.0.40-archive
```

## Quick Start

```bash
cd code

# Native mode (fits in 8GB RAM)
./build/edge_quantum --qubits 26 --verify

# Tiered mode (exceeds RAM, uses NVMe)
./build/edge_quantum --qubits 30 --verify

# Compression mode (exceeds SSD, uses LZ4)
./build/edge_quantum --qubits 35 --circuit random --depth 1

# Full benchmark
python3 comprehensive_benchmark.py
```

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   Execution Mode Summary                      │
  ├──────────┬──────────┬───────────────────────────────────────  │
  │  Mode    │  Qubits  │  Description                            │
  │──────────┼──────────┼───────────────────────────────────────  │
  │  Native  │  ≤28     │  cudaMalloc, no I/O, fastest            │
  │  Tiered  │  29–34   │  UVM + NVMe, async pipeline             │
  │  Tiered  │  35–39+  │  UVM + NVMe + LZ4, ping-pong files     │
  │  +LZ4    │          │  Enables simulation beyond SSD size     │
  └──────────┴──────────┴───────────────────────────────────────  ┘
```

---

## Experiment Coverage

```
  151 total experiment runs across 11 qubit counts

  Qubits:   29  30  31  32  33  34 │ 35  36  37  38  39
  Schemes:  EQ + BMQSim (both)     │ EdgeQuantum only (LZ4)
  Circuits: Random, QV, VQC, QSVM, GHZ, VQE
  Depths:   1, 3, 5  (GHZ: depth=1 only)

  Coverage Matrix (EdgeQuantum):
  ─────────┬─────────────────────────────────────────────
  Circuit  │ 29 30 31 32 33 34 │ 35 36 37 38 39
  ─────────┼─────────────────────────────────────────────
  Random   │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  ✓
  QV       │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  ✓
  VQC      │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  ✓
  QSVM     │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  ✓
  GHZ      │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  ✓
  VQE      │ ✓  ✓  ✓  ✓  ✓  ✓ │ ✓  ✓  ✓  ✓  △
  ─────────┴─────────────────────────────────────────────
                                            △ = in progress
```

---

## Paper

The accompanying paper is submitted to **IEEE Internet of Things Journal**. LaTeX sources are in `paper/`.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

Apache-2.0
