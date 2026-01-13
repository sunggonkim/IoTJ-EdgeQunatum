# EdgeQuantum: Variational Quantum Algorithms on IoT Edge Devices

[![IEEE IoTJ](https://img.shields.io/badge/IEEE-IoT_Journal-blue)](https://ieee-iotj.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![cuQuantum](https://img.shields.io/badge/cuQuantum-23.3-orange)](https://developer.nvidia.com/cuquantum-sdk)

> **IEEE Internet of Things Journal (IoTJ) Submission**  
> Enabling Quantum Circuit Simulation on Resource-Constrained Edge Devices

---

## 📋 Abstract

This work demonstrates the feasibility of running variational quantum algorithms (VQE, QAOA) on IoT edge devices. Using NVIDIA Jetson Orin Nano with cuQuantum SDK, we achieve up to **20-qubit VQE simulation** (1M+ Hilbert dimension) and **100% MaxCut accuracy with QAOA** at 10 qubits.

---

## 🚀 Key Results

| Metric | EdgeQuantum (Ours) | Competitor (Qiskit/SV) |
|--------|-------------------|------------------------|
| **Max Qubits** | **35 (34.3B Amplitudes)** | 26 (GPU limit) |
| **Engine** | **Native cuQuantum (custatevec)** | Custom / NumPy |
| **Compression** | **LZ4 (242x ratio)** | None |
| **Speed (35Q)** | **13.4 min (Total)** | N/A (OOM) |
| **Energy** | **15W (Jetson Nano)** | 400W (A100 GPU) |

### 📊 Native cuQuantum Benchmark
- **28 Qubits**: 7.0s
- **30 Qubits**: 25.8s
- **32 Qubits**: 1.7 min
- **33 Qubits**: 3.4 min
- **34 Qubits**: 6.5 min
- **35 Qubits**: 13.4 min (256GB Data → 1.0GB Disk) - Max 28 qubits

| Qubits | Hilbert Dim | DRAM | Gate Time |
|--------|-------------|------|-----------|
| 27 | 134,217,728 | 1.0GB | 2.1s |
| **28** | **268,435,456** | **2.0GB** | **4.3s** |

### QAOA (MaxCut)

| Qubits | Edges | Approx. Ratio | Time |
|--------|-------|---------------|------|
| 4-10 | 3-20 | **100%** | 5-82s |
| 12 | 30 | 91.3% | 464s |

---

## 🗂️ Repository Structure

```
edgeQuantum-iotj/
├── README.md           # This file
├── paper/              # IEEE IoTJ LaTeX source
│   ├── main.tex        # Main paper
│   ├── IEEEtran.cls    # IEEE class file
│   └── figures/        # Paper figures
├── code/               # Implementation
│   ├── vqe_iot.py      # VQE algorithm
│   ├── qaoa_iot.py     # QAOA MaxCut
│   └── test_cuquantum_*.py  # Benchmarks
└── data/               # Results
    ├── benchmark_results.json
    ├── vqe_log.txt
    └── qaoa_log.txt
```

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA Jetson Orin Nano (JetPack 5.1+)
- CUDA 11.4+
- Python 3.8+

### Installation

```bash
# Clone repository
git clone https://github.com/sunggonkim/IoTJ-EdgeQunatum.git
cd IoTJ-EdgeQunatum

# Install dependencies
pip install cuquantum-python-cu11 cupy-cuda11x scipy psutil
```

### Run Benchmarks

```bash
# VQE benchmark (4-22 qubits)
python3 -u code/vqe_iot.py | tee data/vqe_results.log

# QAOA benchmark (4-14 qubits)
python3 -u code/qaoa_iot.py | tee data/qaoa_results.log
```

---

## 📈 Performance Analysis

### Scaling Behavior

```
VQE Time per Iteration vs Qubits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     │
  9s ┤                        ●
     │                       20q
  2s ┤              ●
     │             18q
  1s ┤        ●  ●
     │       14q 16q
     │   ●  ●
     │  10q 12q
     └────────────────────────────
         4   8  12  16  20  Qubits
```

### Memory Efficiency

| Algorithm | Peak GPU | State Vector Size |
|-----------|----------|-------------------|
| VQE 16q | 0.5 MB | 65K amplitudes |
| VQE 20q | 8.0 MB | 1M amplitudes |
| QAOA 12q | 0.0 MB* | 4K amplitudes |

*QAOA uses diagonal operators - no matrix allocation needed

---

## 🔬 Algorithms

### VQE (Variational Quantum Eigensolver)
- **Ansatz**: Hardware-efficient (RY-RZ + CNOT layers)
- **Hamiltonian**: Transverse-field Ising model
- **Optimizer**: COBYLA (derivative-free)

### QAOA (Quantum Approximate Optimization Algorithm)
- **Problem**: MaxCut on random graphs
- **Layers**: p=1 (extensible to p>1)
- **Cost function**: RZZ diagonal gates (memory-efficient)

---

## 📝 Paper

```
paper/
├── main.tex          # IEEE IoTJ paper
├── IEEEtran.cls      # IEEE template
└── figures/          # Result visualizations
```

Build PDF:
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 📚 Related Work

- **ScaleQsim** (SIGMETRICS 2026): [Paper](https://dl.acm.org/doi/10.1145/3771577) | [Code](https://github.com/Bigdata-HPC-Lab/ScaleQsim)
- **cuQuantum SDK**: [Documentation](https://developer.nvidia.com/cuquantum-sdk)

---

## 📖 Citation

```bibtex
@article{kim2026edgequantum,
  title={EdgeQuantum: Variational Quantum Algorithms on IoT Edge Devices},
  author={Kim, Sunggon and ...},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License

Apache-2.0

---

## 👥 Authors

- **Sunggon Kim** - Seoul National University of Science and Technology
