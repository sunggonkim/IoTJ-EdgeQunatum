# EdgeQuantum

EdgeQuantum is a tiered-memory quantum circuit simulator targeting NVIDIA Jetson-class devices. It supports multiple execution schemes on the same hardware: cuQuantum Native, cuQuantum UVM, BMQSim-like offload, Cirq (CPU), and EdgeQuantum (async tiered I/O with LZ4).

## Repository Layout

```
edgeQuantum/
├── README.md
├── paper/                # LaTeX sources
├── code/                 # C++/Python benchmarks and runner scripts
│   ├── src/              # C++ simulator
│   ├── run_28q_schemes.py # 28-qubit scheme validation
│   └── grand_benchmark.py # Multi-circuit benchmark
└── third_party/          # cuQuantum archives (downloaded)
```

## Requirements (Jetson Orin Nano)

- Ubuntu 20.04 (JetPack 5.x)
- CUDA 11.4 (nvcc 11.4)
- cuQuantum SDK **SBSA archive** (CUDA 11) — use 22.03 for SM 8.7 compatibility
- Python 3.8 + pip
- LZ4 development library

## Installation

### 1) System packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential liblz4-dev python3-pip
```

### 2) cuQuantum (SBSA, CUDA 11)

Download and extract the ARM64 SBSA archive (works on Jetson Orin SM 8.7):

```bash
mkdir -p third_party
cd third_party
wget -O cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz \
  https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz
tar -xvf cuquantum-linux-sbsa-22.03.0.40-archive.tar.xz
```

### 3) Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Build

```bash
cd code
make clean && make CUQUANTUM_ROOT=../third_party/cuquantum-linux-sbsa-22.03.0.40-archive
```

## Run: 28-qubit scheme validation (time + correctness)

```bash
cd code
python3 run_28q_schemes.py
```

Results are written to `code/results_28q_schemes.json`.

## Run: Grand benchmark

```bash
cd code
python3 grand_benchmark.py
```

## Notes

- The cuQuantum SBSA 23.10+ archives include PTX for `sm_80/90` only and fail to initialize on SM 8.7.
- The 22.03 SBSA archive includes PTX for `sm_86`, which successfully JITs on Jetson Orin (SM 8.7).

## License

Apache-2.0
