# Copilot instructions for EdgeQuantum

Purpose: Give AI coding agents the minimal, concrete knowledge to be productive in this repo.

System password: 1234qwer 
Do root stuff and sudo when needed.

- **Big picture:** This repository implements a tiered-memory quantum circuit simulator targeting NVIDIA Jetson-class devices. The C++ simulator core lives under `code/src/` and exposes multiple execution "schemes" (cuQuantum native, cuQuantum UVM, Tiered/EdgeQuantum) via a single binary `code/build/edge_quantum`. Python scripts in `code/` run benchmarks that invoke that binary and also run CPU Cirq experiments.

- **Key files / dirs:** `code/src/` (C++ core: `main.cpp`, `simulator.cpp`, `chunk_manager.cpp`, `io_backend.cpp/hpp`), `code/Makefile` & `code/CMakeLists.txt` (build), `code/build/edge_quantum` (binary), `code/*.py` (benchmarks and runners), `third_party/` (cuQuantum SBSA archives). The paper and figures are in `paper/` but are not needed to build/run experiments.

- **Build / install (exact commands used by developers):**
  - Install system deps: `sudo apt-get install -y build-essential liblz4-dev python3-pip`
  - Extract cuQuantum SBSA archive into `third_party/` (repo includes archives under `third_party/`). Prefer the 22.03 SBSA archive for Jetson Orin (SM 8.7).
  - Python deps: `python3 -m pip install -r requirements.txt`
  - Build (Makefile):
    ```bash
    cd code
    make clean && make CUQUANTUM_ROOT=../third_party/cuquantum-linux-sbsa-22.03.0.40-archive
    ```
  - Alternate build: use `cmake`/`CMakeLists.txt` if you prefer a CMake flow; CMake sets CUDA arch to `sm_87` by default for Jetson Orin.

- **How to run representative flows:**
  - Quick verification: `cd code && ./build/edge_quantum --qubits 26 --verify` (Native mode) or `--qubits 30 --verify` (Tiered mode).
  - Grand benchmark: `cd code && python3 grand_benchmark.py`.
  - Full comprehensive benchmark: `cd code && nohup python3 comprehensive_benchmark.py > benchmark_output.log 2>&1 &`
  - Note: Python benchmarks expect the C++ schemes to be available as `code/build/edge_quantum` and select schemes via runtime flags.

- **Execution modes:**
  - **Native mode (SimMode::Native):** For smaller qubit counts that fit in RAM. Uses `cudaMalloc` for full state vector.
  - **Tiered mode (SimMode::Tiered_Async):** For larger qubit counts exceeding RAM. Uses UVM buffers + NVMe storage. Mode is auto-selected based on available RAM vs state size.
  - Auto selection logic: If `state_size < 80% of free RAM`, use Native; otherwise use Tiered.

- **Memory architecture (critical for Jetson):**
  - UVM (`cudaMallocManaged` with `cudaMemAttachHost`) is the only memory type that supports both POSIX I/O and cuStateVec on Jetson Tegra.
  - `cudaMalloc` (device memory): Works with cuStateVec but NOT with pread/pwrite.
  - `cudaMallocHost` (pinned): Works with I/O but cuStateVec rejects it on Tegra.
  - `cuComplex` is 8 bytes (not 16). State size = `(1ULL << qubits) * sizeof(cuComplex)`.

---

## Current Pipeline Implementation (4-Buffer Async Pipeline)

### Key Performance Characteristics (Samsung PM9A1 NVMe on Jetson Orin)

| Operation | Time (256MB) | Bandwidth | Notes |
|-----------|--------------|-----------|-------|
| **Read (pread)** | 100ms | ~2600 MB/s | O_DIRECT + UVM |
| **Write (pwrite)** | 100ms (SLC) / 450ms (TLC) | Varies | SLC cache ~1.75GB |
| **GPU compute** | 220ms | N/A | cuStateVec gate layer |
| **R+W concurrent** | 208ms each! | 50% penalty | **NVMe contention!** |

### Critical Finding: NVMe Read+Write Contention

**DO NOT overlap Read and Write!** When pread and pwrite run concurrently on NVMe:
- Each operation degrades from 100ms to ~200ms (2x slower)
- Root cause: NVMe controller bandwidth contention
- Solution: Serialize Read and Write, but overlap Write with GPU

### 4-Buffer Pipeline Design

```
Pipeline Flow (per gate layer, 32 chunks):

  C0: buf[0] - Read[0], GPU[0]
  C1: buf[1] - Read[1], Write[0] async + GPU[1] overlap
  C2: buf[2] - Read[2], Write[1] async + GPU[2] overlap  
  C3: buf[3] - Read[3], Write[2] async + GPU[3] overlap
  C4: buf[0] - WaitW[0], Read[4], Write[3] async + GPU[4] overlap
  ...

Key invariants:
  1. Read[c] is SYNC (no concurrent Write)
  2. Write[c-1] overlaps with GPU[c] (different HW!)
  3. WaitW on same buffer ensures buffer is free before Read
  4. 4 buffers = 3 iterations for Write to complete
```

### Why 4 Buffers?

With Write taking up to 450ms (TLC mode) and GPU taking 220ms:
- 2 buffers: WaitW = 450 - 220 = 230ms (waiting every iteration)
- 4 buffers: Write has 3 iterations to complete: 3 × 320ms = 960ms > 450ms → WaitW ≈ 0!

### Implementation (simulator.cpp::process_pipeline)

```cpp
std::future<ssize_t> write_futures[NUM_PIPELINE_BUFS];  // 4 futures

for (size_t c = 0; c < n_chunks; c++) {
    int buf = c % NUM_PIPELINE_BUFS;
    int prev_buf = (c > 0) ? ((c - 1) % NUM_PIPELINE_BUFS) : -1;
    
    // 1. Wait for previous Write on THIS buffer
    if (write_futures[buf].valid()) {
        write_futures[buf].get();
    }
    
    // 2. Read (sync, no concurrent Write!)
    chunk_mgr->read_chunk(*io_read_ptr, c, buf);
    
    // 3. Launch async Write for previous chunk (overlaps with GPU!)
    if (c > 0) {
        write_futures[prev_buf] = std::async(std::launch::async, ...);
    }
    
    // 4. GPU compute
    cudaStreamAttachMemAsync(stream, buffer, AttachGlobal);
    kernel(c, buffer, stream);
    cudaStreamSynchronize(stream);
    cudaStreamAttachMemAsync(stream, buffer, AttachHost);
}

// Drain remaining writes
for (auto& f : write_futures) if (f.valid()) f.get();
```

### Performance Results

| Config | 30q Depth=1 | 30q Depth=5 | Improvement |
|--------|-------------|-------------|-------------|
| Original 2-buffer | 16.3s | 85.99s | baseline |
| **4-buffer pipeline** | **10.5s** | **52.5s** | **35-39%** |

---

## I/O Strategy

- **O_DIRECT** for optimal NVMe performance (bypasses page cache completely)
- O_DIRECT + UVM works on Jetson! (unlike x86)
- File preallocation via `ftruncate` + `posix_fallocate` for sequential throughput
- `cudaStreamAttachMemAsync` to switch buffers:
  - `AttachHost` for I/O (pread/pwrite)
  - `AttachGlobal` for GPU compute

---

## Experimental History & Lessons Learned (CRITICAL)

### ✅ What Works

| Approach | Result | Notes |
|----------|--------|-------|
| **UVM (cudaMallocManaged + cudaMemAttachHost)** | ✅ | Only memory type supporting both I/O AND cuStateVec on Jetson |
| **O_DIRECT + UVM** | ✅ | ~15% faster than buffered I/O |
| **cudaStreamAttachMemAsync mode switching** | ✅ | AttachHost for I/O, AttachGlobal for GPU |
| **Write + GPU overlap** | ✅ | Different HW, true parallelism! |
| **4 buffers × 256MB** | ✅ | Optimal for hiding Write latency |
| **LZ4 compression (when storage tight)** | ✅ | Auto-enable when needed |
| **High-priority CUDA streams** | ✅ | cudaStreamCreateWithPriority |

### ❌ What Doesn't Work

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| **Read + Write concurrent** | ❌ | NVMe contention: each 100ms → 200ms! |
| **io_uring** | ❌ | Stability issues on Jetson kernel 5.10 |
| **cudaMallocHost (pinned)** | ❌ | cuStateVec rejects on Tegra |
| **cudaMalloc + pread** | ❌ | EFAULT (device memory not accessible) |
| **GDS (cuFile)** | ❌ | libcufile.so not available on Jetson |
| **128MB chunks** | ❌ | cuStateVec error (too small) |
| **5+ buffers** | ❌ | No additional benefit over 4 buffers |

### 🔬 Key Experiments

#### R+W Contention Test (Critical Finding!)
```
Test: What happens when pread and pwrite run concurrently?

Write only:  99ms (256MB)
Read only:   105ms (256MB)
R+W concurrent: 208ms EACH! (both take 2x longer!)

Conclusion: NVMe has severe R+W contention.
Must serialize Read and Write, overlap only Write+GPU.
```

#### SLC Cache Exhaustion
```
Samsung PM9A1 has ~1.75GB SLC cache:
- First 7 chunks (W6-W12): 100ms/chunk
- After cache: 450ms/chunk (TLC direct write)

Init phase writes 16GB → SLC cache exhausted
Subsequent writes may hit TLC speeds
4-buffer pipeline absorbs this variance
```

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ GPU (Ampere SM87)                                           │
│   └─ cuStateVec quantum gate execution: 220ms/chunk         │
│   └─ UVM buffer in AttachGlobal mode                        │
├─────────────────────────────────────────────────────────────┤
│ CPU (ARM Cortex-A78AE)                                      │
│   └─ Async Write thread (std::async)                        │
│   └─ Sync Read on UVM buffer                                │
├─────────────────────────────────────────────────────────────┤
│ Storage (Samsung PM9A1 NVMe)                                │
│   └─ O_DIRECT for page cache bypass                         │
│   └─ Read: 100ms/256MB, Write: 100-450ms/256MB              │
│   └─ 16GB state (30q) = 32 × 256MB chunks                   │
├─────────────────────────────────────────────────────────────┤
│ Memory (8GB Unified LPDDR5)                                 │
│   └─ UVM buffers: 4 × 256MB = 1GB                           │
│   └─ cuStateVec workspace: 128MB                            │
└─────────────────────────────────────────────────────────────┘

4-Buffer Pipeline (NO R+W overlap, YES Write+GPU overlap):
  for each chunk c:
    1. WaitW[buf] - wait for previous write on same buffer
    2. Read[c] - sync, no concurrent write
    3. Write[c-1] async - overlaps with GPU below!
    4. GPU[c] - cuStateVec execution
    5. Switch buffer back to Host mode

Time per chunk: Read(100ms) + GPU(220ms) ≈ 320ms
Write(100-450ms) overlaps with GPU → hidden!
```

---

## Where to Look for Bugs

- **Simulator entry & mode selection**: `code/src/main.cpp`, `code/src/simulator.cpp` constructor
- **Pipeline & buffer management**: `code/src/simulator.cpp::process_pipeline()`
- **UVM buffer allocation**: `code/src/chunk_manager.cpp`
- **File I/O**: `code/src/io_backend.hpp`
- **Benchmarks**: `code/comprehensive_benchmark.py`, `code/grand_benchmark.py`

---

## Quick Validation Commands

```bash
# Native mode (fits in RAM)
cd code && ./build/edge_quantum --qubits 26 --verify

# Tiered mode (uses NVMe)
cd code && ./build/edge_quantum --qubits 30 --verify

# Performance test
cd code && ./build/edge_quantum --qubits 30 --circuit random --depth 5
# Expected: ~52s with 4-buffer pipeline
```
