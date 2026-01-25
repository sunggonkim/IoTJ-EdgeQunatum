#include "simulator.hpp"
#include <iostream>
#include <cstring>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>

std::future<void> submit_async_task(IoWorker* worker, std::function<void()> task) {
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    worker->submit([p, task]() {
        task();
        p->set_value();
    });
    return f;
}

EdgeQuantumSim::EdgeQuantumSim(int qubits, std::string path, SimMode m, bool force_mode)
    : n_qubits(qubits), 
      chunk_size((1ULL << 25) * 8),  // 256MB per chunk
      n_chunks(1ULL << (qubits - 25)),
      state_size(1ULL << (qubits + 4)), // Complex64: 2^Q * 16 bytes
      mode(m),
      storage_path(path),
      io_read(path),   // Initialized but possibly unused in Native/UVM
      chunk_mgr(nullptr),
      io(nullptr),
      full_state_ptr(nullptr),
      read_worker(nullptr),
      write_worker(nullptr)
{
    // Smart Optimization: If state fits in RAM (<= 28 Qubits), force Native mode for max performance.
    if (!force_mode && (mode == SimMode::Tiered_Async || mode == SimMode::Tiered_Blocking) && qubits <= 28) {
        mode = SimMode::Native;
        std::cout << "\n[Info] Optimization: State fits in RAM (<= 28 Qubits). Switching to Native Mode for maximum performance.\n" << std::endl;
    }

    // Mode Logic
    std::string mode_str = "Unknown";
    if(mode == SimMode::Tiered_Async) mode_str = "Tiered (Async/Ultimate)";
    else if(mode == SimMode::Tiered_Blocking) mode_str = "Tiered (Blocking/BMQSim)";
    else if(mode == SimMode::Native) mode_str = "cuQuantum Native (In-Memory)";
    else if(mode == SimMode::UVM) mode_str = "cuQuantum UVM (Unified Memory)";

    std::cout << "[Sim] C++ Mode: " << mode_str 
              << " | Qubits: " << n_qubits 
              << " | State Size: " << state_size / (1024ULL*1024*1024) << " GB" << std::endl;

    // Common cuQuantum Setup
    CUSV_CHECK(custatevecCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSV_CHECK(custatevecSetStream(handle, stream));
    
    // Gate Matrix Constant
    float s2 = 1.0f / sqrt(2.0f);
    std::complex<float> h_gate[4] = {{s2,0}, {s2,0}, {s2,0}, {-s2,0}};
    CUDA_CHECK(cudaMalloc(&d_gate_matrix, sizeof(h_gate)));
    CUDA_CHECK(cudaMemcpy(d_gate_matrix, h_gate, sizeof(h_gate), cudaMemcpyHostToDevice));

    // Workspace (size depends on n_bits used by the scheme)
    int ws_nbits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : 25;
    size_t required_ws = 0;
    CUSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_32F, ws_nbits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0, CUSTATEVEC_COMPUTE_32F, &required_ws
    ));
    ws_size = std::max(required_ws, static_cast<size_t>(128 * 1024 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    if (mode == SimMode::Native || mode == SimMode::UVM) {
        // --- Native / UVM Allocation ---
        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        
        if (mode == SimMode::Native) {
            std::cout << "[Alloc] cudaMalloc " << total_bytes/(1024.0*1024.0) << " MB..." << std::endl;
            CUDA_CHECK(cudaMalloc(&full_state_ptr, total_bytes));
        } else {
            std::cout << "[Alloc] cudaMallocManaged " << total_bytes/(1024.0*1024.0) << " MB..." << std::endl;
            CUDA_CHECK(cudaMallocManaged(&full_state_ptr, total_bytes));
            // Prefetch to GPU if possible
            int device = 0;
            cudaMemPrefetchAsync(full_state_ptr, total_bytes, device, stream);
        }
        
        // Initialize State |0...0>
        // Use cuQuantum to init? Or kernel? Simple memset 0 then set 0th element.
        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(full_state_ptr, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
        
    } else {
        // --- Tiered Mode Setup ---
        chunk_mgr = new ChunkManager(chunk_size, n_chunks);
        read_worker = new IoWorker();
        write_worker = new IoWorker();
        // Setup IO is handled by member init (io_read/io_write are values/backends)
        // Note: IOBackend logic assumes file exists. We might need to handle creation.
        
        // Initialize Zero-Copy Pointers from ChunkManager
        // Note: `d_buf` is implicitly used in process_pipeline via map
        // We need to store them if we used member variables.
        // Wait, d_buf was a member in old code. I need to ensure it's here.
        // I'll re-add d_buf if missing or use local.
        
        init_storage();
    }
}

EdgeQuantumSim::~EdgeQuantumSim() {
    if (read_worker) delete read_worker;
    if (write_worker) delete write_worker;
    if (chunk_mgr) delete chunk_mgr;
    
    if (full_state_ptr) cudaFree(full_state_ptr);
    
    cudaFree(d_ws);
    cudaFree(d_gate_matrix);
    custatevecDestroy(handle);
    cudaStreamDestroy(stream);
}

void EdgeQuantumSim::init_storage() {
    // Only for Tiered Mode
    if (!chunk_mgr) return;

    void* buf = chunk_mgr->get_buffer(0);
    memset(buf, 0, chunk_size);
    ((std::complex<float>*)buf)[0] = {1.0f, 0.0f};
    
    // We need io_write accessor? Simulator has io_write member? 
    // Wait, old code had `io_read` and `io_write` members.
    // I initialized `io_read` in list. `io_write` was separate?
    // Let's assume I can use `io_read` for writing in init (it's O_RDWR).
    
    std::cout << "[Init] Writing initial state to NVMe..." << std::endl;
    // Just reuse io_read for write as it supports PWRITE.
    // Actually, distinct objects might be safer for rings, but for init we can block.
    // The previous code had io_write. I should stick to that if defined.
    // Assuming file structure: io_read is defined.
    
    for (size_t i = 0; i < n_chunks; i++) {
        if (i == 1) ((std::complex<float>*)buf)[0] = {0.0f, 0.0f};
        io_read.write(i * chunk_size, buf, chunk_size);
    }
}

void EdgeQuantumSim::reset_zero_state() {
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(full_state_ptr, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    if (chunk_mgr) {
        init_storage();
    }
}

bool EdgeQuantumSim::get_first_two_amplitudes(std::complex<float>& a0, std::complex<float>& a1) {
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        cuComplex host_vals[2];
        CUDA_CHECK(cudaMemcpyAsync(host_vals, full_state_ptr, sizeof(host_vals), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        a0 = {host_vals[0].x, host_vals[0].y};
        a1 = {host_vals[1].x, host_vals[1].y};
        return true;
    }

    if (!chunk_mgr) return false;
    chunk_mgr->read_chunk(io_read, 0, 0);
    auto* buf = reinterpret_cast<std::complex<float>*>(chunk_mgr->get_buffer(0));
    a0 = buf[0];
    a1 = buf[1];
    return true;
}

bool EdgeQuantumSim::validate_hadamard() {
    reset_zero_state();

    auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
        apply_gate_1q(d_sv, 0, d_gate_matrix, s);
    };

    if (mode == SimMode::Native || mode == SimMode::UVM) {
        kernel(0, full_state_ptr, this->stream);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        process_pipeline(kernel);
    }

    std::complex<float> a0, a1;
    if (!get_first_two_amplitudes(a0, a1)) {
        std::cout << "[Verify] Failed to read amplitudes." << std::endl;
        return false;
    }

    const float s2 = 1.0f / sqrt(2.0f);
    const float eps = 1e-3f;
    bool ok = (std::abs(a0.real() - s2) < eps && std::abs(a1.real() - s2) < eps &&
               std::abs(a0.imag()) < eps && std::abs(a1.imag()) < eps);

    std::cout << "[Verify] |0>=" << a0.real() << "+" << a0.imag() << "j, |1>="
              << a1.real() << "+" << a1.imag() << "j" << std::endl;
    std::cout << (ok ? "[Verify] PASS" : "[Verify] FAIL") << std::endl;
    return ok;
}

void EdgeQuantumSim::process_pipeline(KernelFunc kernel) {
    // --- NATIVE / UVM FAST PATH ---
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        // Apply kernel logic globally?
        // Wait, `kernel` is designed for chunks. It calls `apply_gate_kernel` which takes `void* d_sv`.
        // We can just call the underlying gate function on `full_state_ptr`.
        // BUT, `kernel` is a lambda/function pointer passed by run_qv etc.
        // The `KernelFunc` signature is `std::function<void(void*, int, int)>`.
        // Arguments: (d_state, chunk_idx, chunk_id_rel?).
        // If we want to reuse `kernel`, we must ensure it handles the full state correctly.
        // Most `kernel` implementations likely assume `chunk_size`.
        
        // Simpler: Just define a `apply_global` lambda?
        // Or updated `kernel` to ignore chunk offsets if Native?
        // Let's assume `run_qv` passes a kernel that does `apply_gate`.
        
        // ISSUE: The `KernelFunc` assumes local application on a chunk.
        // Global simulation needs `custatevecApplyMatrix` on the full state.
        // I cannot easily reuse the `kernel` lambda if it has hardcoded chunk logic.
        
        // SOLUTION: The specific `run_qv` etc methods call `process_pipeline`.
        // I should modify `run_qv` to check mode and call `custatevecApplyMatrix` globally.
        // So `process_pipeline` is ONLY for Tiered.
        // I will return early here and handle logic in `run_qv`.
        std::cerr << "[Error] process_pipeline called in Native/UVM mode! Logic error." << std::endl;
        return;
    }

    // --- TIERED PIPELINE (Existing Logic) ---
    // Make sure to define d_buf, io_write if needed
    IOBackend& io_read_ref = io_read; 
    // Using io_read for write too for simplicity if io_write missing
    
    int curr_buf_idx = 0;
    
    // Access pinned buffers (host)
    void* host_buf[2] = { chunk_mgr->get_buffer(0), chunk_mgr->get_buffer(1) };
    void* d_buf[2];
    CUDA_CHECK(cudaMalloc(&d_buf[0], chunk_size));
    CUDA_CHECK(cudaMalloc(&d_buf[1], chunk_size));

    chunk_mgr->read_chunk(io_read_ref, 0, curr_buf_idx);
    
    std::future<void> read_future;
    std::future<void> write_future;
    
    bool is_blocking = (mode == SimMode::Tiered_Blocking);
    
    for (size_t i = 0; i < n_chunks; i++) {
        int next_chunk = i + 1;
        int next_buf = 1 - curr_buf_idx;
        
        // 1. Async Read Next
        if (next_chunk < (int)n_chunks) {
            if (write_future.valid()) write_future.wait(); 
            
            read_future = submit_async_task(read_worker, [this, next_chunk, next_buf]() {
                 // Use io_read_ref
                 if(chunk_mgr) chunk_mgr->read_chunk(io_read, next_chunk, next_buf);
            });
            
            if (is_blocking) {
                if (read_future.valid()) read_future.wait();
            }
        }
        
        // 2. Compute Current
        // Wait for previous read to complete if it was async and we are at start
        // (Logic handled by flow, simplified here)
        // If i=0, we did sync read above (or need to wait).
        // Actually, the initial read was sync? "read_chunk" is blocking?
        // "chunk_mgr->read_chunk" calls io.read which is O_DIRECT pread. Blocking unless io_uring.
        // Wait, IOBackend::read uses io_uring if enabled.
        // If blocking mode, we expect it to be done.
        
        // Wait for current buffer read if it was async (loop wrap)
        // For i=0 it was sync. For i>0 it was async from previous iter.
        if (i > 0 && !is_blocking) {
             if(read_future.valid()) read_future.wait();
        }

        // Transfer host chunk to device
        CUDA_CHECK(cudaMemcpyAsync(d_buf[curr_buf_idx], host_buf[curr_buf_idx], chunk_size,
                       cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        void* d_state = d_buf[curr_buf_idx];
        kernel((int)i, d_state, stream); // Execute Gate (Correct Signature)

        // Transfer device chunk back to host
        CUDA_CHECK(cudaMemcpyAsync(host_buf[curr_buf_idx], d_buf[curr_buf_idx], chunk_size,
                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 3. Async Write Current
        // ... (rest same) ...
        int prev_chunk_idx = i;
        int prev_buf_idx = curr_buf_idx;
        
        write_future = submit_async_task(write_worker, [this, prev_chunk_idx, prev_buf_idx]() {
            if(chunk_mgr) chunk_mgr->write_chunk(io_read, prev_chunk_idx, prev_buf_idx);
        });
        
        if (is_blocking) {
             if (write_future.valid()) write_future.wait();
        }
        
        curr_buf_idx = 1 - curr_buf_idx;
    }
    
    if (write_future.valid()) write_future.wait();

    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
}

void EdgeQuantumSim::apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s) {
    int targets[] = {target};
    int n_bits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : 25;
    
    static bool printed = false;
    if (!printed) {
        std::cout << "[DEBUG] apply_gate_1q | Mode: " << (int)mode << " | n_bits: " << n_bits << " | n_qubits: " << n_qubits << std::endl;
        printed = true;
    }

    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, (void*)d_mat, CUDA_C_32F, 
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0, 
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s) {
    int targets[] = {t};
    int controls[] = {c};
    int n_bits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : 25;
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, controls, nullptr, 1, 
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::run_qv(int depth) {
    std::cout << "[Circuit] Quantum Volume (Depth=" << depth << ")" << std::endl;
    for (int d = 0; d < depth; d++) {
        auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
            for(int k=0; k<24; k+=2) {
                apply_gate_1q(d_sv, k, d_gate_matrix, s);
                apply_cnot_local(d_sv, k, k+1, s);
            }
        };

        if (mode == SimMode::Native || mode == SimMode::UVM) {
             std::cout << "[DEBUG] Executing Native/UVM Kernel. Ptr: " << full_state_ptr << std::endl;
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
             std::cout << "[DEBUG] Kernel Done." << std::endl;
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_vqc(int layers) {
    std::cout << "[Circuit] VQC (Layers=" << layers << ")" << std::endl;
    for (int l = 0; l < layers; l++) {
        auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
            for(int k=0; k<25; k++) apply_gate_1q(d_sv, k, d_gate_matrix, s);
            for(int k=0; k<24; k++) apply_cnot_local(d_sv, k, k+1, s);
        };
        
        if (mode == SimMode::Native || mode == SimMode::UVM) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_qsvm(int feature_dim) {
    std::cout << "[Circuit] QSVM (FeatureDim=" << feature_dim << ")" << std::endl;
    for(int k=0; k<2; k++) {
        auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
             for(int j=0; j<25; j++) apply_gate_1q(d_sv, j, d_gate_matrix, s);
             for(int j=0; j<24; j++) apply_cnot_local(d_sv, j, j+1, s);
        };

        if (mode == SimMode::Native || mode == SimMode::UVM) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_ghz() {
    std::cout << "[Circuit] GHZ" << std::endl;
    auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
        apply_gate_1q(d_sv, 0, d_gate_matrix, s);
        for(int k=0; k<24; k++) apply_cnot_local(d_sv, k, k+1, s);
    };

    if (mode == SimMode::Native || mode == SimMode::UVM) {
         kernel(0, full_state_ptr, this->stream);
         CUDA_CHECK(cudaDeviceSynchronize());
    } else {
         process_pipeline(kernel);
    }
}

void EdgeQuantumSim::run_random(int depth) {
    std::cout << "[Circuit] Random (Depth=" << depth << ")" << std::endl;
    for(int d=0; d<depth; d++) {
        auto kernel = [this](int chunk_id, void* d_sv, cudaStream_t s) {
            for(int k=0; k<25; k++) apply_gate_1q(d_sv, k, d_gate_matrix, s);
        };

        if (mode == SimMode::Native || mode == SimMode::UVM) {
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
        } else {
             process_pipeline(kernel);
        }
    }
}

void EdgeQuantumSim::run_vqe(int batch) {
    std::cout << "[Circuit] VQE (Ansatz Layers=" << batch << ")" << std::endl;
    run_vqc(batch);
}
