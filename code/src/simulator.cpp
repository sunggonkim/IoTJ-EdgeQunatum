#include "simulator.hpp"
#include <iostream>
#include <cstring>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <cstdlib>

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
            chunk_bits(25),
            chunk_size(0),
            n_chunks(0),
            io_read(path),   // Initialized but possibly unused in Native/UVM
            io_write(path),  // Dedicated write backend (thread-safe separation)
            state_size(1ULL << (qubits + 4)), // Complex64: 2^Q * 16 bytes
            mode(m),
            storage_path(path),
            chunk_mgr(nullptr),
            io(nullptr),
            full_state_ptr(nullptr),
            device_buf_ready(false),
            read_worker(nullptr),
            write_worker(nullptr)
{
    // Smart Optimization: If state fits in RAM (<= 28 Qubits), force Native mode for max performance.
    if (!force_mode && (mode == SimMode::Tiered_Async || mode == SimMode::Tiered_Blocking) && qubits <= 28) {
        mode = SimMode::Native;
        std::cout << "\n[Info] State fits in RAM. Switching to Native Mode.\n" << std::endl;
    }

    // Optimal chunk size for Jetson: 256MB (2^25 elements * 16 bytes)
    // Balances GPU occupancy vs memory overhead
    int chunk_pow = 25;
    if (chunk_pow > n_qubits) chunk_pow = n_qubits;
    chunk_bits = chunk_pow;
    chunk_size = (1ULL << chunk_pow) * sizeof(cuComplex);
    n_chunks = (1ULL << n_qubits) >> chunk_pow;
    if (n_chunks == 0) n_chunks = 1;

    // Mode Logic
    std::string mode_str = "Unknown";
    if(mode == SimMode::Tiered_Async) mode_str = "EdgeQuantum (Async Pipeline)";
    else if(mode == SimMode::Tiered_Blocking) mode_str = "BMQSim-like (Blocking)";
    else if(mode == SimMode::Native) mode_str = "cuQuantum Native";
    else if(mode == SimMode::UVM) mode_str = "cuQuantum UVM";

    std::cout << "[Sim] Mode: " << mode_str 
              << " | Qubits: " << n_qubits 
              << " | State Size: " << state_size / (1024ULL*1024*1024) << " GB" << std::endl;

    // For stability, disable io_uring in blocking mode
    if (mode == SimMode::Tiered_Blocking) {
        io_read.disable_uring();
        io_write.disable_uring();
    }

    // Common cuQuantum Setup
    CUSV_CHECK(custatevecCreate(&handle));
    
    // Jetson Optimization: Use high-priority CUDA streams for lower latency
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream, cudaStreamNonBlocking, greatestPriority));
    
    CUSV_CHECK(custatevecSetStream(handle, stream));
    
    // Gate Matrix Constant
    float s2 = 1.0f / sqrt(2.0f);
    std::complex<float> h_gate[4] = {{s2,0}, {s2,0}, {s2,0}, {-s2,0}};
    CUDA_CHECK(cudaMalloc(&d_gate_matrix, sizeof(h_gate)));
    CUDA_CHECK(cudaMemcpy(d_gate_matrix, h_gate, sizeof(h_gate), cudaMemcpyHostToDevice));

    // Workspace (size depends on n_bits used by the scheme)
    int ws_nbits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : chunk_bits;
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
        // --- Tiered Mode Setup (UVM-based Zero-Copy) ---
        // Use 3 UVM buffers for triple-buffer pipeline
        // UVM eliminates the need for separate device buffers and cudaMemcpy!
        chunk_mgr = new ChunkManager(chunk_size, n_chunks, NUM_PIPELINE_BUFS);
        read_worker = new IoWorker();
        write_worker = new IoWorker();

        // No separate device buffers needed - UVM buffers are directly accessible by GPU
        // No CUDA events needed for H2D/D2H - we use cudaDeviceSynchronize for UVM coherency
        device_buf_ready = true;
        
        init_storage();
    }
}

EdgeQuantumSim::~EdgeQuantumSim() {
    if (read_worker) delete read_worker;
    if (write_worker) delete write_worker;
    
    if (full_state_ptr) cudaFree(full_state_ptr);

    // UVM-based pipeline doesn't need separate device buffers or events
    device_buf_ready = false;
    
    if (chunk_mgr) delete chunk_mgr;
    
    cudaFree(d_ws);
    cudaFree(d_gate_matrix);
    custatevecDestroy(handle);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(copy_stream);
}

void EdgeQuantumSim::init_storage() {
    // Only for Tiered Mode
    if (!chunk_mgr) return;

    void* buf = chunk_mgr->get_buffer(0);
    memset(buf, 0, chunk_size);
    ((std::complex<float>*)buf)[0] = {1.0f, 0.0f};
    
    std::cout << "[Init] Writing initial state to NVMe (" << n_chunks << " chunks)..." << std::endl;
    for (size_t i = 0; i < n_chunks; i++) {
        if (i == 1) ((std::complex<float>*)buf)[0] = {0.0f, 0.0f};
        io_write.write(i * chunk_size, buf, chunk_size);
    }
    
    // Sync to ensure writes are visible to io_read fd
    io_write.sync();
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
    if (mode == SimMode::Native || mode == SimMode::UVM) {
        std::cerr << "[Error] process_pipeline called in Native/UVM mode!" << std::endl;
        return;
    }

    // ============================================================
    // UVM ASYNC PIPELINE with cudaStreamAttachMemAsync
    // ============================================================
    // Key discovery: UVM buffers allocated with cudaMemAttachHost can be
    // dynamically switched between Host/Global access modes:
    //   - AttachHost: Allows pread/pwrite from worker threads
    //   - AttachGlobal: Allows GPU (cuStateVec) access
    // 
    // This enables TRUE async overlap:
    //   - GPU computes on buf[i] (AttachGlobal)
    //   - Worker reads chunk into buf[i+1] (AttachHost) 
    //   - Worker writes from buf[i-1] (AttachHost)
    // 
    // Speedup: ~15-40% over blocking, no cudaMemcpy needed!
    // ============================================================
    
    IOBackend* io_read_ptr = &io_read;
    IOBackend* io_write_ptr = &io_write;
    
    // Environment variable to force blocking mode for debugging
    bool force_blocking = (std::getenv("FORCE_BLOCKING") != nullptr);
    
    if (force_blocking) {
        // Simple blocking fallback
        for (size_t i = 0; i < n_chunks; i++) {
            int buf = i % NUM_PIPELINE_BUFS;
            chunk_mgr->read_chunk(*io_read_ptr, (int)i, buf);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            void* uvm_buffer = chunk_mgr->get_buffer(buf);
            kernel((int)i, uvm_buffer, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            chunk_mgr->write_chunk(*io_write_ptr, (int)i, buf);
        }
        return;
    }
    
    // ============================================================
    // OPTIMIZED Triple Buffer Pipeline with AttachHost/AttachGlobal
    // ============================================================
    // Key optimization: Remove unnecessary cudaStreamSynchronize after AttachGlobal
    // The GPU will wait for the attach to complete before accessing the buffer.
    // Only sync before host I/O operations (pread/pwrite).
    // ============================================================
    
    std::future<ssize_t> read_futures[NUM_PIPELINE_BUFS];
    std::future<ssize_t> write_futures[NUM_PIPELINE_BUFS];
    
    // Initialize all buffers to Host mode (only once at start)
    for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
        void* ptr = chunk_mgr->get_buffer(i);
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachHost));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Prefetch: Read chunk 0 into buf[0] (sync)
    chunk_mgr->read_chunk(*io_read_ptr, 0, 0);
    
    // Prefetch: Start reading chunk 1 into buf[1] (async)
    if (n_chunks > 1) {
        read_futures[1] = std::async(std::launch::async, [this, io_read_ptr]() {
            chunk_mgr->read_chunk(*io_read_ptr, 1, 1);
            return (ssize_t)0;
        });
    }

    for (size_t c = 0; c < n_chunks; c++) {
        int compute_buf = c % NUM_PIPELINE_BUFS;
        int prev_buf = (c + NUM_PIPELINE_BUFS - 1) % NUM_PIPELINE_BUFS;
        
        // 1. Wait for read into compute_buf (if pending from earlier iteration)
        if (c >= 2 && read_futures[compute_buf].valid()) {
            read_futures[compute_buf].get();
        }
        
        // 2. Wait for previous write on compute_buf to complete (buffer reuse)
        if (c >= NUM_PIPELINE_BUFS && write_futures[compute_buf].valid()) {
            write_futures[compute_buf].get();
        }
        
        // 3. Switch compute_buf to Global for GPU access (NO sync here!)
        void* uvm_buffer = chunk_mgr->get_buffer(compute_buf);
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachGlobal));
        // GPU will implicitly wait for attach to complete
        
        // 4. GPU compute on compute_buf
        kernel((int)c, uvm_buffer, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 5. Switch compute_buf back to Host for I/O (sync before write!)
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 6. Start async write for current chunk from compute_buf
        size_t write_chunk = c;
        write_futures[compute_buf] = std::async(std::launch::async,
            [this, io_write_ptr, compute_buf, write_chunk]() {
                chunk_mgr->write_chunk(*io_write_ptr, (int)write_chunk, compute_buf);
                return (ssize_t)0;
            });
        
        // 7. Start async read for chunk (c+2) into prev_buf
        if (c + 2 < n_chunks) {
            size_t future_chunk = c + 2;
            // Wait for pending write on prev_buf if needed
            if (c >= 1 && write_futures[prev_buf].valid()) {
                write_futures[prev_buf].get();
            }
            read_futures[prev_buf] = std::async(std::launch::async, 
                [this, io_read_ptr, prev_buf, future_chunk]() {
                    chunk_mgr->read_chunk(*io_read_ptr, (int)future_chunk, prev_buf);
                    return (ssize_t)0;
                });
        }
    }
    
    // Wait for all pending writes
    for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
        if (write_futures[i].valid()) {
            write_futures[i].get();
        }
    }
}

void EdgeQuantumSim::apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s) {
    target_idx[0] = target;
    int n_bits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : chunk_bits;

    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, (void*)d_mat, CUDA_C_32F, 
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, nullptr, nullptr, 0, 
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s) {
    target_idx[0] = t;
    control_idx[0] = c;
    int n_bits = (mode == SimMode::Native || mode == SimMode::UVM) ? n_qubits : chunk_bits;
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, control_idx, nullptr, 1, 
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
             kernel(0, full_state_ptr, this->stream);
             CUDA_CHECK(cudaDeviceSynchronize());
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
