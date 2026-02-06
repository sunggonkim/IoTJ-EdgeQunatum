#include "simulator.hpp"
#include <iostream>
#include <cstring>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <sys/statvfs.h>

std::future<void> submit_async_task(IoWorker* worker, std::function<void()> task) {
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    worker->submit([p, task]() {
        task();
        p->set_value();
    });
    return f;
}

// Helper: Get available GPU memory in bytes
static size_t get_available_gpu_memory() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

// Helper: Get available disk space in bytes
static size_t get_available_disk_space(const std::string& path) {
    struct statvfs stat;
    std::string dir = path.substr(0, path.rfind('/'));
    if (dir.empty()) dir = ".";
    if (statvfs(dir.c_str(), &stat) != 0) {
        return 0; // Error - assume no space
    }
    return (size_t)stat.f_bavail * stat.f_frsize;
}

EdgeQuantumSim::EdgeQuantumSim(int qubits, std::string path, std::string mode_str, bool force_mode)
        : n_qubits(qubits),
            chunk_bits(25),
            chunk_size(0),
            n_chunks(0),
            io_read(path),   // Initialized but possibly unused in Native
            io_write(path),  // Dedicated write backend (thread-safe separation)
            state_size((1ULL << qubits) * sizeof(cuComplex)), // 2^Q * 8 bytes (cuComplex is 8 bytes)
            mode(SimMode::Native),  // Default, will be selected below
            storage_path(path),
            compressed_storage(nullptr),
            use_compression(false),
            chunk_mgr(nullptr),
            io(nullptr),
            full_state_ptr(nullptr),
            device_buf_ready(false),
            read_worker(nullptr),
            write_worker(nullptr)
{
    // ============================================================
    // MODE SELECTION
    // ============================================================
    // mode_str: "auto", "native", "uvm", "tiered"
    // force_mode: if true, force the mode even if memory insufficient (may OOM)
    // 
    // native/uvm -> SimMode::Native (full state in GPU memory)
    // tiered     -> SimMode::Tiered (NVMe-backed chunked pipeline)
    // auto       -> Select based on available GPU memory
    // ============================================================
    size_t state_bytes = (1ULL << qubits) * sizeof(cuComplex);
    size_t gpu_free = get_available_gpu_memory();
    size_t gpu_threshold = (gpu_free * 80) / 100;  // 80% of free GPU memory
    
    if (mode_str == "native" || mode_str == "uvm") {
        // Native/UVM mode requested
        if (!force_mode && state_bytes > gpu_threshold) {
            std::cerr << "\n[ERROR] Mode '" << mode_str << "' requested but state (" 
                      << state_bytes / (1024*1024) << " MB) exceeds GPU memory (" 
                      << gpu_free / (1024*1024) << " MB free)." << std::endl;
            std::cerr << "Use --force-mode to attempt anyway (will likely OOM)." << std::endl;
            throw std::runtime_error("OOM: Insufficient GPU memory for native mode");
        }
        mode = SimMode::Native;
        std::cout << "\n[Mode] " << mode_str << " -> Using Native mode (cuQuantum direct)\n" << std::endl;
    } else if (mode_str == "tiered" || mode_str == "blocking" || mode_str == "async") {
        // Tiered mode requested (blocking/async are legacy names)
        mode = SimMode::Tiered;
        std::cout << "\n[Mode] " << mode_str << " -> Using Tiered mode (NVMe pipeline)\n" << std::endl;
    } else {
        // Auto mode: Select based on available GPU memory
        if (state_bytes <= gpu_threshold) {
            mode = SimMode::Native;
            std::cout << "\n[Auto] State (" << state_bytes / (1024*1024) << " MB) fits in GPU memory (" 
                      << gpu_free / (1024*1024) << " MB free). Using Native mode.\n" << std::endl;
        } else {
            mode = SimMode::Tiered;
            std::cout << "\n[Auto] State (" << state_bytes / (1024*1024) << " MB) exceeds GPU memory (" 
                      << gpu_free / (1024*1024) << " MB free). Using Tiered mode.\n" << std::endl;
        }
    }

    // Optimal chunk size for Jetson: 256MB (2^25 elements * 8 bytes)
    // Balances GPU occupancy vs memory overhead
    int chunk_pow = 25;
    if (chunk_pow > n_qubits) chunk_pow = n_qubits;
    chunk_bits = chunk_pow;
    chunk_size = (1ULL << chunk_pow) * sizeof(cuComplex);
    n_chunks = (1ULL << n_qubits) >> chunk_pow;
    if (n_chunks == 0) n_chunks = 1;

    // Mode Logic - display
    std::string mode_display = (mode == SimMode::Native) ? "Native (cuQuantum)" : "Tiered (EdgeQuantum)";

    std::cout << "[Sim] Mode: " << mode_display 
              << " | Qubits: " << n_qubits 
              << " | State Size: " << state_size / (1024ULL*1024*1024) << " GB" << std::endl;

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
    int ws_nbits = (mode == SimMode::Native) ? n_qubits : chunk_bits;
    size_t required_ws = 0;
    CUSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_32F, ws_nbits, d_gate_matrix, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0, CUSTATEVEC_COMPUTE_32F, &required_ws
    ));
    ws_size = std::max(required_ws, static_cast<size_t>(128 * 1024 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

    if (mode == SimMode::Native) {
        // --- Native Allocation (fastest path) ---
        size_t total_bytes = (1ULL << n_qubits) * sizeof(cuComplex);
        std::cout << "[Alloc] cudaMalloc " << total_bytes/(1024.0*1024.0) << " MB..." << std::endl;
        CUDA_CHECK(cudaMalloc(&full_state_ptr, total_bytes));
        
        // Initialize State |0...0>
        CUDA_CHECK(cudaMemset(full_state_ptr, 0, total_bytes));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpy(full_state_ptr, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
        
    } else {
        // --- Tiered Mode Setup (UVM-based Zero-Copy) ---
        // Use 4 UVM buffers for quad-buffer pipeline
        // UVM eliminates the need for separate device buffers and cudaMemcpy!
        chunk_mgr = new ChunkManager(chunk_size, n_chunks, NUM_PIPELINE_BUFS);
        read_worker = new IoWorker();
        write_worker = new IoWorker();

        // Check if compression is needed based on available disk space
        use_compression = should_enable_compression(path, state_size);
        if (use_compression) {
            compressed_storage = new CompressedStorage(path + ".lz4", chunk_size, n_chunks, true);
        }

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
    
    if (compressed_storage) delete compressed_storage;
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
    
    std::cout << "[Init] Writing initial state to NVMe (" << n_chunks << " chunks";
    if (use_compression) std::cout << ", LZ4 ping-pong compressed";
    std::cout << ")..." << std::endl;
    
    size_t total_written = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        if (i == 1) ((std::complex<float>*)buf)[0] = {0.0f, 0.0f};
        
        if (use_compression && compressed_storage) {
            // Use init_write_chunk for ping-pong variable-size compression
            compressed_storage->init_write_chunk(i, buf);
        } else {
            io_write.write(i * chunk_size, buf, chunk_size);
            total_written += chunk_size;
        }
        
        // Progress indicator for large states
        if (n_chunks >= 100 && (i+1) % (n_chunks/10) == 0) {
            std::cout << "[Init] Progress: " << (100*(i+1)/n_chunks) << "%" << std::endl;
        }
    }
    
    // Sync/finalize init phase
    if (use_compression && compressed_storage) {
        compressed_storage->finish_init();  // Finalize ping-pong init, set read_file=A
    } else {
        io_write.sync();
    }
}

void EdgeQuantumSim::reset_zero_state() {
    if (mode == SimMode::Native) {
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
    if (mode == SimMode::Native) {
        cuComplex host_vals[2];
        CUDA_CHECK(cudaMemcpyAsync(host_vals, full_state_ptr, sizeof(host_vals), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        a0 = {host_vals[0].x, host_vals[0].y};
        a1 = {host_vals[1].x, host_vals[1].y};
        return true;
    }

    if (!chunk_mgr) return false;
    
    // Read chunk 0 (using compressed storage if enabled)
    if (use_compression && compressed_storage) {
        compressed_storage->read_chunk(0, chunk_mgr->get_buffer(0));
    } else {
        chunk_mgr->read_chunk(io_read, 0, 0);
    }
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

    if (mode == SimMode::Native) {
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
    if (mode == SimMode::Native) {
        std::cerr << "[Error] process_pipeline called in Native mode!" << std::endl;
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
    
    // Helper lambdas for compressed/uncompressed I/O
    auto read_chunk_data = [this, io_read_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->read_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->read_chunk(*io_read_ptr, (int)chunk_idx, buf_idx);
        }
    };
    
    auto write_chunk_data = [this, io_write_ptr](size_t chunk_idx, int buf_idx) {
        if (use_compression && compressed_storage) {
            compressed_storage->write_chunk(chunk_idx, chunk_mgr->get_buffer(buf_idx));
        } else {
            chunk_mgr->write_chunk(*io_write_ptr, (int)chunk_idx, buf_idx);
        }
    };
    
    // Environment variable to force blocking mode (BMQSim-like behavior)
    bool force_blocking = (std::getenv("FORCE_BLOCKING") != nullptr);
    
    // BMQSim-like blocking mode (only if explicitly requested)
    if (force_blocking) {
        std::cout << "[Pipeline] Blocking mode (BMQSim-like)" << std::endl;
        for (size_t i = 0; i < n_chunks; i++) {
            int buf = i % NUM_PIPELINE_BUFS;
            read_chunk_data(i, buf);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            void* uvm_buffer = chunk_mgr->get_buffer(buf);
            CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachGlobal));
            kernel((int)i, uvm_buffer, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachHost));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            write_chunk_data(i, buf);
        }
        // Swap ping-pong files after each layer (for compression)
        if (use_compression && compressed_storage) {
            compressed_storage->swap_files();
        }
        return;
    }
    
    // EdgeQuantum: Async 4-buffer pipeline
    std::cout << "[Pipeline] Async 4-buffer (EdgeQuantum";
    if (use_compression) std::cout << ", LZ4";
    std::cout << ")" << std::endl;
    
    // ============================================================
    // OPTIMAL Pipeline: GPU + Write overlap, Read after Write done
    // ============================================================
    // UPDATED: 4-buffer pipeline for TLC SSD (Write: 450ms, GPU: 220ms)
    // ============================================================
    // Key findings:
    //   - Write: 450ms (TLC direct after SLC cache exhausted!)
    //   - Read: 100ms, GPU: 220ms
    //   - R+W concurrent: 2x slower (NVMe contention)
    //
    // With 4 buffers, Write[c-1] has 3 iterations to complete:
    //   - iter[c]: R[c] + GPU[c] = 100 + 220 = 320ms
    //   - iter[c+1]: 320ms (using different buf)
    //   - By iter[c+2], Write[c-1] waited 640ms > 450ms ✓
    //
    // Flow:
    //   C0: buf[0], Read[0], GPU[0]
    //   C1: buf[1], Read[1], Write[0] async, GPU[1]
    //   C2: buf[2], Read[2], Write[1] async, GPU[2]
    //   C3: buf[3], Read[3], WaitW[0], Write[2] async, GPU[3]
    //   C4: buf[0], Read[4], WaitW[1], Write[3] async, GPU[4]
    //   ...
    // ============================================================
    
    // Store write futures for each buffer
    std::future<ssize_t> write_futures[NUM_PIPELINE_BUFS];
    
    // Initialize all buffers to Host mode
    for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
        void* ptr = chunk_mgr->get_buffer(i);
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachHost));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Track previous buffer for Write overlap
    int prev_completed_buf = -1;
    
    for (size_t c = 0; c < n_chunks; c++) {
        int buf = c % NUM_PIPELINE_BUFS;
        void* uvm_buffer = chunk_mgr->get_buffer(buf);
        
        // ============================================================
        // OPTIMIZED PIPELINE: GPU+Write overlap (not Read+Write!)
        // ============================================================
        // NVMe contention: R+W concurrent = 2x slower each!
        // Solution: 
        //   1. Complete any pending Write BEFORE Read (avoid R+W overlap)
        //   2. Read
        //   3. Start async Write for PREVIOUS chunk (overlaps with THIS GPU)
        //   4. GPU compute
        // ============================================================
        
        // 1. Wait for THIS buffer's previous Write (buffer reuse safety)
        if (write_futures[buf].valid()) {
            write_futures[buf].get();
        }
        
        // 2. CRITICAL: Wait for ANY pending write to avoid R+W contention!
        for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
            if (write_futures[i].valid()) {
                write_futures[i].get();
            }
        }
        
        // 3. Read chunk c (NO concurrent Write now!)
        read_chunk_data(c, buf);
        
        // 4. Start async Write for PREVIOUS chunk (overlaps with GPU!)
        if (prev_completed_buf >= 0) {
            int prev_c = c - 1;
            write_futures[prev_completed_buf] = std::async(std::launch::async,
                [this, write_chunk_data, prev_completed_buf, prev_c]() {
                    write_chunk_data(prev_c, prev_completed_buf);
                    return (ssize_t)0;
                });
        }
        
        // 5. GPU compute (overlaps with Write[c-1]!)
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachGlobal));
        kernel((int)c, uvm_buffer, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 6. Back to Host mode
        CUDA_CHECK(cudaStreamAttachMemAsync(stream, uvm_buffer, 0, cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        prev_completed_buf = buf;
    }
    
    // Drain: wait for pending write, then write last chunk
    for (int i = 0; i < NUM_PIPELINE_BUFS; i++) {
        if (write_futures[i].valid()) {
            write_futures[i].get();
        }
    }
    // Write last chunk (sync)
    if (prev_completed_buf >= 0) {
        write_chunk_data(n_chunks - 1, prev_completed_buf);
    }
    
    // Swap ping-pong files after each layer (for compression)
    if (use_compression && compressed_storage) {
        compressed_storage->swap_files();
    }
}

void EdgeQuantumSim::apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s) {
    target_idx[0] = target;
    int n_bits = (mode == SimMode::Native) ? n_qubits : chunk_bits;

    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_bits, (void*)d_mat, CUDA_C_32F, 
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, target_idx, 1, nullptr, nullptr, 0, 
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
}

void EdgeQuantumSim::apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s) {
    target_idx[0] = t;
    control_idx[0] = c;
    int n_bits = (mode == SimMode::Native) ? n_qubits : chunk_bits;
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

        if (mode == SimMode::Native) {
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
        
        if (mode == SimMode::Native) {
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

        if (mode == SimMode::Native) {
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

    if (mode == SimMode::Native) {
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

        if (mode == SimMode::Native) {
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
