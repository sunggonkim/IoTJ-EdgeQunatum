#pragma once
#include <vector>
#include <string>
#include <complex>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include "chunk_manager.hpp"
#include "io_backend.hpp"

enum class SimMode {
    Tiered_Async,
    Tiered_Blocking,
    Native,
    UVM
};

// Simple Thread Pool for I/O
class IoWorker {
public:
    IoWorker() : stop(false), is_working(false) {
        worker_thread = std::thread([this]{ run(); });
    }
    
    ~IoWorker() {
        {
            std::lock_guard<std::mutex> lock(m);
            stop = true;
        }
        cv.notify_one();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(m);
            tasks.push_back(task);
        }
        cv.notify_one();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                
                is_working = true;
                task = std::move(tasks.front());
                tasks.erase(tasks.begin());
            }
            task();
            {
                std::lock_guard<std::mutex> lock(m);
                is_working = false;
            }
        }
    }

    std::thread worker_thread;
    std::vector<std::function<void()>> tasks;
    std::mutex m;
    std::condition_variable cv;
    bool stop;
    bool is_working;
};

class EdgeQuantumSim {
    int n_qubits;
    int chunk_bits;
    size_t chunk_size;
    size_t n_chunks;
    
    // Core Components - Dual Rings
    IOBackend io_read;
    IOBackend io_write;
    size_t state_size;
    SimMode mode;
    std::string storage_path;
    
    // Tiered Memory Resources
    ChunkManager* chunk_mgr;
    IOBackend* io;
    
    // Native/UVM Resources
    void* full_state_ptr;

    // UVM-based pipeline configuration
    // With UVM, we don't need separate device buffers - UVM buffers are GPU-accessible!
    static constexpr int NUM_PIPELINE_BUFS = 3;  // 3 UVM buffers for triple pipeline
    bool device_buf_ready;  // Legacy flag, kept for compatibility
    
    // cuQuantum Resources
    // cuQuantum Resources
    custatevecHandle_t handle;
    void* d_ws;
    size_t ws_size;
    cudaStream_t stream;      // compute stream
    cudaStream_t copy_stream; // memcpy stream
    
    // Constants
    void* d_gate_matrix;
    int target_idx[1];
    int control_idx[1];
    
    // Thread Pools
    IoWorker* read_worker;
    IoWorker* write_worker;
    
    // Mode

public:
    EdgeQuantumSim(int qubits, std::string path, SimMode m, bool force_mode=false);
    ~EdgeQuantumSim();
    
    // Benchmarks
    void run_qv(int depth);
    void run_vqc(int layers);
    void run_qsvm(int feature_dim);
    void run_ghz();
    void run_random(int depth);
    void run_vqe(int batch);

    bool validate_hadamard();
    
private:
    void init_storage();
    void reset_zero_state();
    bool get_first_two_amplitudes(std::complex<float>& a0, std::complex<float>& a1);
    
    // Generic Pipeline
    using KernelFunc = std::function<void(int, void*, cudaStream_t)>;
    void process_pipeline(KernelFunc kernel);
    
    // Gate Helpers
    void apply_gate_1q(void* d_sv, int target, const void* d_mat, cudaStream_t s);
    void apply_cnot_local(void* d_sv, int c, int t, cudaStream_t s);
};
