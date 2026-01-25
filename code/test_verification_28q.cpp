#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUSV_CHECK(call) \
    do { \
        custatevecStatus_t err = call; \
        if (err != CUSTATEVEC_STATUS_SUCCESS) { \
            std::cerr << "cuStateVec Error: " << err \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    // Force CUDA runtime initialization and print device info
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA Device: " << prop.name
              << " (SM " << prop.major << "." << prop.minor << ")"
              << ", GlobalMem=" << (prop.totalGlobalMem / (1024.0 * 1024.0))
              << " MB" << std::endl;
    int n_qubits = 28; 
    size_t mem_size = (1ULL << n_qubits) * sizeof(cuComplex); // 4GB
    
    std::cout << "--- EdgeQuantum Correctness Verification (28Q) ---" << std::endl;
    std::cout << "Allocating " << mem_size / (1024.0*1024.0) << " MB for State Vector..." << std::endl;
    
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, mem_size));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(d_sv, 0, mem_size, stream));
    
    // Set |0...0> = 1.0
    cuComplex one = {1.0f, 0.0f};
    CUDA_CHECK(cudaMemcpyAsync(d_sv, &one, sizeof(cuComplex), cudaMemcpyHostToDevice, stream));
    
    // Logger Callback
    // Typedef in custatevec is custatevecLoggerCallback_t
    // But simplest is set level and output to stdout? 
    // Need to set callback.
    // Actually, create handle, then:
    CUSV_CHECK(custatevecLoggerSetLevel(5)); // 5 = Trace? Check enum. 
    CUSV_CHECK(custatevecLoggerSetCallback([](int32_t level, const char* name, const char* msg){
        std::cout << "[LOG " << level << "] " << name << ": " << msg << std::endl;
    }));
    
    custatevecHandle_t handle;
    CUSV_CHECK(custatevecCreate(&handle));
    CUSV_CHECK(custatevecSetStream(handle, stream));
    
    // X Gate Matrix (Pinned Host)
    cuComplex* h_gate;
    CUDA_CHECK(cudaMallocHost(&h_gate, sizeof(cuComplex) * 4));
    h_gate[0] = {0.0f, 0};
    h_gate[1] = {1.0f, 0};
    h_gate[2] = {1.0f, 0};
    h_gate[3] = {0.0f, 0};
    
    int* targets;
    CUDA_CHECK(cudaMallocHost(&targets, sizeof(int)));
    targets[0] = 0;

    // Sanity: 2-qubit X gate should work
    {
        void* d_sv_small;
        CUDA_CHECK(cudaMalloc(&d_sv_small, 4 * sizeof(cuComplex)));
        CUDA_CHECK(cudaMemsetAsync(d_sv_small, 0, 4 * sizeof(cuComplex), stream));
        cuComplex one = {1.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(d_sv_small, &one, sizeof(cuComplex), cudaMemcpyHostToDevice, stream));
        CUSV_CHECK(custatevecApplyMatrix(
            handle, d_sv_small, CUDA_C_32F, 2, h_gate, CUDA_C_32F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_32F, nullptr, 0
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cuComplex host_small[4];
        CUDA_CHECK(cudaMemcpyAsync(host_small, d_sv_small, sizeof(host_small), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (std::abs(host_small[1].x - 1.0f) > 1e-5f) {
            std::cout << "[FAIL] 2-qubit X gate failed. cuStateVec applyMatrix not updating." << std::endl;
            CUDA_CHECK(cudaFree(d_sv_small));
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 1;
        }
        CUDA_CHECK(cudaFree(d_sv_small));
    }
    int controls[] = {};
    
    size_t required_ws_size = 0;
    CUSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_32F, n_qubits, h_gate, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0,
        CUSTATEVEC_COMPUTE_32F, &required_ws_size
    ));
    
    std::cout << "Required Workspace: " << required_ws_size / (1024.0*1024.0) << " MB" << std::endl;
    
    void* d_ws;
    size_t ws_size = required_ws_size; 
    if (ws_size < 128*1024*1024) ws_size = 128*1024*1024; // Min 128
    
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));
    
    std::cout << "Applying X Gate on Qubit 0..." << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure inputs ready
    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply H (Using Device Pointer)
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_qubits, h_gate, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Execution Time: " << diff.count() << " s" << std::endl;
    
    // Verify Result: Depending on qubit ordering, amplitude splits between
    // index 0 and either index 1 (LSB) or index 2^(n-1) (MSB).
    cuComplex res0;
    cuComplex res1;
    cuComplex resMSB;
    size_t idx_msb = 1ULL << (n_qubits - 1);

    CUDA_CHECK(cudaMemcpyAsync(&res0, d_sv, sizeof(cuComplex), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&res1, static_cast<char*>(d_sv) + sizeof(cuComplex), sizeof(cuComplex), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&resMSB, static_cast<char*>(d_sv) + idx_msb * sizeof(cuComplex), sizeof(cuComplex), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "State Vector Samples:" << std::endl;
    std::cout << "  [0]: " << res0.x << " + " << res0.y << "j" << std::endl;
    std::cout << "  [1]: " << res1.x << " + " << res1.y << "j" << std::endl;
    std::cout << "  [2^(n-1)]: " << resMSB.x << " + " << resMSB.y << "j" << std::endl;

    bool pass = true;
    const float expected = 1.0f;
    const float eps = 1e-5f;

    bool lsb_ok = (std::abs(res0.x) < eps) && (std::abs(res1.x - expected) < eps);
    bool msb_ok = (std::abs(res0.x) < eps) && (std::abs(resMSB.x - expected) < eps);

    if (lsb_ok) {
        std::cout << "Detected LSB qubit ordering. (PASS)" << std::endl;
    } else if (msb_ok) {
        std::cout << "Detected MSB qubit ordering. (PASS)" << std::endl;
    } else {
        std::cout << "Amplitude pattern mismatch. (FAIL)" << std::endl;
        pass = false;
    }
    
    if (pass) std::cout << "\n>>> ALL CHECKS PASSED. Computation is VALID. <<<" << std::endl;
    else std::cout << "\n>>> VERIFICATION FAILED. <<<" << std::endl;

    CUDA_CHECK(cudaFreeHost(targets));
    CUDA_CHECK(cudaFreeHost(h_gate));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return pass ? 0 : 1;
}
