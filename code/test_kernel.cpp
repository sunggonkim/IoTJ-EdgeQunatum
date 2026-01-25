#include <iostream>
#include <chrono>
#include <complex>
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
    int n_qubits = 28;
    size_t mem_size = (1ULL << n_qubits) * 16;
    
    std::cout << "Allocating " << mem_size / (1024.0*1024.0) << " MB..." << std::endl;
    
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, mem_size));
    CUDA_CHECK(cudaMemset(d_sv, 0, mem_size));
    
    // Set |0>
    cuComplex one = {1.0f, 0.0f};
    CUDA_CHECK(cudaMemcpy(d_sv, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
    
    custatevecHandle_t handle;
    CUSV_CHECK(custatevecCreate(&handle));
    
    void* d_gate;
    cuComplex h_gate[4] = {{0.707f,0}, {0.707f,0}, {0.707f,0}, {-0.707f,0}};
    CUDA_CHECK(cudaMalloc(&d_gate, sizeof(h_gate)));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, sizeof(h_gate), cudaMemcpyHostToDevice));
    
    void* d_ws;
    size_t ws_size = 64 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));
    
    std::cout << "Executing Kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    int targets[] = {0};
    int controls[] = {1};
    
    // Apply H on 0
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_qubits, d_gate, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
    
    // Apply CNOT 1->0
    targets[0] = 0;
    // For CNOT, matrix is set to Xgate or something?
    // Let's just apply H again to test timing
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_32F, n_qubits, d_gate, CUDA_C_32F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr, nullptr, 0,
        CUSTATEVEC_COMPUTE_32F, d_ws, ws_size
    ));
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "Time: " << diff.count() << " s" << std::endl;
    
    return 0;
}
