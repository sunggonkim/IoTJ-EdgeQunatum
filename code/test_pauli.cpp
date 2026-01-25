#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
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
    int n_qubits = 10;
    std::cout << "sizeof(cuComplex): " << sizeof(cuComplex) << std::endl;
    size_t mem_size = (1ULL << n_qubits) * sizeof(cuComplex);
    
    std::cout << "--- EdgeQuantum Pauli Verification (2Q) - UVM ---" << std::endl;
    
    void* d_sv;
    CUDA_CHECK(cudaMallocManaged(&d_sv, mem_size));
    CUDA_CHECK(cudaMemset(d_sv, 0, mem_size));
    
    // Set |0...0> = 1.0
    cuComplex one = {1.0f, 0.0f};
    CUDA_CHECK(cudaMemcpy(d_sv, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
    
    custatevecHandle_t handle;
    CUSV_CHECK(custatevecCreate(&handle));
    
    // Apply X via Pauli Rotation (RX(pi) ~ -iX)
    // Rotation angle theta = PI
    double theta = M_PI;
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_X};
    int targets[] = {0};
    int controls[] = {};
    
    size_t ws_size = 0; // Check docs, usually 0 for 1Q?
    // Using simple version or GetWorkspace?
    // Let's assume 0 first or huge?
    // custatevecApplyPauliRotation does NOT have GetWorkspaceSize in older versions?
    // v1.1 has it?
    // Let's check signature.
    
    // Using robust allocation just in case
    ws_size = 64 * 1024 * 1024; 
    void* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size));
    
    std::cout << "Applying X Rotation (PI)..." << std::endl;
    
    // Note: Signature in header: (handle, sv, type, nBits, theta, paulis, targets, nTargets, controls, controlVals, nControls)
    // Implicitly NO workspace arg in header printed above?
    // Check line 998 of header in previous turn.
    // custatevecApplyPauliRotation(... nControls); NO WORKSPACE ARG!!!!!
    
    CUSV_CHECK(custatevecApplyPauliRotation(
        handle, d_sv, CUDA_C_32F, n_qubits, theta, paulis, targets, 1, nullptr, nullptr, 0
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cuComplex res[4]; 
    CUDA_CHECK(cudaMemcpy(res, d_sv, sizeof(res), cudaMemcpyDeviceToHost));
    
    std::cout << "State Vector [0..3]:" << std::endl;
    for(int i=0; i<4; i++) {
        std::cout << "  [" << i << "]: " << res[i].x << " + " << res[i].y << "j";
    }
    std::cout << std::endl;
    
    // Expectation: |0> -> -i|1>. So index 0=0, index 1 = (0, -1).
    if (std::abs(res[1].y + 1.0) < 1e-5) std::cout << "PASS" << std::endl;
    else std::cout << "FAIL" << std::endl;

    return 0;
}
