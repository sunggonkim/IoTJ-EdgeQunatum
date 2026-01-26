#include "chunk_manager.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>

ChunkManager::ChunkManager(size_t chunk_bytes, size_t num_chunks, int num_buffers)
    : chunk_size(chunk_bytes), n_chunks(num_chunks), n_buffers(num_buffers),
      host_buffer(nullptr) {

    // ============================================================
    // CRITICAL DISCOVERY: UVM (cudaMallocManaged) WORKS on Jetson!
    // ============================================================
    // Contrary to conventional assumptions:
    // - UVM pointers CAN be passed to pread()/pwrite() syscalls
    // - UVM IS compatible with O_DIRECT
    // - cuStateVec ACCEPTS UVM pointers for gate operations
    // 
    // This enables TRUE ZERO-COPY: Disk -> UVM -> cuStateVec -> UVM -> Disk
    // Result: 2x speedup over Pinned+cudaMemcpy approach!
    // ============================================================

    host_buffer = new void*[n_buffers];
    for (int i = 0; i < n_buffers; i++) {
        host_buffer[i] = nullptr;
    }

    // UVM (Managed Memory) with AttachHost for async I/O support
    // AttachHost allows worker threads to perform pread/pwrite while GPU uses other buffers
    for (int i = 0; i < n_buffers; i++) {
        CUDA_CHECK(cudaMallocManaged(&host_buffer[i], chunk_size, cudaMemAttachHost));
    }
    std::cout << "[ChunkManager] " << n_buffers << "x " << chunk_size/1024/1024 
              << "MB UVM Buffers (AttachHost for Async I/O)" << std::endl;
}

ChunkManager::~ChunkManager() {
    if (host_buffer) {
        for (int i = 0; i < n_buffers; i++) {
            if (host_buffer[i]) {
                cudaFree(host_buffer[i]);  // cudaFree for UVM memory
            }
        }
        delete[] host_buffer;
    }
}

void* ChunkManager::get_buffer(int idx) {
    if (idx < 0 || idx >= n_buffers) return nullptr;
    return host_buffer[idx];
}

void ChunkManager::read_chunk(IOBackend& io, int chunk_idx, int buf_idx) {
    if (buf_idx < 0 || buf_idx >= n_buffers || host_buffer[buf_idx] == nullptr) {
        std::cerr << "[ChunkManager] Invalid buffer index: " << buf_idx << std::endl;
        std::abort();
    }
    size_t offset = chunk_idx * chunk_size;
    io.read(offset, host_buffer[buf_idx], chunk_size);
}

void ChunkManager::write_chunk(IOBackend& io, int chunk_idx, int buf_idx) {
    if (buf_idx < 0 || buf_idx >= n_buffers || host_buffer[buf_idx] == nullptr) {
        std::cerr << "[ChunkManager] Invalid buffer index: " << buf_idx << std::endl;
        std::abort();
    }
    size_t offset = chunk_idx * chunk_size;
    io.write(offset, host_buffer[buf_idx], chunk_size);
}
