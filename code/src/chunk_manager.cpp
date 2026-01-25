#include "chunk_manager.hpp"
#include <iostream>

ChunkManager::ChunkManager(size_t chunk_bytes, size_t num_chunks)
    : chunk_size(chunk_bytes), n_chunks(num_chunks) {
    
    CUDA_CHECK(cudaHostAlloc(&host_buffer[0], chunk_size, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host_buffer[1], chunk_size, cudaHostAllocDefault));
    
    std::cout << "[ChunkManager] Allocated 2x " << chunk_size/1024/1024 << "MB Pinned Buffers" << std::endl;
}

ChunkManager::~ChunkManager() {
    cudaFreeHost(host_buffer[0]);
    cudaFreeHost(host_buffer[1]);
}

void* ChunkManager::get_buffer(int idx) {
    return host_buffer[idx];
}

void ChunkManager::read_chunk(IOBackend& io, int chunk_idx, int buf_idx) {
    size_t offset = chunk_idx * chunk_size;
    io.read(offset, host_buffer[buf_idx], chunk_size);
}

void ChunkManager::write_chunk(IOBackend& io, int chunk_idx, int buf_idx) {
    size_t offset = chunk_idx * chunk_size;
    io.write(offset, host_buffer[buf_idx], chunk_size);
}
