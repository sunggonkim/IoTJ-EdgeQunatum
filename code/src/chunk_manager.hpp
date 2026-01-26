#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "io_backend.hpp"
#include "utils.hpp"

class ChunkManager {
    size_t chunk_size;
    size_t n_chunks;
    int n_buffers;  // 3 for triple-buffer pipeline
    void** host_buffer;
    
public:
    // num_buffers: 3 = triple-buffer (read/compute/write overlap)
    ChunkManager(size_t chunk_bytes, size_t num_chunks, int num_buffers = 3);
    ~ChunkManager();
    
    void* get_buffer(int idx);
    int get_num_buffers() const { return n_buffers; }
    void read_chunk(IOBackend& io, int chunk_idx, int buf_idx);
    void write_chunk(IOBackend& io, int chunk_idx, int buf_idx);
    size_t get_chunk_size() const { return chunk_size; }
};
