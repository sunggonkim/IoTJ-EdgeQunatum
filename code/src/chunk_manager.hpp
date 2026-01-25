#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "io_backend.hpp"
#include "utils.hpp"

class ChunkManager {
    size_t chunk_size;
    size_t n_chunks;
    void* host_buffer[2]; // Double buffering
    
public:
    ChunkManager(size_t chunk_bytes, size_t num_chunks);
    ~ChunkManager();
    
    void* get_buffer(int idx);
    void read_chunk(IOBackend& io, int chunk_idx, int buf_idx);
    void write_chunk(IOBackend& io, int chunk_idx, int buf_idx);
};
