#pragma once
#include <lz4.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/statvfs.h>
#include <mutex>

// ============================================================
// LZ4 Compressed Chunk Storage - Variable Size with Ping-Pong
// ============================================================
// Design: True variable-size compression for maximum space efficiency.
// Uses ping-pong files to handle read-modify-write patterns.
//
// During each gate layer:
//   - Read from file_A (using offset table A)
//   - Write sequentially to file_B (building offset table B)
//   - Swap: A becomes B, B becomes A
//
// This achieves true compression ratios while supporting pipeline.
// ============================================================

class CompressedStorage {
    int fd[2];                    // Ping-pong file descriptors
    std::string paths[2];         // File paths
    std::vector<size_t> offsets[2];  // Offset tables for each file
    int read_file;                // Current read file index (0 or 1)
    int write_file;               // Current write file index
    
    size_t chunk_size;            // Uncompressed chunk size (e.g., 256MB)
    size_t n_chunks;
    bool compression_enabled;
    
    std::vector<char> compress_buf;
    std::vector<char> decompress_buf;
    
    std::mutex io_mutex;
    
    size_t write_offset;          // Current write position in write_file
    size_t total_compressed;      // Stats: total compressed bytes written
    size_t total_uncompressed;    // Stats: total uncompressed bytes

public:
    CompressedStorage(const std::string& filepath, size_t chunk_bytes, size_t num_chunks, bool enable_compression = true)
        : read_file(0),
          write_file(1),
          chunk_size(chunk_bytes),
          n_chunks(num_chunks),
          compression_enabled(enable_compression),
          write_offset(0),
          total_compressed(0),
          total_uncompressed(0) {
        
        // Setup paths
        paths[0] = filepath + ".A";
        paths[1] = filepath + ".B";
        
        // Allocate buffers
        int max_compressed = LZ4_compressBound(chunk_size);
        compress_buf.resize(max_compressed + 8);
        decompress_buf.resize(max_compressed + 8);
        
        // Initialize offset tables
        offsets[0].resize(n_chunks + 1, 0);
        offsets[1].resize(n_chunks + 1, 0);
        
        // Open both files
        for (int i = 0; i < 2; i++) {
            fd[i] = open(paths[i].c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd[i] < 0) {
                perror(("open " + paths[i]).c_str());
                exit(1);
            }
        }
        
        std::cout << "[CompressedStorage] Ping-pong mode (variable-size compression)" << std::endl;
        std::cout << "  Files: " << paths[0] << ", " << paths[1] << std::endl;
        std::cout << "  Chunks: " << n_chunks << " x " << chunk_size/(1024*1024) << " MB" << std::endl;
        std::cout << "  Compression: " << (compression_enabled ? "ON" : "OFF") << std::endl;
    }
    
    ~CompressedStorage() {
        for (int i = 0; i < 2; i++) {
            if (fd[i] >= 0) {
                fsync(fd[i]);
                close(fd[i]);
            }
            // Clean up files
            unlink(paths[i].c_str());
        }
    }
    
    CompressedStorage(const CompressedStorage&) = delete;
    CompressedStorage& operator=(const CompressedStorage&) = delete;
    
    bool is_compression_enabled() const { return compression_enabled; }
    
    // Initialize: write all chunks to file A (building initial offset table)
    void init_write_chunk(size_t chunk_idx, const void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid chunk index: " << chunk_idx << std::endl;
            return;
        }
        
        // During init, always write to file 0 (A)
        offsets[0][chunk_idx] = write_offset;
        
        size_t bytes_written;
        
        if (compression_enabled) {
            int compressed_size = LZ4_compress_default(
                (const char*)data, compress_buf.data() + 4,
                chunk_size, compress_buf.size() - 4
            );
            
            if (compressed_size <= 0) {
                std::cerr << "[CompressedStorage] LZ4 compression failed at init!" << std::endl;
                exit(1);
            }
            
            // Store header + compressed data
            memcpy(compress_buf.data(), &compressed_size, 4);
            bytes_written = 4 + compressed_size;
            
            ssize_t ret = pwrite(fd[0], compress_buf.data(), bytes_written, write_offset);
            if (ret != (ssize_t)bytes_written) {
                perror("pwrite init");
                exit(1);
            }
            
            total_compressed += compressed_size;
            total_uncompressed += chunk_size;
            
            // Debug first few
            if (chunk_idx < 3 || chunk_idx == n_chunks - 1) {
                float ratio = 100.0f * compressed_size / chunk_size;
                std::cout << "[Init] Chunk " << chunk_idx 
                          << ": " << chunk_size/(1024*1024) << "MB -> " 
                          << compressed_size/1024 << "KB (" 
                          << ratio << "%)" << std::endl;
            }
        } else {
            bytes_written = chunk_size;
            ssize_t ret = pwrite(fd[0], data, chunk_size, write_offset);
            if (ret != (ssize_t)chunk_size) {
                perror("pwrite init raw");
                exit(1);
            }
        }
        
        write_offset += bytes_written;
        offsets[0][chunk_idx + 1] = write_offset;  // End offset
    }
    
    // Finalize init phase
    void finish_init() {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        fsync(fd[0]);
        read_file = 0;  // Read from A
        write_file = 1; // Write to B
        write_offset = 0;
        offsets[1][0] = 0;
        
        if (compression_enabled && total_uncompressed > 0) {
            float ratio = 100.0f * total_compressed / total_uncompressed;
            size_t file_size = lseek(fd[0], 0, SEEK_END);
            std::cout << "[CompressedStorage] Init complete. File size: " 
                      << file_size / (1024ULL*1024*1024) << " GB ("
                      << ratio << "% of uncompressed)" << std::endl;
        }
    }
    
    // Read chunk from current read file
    void read_chunk(size_t chunk_idx, void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid read index: " << chunk_idx << std::endl;
            return;
        }
        
        size_t file_offset = offsets[read_file][chunk_idx];
        
        if (compression_enabled) {
            // Read header
            int compressed_size = 0;
            ssize_t ret = pread(fd[read_file], &compressed_size, 4, file_offset);
            if (ret != 4 || compressed_size <= 0) {
                std::cerr << "[CompressedStorage] Bad header at chunk " << chunk_idx 
                          << " offset " << file_offset << std::endl;
                exit(1);
            }
            
            // Read compressed data
            ret = pread(fd[read_file], decompress_buf.data(), compressed_size, file_offset + 4);
            if (ret != compressed_size) {
                std::cerr << "[CompressedStorage] Read failed at chunk " << chunk_idx << std::endl;
                exit(1);
            }
            
            // Decompress
            int decompressed = LZ4_decompress_safe(
                decompress_buf.data(), (char*)data,
                compressed_size, chunk_size
            );
            
            if (decompressed != (int)chunk_size) {
                std::cerr << "[CompressedStorage] Decompress failed! chunk " << chunk_idx
                          << " got " << decompressed << " expected " << chunk_size << std::endl;
                exit(1);
            }
        } else {
            ssize_t ret = pread(fd[read_file], data, chunk_size, file_offset);
            if (ret != (ssize_t)chunk_size) {
                std::cerr << "[CompressedStorage] Raw read failed at chunk " << chunk_idx << std::endl;
                exit(1);
            }
        }
    }
    
    // Write chunk to current write file (must be called in order!)
    void write_chunk(size_t chunk_idx, const void* data) {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        if (chunk_idx >= n_chunks) {
            std::cerr << "[CompressedStorage] Invalid write index: " << chunk_idx << std::endl;
            return;
        }
        
        // Verify sequential write
        if (offsets[write_file][chunk_idx] != write_offset) {
            // First chunk of layer resets
            if (chunk_idx == 0) {
                write_offset = 0;
            }
        }
        offsets[write_file][chunk_idx] = write_offset;
        
        size_t bytes_written;
        
        if (compression_enabled) {
            int compressed_size = LZ4_compress_default(
                (const char*)data, compress_buf.data() + 4,
                chunk_size, compress_buf.size() - 4
            );
            
            if (compressed_size <= 0) {
                std::cerr << "[CompressedStorage] Compression failed!" << std::endl;
                exit(1);
            }
            
            memcpy(compress_buf.data(), &compressed_size, 4);
            bytes_written = 4 + compressed_size;
            
            ssize_t ret = pwrite(fd[write_file], compress_buf.data(), bytes_written, write_offset);
            if (ret != (ssize_t)bytes_written) {
                perror("pwrite chunk");
                exit(1);
            }
        } else {
            bytes_written = chunk_size;
            ssize_t ret = pwrite(fd[write_file], data, chunk_size, write_offset);
            if (ret != (ssize_t)chunk_size) {
                perror("pwrite raw chunk");
                exit(1);
            }
        }
        
        write_offset += bytes_written;
        offsets[write_file][chunk_idx + 1] = write_offset;
    }
    
    // Swap files after each layer (call at end of each gate layer)
    void swap_files() {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        fsync(fd[write_file]);
        
        // Swap
        std::swap(read_file, write_file);
        write_offset = 0;
        offsets[write_file][0] = 0;
    }
    
    void sync() {
        fsync(fd[0]);
        fsync(fd[1]);
    }
    
    size_t get_file_size() {
        return lseek(fd[read_file], 0, SEEK_END);
    }
    
    // Get current read file descriptor (for external async reads if needed)
    int get_read_fd() const { return fd[read_file]; }
    int get_write_fd() const { return fd[write_file]; }
};

// ============================================================
// Helper: Check if compression is needed
// ============================================================
inline bool should_enable_compression(const std::string& path, size_t required_bytes) {
    struct statvfs stat;
    std::string dir = path.substr(0, path.rfind('/'));
    if (dir.empty()) dir = ".";
    
    if (statvfs(dir.c_str(), &stat) != 0) {
        return true;
    }
    
    size_t available = (size_t)stat.f_bavail * stat.f_frsize;
    
    // Need space for 2 ping-pong files, but compressed
    // Assume average 10% compression ratio for quantum states
    // So need: available > required * 0.10 * 2 = required * 0.20
    // But be conservative: available > required * 0.30
    bool need_compression = (available < required_bytes * 11 / 10);
    
    std::cout << "[Storage] Available: " << available / (1024ULL*1024*1024) << " GB, "
              << "Required (raw): " << required_bytes / (1024ULL*1024*1024) << " GB -> "
              << (need_compression ? "COMPRESSION ENABLED" : "compression disabled") 
              << std::endl;
    
    return need_compression;
}
