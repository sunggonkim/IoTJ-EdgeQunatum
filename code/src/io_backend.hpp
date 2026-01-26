#pragma once
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <dlfcn.h>
#include "utils.hpp"

#if __has_include(<cufile.h>)
#include <cufile.h>
#define EDGEQ_HAVE_CUFILE 1
#else
#define EDGEQ_HAVE_CUFILE 0
typedef struct { int err; } CUfileError_t;
typedef void* CUfileHandle_t;
typedef struct { int dummy; } CUfileDescr_t;
#endif

inline int io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return syscall(__NR_io_uring_setup, entries, p);
}

inline int io_uring_enter(int ring_fd, unsigned to_submit, unsigned min_complete, unsigned flags, sigset_t *sig) {
    return syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete, flags, sig);
}

class IOBackend {
    int fd;
    std::string path;
    std::mutex io_mutex;
    
    // io_uring resources
    int ring_fd;
    void* sq_ptr;
    void* cq_ptr;
    struct io_uring_sqe* sqes;
    unsigned* sq_tail;
    unsigned* sq_head;
    unsigned* cq_tail;
    unsigned* cq_head;
    unsigned* sq_array;
    struct io_uring_cqe* cqes;
    unsigned sq_entries;
    unsigned cq_entries;
    unsigned sq_mask;
    unsigned cq_mask;
    bool uring_enabled;

    // GDS (cuFile) dynamic loader
    bool gds_enabled;
    void* gds_lib;
    CUfileHandle_t gds_handle;

    // cuFile function pointers
    typedef CUfileError_t (*cuFileDriverOpen_t)();
    typedef CUfileError_t (*cuFileDriverClose_t)();
    typedef CUfileError_t (*cuFileHandleRegister_t)(CUfileHandle_t*, CUfileDescr_t*);
    typedef CUfileError_t (*cuFileHandleDeregister_t)(CUfileHandle_t);
    typedef CUfileError_t (*cuFileBufRegister_t)(void*, size_t, int);
    typedef CUfileError_t (*cuFileBufDeregister_t)(void*);
    typedef ssize_t (*cuFileRead_t)(CUfileHandle_t, void*, size_t, off_t, off_t);
    typedef ssize_t (*cuFileWrite_t)(CUfileHandle_t, const void*, size_t, off_t, off_t);

    cuFileDriverOpen_t p_cuFileDriverOpen;
    cuFileDriverClose_t p_cuFileDriverClose;
    cuFileHandleRegister_t p_cuFileHandleRegister;
    cuFileHandleDeregister_t p_cuFileHandleDeregister;
    cuFileBufRegister_t p_cuFileBufRegister;
    cuFileBufDeregister_t p_cuFileBufDeregister;
    cuFileRead_t p_cuFileRead;
    cuFileWrite_t p_cuFileWrite;

public:
    IOBackend(const std::string& filepath)
        : fd(-1),
          path(filepath),
          ring_fd(-1),
          sq_ptr(nullptr),
          cq_ptr(nullptr),
          sqes(nullptr),
          sq_tail(nullptr),
          sq_head(nullptr),
          cq_tail(nullptr),
          cq_head(nullptr),
          sq_array(nullptr),
          cqes(nullptr),
          sq_entries(0),
          cq_entries(0),
          sq_mask(0),
          cq_mask(0),
          uring_enabled(false),
          gds_enabled(false),
          gds_lib(nullptr),
          gds_handle(nullptr),
          p_cuFileDriverOpen(nullptr),
          p_cuFileDriverClose(nullptr),
          p_cuFileHandleRegister(nullptr),
          p_cuFileHandleDeregister(nullptr),
          p_cuFileBufRegister(nullptr),
          p_cuFileBufDeregister(nullptr),
          p_cuFileRead(nullptr),
          p_cuFileWrite(nullptr) {
        // Always use O_DIRECT for optimal NVMe performance (bypasses page cache)
        int flags = O_RDWR | O_CREAT | O_DIRECT;
        fd = open(filepath.c_str(), flags, 0644);
        if (fd < 0) {
            // Fallback without O_DIRECT if filesystem doesn't support it
            flags = O_RDWR | O_CREAT;
            fd = open(filepath.c_str(), flags, 0644);
        }
        if (fd < 0) {
            perror("open");
            exit(1);
        }
        
        std::cout << "[IOBackend] Opened " << filepath << " (fd: " << fd << ", O_DIRECT)" << std::endl;
        
        // io_uring disabled - causes stability issues on Jetson
        // GDS (cuFile) disabled - not available on Jetson R35.4
    }

    void disable_uring() {
        uring_enabled = false;
    }
    
    // Disable copy
    IOBackend(const IOBackend&) = delete;
    IOBackend& operator=(const IOBackend&) = delete;
    
    // Enable move if strictly needed, but better avoid for simple debug
    
    ~IOBackend() {
        if (fd >= 0) {
            std::cout << "[IOBackend] Closing fd: " << fd << std::endl;
            close(fd);
        }
        // Unmap io_uring regions before closing ring_fd
        if (uring_enabled && ring_fd >= 0) {
            if (sq_ptr && sq_ptr != MAP_FAILED) {
                munmap(sq_ptr, sq_entries * sizeof(unsigned) + 64);  // Approx size
            }
            if (cq_ptr && cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) {
                munmap(cq_ptr, cq_entries * sizeof(struct io_uring_cqe) + 64);
            }
            if (sqes && (void*)sqes != MAP_FAILED) {
                munmap(sqes, sq_entries * sizeof(struct io_uring_sqe));
            }
        }
        if (ring_fd >= 0) close(ring_fd);
        if (gds_enabled && p_cuFileHandleDeregister) {
#if EDGEQ_HAVE_CUFILE
            p_cuFileHandleDeregister(gds_handle);
#endif
        }
        if (gds_enabled && p_cuFileDriverClose) {
            p_cuFileDriverClose();
        }
        if (gds_lib) dlclose(gds_lib);
    }
    
    void write(size_t offset, void* buf, size_t size) {
        std::lock_guard<std::mutex> lock(io_mutex);
        if (uring_enabled) {
            submit_and_wait(IORING_OP_WRITEV, offset, buf, size);
        } else {
            if (pwrite(fd, buf, size, offset) < 0) { perror("pwrite"); exit(1); }
        }
    }
    
    void sync() {
        if (fd >= 0) {
            fsync(fd);
        }
    }
    
    void read(size_t offset, void* buf, size_t size) {
        std::lock_guard<std::mutex> lock(io_mutex);
        if (uring_enabled) {
            submit_and_wait(IORING_OP_READV, offset, buf, size);
        } else {
            if (pread(fd, buf, size, offset) < 0) { perror("pread"); exit(1); }
        }
    }

    bool is_gds_enabled() const { return gds_enabled; }

    bool register_device_buffer(void* dev_ptr, size_t size) {
        if (!gds_enabled || !p_cuFileBufRegister) return false;
        return p_cuFileBufRegister(dev_ptr, size, 0).err == 0;
    }

    void deregister_device_buffer(void* dev_ptr) {
        if (!gds_enabled || !p_cuFileBufDeregister) return;
        p_cuFileBufDeregister(dev_ptr);
    }

    bool read_device(size_t offset, void* dev_ptr, size_t size) {
        if (!gds_enabled || !p_cuFileRead) return false;
        ssize_t ret = p_cuFileRead(gds_handle, dev_ptr, size, (off_t)offset, 0);
        return ret == (ssize_t)size;
    }

    bool write_device(size_t offset, void* dev_ptr, size_t size) {
        if (!gds_enabled || !p_cuFileWrite) return false;
        ssize_t ret = p_cuFileWrite(gds_handle, dev_ptr, size, (off_t)offset, 0);
        return ret == (ssize_t)size;
    }

private:
    void submit_and_wait(uint8_t opcode, size_t offset, void* buf, size_t size) {
        struct iovec iov = { .iov_base = buf, .iov_len = size };
        
        unsigned tail = *sq_tail;
        unsigned idx = tail & sq_mask;
        struct io_uring_sqe* sqe = &sqes[idx];
        
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = opcode;
        sqe->fd = fd;
        sqe->off = offset;
        sqe->addr = (unsigned long)&iov;
        sqe->len = 1;
        
        sq_array[idx] = idx;
        *sq_tail = tail + 1;
        
        __sync_synchronize();
        
        int ret = io_uring_enter(ring_fd, 1, 1, IORING_ENTER_GETEVENTS, NULL);
        if (ret < 0) {
            perror("io_uring_enter");
            exit(1);
        }
        
        unsigned head = *cq_head;
        __sync_synchronize();
        struct io_uring_cqe* cqe = &cqes[head & cq_mask];
        
        if (cqe->res < 0) {
            std::cerr << "IO Error Op: " << (int)opcode << " Res: " << cqe->res 
                      << " (" << strerror(-cqe->res) << ") fd: " << fd << std::endl;
            exit(1);
        }
        if (cqe->res != (int)size && cqe->res > 0) {
             std::cerr << "Short IO Op: " << (int)opcode << " Res: " << cqe->res 
                       << " Expected: " << size << std::endl;
             exit(1);
        }
        
        *cq_head = head + 1;
        __sync_synchronize();
    }
};
