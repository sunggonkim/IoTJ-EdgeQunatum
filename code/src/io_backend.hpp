#pragma once
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <atomic>
#include "utils.hpp"

inline int io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return syscall(__NR_io_uring_setup, entries, p);
}

inline int io_uring_enter(int ring_fd, unsigned to_submit, unsigned min_complete, unsigned flags, sigset_t *sig) {
    return syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete, flags, sig);
}

class IOBackend {
    int fd;
    std::string path;
    
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

public:
    IOBackend(const std::string& filepath) : path(filepath), uring_enabled(false) {
        fd = open(filepath.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
        if (fd < 0) {
            perror("open O_DIRECT");
            exit(1);
        }
        
        std::cout << "[IOBackend] Opened " << filepath << " (fd: " << fd << ")" << std::endl;
        
        struct io_uring_params params;
        memset(&params, 0, sizeof(params));
        // params.flags = IORING_SETUP_SQPOLL; // Disabled due to EBADF issues
        // params.sq_thread_idle = 2000;
        
        ring_fd = io_uring_setup(64, &params);
        if (ring_fd < 0) {
            perror("io_uring_setup");
            // exit(1);
        }
        
        if (ring_fd >= 0) {
            sq_entries = params.sq_entries;
            cq_entries = params.cq_entries;
            
            size_t sq_size = params.sq_off.array + sq_entries * sizeof(unsigned);
            size_t cq_size = params.cq_off.cqes + cq_entries * sizeof(struct io_uring_cqe);
            
            sq_ptr = mmap(0, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
            cq_ptr = mmap(0, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
            sqes = (struct io_uring_sqe*)mmap(0, sq_entries * sizeof(struct io_uring_sqe), 
                                               PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, 
                                               ring_fd, IORING_OFF_SQES);
            
            if (sq_ptr != MAP_FAILED && cq_ptr != MAP_FAILED && sqes != MAP_FAILED) {
                sq_head = (unsigned*)((char*)sq_ptr + params.sq_off.head);
                sq_tail = (unsigned*)((char*)sq_ptr + params.sq_off.tail);
                sq_array = (unsigned*)((char*)sq_ptr + params.sq_off.array);
                cq_head = (unsigned*)((char*)cq_ptr + params.cq_off.head);
                cq_tail = (unsigned*)((char*)cq_ptr + params.cq_off.tail);
                cqes = (struct io_uring_cqe*)((char*)cq_ptr + params.cq_off.cqes);
                sq_mask = sq_entries - 1;
                cq_mask = cq_entries - 1;
                uring_enabled = true;
            }
        }
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
        if (ring_fd >= 0) close(ring_fd);
    }
    
    void write(size_t offset, void* buf, size_t size) {
        if (uring_enabled) {
            submit_and_wait(IORING_OP_WRITEV, offset, buf, size);
        } else {
            if (pwrite(fd, buf, size, offset) < 0) { perror("pwrite"); exit(1); }
        }
    }
    
    void read(size_t offset, void* buf, size_t size) {
        if (uring_enabled) {
            submit_and_wait(IORING_OP_READV, offset, buf, size);
        } else {
            if (pread(fd, buf, size, offset) < 0) { perror("pread"); exit(1); }
        }
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
