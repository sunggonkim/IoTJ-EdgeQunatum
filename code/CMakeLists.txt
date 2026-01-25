cmake_minimum_required(VERSION 3.18)
project(EdgeQuantum CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# Try to find custatevec manually if not in CUDAToolkit
find_library(CUSTATEVEC_LIB custatevec HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
find_path(CUSTATEVEC_INC custatevec.h HINTS /usr/local/cuda/include /usr/include)

if(NOT CUSTATEVEC_LIB OR NOT CUSTATEVEC_INC)
    message(WARNING "custatevec not found! Please assume it is in standard paths or set CUSTATEVEC_ROOT.")
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS} ${LZ4_INCLUDE_DIRS} ${CUSTATEVEC_INC})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${LZ4_LIBRARY_DIRS})

add_executable(edge_quantum
    src/main.cpp
    src/simulator.cpp
    src/chunk_manager.cpp
    src/io_backend.cpp
)

target_link_libraries(edge_quantum
    CUDA::cudart
    ${LZ4_LIBRARIES}
    ${CUSTATEVEC_LIB}
    pthread
)

# Architectures: Jetson Orin = sm_87
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(edge_quantum PRIVATE -O3 --use_fast_math -arch=sm_87)
endif()
