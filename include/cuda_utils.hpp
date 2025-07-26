#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

//#include <cuda_runtime.h>

// Custom deleter for CUDA device memory
struct CudaDeleter {
    template <typename T>
    void operator()(T* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

// Alias for unique_ptr with CUDA-aware deleter
template <typename T>
using cu_unique_ptr = std::unique_ptr<T, CudaDeleter>;

#endif