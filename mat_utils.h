#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <stdio.h>
#include <memory>
#include "matrix.hpp"
#include "cuda_utils.hpp"

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        }\
    } while (0)

// Creates three matrices A (M,K), B (K,N) and C(M,N) on the host, Allocates
// memory on the device for all three and copies the data from the host to the 
// device. Matrices A and B are initialised with random values, C is initialised 
// to zero. But we don't copy C to the device as it will be used as an output 
template<typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, cu_unique_ptr<T[]>, cu_unique_ptr<T[]>, cu_unique_ptr<T[]>>
    create_abc_matrices(int M, int K, int N);
    

#endif