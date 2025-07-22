#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <stdio.h>

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

// Initialises a MxN matris with random values [0,1)
void init_matrix(float* mat, int M, int N);

// Initialises a MxN matris with zero values
void zero_matrix(float* mat, int M, int N);

// Creates an identity matrix NxN
void identity_matrix(float* mat, int N);

// Creates three matrices A (M,K), B (K,N) and C(M,N) on the host, Allocates
// memory on the device for all three and copies the data from the host to the 
// device. Matrices A and B are initialised with random values, C is initialised 
// to zero. But we don't copy C to the device as it will be used as an output 
void create_matrices(float** h_A, float** h_B, float** h_C, float** d_A, float** d_B, float** d_C, int M, int K, int N);

#endif