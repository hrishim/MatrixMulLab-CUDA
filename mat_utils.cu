#include <stdio.h>
#include "mat_utils.h"


// Initialises an MxN matrix with random values [0,1)
void init_matrix(float* mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = (float)rand() / RAND_MAX;
        }
    }
}

// Initialises a MxN matrix with zero values
void zero_matrix(float* mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = 0.0f;
        }
    }
}
    
// Creates an identity matrix NxN
void identity_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Creates three matrices A (M,K), B (K,N) and C(M,N) on the host, Allocates
// memory on the device for all three and copies the data from the host to the 
// device. Matrices A and B are initialised with random values, C is initialised 
// to zero. But we don't copy C to the device as it will be used as an output 
void create_matrices(float** h_A, float** h_B, float** h_C, float** d_A, float** d_B, float** d_C, int M, int K, int N) {
    *h_A = (float*)malloc(M * K * sizeof(float));
    *h_B = (float*)malloc(K * N * sizeof(float));
    *h_C = (float*)malloc(M * N * sizeof(float));
    init_matrix(*h_A, M, K);
    init_matrix(*h_B, K, N);
    zero_matrix(*h_C, M, N);
    cudaMalloc((void**)d_A, M * K * sizeof(float));
    cudaCheckErrors("Failed to allocate memory for d_A");
    cudaMalloc((void**)d_B, K * N * sizeof(float));
    cudaCheckErrors("Failed to allocate memory for d_B");
    cudaMalloc((void**)d_C, M * N * sizeof(float));
    cudaCheckErrors("Failed to allocate memory for d_C");

    cudaMemcpy(*d_A, *h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy data to d_A");
    cudaMemcpy(*d_B, *h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy data to d_B");
    //cudaMemcpy(*d_C, *h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
}