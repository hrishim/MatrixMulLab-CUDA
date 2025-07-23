#include <stdio.h>
#include <tuple>
#include "mat_utils.h"
#include "matrix.hpp"

// Creates three matrices A (M,K), B (K,N) and C(M,N) on the host, Allocates
// memory on the device for all three and copies the data from the host to the 
// device. Matrices A and B are initialised with random values, C is initialised 
// to zero. But we don't copy C to the device as it will be used as an output 

template<typename T>
std::tuple<
    Matrix<T>, Matrix<T>, Matrix<T>,
    cu_unique_ptr<T[]>, cu_unique_ptr<T[]>, cu_unique_ptr<T[]>
>
create_abc_matrices(int M, int K, int N) {
    Matrix<T> h_A = Matrix<T>::random_matrix(M, K);
    Matrix<T> h_B = Matrix<T>::random_matrix(K, N);
    Matrix<T> h_C = Matrix<T>::zero_matrix(M, N);
    
    T* raw_dA = nullptr;
    T* raw_dB = nullptr;
    T* raw_dC = nullptr;

    cudaMalloc((void**)&raw_dA, M * K * sizeof(T));
    cudaCheckErrors("Failed to allocate memory for raw_dA");
    cudaMalloc((void**)&raw_dB, K * N * sizeof(T));
    cudaCheckErrors("Failed to allocate memory for raw_dB");
    cudaMalloc((void**)&raw_dC, M * N * sizeof(T));
    cudaCheckErrors("Failed to allocate memory for raw_dC");

    cu_unique_ptr<T[]> d_A(raw_dA);
    cu_unique_ptr<T[]> d_B(raw_dB);
    cu_unique_ptr<T[]> d_C(raw_dC);

    cudaMemcpy(d_A.get(), h_A.data.get(), M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy data to d_A");
    cudaMemcpy(d_B.get(), h_B.data.get(), K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy data to d_B");

    return std::make_tuple(std::move(h_A), std::move(h_B), std::move(h_C), std::move(d_A), std::move(d_B), std::move(d_C));
}
