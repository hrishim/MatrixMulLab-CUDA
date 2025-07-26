#include <type_traits>
#include <algorithm> // for std::copy
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "mat_utils.h"

// Constructor
template<typename T>
Matrix<T>::Matrix(int rows, int cols, bool device)
    : M(rows), N(cols), in_device(device) { 
        if (device) {
            T* raw_dev_data;
            cudaMalloc((void**)&raw_dev_data, rows * cols * sizeof(T));
            cudaCheckErrors("Failed to allocate memory for device matrix");
            device_data = cu_unique_ptr<T[]>(raw_dev_data);
        } else {
            device_data = nullptr;
            data = std::make_unique<T[]>(rows * cols);
        }
    }

// Element access
template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    if (in_device) {
        throw std::runtime_error("operator() cannot be used on device matrix data");
    }
    return data[i * N + j];
}

template<typename T>
const T& Matrix<T>::operator()(int i, int j) const {
    if (in_device) {
        throw std::runtime_error("operator() cannot be used on device matrix data");
    }
    return data[i * N + j];
}

// Move constructor
template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : M(other.M), N(other.N), in_device(other.in_device)  {
    // Device matrices must not be moved. This is a logic error.
    assert(!in_device && "Move constructor called on a device matrix, which is not allowed!");
    data = std::move(other.data);
    other.M = 0;
    other.N = 0;
}

// Move assignment
template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        M = other.M;
        N = other.N;
        data = std::move(other.data);

        other.M = 0;
        other.N = 0;
    }
    return *this;
}

// Clone method (deep copy)
template<typename T>
Matrix<T> Matrix<T>::clone() const {
    Matrix<T> result(M, N);
    std::copy(data.get(), data.get() + (M * N), result.data.get());
    return result;
}

// Clone device matrix to host matrix
// Only call on a device matrix
template<typename T>
Matrix<T> Matrix<T>::clone_device_to_host() const {
    if (!in_device) {
        throw std::runtime_error("clone_device_to_host() called on a host matrix");
    }
    Matrix<T> host_mat(M, N, false);
    cudaMemcpy(host_mat.data.get(), device_data.get(), M * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy device matrix to host");
    return host_mat;
}

// Clone host matrix to device matrix
// Only call on a host matrix
template<typename T>
Matrix<T> Matrix<T>::clone_host_to_device() const {
    if (in_device) {
        throw std::runtime_error("clone_host_to_device() called on a device matrix");
    }
    Matrix<T> device_mat(M, N, true);
    cudaMemcpy(device_mat.device_data.get(), data.get(), M * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy host matrix to device");
    return device_mat;
}

template<typename T>
void Matrix<T>::print() const {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << data[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}

// Explicit template instantiations for common types
template struct Matrix<float>;
template struct Matrix<int>;