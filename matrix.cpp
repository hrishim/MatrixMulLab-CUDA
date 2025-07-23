#include <type_traits>
#include <algorithm> // for std::copy
#include <iostream>

#include "matrix.hpp"

// Constructor
template<typename T>
Matrix<T>::Matrix(int rows, int cols)
    : M(rows), N(cols), data(std::make_unique<T[]>(rows * cols)) { }

// Element access
template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    return data[i * N + j];
}

template<typename T>
const T& Matrix<T>::operator()(int i, int j) const {
    return data[i * N + j];
}

// Move constructor
template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : M(other.M), N(other.N), data(std::move(other.data)) {
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