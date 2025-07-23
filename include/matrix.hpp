#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <memory>

template<typename T>
struct Matrix {
    int M;
    int N;
    std::unique_ptr<T[]> data;

    // Constructor
    Matrix(int rows, int cols);

    // Clone method (deep copy)
    Matrix<T> clone() const;

    // Element access
    T& operator()(int i, int j);
    const T& operator()(int i, int j) const;

    // Move constructor and move assignment
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    // Delete copy constructor and copy assignment
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Get size in bytes
    size_t size() const { return M * N * sizeof(T); }
    //print matrix to stdout
    void print() const;

    // Return a random matrix
    static Matrix<T> random_matrix(int M, int N) {
        static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                      "Matrix<T>::random_matrix supports only floating-point or integral types.");
        Matrix<T> mat(M, N);
        if constexpr (std::is_floating_point<T>::value) {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    mat(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        } else if constexpr (std::is_integral<T>::value) {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    mat(i, j) = rand() % 100;  // Produce int in [0, 99]
        }
        return mat;
    }

    // Return a MxN matrix with zero values
    static Matrix<T> zero_matrix(int M, int N) {
        Matrix<T> mat(M, N);
        std::fill(mat.data.get(), mat.data.get() + (M * N), static_cast<T>(0));
        return mat;
    }

    // Return a MxN matrix with identity values
    static Matrix<T> identity_matrix(int M, int N) {
        Matrix<T> mat(M, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                mat(i, j) = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
        return mat;
    }
};
    
#endif