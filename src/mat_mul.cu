#include <stdio.h>
#include <iostream>
#include "mat_utils.h"



// 
__global__ void matmul_serial_k(float* A, float* B, float* C, int M, int K, int N) {

    //Multiply row of A into column of B
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d\nrow: %d, col: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col);
    if((row<M) && (col<N)) {
        float sum = 0;
        for(int i=0; i<K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
} 

void test_print_matrix(){

    Matrix<float> h_A = Matrix<float>::identity_matrix(3, 3);
    h_A.print();
    std::cout << "\n";
    Matrix<float> h_B = Matrix<float>::random_matrix(3, 3);
    h_B.print();
    std::cout << "\n";
    Matrix<float> h_C = Matrix<float>::zero_matrix(3, 3);
    h_C.print();
}

int main() {
    //float *h_A, *h_B, *h_C;
    //float *d_A, *d_B, *d_C;
    int M = 4; 
    int K = 4; 
    int N = 4;

    //test_print_matrix();
    Matrix<float> h_A = Matrix<float>::random_matrix(M, K);
    Matrix<float> h_B = Matrix<float>::identity_matrix(K, N);
    Matrix<float> d_A = h_A.clone_host_to_device();
    Matrix<float> d_B = h_B.clone_host_to_device();
    Matrix<float> d_C = Matrix<float>(M, N, true);
    h_A.print();
    std::cout << "\n";
    h_B.print();
    std::cout << "\n";

    dim3 threads(K, K);
    matmul_serial_k<<<1,threads>>>(d_A.device_data.get(), d_B.device_data.get(), d_C.device_data.get(), M, K, N);
    Matrix<float> h_C = d_C.clone_device_to_host();
    h_C.print();

    //create_matrices(&h_A, &h_B, &h_C, &d_A, &d_B, &d_C, M, K, N);
    //matmul<<<1, 1>>>(d_A, d_B, d_C, M, K, N);
    //cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}