#include <stdio.h>
#include "mat_utils.h"

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {

} 

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int M = 1024; 
    int K = 1024; 
    int N = 1024;

    create_matrices(&h_A, &h_B, &h_C, &d_A, &d_B, &d_C, M, K, N);
    matmul<<<1, 1>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}