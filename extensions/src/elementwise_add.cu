#include "matmul.cuh"
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void add(unsigned const int M, unsigned const int N, const float* A, const float* B, float* C) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int array_idx = (row * N) + col;
    if (row < M && col < N) {
        C[ array_idx ] = A[array_idx] + B[array_idx];
    }
}

void elementwise_add(unsigned const int M, unsigned const int N, const float* A, const float* B, float* C) {
    dim3 dimBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid( (N / BLOCK_SIZE) + 1, (M / BLOCK_SIZE) + 1);
    add<<< dimGrid, dimBlocks >>>(M, N, A, B, C);
    cudaDeviceSynchronize();
}