#include "matmul.cuh"
#include <stdio.h>

#define BLOCK_SIZE 32


/**
 * CUDA kernel that multiplies matrix A (MxK) by matrix B (KxN), and stores result in matrix C (MxN)
 * @param M number of rows in left matrix A, and number of rows resulting matrix result C
 * @param N number of columns in right matrix A, and number of columns in matrix result C
 * @param K number of columns in matrix A, number of rows in matrix B
 * @param A float32 array located on the GPU memory, that represents flattened matrix A
 * @param B float32 array located on the GPU memory, that represents flattened matrix B
 * @param C float32 array located on the GPU memory, that represents flattened result matrix C
 */
__global__ void sgemm(unsigned const int M, unsigned const int N, unsigned const int K, const float* A, const float* B, float* C) {
    __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

    unsigned int
        block_row = threadIdx.x / BLOCK_SIZE,
        block_col = threadIdx.x % BLOCK_SIZE,
        global_row = blockIdx.y * BLOCK_SIZE + block_row,
        global_col = blockIdx.x * BLOCK_SIZE + block_col,
        stride = 0;

    float tmp = 0.0f;

    for(int i = 0; i < K; i += BLOCK_SIZE) {
        unsigned int A_shared_idx = block_row * BLOCK_SIZE + block_col;
        unsigned int B_shared_idx = block_row * BLOCK_SIZE + block_col;

        A_shared[A_shared_idx] = ((stride + block_col) < K) ?
                A[(global_row * K) + stride + block_col] : 0.0f;

        B_shared[B_shared_idx] = ((stride + block_row) < K) ?
                B[(stride + block_row) * N + global_col] : 0.0f;

        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++)
            tmp += A_shared[block_row * BLOCK_SIZE + j] * B_shared[j * BLOCK_SIZE + block_col];

        __syncthreads();
        stride += BLOCK_SIZE;
    }

    if(global_row < M && global_col < N )
        C[global_row * N + global_col] = tmp;



}

void matmul(unsigned const int M, unsigned const int N, unsigned const int K, const float* A, const float* B, float* C) {
    dim3 dimBlocks(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid( (N / BLOCK_SIZE) + 1, (M / BLOCK_SIZE) + 1);
    sgemm<<< dimGrid, dimBlocks >>>(M, N, K, A, B, C);
    cudaDeviceSynchronize();
}



