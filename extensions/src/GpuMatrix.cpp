#include "GpuMatrix.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "matmul.cuh"


// TODO: REMOVE!!!
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
   else {
      printf("CUDA SUCCESS! \n");
   }
}

GpuMatrix::GpuMatrix(float* host_mat, unsigned int M, unsigned int N) {
    this->host_mat = host_mat;
    this->M = M;
    this->N = N;
    cudaMalloc((void**) (&(this->device_mat) ), (M * N) * sizeof(float));
    cudaMemcpy(this->device_mat, this->host_mat, (M * N) * sizeof(float), cudaMemcpyHostToDevice);
}

GpuMatrix::~GpuMatrix() {
    std::cout << "DESTRUCT CALLED" << std::endl;
    gpuErrchk( cudaFree(this->device_mat) );
}

float *GpuMatrix::get_device_mat() {
    return this->device_mat;
}

float *GpuMatrix::get_host_mat() {
    return this->host_mat;
}

void GpuMatrix::loadFromDevice() {
    cudaMemcpy(this->host_mat, this->device_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void GpuMatrix::loadToDevice() {
    cudaMemcpy(this->device_mat, this->host_mat, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void GpuMatrix::mul(unsigned int M, unsigned int N, unsigned int K, GpuMatrix* B, GpuMatrix* C) {
    matmul(M, N, K,this->device_mat, B->device_mat, C->device_mat);
    cudaDeviceSynchronize();
}

void sayHi() {
    std::cout << "HI!!!!" << std::endl;
}
