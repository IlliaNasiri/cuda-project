#include "../src/GpuMatrix.h"
#include <iostream>

void fill_ones(float* arr, unsigned int size) {
    for(int i = 0; i < size; i++)
        arr[i] = 1;
}

void fill_consecutive(float* arr, unsigned int size) {
    for(int i = 0; i < size; i++)
        arr[i] = i;
}

void print_array(float* arr, unsigned int size, int M, int N) {
    for(int i = 0; i < size; i++) {
        if( i % N == 0)
            std::cout << std::endl;
        std::cout << arr[i] << " ,";
    }
    std::cout << std::endl;
}

int main() {

    unsigned const int M = 1000;
    unsigned const int K = 1000;
    unsigned const int N = 1000;

    float* host_A = new float[M * K];
    float* host_B = new float[K * N];
    float* host_C = new float[M * N];

    fill_consecutive(host_A, M*K);
    fill_consecutive(host_B, K*N);

    GpuMatrix* A = new GpuMatrix(host_A, M, K);
    GpuMatrix* B = new GpuMatrix(host_B, K, N);
    GpuMatrix* C = new GpuMatrix(host_C, K, N);

    A->mul(M,N,K, B, C);

    C->loadFromDevice();

    std::cout << C->get_host_mat()[0] << std::endl;

    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    delete A;
    delete B;
    delete C;

    return 0;

}

