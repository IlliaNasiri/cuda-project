#ifndef PY_EXT_222_MATMUL_CUH
#define PY_EXT_222_MATMUL_CUH

/**
 * A wrapper function around src kernel matrix multiplication of matrices A, B, whose result is stored in matrix C.
 * Determines the appropriate block and grid size, invokes the kernel on the GPU,
 * and then waits for the GPU to finish before returning control back to the CPU.
 * @param M number of rows in left matrix A, and number of rows resulting matrix result C
 * @param N number of columns in right matrix A, and number of columns in matrix result C
 * @param K number of columns in matrix A, number of rows in matrix B
 * @param A float32 array located ON THE GPU MEMORY, that represents flattened matrix A
 * @param B float32 array located ON THE GPU MEMORY, that represents flattened matrix B
 * @param C float32 array located ON THE GPU MEMORY, that represents flattened result matrix C
 */
void matmul(unsigned const int M, unsigned const int N, unsigned const int K, const float* A, const float* B, float* C);

#endif //PY_EXT_222_MATMUL_CUH
