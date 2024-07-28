#ifndef PY_EXT_222_GPUMATRIX_H
#define PY_EXT_222_GPUMATRIX_H

class GpuMatrix {
private:
    float* device_mat;
    float* host_mat;
    unsigned int M;
    unsigned int N;
public:
    GpuMatrix(float* host_mat, unsigned int M, unsigned int N);
    ~GpuMatrix();
    float* get_device_mat();
    float* get_host_mat();
    void loadFromDevice();
    void loadToDevice();

    void mul(unsigned int M, unsigned int N, unsigned int K, GpuMatrix* B, GpuMatrix* C);
    void add(unsigned int M, unsigned int N, GpuMatrix* B, GpuMatrix* C);
};

void sayHi();

#endif //PY_EXT_222_GPUMATRIX_H
