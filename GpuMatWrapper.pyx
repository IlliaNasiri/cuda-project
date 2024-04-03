#distutils: language=c++
cimport numpy as np

cdef extern from "GpuMatrix.h":
    void sayHi();

    cdef cppclass GpuMatrix:
        GpuMatrix(float * host_mat, unsigned int M, unsigned int N);
        float * get_device_mat();
        float * get_host_mat();
        void loadFromDevice();
        void loadToDevice();

        void mul(unsigned int M, unsigned int N, unsigned int K, GpuMatrix* B, GpuMatrix* C);

cpdef void aaa():
    sayHi()

cdef class PyGpuMatrix:
    cdef GpuMatrix *thisptr  # Pointer to C++ object

    def __cinit__(self, np.ndarray[np.float32_t, ndim=2] mat, M, N):
        self.thisptr = new GpuMatrix(&mat[0,0], M, N)

    def __dealloc__(self):
        del self.thisptr

    def loadFromDevice(self):
        self.thisptr.loadFromDevice()

    def loadToDevice(self):
        self.thisptr.loadToDevice()

    def mul(self, M, N, K, PyGpuMatrix B, PyGpuMatrix C):
        self.thisptr.mul(M, N, K, B.thisptr, C.thisptr)
