from gpumat import PyGpuMatrix
import numpy as np

import time

M = 12000
K = 12000
N = 12000

k = np.ones((M, N), dtype=np.float32)

A = PyGpuMatrix(np.ones((M, K), dtype=np.float32), M, K)
B = PyGpuMatrix(np.ones((K, N), dtype=np.float32), K, N)
C = PyGpuMatrix(k, M, N)

start = time.time()
print("Start Multiplication!")
A.mul(M, N, K, B, C)
C.loadFromDevice()

stop = time.time()
print("Done!")

print("Time elapsed: ", stop - start)

print(k)



