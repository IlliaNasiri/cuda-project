import numpy as np
import subprocess

def install_extension():
    res = subprocess.run(["pip", "install", "."], capture_output=True, text=True)
    print(res.stdout)

install_extension()

from gpumat import PyGpuMatrix
import time

M = 10000
K = 10000
N = 10000

k = np.ones((M, N), dtype=np.float32)

A = PyGpuMatrix(np.ones((M, K), dtype=np.float32), M, K)
B = PyGpuMatrix(np.ones((K, N), dtype=np.float32), K, N)
C = PyGpuMatrix(k, M, N)

start = time.time()
print("Start Multiplication!")
# A.mul(M, N, K, B, C)
A.add(M, N, B, C)
C.loadFromDevice()

stop = time.time()
print("Done!")

print("Time elapsed: ", stop - start)

print(k)



