# Motivation and Project Description #
Ever since I have learnt about deep learning, I was curious to find out how Gradient descent works. Also, I really wanted to understand how exactly GPUs speed it up inference and training steps in deep learning; 
therefore, I have decided to take on this project to answer these questions for myself.

# How does this work: #
The way this project works is as follows: there exists some CUDA code (with .cuh and .cu) extensions, that is compiled as a shared library (.so) via the nvcc compiler. Then, using Cython, and Setup.py, it is built as an extension that is callable from Python as a library. 

# Installation: # 

**REQUIREMENTS:** As for the currect phase of development, the requirements are as follows:
1. Linux (Windows subsystem for linux could be used as well)
2. CUDA capable GPU
3. CUDA toolkit, installable via:
   ```sudo apt install nvidia-cuda-toolkit   ``` (Read  below if you do not want to install cuda toolkit)
5. Python Libraries: numpy, cython
6. Some patience with the installation 

**IF YOU CANNOT FULFILL ANY OF THE REQUIREMENTS, OR DO NOT WANT TO INSTALL CUDA TOOLKIT**: you can easily repeat the steps given below on in google colab. 

**Linux Terminal:**

```shell
mkdir cuda-project
git clone https://github.com/IlliaNasiri/cuda-project ./cuda-project
cd cuda-project/extensions/src
make
cd ../../
pip install . 
```

**Colab: MAKE SURE YOU ARE CONNECTED TO A GPU!** 

```shell
!mkdir cuda-project
!git clone https://github.com/IlliaNasiri/cuda-project ./cuda-project
%cd cuda-project/extensions/src
!make
%cd ../../
!pip install . 

```

# Usage: #
At this stage of development, the only implemented functionality constitutes matrix multiplication which is one of fundamental operations required to make neural networks work.

To tests its working you can the following code once you have installed the library: 


```python
from gpumat import PyGpuMatrix
import numpy as np

import time

M = 5000
K = 5000
N = 5000

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

```


# TO-DOs #
* (DONE) Matrix Multiplication
* Elementwise operations (multiplication, addition, negation)
* Matrix Transpose (used in the backward pass)
* Random matrix initialization (used for initializing weights)
* Write an interface so that it could be invoked through Python
* Implement CUDA ReLU on a matrix

  *Once this functionality has been implemented:*
* Implementing the forward, and backward passes for a *linear layer*
* Implementing ReLU activation
* Implementig forward and backward passes for a Convolutional layer
* Pooling layer
* Implementing loss function (MSE and MAE), with it's forward/backward computations. 


