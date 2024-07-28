# Motivation and Project Description #
Ever since I have learnt about deep learning, I was curious to find out how Gradient descent works. Also, I really wanted to understand how exactly GPUs speed it up inference and training steps in deep learning; 
therefore, I have decided to take on this project to answer these questions for myself.

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

**Colab:**

```shell
!mkdir cuda-project
!git clone https://github.com/IlliaNasiri/cuda-project ./cuda-project
%cd cuda-project/extensions/src
!make
%cd ../../
!pip install . 

```




