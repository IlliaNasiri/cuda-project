CC = gcc
CXX = g++
NVC = nvcc

LDLIBS = -lglfw -lGLEW -lGL

all: libmatmul.so libgpumat.so

libgpumat.so: GpuMatrix.cpp
	nvcc $^ -o $@ --shared --compiler-options -fPIC

libmatmul.so: matmul.cu
	nvcc $^ -o $@ --shared --compiler-options -fPIC

clean:
	rm -rf *.so