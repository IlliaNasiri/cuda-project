all: testMatmul

testMatmul: matmul_tester.cpp ../src/GpuMatrix.cpp ../src/matmul.cu
	nvcc $^ -o $@
