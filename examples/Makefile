CXX=g++
CC=gcc
HCC=hipcc
PGCC=/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/bin/pgc++
SM=61
NVCC=nvcc -allow-unsupported-compiler -arch=compute_$(SM) -code=sm_$(SM)
MPICC=mpicc
CLANG=CLANG
ICC=icpc
AR=ar
RANLIB=ranlib

NVCC_FLAGS=-O3 -std=c++17
CXX_FLAGS=-O3 -Wall -Wextra -std=c++17
C_FLAGS=-O3 -Wall -Wextra -std=c17

OCL_INCLUDE=/opt/cuda/include
OCL_LIB=/opt/cuda/lib64
MPI_INCLUDE=/usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIB=/usr/lib/x86_64-linux-gnu/openmpi/lib/
CUDA_INCLUDE=/opt/cuda/include
CUDA_LIB=/opt/cuda/lib64
ROC_INCLUDE=/opt/cuda/include
ROC_LIB=/opt/cuda/lib64
OCV_INCLUDE=/usr/include/opencv4
OCV_LIB=/usr/lib64
MDNN_INCLUDE=
MDNN_LIB=
MAGMA_INCLUDE=
MAGMA_LIB=
BLAS_INCLUDE=
BLAS_LIB=
BLAS=openblas
PARA_INCLUDE=
PARA_LIB=


.PHONY: axpy
axpy:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" axpy/axpy.cu -o build/axpy/axpy -I axpy
	build/axpy/axpy build/axpy/axpy.data
	gnuplot axpy/axpy.gp > axpy/axpy_fit.log

bubble:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" bubble.cu -o build/bubblesort

cluster:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" cluster.cu -o build/cluster -I $(MPI_INCLUDE) -L $(MPI_LIB) -lmpi

curand:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" curand.cu -o build/curand -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcurand

device_test:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" device_test.cu -o build/device_test
	$(CXX) $(CXX_FLAGS) device_test.cpp -o build/device_test -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcudart

dynpar:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" dynpar.cu -o build/dynamic_parallelism -rdc=true -lcudadevrt

heat:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" heat.cu -o build/heat

nbody:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" nbody.cu -o build/nbody

nccl:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" nccl.cu -o build/nccl -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lnccl

pci:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" nbody.cu -o build/bandwidth_test

pi:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" pi.cu -o build/pi -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcurand

pipe:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" pipe.cu -o build/pipeline -DDEBUG_CUDA

reduction: libgpuerror_cuda.a
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" reduction.cu -o build/reduction build/libgpuerror_cuda.a -DDEBUG_CUDA

reverse:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" reverse_shared.cu -o build/reverse

thrust:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" thrust.cu -o build/thrust

oacc:
	$(PGCC) -ta=tesla -Minfo=accel oacc_test.c -o build/oacc

omp:
	$(CC) -O3 -Wall -Wextra -std=c17 omp_test.c -o build/omp -fopenmp

ising:
	$(CXX) $(CXX_FLAGS) ising_cluster.cpp -I $(MPI_INCLUDE) -L $(MPI_LIB) -lmpi_cxx -lmpi -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcudart -fopenmp -o build/ising


gpuerror:
	make libgpuerror_cuda.a 
	make libgpuerror_ocl.a 
	make libgpuerror.a

libgpuerror_cuda.a:
	$(CXX) $(CXX_FLAGS) -c -o gpuerror_cuda.o -fPIC -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcudart -DDEBUG_CUDA gpuerror.cpp
	$(AR) crs build/libgpuerror_cuda.a gpuerror_cuda.o
	rm gpuerror_cuda.o
	ranlib build/libgpuerror_cuda.a

libgpuerror_ocl.a:
	$(CXX) $(CXX_FLAGS) -c -o gpuerror_ocl.o -fPIC -I $(OCL_INCLUDE) -L $(OCL_LIB) -lOpenCL -DDEBUG_OCL gpuerror.cpp
	ar crs build/libgpuerror_ocl.a gpuerror_ocl.o
	rm gpuerror_ocl.o
	ranlib build/libgpuerror_ocl.a

libgpuerror.a:
	$(CXX) $(CXX_FLAGS) -c -o gpuerror.o -fPIC -I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcudart -DDEBUG_CUDA -I $(OCL_INCLUDE) -L $(OCL_LIB) -lOpenCL -DDEBUG_OCL gpuerror.cpp
	$(AR) crs build/libgpuerror.a gpuerror.o
	rm gpuerror.o
	$(RANLIB) build/libgpuerror.a


convolution:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" ML/conv.cu -o build/ML/convolution -I $(OCV_INCLUDE) -L $(OCV_LIB) -lcudnn -lopencv_core

pooling:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" ML/pooling.cu -o build/ML/pooling -I $(OCV_INCLUDE) -lcudnn

magmadnn:
	$(CXX) $(CXX_FLAGS) ML/magmadnn.cpp -o build/magmadnn -I $(MDNN_INCLUDE) -L $(MDNN_LIB) -lmagmadnn -lcudnn


linalg: matmul cusparse cusolver magma lu_magma_cpu lu_magma_gpu paralution

cusparse:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" linalg/cusparse.cu -o build/linalg/cusparse -lcusparse

cusolver:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" linalg/cusolver.cu -o build/linalg/cusolver -lcusolver

lu_magma_cpu:
	$(CC) $(C_FLAGS) -fopenmp -o build/linalg/lu_magma_cpu linalg/lu_magma_cpu.c -DADD -I $(MAGMA_INCLUDE) -L $(MAGMA_LIB) -I $(BLAS_INCLUDE) -L $(BLAS_LIB) -lmagma -l$(BLAS)

lu_magma_gpu:
	$(CC) $(C_FLAGS) -fopenmp -o build/linalg/lu_magma_gpu linalg/lu_magma_gpu.c -DADD -I $(MAGMA_INCLUDE) -L $(MAGMA_LIB) -I $(BLAS_INCLUDE) -L $(BLAS_LIB) -lmagma -l$(BLAS)

magma:
	$(CC) $(C_FLAGS) -fopenmp -o build/linalg/magma linalg/magma.c -I $(MAGMA_INCLUDE) -L $(MAGMA_LIB) -I $(BLAS_INCLUDE) -L $(BLAS_LIB) -lmagma -l$(BLAS)

matmul:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" linalg/matmul.cu -o build/linalg/matmul

paralution:
	$(CXX) $(CXX_FLAGS) linalg/para.cpp -o build/linalg/paralution -fopenmp -I $(PARA_INCLUDE) -L $(PARA_LIB) -lparalution


vadd: vadd_const vadd_nonconst vadd_hip vadd_managed vadd_cl vadd_clcpp

vadd_nonconst:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" vecadd/vecadd.cu -o build/vecadd/vecadd_nonconst

vadd_const:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" vecadd/vecadd_const.cu -o build/vecadd/vecadd_const

vadd_managed:
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" vecadd/vecadd_managed.cu -o build/vecadd/vecadd_managed

vadd_hip:
	$(HCC) $(CXX_FLAGS) vecadd/vecadd.hip.cu -o build/vecadd/vecadd_hip

vadd_cl:
	$(CC) $(C_FLAGS) vecadd/cl_vecadd.c -o build/vecadd/vecadd_cl -I $(OCL_INCLUDE) -L $(OCL_LIB) -lOpenCL

vadd_clcpp:
	$(CXX) $(CXX_FLAGS) vecadd/cl_vecadd.cpp -o build/vecadd/vecadd_clcpp -I $(OCL_INCLUDE) -L $(OCL_LIB) -lOpenCL

vecadd_llvm:
	$(CLANG) -emit-llvm build/vecadd/cl_vecadd.cl -o build/vecadd/cl_vecadd.bc -c -S -Xclang -finclude-default-header


mkl_fft.o:
	$(ICC) fft/mkl_fft.c -qmkl -qtbb -fPIE -cxxlib-nostd -no-cxxlib -c

fft: mkl_fft.o
	$(CXX) $(CXX_FLAGS) -I . -I fft -o build/fft/fft fft/fft_test.cpp mkl_fft.o \
	-I $(OCL_INCLUDE) -L $(OCL_LIB) -lOpenCL -lclFFT \
	-I $(CUDA_INCLUDE) -L $(CUDA_LIB) -lcudart -lcufft \
	-lfftw3_omp -lfftw3 -fopenmp -lm \
	-lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -ldl \
	-DDEBUG_CUDA -DDEBUG_OCL build/libgpuerror.a

	rm mkl_fft.o

bench_fft: fft
	build/fft/fft > build/fft/output.dat


doc:
	$(DOX) Doxyfile


clean:
	rm -rf build/*
	rm -rf ../doc/*