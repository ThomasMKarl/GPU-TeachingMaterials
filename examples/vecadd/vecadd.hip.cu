#include "hip/hip_runtime.h"
/* CUDA example for the addition of two vectors */
#include <stdio.h>

#define HIP_CALL(x) do { hipError_t err = x; if(err!=hipSuccess) { \
    printf("'%s' at %s:%d\n",hipGetErrorString(x),__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = 12800;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    for(uint i = 0; i < N; ++i)
    {
        h_A[i] = i;
	h_B[i] = 2.0*i;
    }

    // Allocate vectors in device memory
    float* d_A;
    HIP_CALL(hipMalloc(&d_A, size));
    float* d_B;
    HIP_CALL(hipMalloc(&d_B, size));
    float* d_C;
    HIP_CALL(hipMalloc(&d_C, size));

    exit(-1);

    // Copy vectors from host memory to device memory
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(VecAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    for(uint i = 0; i < N; ++i) printf("%f\n", h_C[i]);

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
            
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}
