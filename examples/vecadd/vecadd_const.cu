/* CUDA example for the addition of two vectors, demonstration of constant memory */
#include <stdio.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

__constant__ float d_A[12800];

// Device code
__global__ void VecAdd(float *B, float *C, uint N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = d_A[i] + B[i];
}

// Host code
int main(int argc, char **argv)
{
    uint N = 12800;
    uint size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    for (uint i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2.0 * i;
    }

    // Allocate vectors in device memory
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_A, h_A, size);
    // Invoke kernel
    uint threadsPerBlock = 256;
    uint blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (uint i = 0; i < N; ++i)
        printf("%f\n", h_C[i]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}
