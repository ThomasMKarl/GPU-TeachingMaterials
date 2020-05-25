/* CUDA example for the addition of two vectors */
#include <stdio.h>

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

    // Allocate vectors in device memory
    float* A;
    cudaMallocManaged(&A, size);
    float* B;
    cudaMallocManaged(&B, size);
    float* C;
    cudaMallocManaged(&C, size);

    // Initialize input vectors
    for(uint i = 0; i < N; ++i)
    {
        A[i] = i;
	B[i] = 2.0*i;
    }

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    for(uint i = 0; i < N; ++i) printf("%f\n", C[i]);

    // Free device memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
