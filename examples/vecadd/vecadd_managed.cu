/* CUDA example for the addition of two vectors */
#include <stdio.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

// Device code
__global__ void VecAdd(float* A, float* B, float* C, uint N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main(int argc, char **argv)
{
    uint N = 12800;
    uint size = N * sizeof(float);

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
    uint threadsPerBlock = 256;
    uint blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    for(uint i = 0; i < N; ++i) printf("%f\n", C[i]);

    // Free device memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
