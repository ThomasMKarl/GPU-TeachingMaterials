/* CUDA example for the addition of two vectors */
#include <stdio.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

// Device code
__global__ void VecAdd(float *A, float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main(int argc, char **argv)
{
    size_t N = 12800;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2.0f * i;
    }

    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool error{false};
    for (size_t i = 0; i < N; ++i)
    {
        if(h_C[i] != 3.0f*i)
        {
            printf("Error at element index %zd!\n", i);
            error = true;
            break;
        }
    }
    if (!error)
        printf("No errors.\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}
