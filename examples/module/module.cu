//compile with nvcc into shared or static library
#include "module.h"

void test(float *A)
{
  kernel<<<1,100>>>(A);
  CUDA_CALL(cudaDeviceSynchronize());
}

__global__ void kernel(float *A)
{
  int i = threadIdx.x;
  int a = 5;
  int b = 5*a;
  A[0] = b;
  printf("%d",i);
}
