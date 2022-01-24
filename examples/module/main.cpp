/* demonstrates the interoperability of g++ toolchains with nvcc compiled GPU code */
#include<cuda.h>
#include<cuda_runtime.h>

#include "module.h" //include some gpu code


int main(int argc, char **argv)
{
  float *d_A;
  CUDA_CALL(cudaMalloc(&d_A, sizeof(float)));
  test(d_A);

  printf("done\n");
  return 0;
}
