/* Comparison of a normal vector addition with an asynchronous one, the latter is divided in smaller memory batches */
#include<stdio.h>
#include<gpuerror.h>

////////////////////////////////////////////////////////////////

int VecAdd      (float*, float*, float*, uint);
int VecAddAsync (float*, float*, float*, uint);
__global__ void cuVecAdd(float*, float*, float*, uint);
  
int main()
{
  cudaDeviceTest();
  
  uint N = 1280000;
  uint size = N*sizeof(float);
  
  float *h_A, *h_B, *h_C;
  CUDA_CALL(cudaMallocHost(&h_A, size));
  CUDA_CALL(cudaMallocHost(&h_B, size));
  CUDA_CALL(cudaMallocHost(&h_C, size));
  for(uint i = 0; i < N; ++i)
  {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  CUDA_CALL(VecAdd(h_A, h_B, h_C, N));
  CUDA_CALL(VecAddAsync(h_A, h_B, h_C, N));
  for(uint i = 0; i < 1; ++i) printf("%f\n", h_C[12345]);

  CUDA_CALL(cudaFreeHost(h_A));
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C));

  return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////

int VecAdd(float *A, float *B, float *C, uint N)
{
  unsigned long long size = N*sizeof(float);
  
  float *d_A; CUDA_CALL(cudaMalloc(&d_A, size));
  float *d_B; CUDA_CALL(cudaMalloc(&d_B, size));
  float *d_C; CUDA_CALL(cudaMalloc(&d_C, size));
  CUDA_CALL(cudaMemcpyAsync(d_A, A, size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyAsync(d_B, B, size, cudaMemcpyHostToDevice));
  
  uint threadsPerBlock = 128;
  uint blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  cuVecAdd<<<blocks,threadsPerBlock>>>(d_A, d_B, d_C, N);

  CUDA_CALL(cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaPeekAtLastError());
  #ifdef DEBUG_CUDA
    CUDA_CALL(cudaDeviceSynchronize());
  #endif

  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));

  return EXIT_SUCCESS;
}

int VecAddAsync(float *A, float *B, float *C, uint N)
{
  uint threadsPerBlock = 128;
  uint stride = 10*threadsPerBlock;
  uint blocks = (stride + threadsPerBlock - 1) / threadsPerBlock;
  
  unsigned long long size = stride*sizeof(float);
  
  float *d_A; CUDA_CALL(cudaMalloc(&d_A, size));
  float *d_B; CUDA_CALL(cudaMalloc(&d_B, size));
  float *d_C; CUDA_CALL(cudaMalloc(&d_C, size));
  
  //////////////////////////////////////////////////////////////
  
  uint num = (N - stride + 1) / stride;
  for(uint i = 0; i < num; ++i)
  {
    CUDA_CALL(cudaMemcpyAsync(d_A, A+i*stride,
			      size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_B, B+i*stride,
			      size, cudaMemcpyHostToDevice));
  
    cuVecAdd<<<blocks,threadsPerBlock>>>(d_A, d_B, d_C, stride);
    #ifdef DEBUG_CUDA
      CUDA_CALL(cudaPeekAtLastError());
      CUDA_CALL(cudaDeviceSynchronize());
    #endif

    CUDA_CALL(cudaMemcpyAsync(C+i*stride, d_C,
			      size, cudaMemcpyDeviceToHost));
  }

  CUDA_CALL(cudaMemcpyAsync(d_A, A+num*stride,
			    size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyAsync(d_B, B+num*stride,
			    size, cudaMemcpyHostToDevice));
  
  cuVecAdd<<<blocks,threadsPerBlock>>>(d_A, d_B, d_C, N-num*stride);
  #ifdef DEBUG_CUDA
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
  #endif

  CUDA_CALL(cudaMemcpyAsync(C+num*stride, d_C,
			    size, cudaMemcpyDeviceToHost));

  //////////////////////////////////////////////////////////////
  
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));

  return EXIT_SUCCESS;
}

__global__
void cuVecAdd(float *A, float *B, float *C, uint N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N) C[i] = A[i] + B[i];
}
