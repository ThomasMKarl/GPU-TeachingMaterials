/* implementation of a max-reduction with five levels of optimization */
#include<stdio.h>

__global__ void PR_v1(int *d)
{
  uint tid = threadIdx.x;
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  
  extern __shared__ int sm[];
  sm[tid] = d[i];//copy to SM
  
  for(uint stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    if(tid % (2*stride) == 0) sm[tid] += sm[tid + stride];
  }
  
  if(tid == 0) d[blockIdx.x] = sm[0]; //copy back
  //d[blockIdx.x] containts the sum of the block
}

__global__ void PR_v2(int *d)
{
  uint tid = threadIdx.x;
  uint i = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ int sm[]; 
  sm[tid] = d[i];//copy to SM
  
  uint index;
  for (uint stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    index = 2*stride*tid; //replace the divergent branch with a non-divergent one
    if(index < blockDim.x) sm[index] += sm[index + stride];
  }
  
  if(tid == 0) d[blockIdx.x] = sm[0]; //copy back
  //d[blockIdx.x] containts the sum of the block
}

__global__ void PR_v3(int *d)
{
  uint tid = threadIdx.x;
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  
  extern __shared__ int sm[];
  sm[tid] = d[i];//copy to SM
  
  for(uint stride = blockDim.x/2; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if(tid < stride) sm[tid] += sm[tid + stride]; //replace the strided loop with a reversed one
  }
  
  if(tid == 0) d[blockIdx.x] = sm[0]; //copy back
  //d[blockIdx.x] containts the sum of the block
}

__global__ void PR_v4(int *d)
{
  uint tid = threadIdx.x;
  uint i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  
  extern __shared__ int sm[]; 
  sm[tid]= d[i] + d[i + blockDim.x]; //do the first reduction during the load from GM 
                                 //replace single load with two loads
  for (uint stride = blockDim.x/2; stride > 0; stride >>= 1) //use only the half the blocks
  {
    __syncthreads();
    if(tid < stride) sm[tid] += sm[tid + stride];
  }
  
  if(tid == 0) d[blockIdx.x] = sm[0]; //copy back
  //d[blockIdx.x] containts the sum of the block
}

__global__ void PR_v5(int *d)
{
  //Number of active threads decreases with the number of iterations
  //for stride<=32 we have only one warp 
  //warp runs the same instruction (SIMD)
  //for stride<=32 __syncthreads() is not necessary
  //unroll last 6 iterations
  
  uint tid = threadIdx.x;
  uint i = blockIdx.x*(blockDim.x*2)+threadIdx.x;

  extern __shared__ int sm[]; 
  sm[tid] = d[i] + d[i + blockDim.x];
  
  for (uint stride = blockDim.x/2; stride > 32; stride >>= 1)
  {
    __syncthreads();
    if(tid<stride) sm[tid] += sm[tid + stride];
  }
  if (tid < 32)
  {
    sm[tid] += sm[tid + 32];
    sm[tid] += sm[tid + 16];
    sm[tid] += sm[tid +  8];
    sm[tid] += sm[tid +  4];
    sm[tid] += sm[tid +  2];
    sm[tid] += sm[tid +  1];
  }
  
  if(tid == 0) d[blockIdx.x] = sm[0]; //copy back
  //d[blockIdx.x] containts the sum of the block
}


///////////////////////////////////////////////////////////
int main()
{
  //all elements are in the shared memory
  uint n = 2<<11;//# of integers
  uint size = n*sizeof(int);//datasize [B]
    
  uint maxThreads = 128; //threads per block
  uint threads = (n < maxThreads) ? n : maxThreads;
  uint blocks = n/threads; //# of blocks!!

  uint smSize = threads*sizeof(int);//shared mem

  int *hOut;
  hOut = (int*)malloc(size);
  for(uint i = 0; i < n; ++i) hOut[i] = 1;

  int *d;
  cudaMalloc(&d, size);
  cudaMemcpy(d, hOut, size, cudaMemcpyHostToDevice);
  //note - size of the data is the blocksize

  PR_v1<<<blocks, threads, smSize>>>(d);  

  uint toDo = 0;
  if (blocks>1) toDo = 1 + blocks/maxThreads;
  for(uint i = 0; i < toDo; ++i)
  {
    threads = (blocks < maxThreads) ? blocks : maxThreads;
    blocks /= threads;
    PR_v5<<<blocks, threads, smSize>>>(d);
  }
  cudaMemcpy(hOut, d, sizeof(int), cudaMemcpyDeviceToHost); //hOut[0] is the result

  printf("sum of a %d dimensional unit-vector: %d\n", n, hOut[0]);
  
  return 0;
}
