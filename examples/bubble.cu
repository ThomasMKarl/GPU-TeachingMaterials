/* implemantation of the bubble-sort algorithm in CUDA */
#include <stdio.h>

#define BLOCK_SIZE 192

uint bubblesort(float*, unsigned long long);
cudaError_t cuda_bubblesort(float*, unsigned long long);
__host__ __device__ void swap(float &, float &);  
__global__ void bubble_kernel1(float*, unsigned long long, unsigned short int*);
__global__ void bubble_kernel2(float*, unsigned long long, unsigned short int*);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
  unsigned long long size = 3000;
  
  float *h_array = (float*)malloc(size*sizeof(float));
  for(unsigned long long i = 0; i < size; ++i) h_array[i] = float(size-i);

  cuda_bubblesort(h_array, size);
  //bubblesort(h_array, size);

  //for(uint i = 0; i < size; ++i)
  //{
    //printf("element %d: %f\n", i, h_array[i]);
  //}

  free(h_array);
  
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__
void swap(float &x, float &y)
{
  x = x + y;
  y = x - y;
  x = x - y;
}

uint bubblesort(float *h_array, unsigned long long size)
{
  uint newn, counter = 0;

  do
  {
    newn = 1;
    for(uint i = 0; i < size-1; ++i)
    {
      if(h_array[i] > h_array[i+1])
      {
	swap(h_array[i], h_array[i+1]);

	newn = i+1;
      }
    }
    
    size = newn;
    
    ++counter;
    
  } while(size > 1);

  printf("sorted after %d steps.\n", counter);
  
  return 0;
}

cudaError_t cuda_bubblesort(float *h_array, unsigned long long size)
{
  cudaError_t error;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  float *d_array;
  error = cudaMalloc(&d_array, size*sizeof(float));
	     
  error = cudaMemcpyAsync(d_array, h_array, size*sizeof(float), cudaMemcpyHostToDevice, stream);


  /////////////////////////////////////////

  unsigned long long number_of_blocks = ( ((size+1)/2) + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
  
  uint *h_swapped = (uint*)malloc(sizeof(uint));
  unsigned short int *d_swapped;
  error = cudaMalloc(&d_swapped, sizeof(unsigned short int));

  unsigned long long counter = 0;
  
  do
  {
    h_swapped[0] = 0;

    error = cudaMemcpyAsync(d_swapped, h_swapped, sizeof(unsigned short int),
			    cudaMemcpyHostToDevice, stream);

    bubble_kernel1<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(d_array, size, d_swapped);
    bubble_kernel2<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(d_array, size, d_swapped);
    
    error = cudaMemcpyAsync(h_swapped, d_swapped, sizeof(unsigned short int),
			    cudaMemcpyDeviceToHost, stream);

    ++counter;
  } while(h_swapped[0]);
  
  /////////////////////////////////////////

  cudaDeviceSynchronize();
  error = cudaMemcpy(h_array, d_array, size*sizeof(float), cudaMemcpyDeviceToHost);

  error = cudaFree(d_array);
  
  printf("sorted after %d steps.\n", counter);
	     
  return error;
}

__global__
void bubble_kernel1(float *d_array, unsigned long long size, unsigned short int *d_swapped)
{
  unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x;

  if(2*i < size)
  {
  if(d_array[2*i] > d_array[2*i+1])
  {
    swap(d_array[2*i], d_array[2*i+1]); 
    d_swapped[0] = 1;
  }
  }
}

__global__
void bubble_kernel2(float *d_array, unsigned long long size, unsigned short int *d_swapped)
{
  unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x;

  if(2*i+1 < size-1)
  {
  if(d_array[2*i+1] > d_array[2*i+2])
  {
    swap(d_array[2*i+1], d_array[2*i+2]);
    d_swapped[0] = 1;
  }
  }
}
