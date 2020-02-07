/* Demonstration of dynamic parllelism */
//compile: nvcc dynpar.cu -arch=compute_61 -code=sm_61 -rdc=true -lcudadevrt
#include<stdio.h>

__global__ void kernel_parent(float* data, uint* N);
__global__ void kernel_child (float* data, uint* N);
  
int main()
{
  uint N = 19200;
  float *host = (float*)malloc(N*sizeof(float));

  float *device;
  cudaMalloc(&device, N*sizeof(float));
  cudaMemcpy(device, host, N*sizeof(float), cudaMemcpyHostToDevice);

  // spawns a kernel with one thread
  kernel_parent<<<1,1>>>(device, &N);

  cudaError_t error = cudaDeviceSynchronize();
  if(error == cudaSuccess) return 0;
  else
  {
    printf(cudaGetErrorString(error));
    return -1;
  }
}

/////////////////////////////////////////////////////////

__global__ void kernel_parent(float* data, uint* N)
{
  uint newN = *N/2;
  if(newN == 0) return; //stops when each thread has exactly one point
  //spawn two new kernels with one thread each
  //copy half the data

  cudaStream_t stream1; cudaStreamCreateWithFlags(&stream1, cudaStreamDefault);
  float *device1;
  cudaMalloc(&device1, newN*sizeof(float));
  cudaMemcpyAsync(device1, data, newN*sizeof(float), cudaMemcpyDeviceToDevice, stream1);
  
  kernel_child <<<1, 1, 0, stream1>>>(device1, &newN);
  
  cudaStream_t stream2; cudaStreamCreateWithFlags(&stream2, cudaStreamDefault);
  float *device2;
  cudaMalloc(&device2, newN*sizeof(float));
  cudaMemcpyAsync(device2, data+*N/2, newN*sizeof(float), cudaMemcpyDeviceToDevice, stream2);
  
  kernel_child <<<1, 1, 0, stream2>>>(device2, &newN);
}

__global__ void kernel_child(float* data, uint *N)
{
  uint newN = *N/2;
  if(newN == 0) return; //stops when each thread has exactly one point
  //spawn two new kernels with one thread each
  //copy half the data
  
  cudaStream_t stream1; cudaStreamCreateWithFlags(&stream1, cudaStreamDefault);
  float *device1;
  cudaMalloc(&device1, newN*sizeof(float));
  cudaMemcpyAsync(device1, data, newN*sizeof(float), cudaMemcpyDeviceToDevice, stream1);
  
  kernel_child <<<1, 1, 0, stream1>>>(device1, &newN);
  
  cudaStream_t stream2; cudaStreamCreateWithFlags(&stream2, cudaStreamDefault);
  float *device2;
  cudaMalloc(&device2, newN*sizeof(float));
  cudaMemcpyAsync(device2, data+*N/2, newN*sizeof(float), cudaMemcpyDeviceToDevice, stream2);
  
  kernel_child <<<1, 1, 0, stream2>>>(device2, &newN);
}
