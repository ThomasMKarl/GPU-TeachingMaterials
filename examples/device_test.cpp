/* searches for all CUDA-enabled devices and prints specifications */
#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char *argv[]) 
{
  int ndev;
  cudaError_t err = cudaGetDeviceCount(&ndev);

  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));

  if (ndev == 0) 
  {
    fprintf(stderr, "ERROR: No devices. exiting...\n");
    return 1;
  }
  
  printf("Number of devices: %d\n ", ndev);
  cudaSetDevice(0);

  for (int i = 0; i < ndev; ++i) 
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Shared Memory: %zd\n", prop.sharedMemPerBlock);
    printf("Global Memory: %zd\n", prop.totalGlobalMem);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warpsize: %d\n", prop.warpSize);
    printf("Memory Pitch: %zd\n", prop.memPitch);
    printf("Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Thread Dimensions: %d %d %d\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Grid Dimensions: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2]);
    printf("Constant Memory: %ld\n", prop.totalConstMem);
    printf("Major Revision Number: %d\n", prop.major);
    printf("Minor Revision Number: %d\n", prop.minor);
    printf("Alignment Requirement: %ld\n", prop.textureAlignment);
    printf("Device Overlap: %d\n", prop.deviceOverlap);
    printf("Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Timeout enabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf("Integrated GPU: %d\n", prop.integrated);
    printf("Map Host Memory: %d\n", prop.canMapHostMemory);
    printf("Compute Mode: %d\n", prop.computeMode);
    printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
    printf("ECC enabled: %d\n", prop.ECCEnabled);
    printf("PCI Bus ID: %d\n", prop.pciBusID);
    printf("TCC Driver: %d\n\n", prop.tccDriver);
  }

  return 0;
}
