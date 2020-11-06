/* utilizing full duplex of PCIe */
/* demonstrates usage of streams */
#include<stdio.h>

int main()
{
  int *h_A, *h_B, *d_A, *d_B;
  
  printf("# MB msec.\n");
  for(unsigned long long N = 1000000; N < 1000000000; N += 50000000)
  {
    cudaMallocHost(&h_A, N*sizeof(int));
    cudaMalloc(&d_A, N*sizeof(int));
    cudaMallocHost(&h_B, N*sizeof(int));
    cudaMalloc(&d_B, N*sizeof(int));
    
    cudaMemcpy(d_A, h_A, N*sizeof(int), cudaMemcpyHostToDevice);
    
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  
  cudaEventRecord(start);
  
    cudaMemcpyAsync(d_B, h_B, N*sizeof(int),
		    cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(h_A, d_A, N*sizeof(int),
		    cudaMemcpyDeviceToHost, stream2);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  
    printf("%f %f\n", 2*N*sizeof(int)/(1024.0f*1024.0f), milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
  }

  return 0;
}
