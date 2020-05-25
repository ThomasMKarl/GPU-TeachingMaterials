/* measuring the performance of haxpy, saxpy and daxpy */

//download gnuplot: http://ftp.cstug.cz/pub/CTAN/graphics/gnuplot/5.2.6/

/*
nvcc haxpy.cu -arch=compute_61 -code=sm_61 -O3 -std=c++14 -Xcompiler "-O3 -Wall -Wextra -std=c++14" -o axpy
./axpy > axpy.plot
gnuplot
	load "axpy.gnuplot"
*/
#include <stdio.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__
void haxpy(uint n, half a, const half *x, half *y)
{
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
  int n2 = n/2;
  half2 *x2 = (half2*)x, *y2 = (half2*)y;

  for (int i = start; i < n2; i+= stride) 
    y2[i] = __hfma2(__halves2half2(a, a), x2[i], y2[i]);

	// first thread handles singleton for odd arrays
  if (start == 0 && (n%2))
  	y[n-1] = __hfma(a, x[n-1], y[n-1]);   

#else
  for (int i = start; i < n; i+= stride) {
    y[i] = __float2half(__half2float(a) * __half2float(x[i]) 
      + __half2float(y[i]));
  }
#endif
}

__global__
void saxpy(uint n, float a, const float *x, float *y)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < n)
    y[index] += a*x[index];
}

__global__
void daxpy(uint n, double a, const double *x, double *y)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < n)
    y[index] += a*x[index];
}

//////////////////////////////////////////////////////////////////////////////

int main()
{
  const uint blockSize = 128;
  uint nBlocks;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds;
  
  printf("#size half single double\n");
  for(uint n = 12800; n < 12800000; n += 128000)
  {
  nBlocks = (n + blockSize - 1) / blockSize;
  
  ////////////////// half /////////////////////
  
  {
  half a = approx_float_to_half(2.0f);

  half *h_x = (half*)malloc(n * sizeof(half));
  half *h_y = (half*)malloc(n * sizeof(half));
  
  for (uint i = 0; i < n; i++)
  {
    h_x[i] = approx_float_to_half(1.0f);
    h_y[i] = approx_float_to_half((float)i);
  }
  
  half *d_x;
  half *d_y;
  checkCuda(cudaMalloc(&d_x, n * sizeof(half)));
  checkCuda(cudaMalloc(&d_y, n * sizeof(half)));
  checkCuda(cudaMemcpy(d_x, h_x, n * sizeof(half), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_y, h_y, n * sizeof(half), cudaMemcpyHostToDevice));
  
    cudaEventRecord(start);
      haxpy<<<nBlocks, blockSize>>>(n, a, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%d %f ", n, milliseconds);
  

  checkCuda(cudaMemcpy(h_y, d_y, n * sizeof(half), cudaMemcpyDeviceToHost));
  
  checkCuda(cudaFree(d_x));
  checkCuda(cudaFree(d_y));
  
  //for (int i = 0; i < n; i++) printf("%f\n", half_to_float(h_y[i]));

  free(h_x);
  free(h_y);
  }
  
  ////////////////// single /////////////////////
  
  {
  float a = 2.0f;

  float *h_x = (float*)malloc(n * sizeof(float));
  float *h_y = (float*)malloc(n * sizeof(float));
  
  for (uint i = 0; i < n; i++)
  {
    h_x[i] = 1.0f;
    h_y[i] = (float)i;
  }
  
  float *d_x;
  float *d_y;
  checkCuda(cudaMalloc(&d_x, n * sizeof(float)));
  checkCuda(cudaMalloc(&d_y, n * sizeof(float)));
  checkCuda(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
  
    cudaEventRecord(start);
      saxpy<<<nBlocks, blockSize>>>(n, a, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ", milliseconds);  

  checkCuda(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  
  checkCuda(cudaFree(d_x));
  checkCuda(cudaFree(d_y));
  
  //for (int i = 0; i < n; i++) printf("%f\n", half_to_float(h_y[i]));

  free(h_x);
  free(h_y);
  }
  
  ////////////////// double /////////////////////

  {
  double a = 2.0;

  double *h_x = (double*)malloc(n * sizeof(double));
  double *h_y = (double*)malloc(n * sizeof(double));
  
  for (uint i = 0; i < n; i++)
  {
    h_x[i] = 1.0;
    h_y[i] = i;
  }
  
  double *d_x;
  double *d_y;
  checkCuda(cudaMalloc(&d_x, n * sizeof(double)));
  checkCuda(cudaMalloc(&d_y, n * sizeof(double)));
  checkCuda(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice));
  
    cudaEventRecord(start);
      daxpy<<<nBlocks, blockSize>>>(n, a, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f\n", milliseconds); 
  

  checkCuda(cudaMemcpy(h_y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
  
  checkCuda(cudaFree(d_x));
  checkCuda(cudaFree(d_y));
  
  //for (int i = 0; i < n; i++) printf("%f\n", half_to_float(h_y[i]));

  free(h_x);
  free(h_y);
  }
  }
  ///////////////////////////////////////////////////////////////////////////////
  
  return 0;
}

