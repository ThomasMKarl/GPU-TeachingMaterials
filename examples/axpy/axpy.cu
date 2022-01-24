/* measuring the performance of haxpy, saxpy and daxpy */
// download gnuplot: http://ftp.cstug.cz/pub/CTAN/graphics/gnuplot/5.2.6/
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"


inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG_CUDA)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T> __global__
void saxpy(unsigned long long int n, T a, const T *x, T *y)
{
  unsigned long long int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n)
    y[index] += a * x[index];
}

template <> __global__
void saxpy(unsigned long long int n, half a, const half *x, half *y)
{
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
  int n2 = n / 2;
  half2 *x2 = (half2 *)x, *y2 = (half2 *)y;

  for (int i = start; i < n2; i += stride)
    y2[i] = __hfma2(__halves2half2(a, a), x2[i], y2[i]);

  // first thread handles singleton for odd arrays
  if (start == 0 && (n % 2))
    y[n - 1] = __hfma(a, x[n - 1], y[n - 1]);

#else
  for (int i = start; i < n; i += stride) {
    y[i] =
        __float2half(__half2float(a) * __half2float(x[i]) + __half2float(y[i]));
  }
#endif
}

//////////////////////////////////////////////////////////////////////////////

template <typename T> 
class DeviceCoordinates
{
public:
  DeviceCoordinates() = default;

  DeviceCoordinates(const size_t number_) : number(number_)
  {
    checkCuda(cudaMalloc(&x, number * sizeof(T)));
    checkCuda(cudaMalloc(&y, number * sizeof(T)));
  }

  ~DeviceCoordinates()
  {
    cudaFree(x);
    cudaFree(y);
  }

  std::pair<T *, T *> data() const { return std::make_pair(x, y); }

  size_t size() const { return number; }

private: 
  T *x{};
  T *y{};
  size_t number{0};
};

template <typename T> 
class Coordinates
{
public:
  Coordinates() = default;
  Coordinates(const size_t number)
  {
    x.resize(number);
    y.resize(number);
  }

  std::pair<T *, T *> data() const
  {
    return std::make_pair<T *, T *>(x.data(), y.data());
  }

  void initWithNumber(const T number)
  {
    std::fill(x.begin(), x.end(), number);
    std::fill(y.begin(), y.end(), number);
  }

  std::pair<T, T> getCoord(const size_t index) const
  {
    return std::make_pair<T, T>(x[index], y[index]);
  }

  void setCoord(const size_t index, const std::pair<T, T> input)
  {
    x[index] = input.first;
    y[index] = input.second;
  }

  cudaError_t upload(DeviceCoordinates<T> &device)
  {
    cudaError_t result =
        cudaMemcpy(device.data().first, x.data(), x.size() * sizeof(T),
                   cudaMemcpyHostToDevice);
    if (result != cudaSuccess) return result;

    return cudaMemcpy(device.data().second, y.data(), y.size() * sizeof(T),
                      cudaMemcpyHostToDevice);
  }

  cudaError_t download(DeviceCoordinates<T> &device)
  {
    return cudaMemcpy(y.data(), device.data().second, y.size() * sizeof(T),
                      cudaMemcpyDeviceToHost);
  }

private : 
  std::vector<T> x{};
  std::vector<T> y{};
};


template <typename T>
cudaError_t saxpy_wrapper(const T scalar, DeviceCoordinates<T> &input)
{
  const uint blockSize{256};
  size_t nBlocks = std::ceil(input.size() / blockSize);
  saxpy<<<nBlocks, blockSize>>>(input.size(), scalar, input.data().first, input.data().second);
  cudaError_t result = cudaSuccess;
#ifdef DEBUG_CUDA
  result = cudaDeviceSynchronize();
#endif
  return result;
}


class Recorder 
{
public:
  Recorder()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  ~Recorder()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void record()
  {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
  }
  
  void reset()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  float getTime() const { return milliseconds; }

private:
  float milliseconds{};
  cudaEvent_t start{};
  cudaEvent_t stop{};
};


int main(int argc, const char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Error: usage: " << argv[0] << " <plot file>\n";
    return EXIT_FAILURE;
  }
  std::string filename{argv[1]};
  std::ofstream stream{filename, std::ios::out};
  if (stream.bad())
  {
    std::cerr << "Error: Could not write file" << filename << "\n";
    return EXIT_FAILURE;
  }

  stream << "#size half single double\n";
  for (size_t n = 12800; n < 12800000; n += 128000)
  {
    ////////////////// half /////////////////////
    {
      Coordinates<half> host{n};
      host.initWithNumber(approx_float_to_half(1.23f));
      DeviceCoordinates<half> device{n};
      checkCuda(host.upload(device));

      Recorder rec{};
      checkCuda(saxpy_wrapper(approx_float_to_half(2.0f), device));
      rec.record();

      checkCuda(host.download(device));

      stream << n << " " << rec.getTime() << " ";
    }
    ////////////////// single /////////////////////
    {
      Coordinates<float> host{n};
      host.initWithNumber(1.23f);
      DeviceCoordinates<float> device{n};
      checkCuda(host.upload(device));

      Recorder rec{};
      checkCuda(saxpy_wrapper(2.0f, device));
      rec.record();

      checkCuda(host.download(device));

      stream << rec.getTime() << " ";
    }
    ////////////////// double /////////////////////
    {
      Coordinates<double> host{n};
      host.initWithNumber(1.23);
      DeviceCoordinates<double> device{n};
      checkCuda(host.upload(device));

      Recorder rec{};
      checkCuda(saxpy_wrapper(2.0, device));
      rec.record();

      checkCuda(host.download(device));

      stream << rec.getTime() << "\n";
    }
  }
  ///////////////////////////////////////////////////////////////////////////////

  return EXIT_SUCCESS;
}
