/* measuring the performance of saxpy with THRUST device vectors and transformations */
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    return a * x + y;
  }
};

void saxpy_fast(float A, thrust::device_vector<float> &X, thrust::device_vector<float> &Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float> &X, thrust::device_vector<float> &Y)
{
  thrust::device_vector<float> temp(X.size());

  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);

  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

__global__ void mysaxpy(uint n, float a, const float *x, float *y)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n)
    y[index] += a * x[index];
}

int main()
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds;

  uint nBlocks;
  const uint blockSize = 256;

  for (uint size = 10000; size < 10000000; size += 10000)
  {
    printf("%d ", size);

    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    for (uint i = 0; i < size; ++i)
    {
      h_x.push_back(1.0f);
      h_y.push_back(2.0f);
    }

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    cudaEventRecord(start);
    saxpy_slow(3.0f, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);
    //h_y = d_y;
    //for(uint i = 0; i < h_y.size(); ++i) std::cout << h_y[i] << std::endl;

    printf(" ");

    cudaEventRecord(start);
    saxpy_fast(3.0f, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);
    //h_y = d_y;
    //for(uint i = 0; i < h_y.size(); ++i) std::cout << h_y[i] << std::endl;

    printf(" ");

    nBlocks = (size + blockSize - 1) / blockSize;
    cudaEventRecord(start);
    mysaxpy<<<nBlocks, blockSize>>>(size, 3.0f, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f\n", milliseconds);
    //h_y = d_y;
    //for(uint i = 0; i < h_y.size(); ++i) std::cout << h_y[i] << std::endl;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
