/* solving the 2d stationary heat conduction equation (serial and with CUDA) */
/* Demonstrates usage of custom kernels on THRUST device vectors */
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

#define BLOCK_SIZE 16

void jacobi(float *, uint, uint, uint, float);
void cuda_jacobi(float *, uint, uint, uint, float);
__global__ void jacobi_kernel(float *, float *, uint, uint, uint, float);

/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  uint n = 100;
  n += 2;
  uint m = 100;
  m += 2;
  float *u = (float *)malloc(n * m * sizeof(float));
  uint i, j;
  for (uint k = 0; k < n * m; ++k)
  {
    i = k % n;
    j = k / n;
    if (i == 0)
      u[k] = 1.0f;
    else
      u[k] = 0.0f;
  }

  //jacobi(u, n, m, 1000, 0.0001);
  cuda_jacobi(u, n, m, 200, 0.0001);

  for (uint k = 0; k < n * m; ++k)
  {
    i = k % n;
    j = k / n;
    if (i == n)
      printf("\n");
    printf("%f ", u[i * n + j]);
  }

  free(u);

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////

void jacobi(float *u, uint n, uint m, uint steps, float eps)
{
  uint iter = 0;
  float help, error = eps;

  uint i, j;

  while (error >= eps && iter < steps)
  {
    printf("iteration %d\n", iter);

    error = 0.0f;

    for (uint k = 0; k < (n - 1) * (m - 1); k += 2)
    {
      i = k % n + 1;
      j = k / n + 1;

      if (i < n - 1 && j < m - 1)
      {
        help = u[i * n + j];

        u[i * n + j] = 0.25 * (u[i * n + j + 1] + u[i * n + j - 1] + u[(i + 1) * n + j] + u[(i - 1) * n + j]);

        help = fabs(help - u[i * n + j]);
        if (error < help)
          error = help;
      }
    }

    for (uint k = 1; k < (n - 1) * (m - 1); k += 2)
    {
      i = k % n + 1;
      j = k / n + 1;

      if (i < n - 1 && j < m - 1)
      {
        help = u[i * n + j];

        u[i * n + j] = 0.25f * (u[i * n + j + 1] + u[i * n + j - 1] + u[(i + 1) * n + j] + u[(i - 1) * n + j]);

        help = fabs(help - u[i * n + j]);
        if (error < help)
          error = help;
      }
    }

    ++iter;
  }
}

/////////////////////////////////////////////////////////////////////////////////////

void cuda_jacobi(float *u, uint n, uint m, uint steps, float eps)
{
  float *d_u;
  cudaMalloc(&d_u, n * m * sizeof(float));
  cudaMemcpy(d_u, u, n * m * sizeof(float), cudaMemcpyHostToDevice);

  thrust::device_vector<float> error(n * m);

  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

  uint iter = 0;

  float err = eps;
  while (err >= eps && iter < steps)
  {
    //printf("##############\niteration %d\n##############\n", iter);

    jacobi_kernel<<<blocks, threads>>>(d_u, thrust::raw_pointer_cast(error.data()), n, m, steps, eps);

    err = thrust::reduce(error.begin(), error.end(), (float)0.0f, thrust::maximum<float>());

    ++iter;
  }

  cudaMemcpy(u, d_u, n * m * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_u);
}

__global__ void jacobi_kernel(float *u, float *error, uint n, uint m, uint steps, float eps)
{
  float help;

  uint i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  uint j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  uint g_index = (i - 1) * n + j - 1;

  if (i < n - 1 && j < m - 1)
  {
    help = u[i * n + j];

    u[i * n + j] = 0.25f * (u[i * n + j + 1] + u[i * n + j - 1] + u[(i + 1) * n + j] + u[(i - 1) * n + j]);

    //printf("%d %d %f\n", i, j, u[i*n + j]);

    error[g_index] = fabs(help - u[i * n + j]);
  }
}
