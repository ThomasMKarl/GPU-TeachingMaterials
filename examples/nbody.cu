/* nbody problem: verlet algorithm for particles in a 1/r^2 potential (serial and with CUDA) */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

#define CUDA_ERROR_CHECK

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
    exit(-1);
  }

  err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

float3 calc_force(float3 *, uint, uint, float);
void nbody_serial(float3 *, float3 *, float, float, uint, float3, float, float);
//////////////////////////////////////////////////////////////////////////////
void nbody_cuda(float3 *, float3 *, float, float, uint, float3, float, float);
__global__ void nbody_forces(float3 *, float3 *, float, uint, float, float);
__global__ void nbody_kernel(float3 *, float3 *, float3 *, uint, float3);

int main(int argc, char **argv)
{
  uint n = 10;
  float3 size = make_float3(1.0, 1.0, 1.0);

  float t_end = 10.0f;
  float delta_t = 1.0f;

  srand(time(NULL));

  float3 *x0 = (float3 *)malloc(n * sizeof(float3));
  float3 *x1 = (float3 *)malloc(n * sizeof(float3));
  for (uint i = 0; i < n; ++i)
  {
    x0[i].x = (rand() % 10000) / 10000.0;
    x0[i].y = (rand() % 10000) / 10000.0;
    x0[i].z = (rand() % 10000) / 10000.0;

    x1[i].x = (rand() % 10000) / 10000.0;
    x1[i].y = (rand() % 10000) / 10000.0;
    x1[i].z = (rand() % 10000) / 10000.0;

    //printf("%f %f %f %f %f %f\n", x0[i].x, x0[i].y, x0[i].z, x1[i].x, x1[i].y, x1[i].z);
  }

  float m = 1.0f;
  float radius = 0.5f;

  /////////////////////////////////////////////////////////////////////////

  //nbody_serial(x0, x1, t_end, delta_t, n, size, m, radius);
  nbody_cuda(x0, x1, t_end, delta_t, n, size, m, radius);

  /////////////////////////////////////////////////////////////////////////

  for (uint i = 0; i < n; ++i)
    printf("%f %f %f\n", x1[i].x, x1[i].y, x1[i].z);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////

float3 calc_force(float3 *x, uint i, uint n, float rad)
{
  float3 force;
  force.x = 0.0f;
  force.y = 0.0f;
  force.z = 0.0f;

  float dist, dif_x, dif_y, dif_z, constant = 1.0f; //=1/(4*pi*epsilon_0*epsilon_r)
  for (uint j = 0; j < n; ++j)
  {
    dif_x = x[i].x - x[j].x;
    dif_y = x[i].y - x[j].y;
    dif_z = x[i].z - x[j].z;

    dist = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

    if (dist <= rad)
    {
      dist = constant / (dist * dist * dist);
      force.x += constant * dif_x;
      force.y += constant * dif_y;
      force.z += constant * dif_z;
    }
  }

  return force;
}

void nbody_serial(float3 *x0, float3 *x1, float t_end, float delta_t, uint n, float3 size,
                  float m = 1.0f, float radius = 1.0f)
{
  float3 *buf = (float3 *)malloc(n * sizeof(float3));

  float3 acc;
  for (float t = 0.0f; t <= t_end; t += delta_t)
  {
    //printf("t = %f\n", t);
    for (uint i = 0; i < n; ++i)
    {
      //printf("particle %d: ", i);

      acc = calc_force(x1, i, radius, n);
      acc.x = acc.x / m * delta_t * delta_t;
      acc.y = acc.y / m * delta_t * delta_t;
      acc.z = acc.z / m * delta_t * delta_t;

      buf[i].x = x1[i].x;
      buf[i].y = x1[i].y;
      buf[i].z = x1[i].z;

      x1[i].x = 2 * x1[i].x - x0[i].x + acc.x;
      x1[i].y = 2 * x1[i].y - x0[i].y + acc.y;
      x1[i].z = 2 * x1[i].z - x0[i].z + acc.z;

      while (x1[i].x >= size.x)
        x1[i].x -= size.x;
      while (x1[i].y >= size.y)
        x1[i].y -= size.y;
      while (x1[i].z >= size.z)
        x1[i].z -= size.z;
      while (x1[i].x < 0)
        x1[i].x += size.x;
      while (x1[i].y < 0)
        x1[i].y += size.y;
      while (x1[i].z < 0)
        x1[i].z += size.z;

      //printf("%f %f %f\n", x1[i].x, x1[i].y, x1[i].z);

      x0[i].x = buf[i].x;
      x0[i].y = buf[i].y;
      x0[i].z = buf[i].z;
    }
  }
}

///////////////////////////////////////////////////

#define THREADS 256

void nbody_cuda(float3 *x0, float3 *x1, float t_end, float delta_t, uint n, float3 size,
                float m = 1.0f, float radius = 1.0f)
{
  uint threads_per_block = THREADS;
  uint blocks = (n + threads_per_block - 1) / threads_per_block;

  float3 *d_x0;
  cudaMalloc(&d_x0, n * sizeof(float3));
  CudaCheckError();
  cudaMemcpy(d_x0, x0, n * sizeof(float3), cudaMemcpyHostToDevice);
  CudaCheckError();
  float3 *d_x1;
  cudaMalloc(&d_x1, n * sizeof(float3));
  CudaCheckError();
  cudaMemcpy(d_x1, x1, n * sizeof(float3), cudaMemcpyHostToDevice);
  CudaCheckError();

  float3 *d_forces;
  cudaMalloc(&d_forces, n * sizeof(float3));
  CudaCheckError();

  for (float t = 0.0f; t <= t_end; t += delta_t)
  {
    //printf("t = %f\n", t);
    nbody_forces<<<blocks, threads_per_block>>>(d_x0, d_forces, delta_t, n, m, radius);
    CudaCheckError();
    nbody_kernel<<<blocks, threads_per_block>>>(d_x0, d_x1, d_forces, n, size);
    CudaCheckError();
  }

  cudaMemcpy(x1, d_x1, n * sizeof(float3), cudaMemcpyDeviceToHost);
  CudaCheckError();
}

__global__ void nbody_forces(float3 *x, float3 *forces, float delta_t, uint n, float m, float rad)
{
  uint g_index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 force;
  force.x = 0.0f;
  force.y = 0.0f;
  force.z = 0.0f;

  float dist, dif_x, dif_y, dif_z, constant = 1.0f; //=1/(4*pi*epsilon_0*epsilon_r)
  for (uint j = 0; j < n; ++j)
  {
    dif_x = x[g_index].x - x[j].x;
    dif_y = x[g_index].y - x[j].y;
    dif_z = x[g_index].z - x[j].z;

    dist = sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z);

    if (dist <= rad)
    {
      dist = constant / (dist * dist * dist);
      force.x += constant * dif_x;
      force.y += constant * dif_y;
      force.z += constant * dif_z;
    }

    __syncthreads();
  }

  forces[g_index].x = force.x / m * delta_t * delta_t;
  forces[g_index].y = force.y / m * delta_t * delta_t;
  forces[g_index].z = force.z / m * delta_t * delta_t;
}

__global__ void nbody_kernel(float3 *x0, float3 *x1, float3 *forces, uint n, float3 size)
{
  uint g_index = blockIdx.x * blockDim.x + threadIdx.x;
  uint s_index = threadIdx.x;

  __shared__ float3 buf[THREADS];
  __shared__ float3 s_x1[THREADS];

  if (g_index < n)
  {
    //printf("particle %d: ", g_index);

    buf[s_index].x = x1[g_index].x;
    buf[s_index].y = x1[g_index].y;
    buf[s_index].z = x1[g_index].z;

    s_x1[s_index].x = buf[s_index].x;
    s_x1[s_index].y = buf[s_index].y;
    s_x1[s_index].z = buf[s_index].z;

    s_x1[s_index].x = 2 * buf[s_index].x - x0[g_index].x + forces[g_index].x;
    s_x1[s_index].y = 2 * buf[s_index].y - x0[g_index].y + forces[g_index].y;
    s_x1[s_index].z = 2 * buf[s_index].z - x0[g_index].z + forces[g_index].z;

    while (s_x1[s_index].x >= size.x)
      s_x1[s_index].x -= size.x;
    while (s_x1[s_index].y >= size.y)
      s_x1[s_index].y -= size.y;
    while (s_x1[s_index].z >= size.z)
      s_x1[s_index].z -= size.z;
    while (s_x1[s_index].x < 0)
      s_x1[s_index].x += size.x;
    while (s_x1[s_index].y < 0)
      s_x1[s_index].y += size.y;
    while (s_x1[s_index].z < 0)
      s_x1[s_index].z += size.z;

    __syncthreads();

    x1[g_index].x = s_x1[s_index].x;
    x1[g_index].y = s_x1[s_index].y;
    x1[g_index].z = s_x1[s_index].z;
    //printf("%f %f %f\n", x1[g_index].x, x1[g_index].y, x[g_index].z);

    x0[g_index].x = buf[s_index].x;
    x0[g_index].y = buf[s_index].y;
    x0[g_index].z = buf[s_index].z;
  }
}
