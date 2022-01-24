/* a simple demonstration of multi-device management with CUDA and MPI */
//execute: mpiexec -H <name1>,<name2> ./cluster
#include <stdio.h>

#include <mpi.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

#define BLOCK_SIZE 256

__global__ void multiply(const double *, const double *, double *, uint);

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  cudaDeviceProp deviceProp;
  for (int device = 0; device < deviceCount; ++device)
  {
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Host %s with rank %d (of %d) has %d device(s). Its device #%d is %s and has compute capability %d.%d.\n", processor_name, world_rank, world_size, deviceCount, device + 1, deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  uint size = 100 * BLOCK_SIZE;

  double *v1 = (double *)malloc(size * sizeof(double));
  double *v2 = (double *)malloc(size * sizeof(double));
  double *res = (double *)malloc(size * sizeof(double));

  for (uint i = 0; i < size; ++i)
  {
    v1[i] = 0.0001 * i;
    v2[i] = 0.0002 * i;
  }

  double *dv1;
  double *dv2;
  double *dres;
  cudaMalloc(&dv1, size * sizeof(double));
  cudaMalloc(&dv2, size * sizeof(double));
  cudaMalloc(&dres, size * sizeof(double));

  cudaMemcpy(dv1, v1, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dv2, v2, size * sizeof(double), cudaMemcpyHostToDevice);

  multiply<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dv1, dv2, dres, size);

  cudaMemcpy(res, dres, size * sizeof(double), cudaMemcpyDeviceToHost);

  for (uint i = 0; i < size; ++i)
    printf("number %d: %f * %f = %f\n", size + i, v1[i], v2[i], res[i]);

  MPI_Finalize();

  return 0;
}

__global__ void multiply(const double *v1, const double *v2, double *res, uint size)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
    res[index] = v1[index] * v2[index];
}
