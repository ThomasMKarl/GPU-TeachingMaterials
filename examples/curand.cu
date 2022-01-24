/* a demonstration of the usage of cuRAND (host and device API)*/
#include<stdio.h>
#include<time.h>

#include<curand.h>
#include<curand_kernel.h>


#ifdef _WIN32
using uint = unsigned int;
#endif

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__
void setup_kernel(curandState *state, uint n)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(NULL, id, 0, &state[id]);
}

__global__
void generate_kernel(curandState *state, float *result, uint n)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if(id < n)
  {
    curandState localState = state[id];

    float x = curand_uniform(&localState);
    /* use x in device */
    result[id] = x;

    state[id] = localState;
  }
}


int main(int argc, char **argv)
{
    size_t n = 10;
    uint block_size = 128;
    
    ////////////////////////////////////////////////////////////////////////////////////////
    
    float *hostResults = (float*)malloc(n*sizeof(float));
    float *devResults;
    cudaMalloc(&devResults, n*sizeof(float));
    
    curandState *devStates;
    cudaMalloc((void **)&devStates , n*sizeof(curandState));
    setup_kernel<<<(n+block_size-1)/block_size, block_size>>>(devStates, n);
    generate_kernel<<<(n+block_size-1)/block_size, block_size>>>(devStates, devResults, n);
    cudaFree(devStates);

    cudaMemcpy(hostResults, devResults, n*sizeof(float), cudaMemcpyDeviceToHost);

    for(uint i = 0; i < n; i++) printf("%f\n", hostResults[i]);
    
    ////////////////////////////////////////////////////////////////////////////////////////
    printf("\n");
    
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    long long seed = time(NULL);
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CALL(curandGenerateUniform(gen, devResults, n));
    CURAND_CALL(curandDestroyGenerator(gen));

    cudaMemcpy(hostResults, devResults, n*sizeof(float), cudaMemcpyDeviceToHost);

    for(uint i = 0; i < n; i++) printf("%f\n", hostResults[i]);
    
    ////////////////////////////////////////////////////////////////////////////////////////
    
    cudaFree(devResults);
    free(hostResults);
  
    return 0;
}
