//nvcc -std=c++14 -O3 -o fft clfft_test.cu mkl_fft.o -lOpenCL -lclFFT -lcufft -lfftw3_omp -lfftw3 -lm -gencode arch=compute_60,code=sm_60 -Xcompiler "-fopenmp -Wall -std=c++14" -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl

#include <stdlib.h>
#include <stdio.h>
#include <cufft.h>

int hip_test(float *X, uint N);

int main(int argc, char* argv[])
{
    float *X;
    long long max = pow(2,30); 
    
    printf("#size FFTW clFFT cuFFT\n");
    for(uint N = 2; N <= max; N *= 2)
    {
        X = (float *)malloc(N*2*sizeof(float));
        for(unsigned int i = 0; i < 2*N; ++i) X[i] = 2.341;
 
        printf("%d ", N);
        hip_test(X,N);
    }
    
    return 0;
}


int hip_test(float *X, uint N)
{
      hipEvent_t start, stop;
      hipEventCreate(&start);
      hipEventCreate(&stop);

//////////////////////////////////////////////////////////////////////////////////
    cufftComplex *hdata = (cufftComplex *)malloc(sizeof(cufftComplex)*N);
    for(unsigned int i = 0; i < N; ++i)
    {
        hdata[i].x = X[2*i];
        hdata[i].y = X[2*i+1];
    }

      hipEventRecord(start);

    cufftHandle plan;
    
    cufftComplex *ddata;
    hipMalloc((void **)&ddata, sizeof(cufftComplex)*N);      
    hipMemcpy(ddata, hdata, sizeof(cufftComplex)*N, hipMemcpyHostToDevice);
    
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, ddata, ddata, CUFFT_FORWARD);
      
    hipMemcpy(hdata, ddata, sizeof(cufftComplex)*N, cudaMemcpyDeviceToHost);
    //for(unsigned int i = 0; i < N; ++i) printf("%f\n%f\n", hdata[i].x, hdata[i].y);
      
      
    cufftDestroy(plan);
    hipFree(ddata);
      
      hipEventRecord(stop);
      hipEventSynchronize(stop);
      float milliseconds = 0;
      hipEventElapsedTime(&milliseconds, start, stop);
      printf("%f ", milliseconds);

    free(hdata);

    return 0;
}
