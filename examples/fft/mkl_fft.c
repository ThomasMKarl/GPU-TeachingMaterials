//icc mkl_fft.c -mkl -tbb -fPIE -cxxlib-nostd -no-cxxlib -c
#include "mkl_fft.h"
  
MKL_LONG mkl_test(float *X, size_t N)
{
  MKL_LONG status;
  
  float _Complex *in = (float _Complex*)malloc(N*sizeof(float _Complex));
  for(uint i = 0; i < N; ++i)
  {
    in[i] = X[2*i] + X[2*i+1]*I;
    //printf("%d %d\n", X[2*i], X[2*i+1]*I);
  }

  float milliseconds = second();

  DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_SINGLE,
				                  DFTI_COMPLEX, 1, N);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, in);
  status = DftiFreeDescriptor(&my_desc1_handle);

    milliseconds = 1000.0*(second() - milliseconds);
    printf("%f\n", milliseconds);
    
  return status;
}
