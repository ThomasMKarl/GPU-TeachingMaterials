#include "gpuerror.h"
#include <iostream>

#include <fftw3.h>
#include <omp.h>

#include <clFFT.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "mkl_fft.h"


#define PLATFORM_N 0
#define DEVICE_N 0
#define OMP_NUM_THREADS 12

int omp_test(std::vector<float> &X);
int cuda_test(std::vector<float> &X);
int cl_test(std::vector<float> &X);

int main(int argc, char *argv[]) 
{
  size_t max = pow(2, 20);

  std::vector<float> X;

  std::cout << "#size FFTW clFFT cuFFT mklFFT\n";
  for (size_t N = 2; N <= max; N *= 2) 
  {
    for (size_t i = 0; i < 2 * N; ++i) 
    {
      // std::cout << i+1 << "/" << 2*N << "\n";
      X.push_back(2.341f);
    }

    std::cout << N << " ";
    omp_test(X);
    cl_test(X);
    cuda_test(X);
    mkl_test(X.data(),N);
  }

  return EXIT_SUCCESS;
}

int cuda_test(std::vector<float> &X) 
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //////////////////////////////////////////////////////////////////////////////////

  thrust::host_vector<cufftComplex> hdata(X.size() / 2);
  for (unsigned int i = 0; i < X.size() / 2; ++i) 
  {
    hdata[i].x = X[2 * i];
    hdata[i].y = X[2 * i + 1];
  }

  cudaEventRecord(start);

  {
    thrust::device_vector<cufftComplex> ddata;
    ddata = hdata;

    cudaEventRecord(start);

    cufftHandle plan;
    CUFFT_CALL(cufftPlan1d(&plan, ddata.size(), CUFFT_C2C, 1));
    CUFFT_CALL(cufftExecC2C(plan, thrust::raw_pointer_cast(ddata.data()),
                            thrust::raw_pointer_cast(ddata.data()),
                            CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan));

    cudaEventRecord(stop);

    hdata = ddata;
  }

  CUDA_CALL(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << milliseconds << " ";

  // for(unsigned int i = 0; i < N; ++i) printf("%f\n%f\n", hdata[i].x,
  // hdata[i].y);

  return EXIT_SUCCESS;
}

int omp_test(std::vector<float> &X) 
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //////////////////////////////////////////////////////////////////////////////////

  fftw_complex *in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (X.size() / 2));
  for (unsigned int i = 0; i < X.size() / 2; ++i) {
    in[0][i] = X[2 * i];
    in[1][i] = X[2 * i + 1];
    // printf("%f\n%f\n", in[0][i], in[1][i]);
  }

  cudaEventRecord(start);

  fftw_plan p;
  fftw_init_threads();

  fftw_plan_with_nthreads(omp_get_max_threads());
  p = fftw_plan_dft_1d(X.size() / 2, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  fftw_cleanup_threads();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%f ", milliseconds);

  // for(unsigned int i = 0; i < N; ++i) printf("%f\n%f\n", in[0][i], in[1][i]);
  fftw_free(in);

  return EXIT_SUCCESS;
}

int cl_test(std::vector<float> &X) 
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //////////////////////////////////////////////////////////////////////////////////

  try {
    std::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);

    cl_context_properties cprops[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platformList[PLATFORM_N])(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, cprops);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cl::CommandQueue queue(context, devices[DEVICE_N]);

    cudaEventRecord(start);

    /* FFT library realted declarations */
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    std::vector<size_t> clLengths;
    clLengths.push_back(X.size() / 2);

    /* Create a default plan for a complex FFT. */
    OCL_CALL(clfftCreateDefaultPlan(&planHandle, context.get(), dim,
                                    clLengths.data()));

    /* Set plan parameters. */
    OCL_CALL(clfftSetPlanPrecision(planHandle, CLFFT_SINGLE));
    OCL_CALL(clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED,
                            CLFFT_COMPLEX_INTERLEAVED));
    OCL_CALL(clfftSetResultLocation(planHandle, CLFFT_INPLACE));

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    OCL_CALL(clfftInitSetupData(&fftSetup));
    OCL_CALL(clfftSetup(&fftSetup));

    /* Prepare OpenCL memory objects and place data inside them. */
    {
      cl::Buffer bufX =
          cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     X.size() * sizeof(float), X.data());

      /* Bake the plan. */
      cl_command_queue q = queue.get();
      cl_mem x = bufX.get();

      OCL_CALL(clfftBakePlan(planHandle, 1, &q, NULL, NULL));

      /* Execute the plan. */
      OCL_CALL(clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &q, 0, NULL,
                                     NULL, &x, NULL, NULL));
      /* Release the plan. */
      OCL_CALL(clfftDestroyPlan(&planHandle));

      bufX = x;
      queue = q;

      /* Wait for calculations to be finished. */
      queue.finish();

      cudaEventRecord(stop);

      /* Release the plan. */
      OCL_CALL(clfftDestroyPlan(&planHandle));

      /* Fetch results of calculations. */
      queue.enqueueMapBuffer(bufX, CL_TRUE, CL_MAP_READ, 0,
                             X.size() * sizeof(float));
    }

    /* Release clFFT library. */
    OCL_CALL(clfftTeardown());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ", milliseconds);

    // for(unsigned int i = 0; i < 2*N; ++i)
    // std::cout << X[i] << "\n";
  } catch (cl::Error const &cpperr) {
    std::cerr << "ERROR: " << cpperr.what() << " (" << cpperr.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
