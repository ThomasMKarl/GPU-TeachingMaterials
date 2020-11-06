/* Header for gpuerror.cpp */
#ifndef CUERROR_H
#define CUERROR_H

#include<stdio.h>

#ifdef DEBUG_CUDA

#include<cuda.h>
#include<cuda_runtime.h>

#define CUDA_CALL(x) do { cudaError_t err = x; if(err!=cudaSuccess) { \
    printf("'%s' at %s:%d\n",cudaGetErrorString(x),__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {	\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) {	\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) {\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_STATUS_SUCCESS) {	\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) {\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUDNN_CALL(x) do { if((x)!=CUDNN_STATUS_SUCCESS) {	\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

cudaError_t cudaDeviceTest();

#endif


///////////////////////////////////////////////////////////////////

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef DEBUG_OCL

#define OCL_CALL(x) do { cl_int r = x ;if((r)!=CL_SUCCESS) { \
      printf("Error %s (%d) at line %d\n",oclGenErrorString(r),r,__LINE__); \
    return EXIT_FAILURE;}} while(0)

const char* oclGenErrorString(int);
int oclKernelTest(int, cl_program, cl_device_id);
#endif

#ifndef DEBUG_CUDA
#define CUDA_CALL(x) do {x;} while(0)
#define CURAND_CALL(x) do {x;} while(0)
#define CUBLAS_CALL(x) do {x;} while(0)
#define CUSPARSE_CALL(x) do {x;} while(0)
#define CUFFT_CALL(x) do {x;} while(0)
#define CUSOLVER_CALL(x) do {x;} while(0)
#define CUDNN_CALL(x) do {x;} while(0)
#define NCCLCHECK(x) do {x;} while(0)
#endif

#ifndef DEBUG_OCL
#define OCL_CALL(x) do {x;} while(0)
const char *oclGenErrorString(int);
int oclKernelTest(int, cl_program, cl_device_id);
#endif

int oclReadFile(char *, char*, size_t*);

#endif
