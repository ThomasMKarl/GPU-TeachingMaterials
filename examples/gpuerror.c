/* provides some additional functions for error handling */
//compile as shared library
//with -DDEBUG_CUDA for CUDA   error handling only
//with -DDEBUG_OCL  for OpenCL error handling only
#include "gpuerror.h"

#ifdef DEBUG_CUDA
cudaError_t cudaDeviceTest()
{
  int ndev;
  cudaError_t err = cudaGetDeviceCount(&ndev);
  if(err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
  if(ndev == 0) 
  {
    fprintf(stderr, "ERROR: No devices. exiting...\n");
    exit(1);
  }
  printf("Number of devices: %d\n ", ndev);
  cudaSetDevice(0);

  for(int i = 0; i < ndev; ++i)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("Shared Memory: %d\n",
           prop.sharedMemPerBlock);
    printf("Global Memory: %d\n",
           prop.totalGlobalMem);
    printf("Registers per Block: %d\n",
           prop.regsPerBlock);
    printf("Warpsize: %d\n",
           prop.warpSize);
    printf("Memory Pitch: %d\n",
           prop.memPitch);
    printf("Threads per Block: %d\n",
           prop.maxThreadsPerBlock);
    printf("Thread Dimensions: %d %d %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Grid Dimensions: %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Constant Memory: %d\n",
           prop.totalConstMem);       
    printf("Major Revision Number: %d\n",
          prop.major);        
    printf("Minor Revision Number: %d\n",
          prop.minor);        
    printf("Alignment Requirement: %d\n",
          prop.textureAlignment);
    printf("Device Overlap: %d\n",
          prop.deviceOverlap);       
    printf("Number of Multiprocessors: %d\n",
          prop.multiProcessorCount);
    printf("Timeout enabled: %d\n",
          prop.kernelExecTimeoutEnabled);
    printf("Integrated GPU: %d\n",
          prop.integrated);        
    printf("Map Host Memory: %d\n",
          prop.canMapHostMemory);        
    printf("Compute Mode: %d\n",
          prop.computeMode);        
    printf("Concurrent Kernels: %d\n",
          prop.concurrentKernels);      
    printf("ECC enabled: %d\n",
          prop.ECCEnabled);      
    printf("PCI Bus ID: %d\n",
          prop.pciBusID);             
    printf("TCC Driver: %d\n\n",
          prop.tccDriver);
  }

  return err;
}
#endif

#ifdef DEBUG_OCL
const char * oclGenErrorString(cl_int error)
{
  //see cl.h for a list of errors
  switch(error)
  {
    case CL_SUCCESS:
    return "CL_SUCCESS";
    
    case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
    
    case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
    
    case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
    
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    
    case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
    
    case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
    
    case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
    
    case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
    
    case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
    
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    
    case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
    
    case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
    
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    
    case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
    
    case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
    
    case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
    
    case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
    
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    
    case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
    
    case CL_INVALID_DEVICE_TYPE: 
    return "CL_INVALID_DEVICE_TYPE";
    
    case CL_INVALID_PLATFORM: 
    return "CL_INVALID_PLATFORM";
    
    case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
 
    case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
    
    case CL_INVALID_QUEUE_PROPERTIES: 
    return "CL_INVALID_QUEUE_PROPERTIES";
    
    case CL_INVALID_COMMAND_QUEUE: 
    return "CL_INVALID_COMMAND_QUEUE";
    
    case CL_INVALID_HOST_PTR :  
    return "CL_INVALID_HOST_PTR";
    
    case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
    
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    
    case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
    
    case CL_INVALID_SAMPLER: 
    return "CL_INVALID_SAMPLER";
    
    case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
    
    case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
    
    case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
    
    case CL_INVALID_PROGRAM_EXECUTABLE:  
    return "CL_INVALID_PROGRAM_EXECUTABLE";
    
    case CL_INVALID_KERNEL_NAME:  
    return "CL_INVALID_KERNEL_NAME";
    
    case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
    
    case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
    
    case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX ";
    
    case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE ";
    
    case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE ";
    
    case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS ";
    
    case CL_INVALID_WORK_DIMENSION:  
    return "CL_INVALID_WORK_DIMENSION";
    
    case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
    
    case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE ";
    
    case CL_INVALID_GLOBAL_OFFSET: 
    return "CL_INVALID_GLOBAL_OFFSET";
    
    case CL_INVALID_EVENT_WAIT_LIST: 
    return "CL_INVALID_EVENT_WAIT_LIST";
    
    case CL_INVALID_EVENT:   
    return "CL_INVALID_EVENT";
    
    case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
    
    case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
    
    case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
    
    case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
    
    case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
    
    case CL_INVALID_PROPERTY:
    return "CL_INVALID_PROPERTY";
    
    case CL_INVALID_IMAGE_DESCRIPTOR:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
    
    case CL_INVALID_COMPILER_OPTIONS:
    return "CL_INVALID_COMPILER_OPTIONS";
    
    case CL_INVALID_LINKER_OPTIONS:
    return "CL_INVALID_LINKER_OPTIONS";
    
    case CL_INVALID_DEVICE_PARTITION_COUNT: 
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
    
    /*case CL_INVALID_PIPE_SIZE:   
    return "CL_INVALID_PIPE_SIZE";
    
    case CL_INVALID_DEVICE_QUEUE: 
    return "CL_INVALID_DEVICE_QUEUE";*/
    
    default:
    return "'newer error than 2.0'"; 
  }
}

int oclKernelTest(int error, cl_program program, cl_device_id device_id)
{
  if(error != 0)
  {
    size_t sz;
    char * bf;

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, 0, NULL, &sz);
    bf = (char*)malloc((sz+1) * sizeof(char));
    if(bf)
    {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sz+1, bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &sz);
    bf = (char*)malloc((sz+1) * sizeof(char));
    if(bf)
    {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sz+1, bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    bf = (char*)malloc((sz+1) * sizeof(char));
    if(bf)
    {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sz+1, bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
  }
  return error;
}
#endif

#ifndef DEBUG_OCL
  const char *oclGenErrorString(int error){return "";}
  int oclKernelTest(int error, cl_program program, cl_device_id device_id){return 0;} 
#endif

/*int oclReadFile(char *path, char *source_str, size_t *source_size) 
{
  FILE *fp;
  fp = fopen(path, "r");
  if(!fp)
  {
    fprintf(stderr, "Failed to load the File %s. Check permissions.\n", path);
    return EXIT_FAILURE;
  }
  
  fseek(fp, 0, SEEK_END);
  size_t s = ftell(fp);
  fseek(fp, 0, SEEK_SET); 
  source_str = (char*)malloc(s);
  
  *source_size = fread(source_str, 1, s, fp);
  
  fclose(fp);
  
  return EXIT_SUCCESS;
  }*/
