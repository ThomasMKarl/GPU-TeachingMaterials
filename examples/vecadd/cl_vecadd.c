/* OpenCL host code example for the addition of two vectors */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/opencl.h>


#define OCL_CALL(x)                                                            \
  do {                                                                         \
    cl_int r = x;                                                              \
    if ((r) != CL_SUCCESS) {                                                   \
      printf("Error %d at line %d\n", r, __LINE__);                            \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

int oclKernelTest(int error, cl_program program, cl_device_id device_id) {
  if (error != 0) {
    size_t sz;
    char *bf;

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, 0, NULL,
                          &sz);
    bf = (char *)malloc((sz + 1) * sizeof(char));
    if (bf) {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sz + 1,
                            bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, 0, NULL,
                          &sz);
    bf = (char *)malloc((sz + 1) * sizeof(char));
    if (bf) {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS,
                            sz + 1, bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &sz);
    bf = (char *)malloc((sz + 1) * sizeof(char));
    if (bf) {
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sz + 1,
                            bf, NULL);
      bf[sz] = 0;
      fprintf(stderr, "\n%s\n", bf);
      free(bf);
    }
  }
  return error;
}

int oclReadFile(char *path, char *source_str, size_t *source_size) {
  FILE *fp;
  fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load the File %s. Check permissions.\n", path);
    return EXIT_FAILURE;
  }

  fseek(fp, 0, SEEK_END);
  size_t s = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  source_str = (char *)malloc(s);

  *source_size = fread(source_str, 1, s, fp);
  printf("%s", source_str);

  fclose(fp);

  return EXIT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("usage: %s <platformID> <deviceID>", argv[0]);
    return -1;
  }
  int p = atoi(argv[1]);
  int d = atoi(argv[2]);

  int ret;

  // cl_... für genormte Datentypen
  cl_uint size = 2560; // Größe der Vektoren

  // Allozieren von Hostspeicher
  cl_float *h_x = (cl_float *)malloc(size * sizeof(cl_float));
  cl_float *h_y = (cl_float *)malloc(size * sizeof(cl_float));
  cl_float *h_res = (cl_float *)malloc(size * sizeof(cl_float));
  for (cl_uint i = 0; i < size; ++i) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }

  ///////////////////////////////////////////////////////////////////

  // Einlesen der Kerneldatei als string
  char *source_str;
  size_t source_size = 0;
  char path[] = "cl_vecadd.cl";
  FILE *fp;
  fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load the File %s. Check permissions.\n", path);
    return EXIT_FAILURE;
  }

  fseek(fp, 0, SEEK_END);
  size_t s = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  source_str = (char *)malloc(s);

  source_size = fread(source_str, 1, s, fp);

  fclose(fp);

  // ret = oclReadFile("cl_vecadd.spir", source_str, &source_size);

  ///////////////////////////////////////////////////////////////////

  // Zuordnung der Platformen
  // platform_id kann ein array von mehreren ids sein
  cl_uint ret_num_platforms;
  OCL_CALL(clGetPlatformIDs(0, NULL, &ret_num_platforms));

  cl_platform_id *platform_ids =
      (cl_platform_id *)malloc(ret_num_platforms * sizeof(cl_platform_id));
  OCL_CALL(clGetPlatformIDs(ret_num_platforms, platform_ids, NULL));

  // Zuordnung der Devices einer Platform
  // device_id kann ein array von mehreren ids sein
  cl_uint ret_num_devices;
  OCL_CALL(clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, 0, NULL,
                          &ret_num_devices));

  cl_device_id *device_ids =
      (cl_device_id *)malloc(ret_num_devices * sizeof(cl_device_id));
  OCL_CALL(clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, ret_num_devices,
                          device_ids, NULL));
  cl_device_id device_id = device_ids[d];

  char param[128] = {0};
  size_t size_post = 0;
  OCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, param, &size_post));
  printf("My name is: %s\n", param);

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  OCL_CALL(ret);

  // ordnet context und device ine eine Warteschlange ein

  // clCreateCommandQueueWithProperties ab Version 2.0
  // const cl_command_queue_properties properties =
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE; cl_command_queue command_queue =
  // clCreateCommandQueueWithProperties(context, device_id, &properties, &ret);
  // OCL_CALL(ret);
  cl_command_queue command_queue =
      clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
  OCL_CALL(ret);

  // Allozieren von Devicespeicher und Zuordnen zum context
  // constant Memory kann so alloziert werden
  // kann direkt aus Hostpointer erstellt werden
  cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              size * sizeof(cl_float), NULL, &ret);
  OCL_CALL(ret);
  cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              size * sizeof(cl_float), NULL, &ret);
  OCL_CALL(ret);
  cl_mem d_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                size * sizeof(cl_float), NULL, &ret);
  OCL_CALL(ret);

  // Kopieren aud Device als Teil der command queue
  // CL_TRUE für blocking (CL_FALSE entspricht cudaMemcpyAsync(...))
  // 0 für keine Offsets
  OCL_CALL(clEnqueueWriteBuffer(command_queue, d_x, CL_TRUE, 0,
                                size * sizeof(cl_float), h_x, 0, NULL, NULL));
  OCL_CALL(clEnqueueWriteBuffer(command_queue, d_y, CL_TRUE, 0,
                                size * sizeof(cl_float), h_y, 0, NULL, NULL));
  /////////

  // Kompilieren einer oder mehrerer OCL Dateien (.cl) zur Laufzeit
  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                (const size_t *)&source_size, &ret);
  OCL_CALL(ret);

  // evtl. für Parameter (include, define) für den OCL compiler
  OCL_CALL(clBuildProgram(program, 1, &device_id,
                          "-cl-std=CL1.2 -cl-single-precision-constant -w",
                          NULL, NULL));
  oclKernelTest(ret, program, device_id);

  // cl_program program = clCreateProgramWithBinary(context, 1, &device_id,
  // (const size_t *)&source_size, (const unsigned char **)&source_str, NULL,
  // &ret); OCL_CALL(ret); OCL_CALL(clBuildProgram(program, 1, &device_id,
  // "-cl-std=CL1.2 -cl-single-precision-constant -w", NULL, NULL));
  // oclKernelTest(ret, program, device_id);

  // Erstellen eines Kernels aus dem Programm
  // Angabe des Namens des Kernels als string
  // Jedes Programm hat eigenen namespace
  cl_kernel kernel_vecadd = clCreateKernel(program, "vecadd", &ret);
  OCL_CALL(ret);

  ////////////////////////////////////////////////////////////////////////////////////////////

  // Setzen der Kernelargumente, buffer oder Variablen
  size_t narg = 0;
  OCL_CALL(clSetKernelArg(kernel_vecadd, narg++, sizeof(cl_mem), (void *)&d_x));
  OCL_CALL(clSetKernelArg(kernel_vecadd, narg++, sizeof(cl_mem), (void *)&d_y));
  OCL_CALL(
      clSetKernelArg(kernel_vecadd, narg++, sizeof(cl_mem), (void *)&d_res));
  OCL_CALL(clSetKernelArg(kernel_vecadd, narg++, sizeof(cl_uint), &size));

  // Einreihen des Kernels in Warteschlange und Zuordnung zu einem Event
  cl_event vecadd_event;
  // muss size_t sein
  size_t local_item_size = 256; // Größe der Workgroup
  size_t global_item_size = ((size + local_item_size - 1) / local_item_size) *
                            local_item_size; // Zahl der Workitems gesamt
  OCL_CALL(clEnqueueNDRangeKernel(command_queue, kernel_vecadd, 1, NULL,
                                  &global_item_size, &local_item_size, 0, NULL,
                                  &vecadd_event));

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Warten auf Event (out of order execution eingestellt!) und Lesen des
  // buffers
  OCL_CALL(clEnqueueReadBuffer(command_queue, d_res, CL_TRUE, 0,
                               size * sizeof(cl_float), h_res, 1, &vecadd_event,
                               NULL));

  // OCL_CALL(clFlush(command_queue)); //wartet auf das Erstellen der Queue
  OCL_CALL(clFinish(command_queue)); // wartet auf das Ausführen der Queue

  // Zerstören der Objekte
  OCL_CALL(clReleaseKernel(kernel_vecadd));
  OCL_CALL(clReleaseProgram(program));

  OCL_CALL(clReleaseMemObject(d_x));
  OCL_CALL(clReleaseMemObject(d_y));
  OCL_CALL(clReleaseMemObject(d_res));

  OCL_CALL(clReleaseCommandQueue(command_queue));
  OCL_CALL(clReleaseContext(context));

  // Ausgabe auf Host
  for (cl_uint i = 0; i < 5; ++i) {
    printf("element %d: %f\n", i, h_res[i]);
  }

  // Speicher freigeben
  free(h_x);
  free(h_y);
  free(h_res);

  return EXIT_SUCCESS;
}
