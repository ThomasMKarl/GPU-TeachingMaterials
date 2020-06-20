/* OpenCL C++ Wrapper API host code example for the addition of two vectors */
//compile g++ -O3 -Wall -std=c++17 cl_vecadd.cpp -lOpenCL

// Enable OpenCL C++ exceptions
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char *argv[])
{
  if(argc != 3)
  {
  	printf("usage: %s <platformID> <deviceID>", argv[0]);
  	return -1;
  }
  int p = atoi(argv[1]);
  int d = atoi(argv[2]);

  //cl_... für genormte Datentypen
  cl_uint size = 2560; //Größe der Vektoren
  //muss size_t sein
  size_t local_item_size = 256; //Größe der Workgroup
  size_t global_item_size = ((size+local_item_size-1)/local_item_size)*local_item_size; //Zahl der Workitems gesamt

  ///////////////////////////////////////////////////////////////////

  //Einlesen der Kerneldatei als string
  std::ifstream file("cl_vecadd.cl");
  std::string source_str((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());

  /////////
  //Allozieren von Hostspeicher
  std::vector<cl_float> h_x(size*sizeof(cl_float));
  std::vector<cl_float> h_y(size*sizeof(cl_float));

  
  
  //cl_float* h_x   = (float *)malloc(size*sizeof(cl_float));
  //cl_float* h_y   = (float *)malloc(size*sizeof(cl_float));
  for(cl_uint i = 0; i < size; ++i)
  {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }

  try
  {
    std::vector<cl::Platform> platformList;

    //Zuordnung der Platformen
    cl::Platform::get(&platformList);

    //Eigenschaften Abfragen
    cl_context_properties cprops[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[p])(), 0};
    
    cl::Context context(CL_DEVICE_TYPE_ALL, cprops);

    //Zuordnung der Devices einer Platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    //Kompilieren einer oder mehrerer OCL Dateien (.cl) zur Laufzeit
    cl::Program program(context, source_str, false);
    program.build(devices);

    
    //Allozieren von Devicespeicher und Zuordnen zum context
    //constant Memory kann so alloziert werden
    //kann direkt aus Hostpointer erstellt werden
    cl::Buffer d_x = cl::Buffer(
      context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
      size*sizeof(cl_float), 
      (void*)&h_x[0]);

    //queue.enqueueMapBuffer(&d_x, CL_TRUE, NULL, 0, size*sizeof(cl_float));

    cl::Buffer d_y = cl::Buffer(
      context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
      size*sizeof(cl_float), 
      (void*)&h_y[0]);

    cl::Buffer d_res = cl::Buffer(
      context, 
      CL_MEM_WRITE_ONLY, 
      size*sizeof(cl_float));

    //Erstellen eines Kernels aus dem Programm
    //Angabe des Namens des Kernels als string
    cl::Kernel kernel(program, "vecadd");
    uint narg = 0;
    kernel.setArg(narg++, d_x);
    kernel.setArg(narg++, d_y);
    kernel.setArg(narg++, d_res);
    kernel.setArg(narg++, size);
    
    //ordnet context und device ine eine Warteschlange ein
    cl::CommandQueue queue(context, devices[d]);

    queue.enqueueNDRangeKernel(
      kernel, 
      cl::NullRange, 
      cl::NDRange(global_item_size), 
      cl::NDRange(local_item_size));
 
    //Lesen des buffers
    cl_float *h_res = (cl_float*)queue.enqueueMapBuffer(
      d_res,
      CL_TRUE,
      CL_MAP_READ,
      0,
      size*sizeof(cl_float));

    //Ausgabe auf Host
    for(cl_uint i = 0; i < size; ++i)
    {
      printf("element %d: %f\n", i, h_res[i]);
    }

    //Speicher freigeben
    queue.enqueueUnmapMemObject(d_res, (void*)h_res);
 
    //Destruktoren rufen...
  }
  catch (cl::Error const &err)
  {
    std::cerr << "ERROR: " << err.what() << " (" << err.err() << ")" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
