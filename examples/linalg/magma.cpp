/* simple magma example: create a vector and find maximum value */
//compile: g++ -std=c++11 -O3 -Wall -Wextra -pedantic -fopenmp -o magma 
// magma.cpp
// -lmagma -lopenblas
  
#include <stdlib.h>
#include <stdio.h>
#include "magma_v2.h"
int main(int argc, char **argv)
{
  magma_init(); //initialize Magma
  magma_queue_t queue = NULL;
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);
  magma_int_t m = 1024000; //length of a
  
  float *a;
  //a - m - vector on the host
  float *d_a;
  //d_a - m - vector a on the device
  
  // allocate array on the host
  magma_smalloc_cpu(&a, m);
  
  // host memory for a
  // allocate array on the device 
  magma_smalloc(&d_a, m);
  
  //device memory for a
  //a ={sin(0) , sin(1), ...,sin(m-1)}
  for(int j = 0; j < m; j++) a[j] = sin((float) j);
  
  //copy data from host to device
  magma_ssetvector (m, a, 1, d_a, 1, queue); //copy a -> d_a
  
  // find the smallest index of the element of d_a with maximum
  // absolute value
  int i = magma_isamax(m, d_a, 1, queue);

  printf("max |a[i]|: %f\n", fabs(a[i -1]));
  printf("fortran index: %d\n", i);
  
  magma_free_cpu(a);
  //free hostmagma_free(d_a);
  //free device_magma_queue_destroy(queue);
  magma_finalize();
  
  return 0;
}
