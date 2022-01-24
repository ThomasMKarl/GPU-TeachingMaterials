/* solving a dense system of linear equations with LU-decompositon in MAGMA (GPU interface) */
#include<stdlib.h>
#include<stdio.h>

#include "magma_v2.h"
#include "magma_lapack.h"


int main(int argc, char *argv[])
{
  magma_init();
  magma_queue_t queue;
  magma_int_t  dev = 0;
  magma_queue_create(dev, &queue);
  
  magma_int_t m = 8192;
  magma_int_t n = 1;
  magma_int_t mm = m*m;
  
  
  float *a;                          
  float *b;                          
  float *x;
  magma_smalloc_cpu(&a, mm);   
  magma_smalloc_cpu(&b, m);  
  magma_smalloc_cpu(&x, m);
  
  float *d_a;
  float *d_b; 
  magma_smalloc(&d_a, mm);
  magma_smalloc(&d_b, m);
  
  magma_int_t *piv;
  piv=(magma_int_t *)malloc(m*sizeof(magma_int_t));

  
  magma_int_t ione = 1;         
  magma_int_t ISEED[4] = {0,0,0,1};
  lapackf77_slarnv(&ione, ISEED, &mm, a);            
  lapackf77_slarnv(&ione, ISEED, &m,  x);
  printf("upper  left  corner  of the  expected  solution :\n");
  magma_sprint(4, 1, x, m);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  blasf77_sgemm("N", "N", &m, &n, &n, &alpha, a, &m, x, &m, &beta, b, &m);
  magma_ssetmatrix(m, m, a, m, d_a, m, queue);
  magma_ssetmatrix(m, n, b, m, d_b ,m, queue);

  
  real_Double_t gpu_time = magma_sync_wtime(NULL);
  magma_int_t info;
  magma_sgesv_gpu(m, n, d_a, m, piv, d_b, m, &info);
  gpu_time = magma_sync_wtime(NULL) - gpu_time;
  printf("magma_sgesv_gpu time: %7.5f sec.\n", gpu_time);

  magma_sgetmatrix(m, n, d_b, m, x, m, queue);
  printf("upper  left  corner  of the  solution :\n");
  magma_sprint(4, 1, x, m);

  free(a);
  free(b);
  free(x);
  free(piv);
  magma_free(d_a);
  magma_free(d_b);

  magma_queue_destroy(queue);   
  magma_finalize();
  
  return  0;
}
