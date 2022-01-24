/* solving a dense system of linear equations with LU-decompositon in MAGMA (CPU interface) */
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
  
  magma_int_t m = 4;
  magma_int_t n = 1;
  magma_int_t mm = m*m;                      
  
  float *a;                          
  float *b;                          
  float *x;
  magma_int_t *piv;
  magma_smalloc_pinned(&a, mm);
  magma_smalloc_pinned(&b, m);  
  magma_smalloc_pinned(&x, m);   
  piv=(magma_int_t *)malloc(m*sizeof(magma_int_t));

  magma_int_t ione = 1;         
  magma_int_t ISEED[4] = {0,0,0,1};
  lapackf77_slarnv(&ione, ISEED, &mm, a);
  //magma_sprint(16, 1, a, mm);         
  lapackf77_slarnv(&ione, ISEED, &m,  x);
  printf("upper  left  corner  of the  expected  solution :\n");
  magma_sprint(4, 1, x, m);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  blasf77_sgemm("N", "N", &m, &n, &n, &alpha, a, &m, x, &m, &beta, b, &m);
  //magma_sprint(4, 1, b, m);
  
  real_Double_t gpu_time = magma_sync_wtime(NULL);
  magma_int_t info;
  magma_sgesv(m, n, a, m, piv, b, m, &info);
  gpu_time = magma_sync_wtime(NULL) - gpu_time;
  printf("magma_sgesv time: %7.5f sec.\n", gpu_time);
  
  printf("upper  left  corner  of the  solution :\n");
  magma_sprint(4, 1, b, m);

  magma_free_pinned(a);
  magma_free_pinned(b);
  magma_free_pinned(x);
  free(piv);
  
  magma_queue_destroy(queue);
  magma_finalize();
  
  return  0;
}
