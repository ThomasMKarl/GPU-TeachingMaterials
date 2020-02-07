/* solving the 2d stationary heat conduction equation with OpenMP (>= v4.5) */
#include <math.h>

int main(void)
{
  float error;
  float tol = 0.01;
  int iter = 0;
  int iter_max = 1000;

  int n = 1000;
  int m = 1000;
  float A[n][m];
  float Anew[n][m];
  for(int j = 0; j < n; ++j)
  {
      for(int i = 0; i < m; ++i) A[j][i] = 1.0f;
  }

  #pragma omp target data map(alloc: Anew) map(A)
  //while(error > tol && iter < iter_max)
  while(iter < iter_max)
  {
    error = 0.0f;
    #pragma omp target teams distribute parallel for reduction(max: error)
    for(int j = 0; j < n; ++j)
    {
      for(int i = 0; i < m; ++i)
      {
        Anew[j][i] = 0.25*(A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
        error = fmax(error, fabs(Anew[j][i] - A[j][i]));
      }
    }
    #pragma omp target teams distribute parallel for
    for(int j = 0; j < n; ++j)
    {
      for(int i = 0; i < m; ++i) A[j][i] = Anew[j][i];
    }
    iter++;
  }
}
