/* solving the 2d stationary heat conduction equation with OpenMP (>= v4.5) */

float max(float a, float b)
{
  if (a > b)
    return a;
  return b;
}

float dist(float a)
{
  if (a < 0)
    return -a;
  return a;
}

int main(int argc, char **argv) 
{
  float tol = 0.01f;
  float error = tol;
  int iter = 0;
  int iter_max = 1000;

  int n = 1000;
  int m = 1000;
  float A[n][m];
  float Anew[n][m];
  for (int j = 0; j < n; ++j)
  {
    for (int i = 0; i < m; ++i)
      A[j][i] = 1.0f;
  }

#pragma omp target data map(alloc : Anew) map(A)
  while (error >= tol && iter < iter_max)
  {
    error = 0.0f;
#pragma omp target teams distribute parallel for reduction(max : error)
    for (int j = 0; j < n; ++j)
    {
      for (int i = 0; i < m; ++i)
      {
        Anew[j][i] =
            0.25f * (A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i]);
        error = max(error, dist(Anew[j][i] - A[j][i]));
      }
    }
#pragma omp target teams distribute parallel for
    for (int j = 0; j < n; ++j)
    {
      for (int i = 0; i < m; ++i)
        A[j][i] = Anew[j][i];
    }
    ++iter;
  }
}
