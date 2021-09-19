/* solving the 2d stationary heat conduction equation with OpenACC */
//compile: nvc/pgcc -ta=tesla -Minfo=accel oacc_test.c
//profile: nvprof/pgprof ./a.out


#include<math.h>

int main(void)
{
  int n = 1000;
  int m = 1000;
  float A[n][m];
  for(int j = 0; j < n; j++)
  {
    for(int i = 0; i < m; i++)
    {
      if(i == 0 || i == m-1 || j == 0 || j == n-1) A[j][i] = 1.0f;
      else                                         A[j][i] = 0.0f;
    }
  }
  float Anew[n][m];
  
  float tol = 0.0001;
  float err = tol;
  int iter_max = 1000;
  int iter = 0;
  #pragma acc data copy(A[:n*m]) copyin(Anew[:n*m])
  while(err >= tol && iter < iter_max)
  {
    err=0.0;
    #pragma acc parallel loop reduction(max:err) copyin(A[0:n*m])
    for(int j = 1; j < n-1; j++)
    {
      for(int i = 1; i < m-1; i++)
      {
	Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
        err = fmax(err, fabs(Anew[j][i]-A[j][i]));
      }
    }

    #pragma acc parallel loop copyin(Anew[0:n*m]) copyout(A[0:n*m])
    for(int j = 1; j < n-1; j++)
    {
      for(int i = 1; i < m-1; i++)
      {
        A[j][i] = Anew[j][i];
      }
    }
    iter++;
  }
}
