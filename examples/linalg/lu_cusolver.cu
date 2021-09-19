/* solving a dense system of linear equations with LU-decompositon in cuSOLVER */
#include <stdio.h>
#include <stdlib.h>

#include <cusolverDn.h>


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++)
    {
        for(int col = 0 ; col < n ; col++)
	{
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
/*       | 1 2 3  |
 *   A = | 4 5 6  |
 *       | 7 8 10 |
 *
 * without pivoting: A = L*U
 *       | 1 0 0 |      | 1  2  3 |
 *   L = | 4 1 0 |, U = | 0 -3 -6 |
 *       | 7 2 1 |      | 0  0  1 |
 *  
 * with pivoting: P*A = L*U
 *       | 0 0 1 |
 *   P = | 1 0 0 |
 *       | 0 1 0 |
 *
 *       | 1       0     0 |      | 7  8       10     |
 *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
 *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
 */
    const int m = 3;
    const int lda = m;
    const int ldb = m;
    double A[lda*m] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    double B[m] = {1.0, 2.0, 3.0};
    double X[m]; /* X = A\B */
    double LU[lda*m]; /* L and U */
    int Ipiv[m];      /* host copy of pivoting sequence */
    int info;     /* host copy of error info */

//////////////////////////////////////////////////////////////////

/* step 1: create cusolver handle, bind a stream */
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

/* step 2: copy A to device */
    double *d_A;
    cudaMalloc(&d_A,    sizeof(double)*lda*m);
    double *d_B;
    cudaMalloc(&d_B,    sizeof(double)*m);
    int *d_Ipiv;
    cudaMalloc(&d_Ipiv, sizeof(int)*m);
    int *d_info;
    cudaMalloc(&d_info, sizeof(int));

    cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
    
/* step 3: query working space of getrf */
    int lwork; /* size of workspace */
    double *d_work; /* device workspace for getrf */
    cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
    cudaMalloc(&d_work, sizeof(double)*lwork);

/* step 4: LU factorization */
    const int pivot_on = 1;
    if (pivot_on) cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info);
    else          cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL,   d_info);
    cudaDeviceSynchronize();

    if (pivot_on) cudaMemcpy(Ipiv, d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(LU   , d_A   , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if(0 > info)
    {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /*
 * step 5: solve A*X = B 
 *       | 1 |       | -0.3333 |
 *   B = | 2 |,  X = |  0.6667 |
 *       | 3 |       |  0      |
 *
 */
    if (pivot_on) cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info);
    else          cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, NULL,   d_B, ldb, d_info);
    cudaDeviceSynchronize();

    cudaMemcpy(X , d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);

    printf("(L-1)+U =\n");
    printMatrix(m, m, LU, lda, "LU");
    printf("\nX =\n");
    printMatrix(m, 1, X, ldb, "X");
    printf("\n");
    

/* free resources */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Ipiv);
    cudaFree(d_info);
    cudaFree(d_work);

    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}
