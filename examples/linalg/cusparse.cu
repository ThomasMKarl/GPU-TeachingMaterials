/* demonstrates usage of cuSPARSE library*/
/* Creation of sparse matrices and vectors, examples for all three levels */
#include<iostream>

#include<cusparse.h>


int main(int argc, char *argv[])
{
    /* --HOST-- */
    
    /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0|
       |    4.0        |
       |5.0     6.0 7.0|
       |    8.0     9.0| */
    int n   = 4;
    int nnz = 9;
    int *cooRowIndexHostPtr =    (int*)malloc(nnz*sizeof(int));
    int *cooColIndexHostPtr =    (int*)malloc(nnz*sizeof(int));
    double *cooValHostPtr   = (double*)malloc(nnz*sizeof(double));

      cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;
      cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;
      cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;
      cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;
      cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;
      cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
      cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;
      cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;
      cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0;

    /* create a sparse and dense vector */
    /* xVal= [100.0 200.0 400.0]   (sparse)
       xInd= [0     1     3    ]
       y   = [10.0 20.0 30.0 40.0 | 50.0 60.0 70.0 80.0] (dense) */
    int nnz_vector = 3;
    int    *xIndHostPtr =    (int*)malloc(nnz_vector*sizeof(int));
    double *xValHostPtr = (double*)malloc(nnz_vector*sizeof(double));
    double *yHostPtr    = (double*)malloc(2*n       *sizeof(double));
    double *zHostPtr    = (double*)malloc(2*(n+1)   *sizeof(double));

      yHostPtr[0] = 10.0;  xIndHostPtr[0]=0; xValHostPtr[0]=100.0;
      yHostPtr[1] = 20.0;  xIndHostPtr[1]=1; xValHostPtr[1]=200.0;
      yHostPtr[2] = 30.0;
      yHostPtr[3] = 40.0;  xIndHostPtr[2]=3; xValHostPtr[2]=400.0;
      yHostPtr[4] = 50.0;
      yHostPtr[5] = 60.0;
      yHostPtr[6] = 70.0;
      yHostPtr[7] = 80.0;

    /* --DEVICE--  */
    
    /* allocate GPU memory and copy the matrix and vectors into it */

    int *cooRowIndex; cudaMalloc(&cooRowIndex, nnz*sizeof(int));
    int *cooColIndex; cudaMalloc(&cooColIndex, nnz*sizeof(int));
    double *cooVal;   cudaMalloc(&cooVal,      nnz*sizeof(double));
    double *y;        cudaMalloc(&y,           2*n*sizeof(double));
    int    *xInd;     cudaMalloc(&xInd, nnz_vector*sizeof(int));
    double *xVal;     cudaMalloc(&xVal, nnz_vector*sizeof(double));

    cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
                            (size_t)(nnz*sizeof(int)),           cudaMemcpyHostToDevice);
    cudaMemcpy(cooColIndex, cooColIndexHostPtr,
                            (size_t)(nnz*sizeof(int)),           cudaMemcpyHostToDevice);
    cudaMemcpy(cooVal,      cooValHostPtr,
	                        (size_t)(nnz*sizeof(double)),        cudaMemcpyHostToDevice);
    cudaMemcpy(y,           yHostPtr,
                            (size_t)(2*n*sizeof(double)),        cudaMemcpyHostToDevice);
    cudaMemcpy(xInd,        xIndHostPtr,
                            (size_t)(nnz_vector*sizeof(int)),    cudaMemcpyHostToDevice);
    cudaMemcpy(xVal,        xValHostPtr,
                            (size_t)(nnz_vector*sizeof(double)), cudaMemcpyHostToDevice);

    /* initialize cusparse library */
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "CUSPARSE Library initialization failed!\n";
        return EXIT_FAILURE;
    }

    /* create and setup matrix descriptor */
    cusparseMatDescr_t descr;
    status = cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
    int *csrRowPtr;
    cudaMalloc(&csrRowPtr, (n+1)*sizeof(int));
    status = cusparseXcoo2csr(handle, cooRowIndex, nnz, n, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);

    /* exercise Level 1 routines (scatter vector elements) */
    status = cusparseDsctr(handle, nnz_vector, xVal, xInd, &y[n], CUSPARSE_INDEX_BASE_ZERO);

    /* exercise Level 2 routines (csrmv) */
    double dtwo   = 2.0;
    double dthree = 3.0;
    status = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                            &dtwo, descr, cooVal, csrRowPtr, cooColIndex, &y[0], &dthree, &y[n]);

    /* exercise Level 3 routines (csrmm) */
    double *z;
    cudaMalloc(&z,   2*(n+1)*sizeof(double));
    cudaMemset(z, 0, 2*(n+1)*sizeof(double));
    
    double dzero = 0.0;
    double dfive = 5.0;
    status = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n,
                            nnz, &dfive, descr, cooVal, csrRowPtr, cooColIndex, y, n, &dzero, z, n+1);

    /* copy final result */
    cudaMemcpy(yHostPtr, y, (size_t)(2*n*sizeof(double)),     cudaMemcpyDeviceToHost);
    cudaMemcpy(zHostPtr, z, (size_t)(2*(n+1)*sizeof(double)), cudaMemcpyDeviceToHost);

    /* --HOST-- */

    /* cleanup */
    status = cusparseDestroyMatDescr(descr);
    status = cusparseDestroy(handle);

    free(xIndHostPtr);
    free(xValHostPtr);
    free(cooRowIndexHostPtr);
    free(cooColIndexHostPtr);
    free(cooValHostPtr);

    cudaFree(y);
    cudaFree(z);
    cudaFree(xInd);
    cudaFree(xVal);
    cudaFree(csrRowPtr);
    cudaFree(cooRowIndex);
    cudaFree(cooColIndex);
    cudaFree(cooVal);

    /* check result */
    if ((zHostPtr[0] != 950.0)    ||
        (zHostPtr[1] != 400.0)    ||
        (zHostPtr[2] != 2550.0)   ||
        (zHostPtr[3] != 2600.0)   ||
        (zHostPtr[4] != 0.0)      ||
        (zHostPtr[5] != 49300.0)  ||
        (zHostPtr[6] != 15200.0)  ||
        (zHostPtr[7] != 132300.0) ||
        (zHostPtr[8] != 131200.0) ||
        (zHostPtr[9] != 0.0)      ||
        (yHostPtr[0] != 10.0)     ||
        (yHostPtr[1] != 20.0)     ||
        (yHostPtr[2] != 30.0)     ||
        (yHostPtr[3] != 40.0)     ||
        (yHostPtr[4] != 680.0)    ||
        (yHostPtr[5] != 760.0)    ||
        (yHostPtr[6] != 1230.0)   ||
        (yHostPtr[7] != 2240.0))
    {
        std::cout << "example test FAILED!\n";
	
	free(yHostPtr);
	free(zHostPtr);
	
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "example test PASSED.\n";
	
	free(yHostPtr);
	free(zHostPtr);
	
        return EXIT_SUCCESS;
    }
}
