#include "MathLib.H"
#include "cudaErrorCheck.H"

#ifdef AMREX_USE_CUDA
    // CUDA runtime
    #include <cuda_runtime.h>
    
    // CUSOLVER
    #include "cusolverSp.h"
    #include "cusolverSp_LOWLEVEL_PREVIEW.h"
    
    // CUBLAS
    #include <cublas_v2.h>
#endif

void MathLib::MatrixMatrixMultiply(ComplexType* d_C,
                                   const ComplexType* d_A, 
                                   const ComplexType* d_B,
                                   unsigned int A_rows, 
                                   unsigned int A_cols, 
                                   unsigned int B_cols) 
{
#ifdef AMREX_USE_GPU
#ifdef AMREX_USE_CUDA
    cublasHandle_t handle;
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    checkCudaErrors(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // see: https://docs.nvidia.com/cuda/archive/10.0/cublas/index.html

    cublasStatus_t status =cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    A_rows, B_cols, A_cols, &alpha, 
                    reinterpret_cast<const cuDoubleComplex*>(d_A), A_rows, 
                    reinterpret_cast<const cuDoubleComplex*>(d_B), A_cols, &beta, 
                    reinterpret_cast<cuDoubleComplex*>(d_C), A_rows);
    checkCudaErrors(status);

    checkCudaErrors(cublasDestroy(handle));
#elif AMREX_USE_HIP
    /*HIP version*/
#elif AMREX_USE_SYCL
    /*SYCL version*/
#endif   
#else
    /*LAPACK version*/
#endif
}