/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels_outofplace.hpp"

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_gels_outofplace_strided_batched_impl(rocblas_handle handle,
                                                              rocblas_operation trans,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              const rocblas_int nrhs,
                                                              U A,
                                                              const rocblas_int lda,
                                                              const rocblas_stride strideA,
                                                              U B,
                                                              const rocblas_int ldb,
                                                              const rocblas_stride strideB,
                                                              U X,
                                                              const rocblas_int ldx,
                                                              const rocblas_stride strideX,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gels_outofplace_strided_batched", "--trans", trans, "-m", m, "-n", n,
                        "--nrhs", nrhs, "--lda", lda, "--strideA", strideA, "--ldb", ldb, "--strideB",
                        strideB, "--ldx", ldx, "--strideX", strideX, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gels_outofplace_argCheck<COMPLEX>(
        handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftB = 0;
    const rocblas_int shiftX = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling GEQRF/GELQF, ORMQR/ORMLQ, and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr, size_ipiv;
    // extra requirements to copy B
    size_t size_savedB;
    rocsolver_gels_outofplace_getMemorySize<false, true, T>(
        m, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv, &size_savedB, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
            size_trfact_workTrmm_invA_arr, size_ipiv, size_savedB);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *diag_trfac_invA, *trfact_workTrmm_invA_arr,
        *ipiv, *savedB;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv,
                              size_savedB);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA_arr = mem[4];
    ipiv = mem[5];
    savedB = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gels_outofplace_template<false, true, T>(
        handle, trans, m, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, X, shiftX, ldx,
        strideX, info, batch_count, (T*)scalars, (T*)work_x_temp, (T*)workArr_temp_arr,
        (T*)diag_trfac_invA, (T**)trfact_workTrmm_invA_arr, (T*)ipiv, (T*)savedB, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_outofplace_strided_batched(rocblas_handle handle,
                                                          rocblas_operation trans,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          float* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          float* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          float* X,
                                                          const rocblas_int ldx,
                                                          const rocblas_stride strideX,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_strided_batched_impl<float>(handle, trans, m, n, nrhs, A, lda,
                                                                 strideA, B, ldb, strideB, X, ldx,
                                                                 strideX, info, batch_count);
}

rocblas_status rocsolver_dgels_outofplace_strided_batched(rocblas_handle handle,
                                                          rocblas_operation trans,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          double* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          double* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          double* X,
                                                          const rocblas_int ldx,
                                                          const rocblas_stride strideX,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_strided_batched_impl<double>(handle, trans, m, n, nrhs, A, lda,
                                                                  strideA, B, ldb, strideB, X, ldx,
                                                                  strideX, info, batch_count);
}

rocblas_status rocsolver_cgels_outofplace_strided_batched(rocblas_handle handle,
                                                          rocblas_operation trans,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_float_complex* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_float_complex* X,
                                                          const rocblas_int ldx,
                                                          const rocblas_stride strideX,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_strided_batched_impl<rocblas_float_complex>(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, X, ldx, strideX, info,
        batch_count);
}

rocblas_status rocsolver_zgels_outofplace_strided_batched(rocblas_handle handle,
                                                          rocblas_operation trans,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_double_complex* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_double_complex* X,
                                                          const rocblas_int ldx,
                                                          const rocblas_stride strideX,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_strided_batched_impl<rocblas_double_complex>(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, X, ldx, strideX, info,
        batch_count);
}

} // extern C
