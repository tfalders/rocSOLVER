/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels_outofplace.hpp"

template <typename T, typename U, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_gels_outofplace_batched_impl(rocblas_handle handle,
                                                      rocblas_operation trans,
                                                      const rocblas_int m,
                                                      const rocblas_int n,
                                                      const rocblas_int nrhs,
                                                      U A,
                                                      const rocblas_int lda,
                                                      U B,
                                                      const rocblas_int ldb,
                                                      U X,
                                                      const rocblas_int ldx,
                                                      rocblas_int* info,
                                                      const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gels_outofplace_batched", "--trans", trans, "-m", m, "-n", n, "--nrhs",
                        nrhs, "--lda", lda, "--ldb", ldb, "--ldx", ldx, "--batch_count", batch_count);

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

    // batched execution
    const rocblas_stride strideA = 0;
    const rocblas_stride strideB = 0;
    const rocblas_stride strideX = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling GEQRF/GELQF, ORMQR/ORMLQ, and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr, size_ipiv;
    // extra requirements to copy B
    size_t size_savedB;
    rocsolver_gels_outofplace_getMemorySize<true, false, T>(
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
    return rocsolver_gels_outofplace_template<true, false, T>(
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

rocblas_status rocsolver_sgels_outofplace_batched(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  float* const A[],
                                                  const rocblas_int lda,
                                                  float* const B[],
                                                  const rocblas_int ldb,
                                                  float* const X[],
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_batched_impl<float>(handle, trans, m, n, nrhs, A, lda, B, ldb,
                                                         X, ldx, info, batch_count);
}

rocblas_status rocsolver_dgels_outofplace_batched(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  double* const A[],
                                                  const rocblas_int lda,
                                                  double* const B[],
                                                  const rocblas_int ldb,
                                                  double* const X[],
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_batched_impl<double>(handle, trans, m, n, nrhs, A, lda, B, ldb,
                                                          X, ldx, info, batch_count);
}

rocblas_status rocsolver_cgels_outofplace_batched(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  rocblas_float_complex* const A[],
                                                  const rocblas_int lda,
                                                  rocblas_float_complex* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_float_complex* const X[],
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_batched_impl<rocblas_float_complex>(
        handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, batch_count);
}

rocblas_status rocsolver_zgels_outofplace_batched(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  rocblas_double_complex* const A[],
                                                  const rocblas_int lda,
                                                  rocblas_double_complex* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_double_complex* const X[],
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_gels_outofplace_batched_impl<rocblas_double_complex>(
        handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, batch_count);
}

} // extern C