/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvj_hegvj.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvj_hegvj_impl(rocblas_handle handle,
                                          const rocblas_eform itype,
                                          const rocblas_evect evect,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          U B,
                                          const rocblas_int ldb,
                                          const S abstol,
                                          S* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          S* W,
                                          rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvj" : "hegvj");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb, "--abstol", abstol, "--max_sweeps", max_sweeps);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygvj_hegvj_argCheck(handle, itype, evect, uplo, n, A, lda, B,
                                                       ldb, residual, max_sweeps, n_sweeps, W, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_sygvj_hegvj_getMemorySize<false, false, T, S>(itype, evect, uplo, n, batch_count,
                                                            &size_scalars, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_workArr);

    // memory workspace allocation
    void *scalars, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    workArr = mem[1];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvj_hegvj_template<false, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, abstol,
        residual, max_sweeps, n_sweeps, W, strideW, info, batch_count, (T*)scalars, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver_sygvj_hegvj_impl<float>(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol,
                                             residual, max_sweeps, n_sweeps, W, info);
}

rocblas_status rocsolver_dsygvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver_sygvj_hegvj_impl<double>(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol,
                                              residual, max_sweeps, n_sweeps, W, info);
}

rocblas_status rocsolver_chegvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver_sygvj_hegvj_impl<rocblas_float_complex>(handle, itype, evect, uplo, n, A, lda,
                                                             B, ldb, abstol, residual, max_sweeps,
                                                             n_sweeps, W, info);
}

rocblas_status rocsolver_zhegvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver_sygvj_hegvj_impl<rocblas_double_complex>(handle, itype, evect, uplo, n, A, lda,
                                                              B, ldb, abstol, residual, max_sweeps,
                                                              n_sweeps, W, info);
}

} // extern C
