/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygv_hegv.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_sygv_hegv_strided_batched_impl(rocblas_handle handle,
                                                        const rocblas_eform itype,
                                                        const rocblas_evect jobz,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        U A,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        U B,
                                                        const rocblas_int ldb,
                                                        const rocblas_stride strideB,
                                                        S* D,
                                                        const rocblas_stride strideD,
                                                        S* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    const char* name = (!is_complex<T> ? "sygv_strided_batched" : "hegv_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", jobz, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--bsa", strideA, "--ldb", ldb, "--bsb", strideB, "--bsc", strideD,
                        "--bsp", strideE, "--batch", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygv_hegv_argCheck(handle, itype, jobz, uplo, n, lda, ldb, A, B,
                                                     D, E, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling TRSM, SYGST/HEGST, and SYEV/HEEV)
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTRF and SYEV/HEEV
    size_t size_pivots_workArr;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygv_hegv_getMemorySize<false, T, S>(itype, jobz, uplo, n, batch_count, &size_scalars,
                                                   &size_work1, &size_work2, &size_work3,
                                                   &size_work4, &size_pivots_workArr, &size_iinfo);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots_workArr,
                                                      size_iinfo);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivots_workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivots_workArr, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots_workArr = mem[5];
    iinfo = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygv_hegv_template<false, true, T>(
        handle, itype, jobz, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, D, strideD,
        E, strideE, info, batch_count, (T*)scalars, work1, work2, work3, work4, pivots_workArr,
        (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygv_strided_batched(rocblas_handle handle,
                                               const rocblas_eform itype,
                                               const rocblas_evect jobz,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               float* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               float* B,
                                               const rocblas_int ldb,
                                               const rocblas_stride strideB,
                                               float* D,
                                               const rocblas_stride strideD,
                                               float* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_sygv_hegv_strided_batched_impl<float, float>(
        handle, itype, jobz, uplo, n, A, lda, strideA, B, ldb, strideB, D, strideD, E, strideE,
        info, batch_count);
}

rocblas_status rocsolver_dsygv_strided_batched(rocblas_handle handle,
                                               const rocblas_eform itype,
                                               const rocblas_evect jobz,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               double* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               double* B,
                                               const rocblas_int ldb,
                                               const rocblas_stride strideB,
                                               double* D,
                                               const rocblas_stride strideD,
                                               double* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_sygv_hegv_strided_batched_impl<double, double>(
        handle, itype, jobz, uplo, n, A, lda, strideA, B, ldb, strideB, D, strideD, E, strideE,
        info, batch_count);
}

rocblas_status rocsolver_chegv_strided_batched(rocblas_handle handle,
                                               const rocblas_eform itype,
                                               const rocblas_evect jobz,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               rocblas_float_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               rocblas_float_complex* B,
                                               const rocblas_int ldb,
                                               const rocblas_stride strideB,
                                               float* D,
                                               const rocblas_stride strideD,
                                               float* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_sygv_hegv_strided_batched_impl<float, rocblas_float_complex>(
        handle, itype, jobz, uplo, n, A, lda, strideA, B, ldb, strideB, D, strideD, E, strideE,
        info, batch_count);
}

rocblas_status rocsolver_zhegv_strided_batched(rocblas_handle handle,
                                               const rocblas_eform itype,
                                               const rocblas_evect jobz,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               rocblas_double_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               rocblas_double_complex* B,
                                               const rocblas_int ldb,
                                               const rocblas_stride strideB,
                                               double* D,
                                               const rocblas_stride strideD,
                                               double* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_sygv_hegv_strided_batched_impl<double, rocblas_double_complex>(
        handle, itype, jobz, uplo, n, A, lda, strideA, B, ldb, strideB, D, strideD, E, strideE,
        info, batch_count);
}

} // extern C
