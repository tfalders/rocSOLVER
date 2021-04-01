/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_ormlq_unmlq.hpp"
#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver.h"

enum copymat_direction
{
    copymat_to_buffer,
    copymat_from_buffer
};

template <typename T, typename U>
__global__ void masked_copymat(copymat_direction direction,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* buffer,
                               const rocblas_int* mask)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const rocblas_int ldw = m;
    const rocblas_stride strideW = rocblas_stride(ldw) * n;

    if(i < m && j < n && mask[b])
    {
        T* Wp = &buffer[b * strideW];
        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        if(direction == copymat_to_buffer)
            Wp[i + j * ldw] = Ap[i + j * lda];
        else // direction == copymat_from_buffer
            Ap[i + j * lda] = Wp[i + j * ldw];
    }
}

template <typename T, typename U>
__global__ void gels_set_zero(const rocblas_int k1,
                              const rocblas_int k2,
                              const rocblas_int nrhs,
                              U B,
                              const rocblas_int shiftB,
                              const rocblas_int ldb,
                              const rocblas_stride strideB,
                              const rocblas_int* info)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k2 - k1 && j < nrhs && !info[b])
    {
        T* Bp = load_ptr_batch<T>(B, b, shiftB, strideB);

        Bp[(i + k1) + j * ldb] = 0;
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_gels_getMemorySize(const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int nrhs,
                                  const rocblas_int batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work_x_temp,
                                  size_t* size_workArr_temp_arr,
                                  size_t* size_diag_trfac_invA,
                                  size_t* size_trfact_workTrmm_invA_arr,
                                  size_t* size_ipiv_savedB)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_diag_trfac_invA = 0;
        *size_trfact_workTrmm_invA_arr = 0;
        *size_ipiv_savedB = 0;
        return;
    }

    size_t gexxf_scalars, gexxf_work, gexxf_workArr, gexxf_diag, gexxf_trfact;
    size_t ormxx_scalars, ormxx_work, ormxx_workArr, ormxx_trfact, ormxx_workTrmm;
    size_t trsm_x_temp, trsm_x_temp_arr, trsm_invA, trsm_invA_arr;

    if(m >= n)
    {
        rocsolver_geqrf_getMemorySize<T, BATCHED>(m, n, batch_count, &gexxf_scalars, &gexxf_work,
                                                  &gexxf_workArr, &gexxf_diag, &gexxf_trfact);

        rocsolver_ormqr_unmqr_getMemorySize<T, BATCHED>(rocblas_side_left, m, nrhs, n, batch_count,
                                                        &ormxx_scalars, &ormxx_work, &ormxx_workArr,
                                                        &ormxx_trfact, &ormxx_workTrmm);

        ROCSOLVER_ASSUME_X(gexxf_scalars == ormxx_scalars, "GEQRF and ORMQR use the same scalars");
    }
    else
    {
        rocsolver_gelqf_getMemorySize<T, BATCHED>(m, n, batch_count, &gexxf_scalars, &gexxf_work,
                                                  &gexxf_workArr, &gexxf_diag, &gexxf_trfact);

        rocsolver_ormlq_unmlq_getMemorySize<T, BATCHED>(rocblas_side_left, n, nrhs, m, batch_count,
                                                        &ormxx_scalars, &ormxx_work, &ormxx_workArr,
                                                        &ormxx_trfact, &ormxx_workTrmm);

        ROCSOLVER_ASSUME_X(gexxf_scalars == ormxx_scalars, "GELQF and ORMLQ use the same scalars");
    }

    rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_left, std::min(m, n), nrhs, batch_count,
                                     &trsm_x_temp, &trsm_x_temp_arr, &trsm_invA, &trsm_invA_arr);

    // TODO: rearrange to minimize total size
    *size_scalars = gexxf_scalars;
    *size_work_x_temp = std::max({gexxf_work, ormxx_work, trsm_x_temp});
    *size_workArr_temp_arr = std::max({gexxf_workArr, ormxx_workArr, trsm_x_temp_arr});
    *size_diag_trfac_invA = std::max({gexxf_diag, ormxx_trfact, trsm_invA});
    *size_trfact_workTrmm_invA_arr = std::max({gexxf_trfact, ormxx_workTrmm, trsm_invA_arr});
    // size_ipiv = sizeof(T) * std::min(m, n) * batch_count, which is always less than size_savedB
    *size_ipiv_savedB = sizeof(T) * std::min(m, n) * nrhs * batch_count;
}

template <bool COMPLEX, typename T>
rocblas_status rocsolver_gels_argCheck(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       T A,
                                       const rocblas_int lda,
                                       T B,
                                       const rocblas_int ldb,
                                       rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if((COMPLEX && trans == rocblas_operation_transpose)
       || (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || nrhs < 0 || lda < m || ldb < m || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m * n && !A) || ((m * nrhs || n * nrhs) && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gels_template(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       U B,
                                       const rocblas_int shiftB,
                                       const rocblas_int ldb,
                                       const rocblas_stride strideB,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* scalars,
                                       T* work_x_temp,
                                       T* workArr_temp_arr,
                                       T* diag_trfac_invA,
                                       T** trfact_workTrmm_invA_arr,
                                       T* ipiv_savedB,
                                       bool optim_mem)
{
    ROCSOLVER_ENTER("gels", "trans:", trans, "m:", m, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BLOCKSIZE + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    hipLaunchKernelGGL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if B is empty
    if(nrhs == 0)
        return rocblas_status_success;

    // quick return if A is empty
    if(m == 0 || n == 0)
    {
        rocblas_int rowsB = std::max(m, n);
        rocblas_int blocksx = (rowsB - 1) / 32 + 1;
        rocblas_int blocksy = (nrhs - 1) / 32 + 1;
        hipLaunchKernelGGL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                           stream, rowsB, nrhs, B, shiftB, ldb, strideB);

        return rocblas_status_success;
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants in host memory
    const rocblas_stride strideP = std::min(m, n);
    const rocblas_int check_threads = std::min(((std::min(m, n) - 1) / 64 + 1) * 64, BLOCKSIZE);
    const rocblas_int copyblocksx = (nrhs - 1) / 32 + 1;
    const rocblas_int copyblocksy = (std::min(m, n) - 1) / 32 + 1;
    const T one = 1;

    // TODO: apply scaling to improve accuracy over a larger range of values

    if(m >= n)
    {
        // compute QR factorization of A
        rocsolver_geqrf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, ipiv_savedB, strideP, batch_count, scalars,
            work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

        if(trans == rocblas_operation_none)
        {
            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose, m, nrhs, n, A,
                shiftA, lda, strideA, ipiv_savedB, strideP, B, shiftB, ldb, strideB, batch_count,
                scalars, (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
                (T**)trfact_workTrmm_invA_arr);

            // do the equivalent of trtrs
            hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1),
                               dim3(1, check_threads, 1), 0, stream, n, A, shiftA, lda, strideA,
                               info);

            // save elements of B that will be overwritten by TRSM for cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                               strideB, ipiv_savedB, info);

            // solve RX = Q'B, overwriting B with X
            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper,
                                         rocblas_operation_none, rocblas_diagonal_non_unit, n, nrhs,
                                         &one, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                         batch_count, optim_mem, work_x_temp, workArr_temp_arr,
                                         diag_trfac_invA, trfact_workTrmm_invA_arr);

            // restore elements of B that were overwritten by TRSM in cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB,
                               ldb, strideB, ipiv_savedB, info);
        }
        else
        {
            // do the equivalent of trtrs
            hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1),
                               dim3(1, check_threads, 1), 0, stream, n, A, shiftA, lda, strideA,
                               info);

            // save elements of B that will be overwritten by TRSM for cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                               strideB, ipiv_savedB, info);

            rocblasCall_trsm<BATCHED, T>(
                handle, rocblas_side_left, rocblas_fill_upper,
                rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit, n, nrhs, &one, A,
                shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work_x_temp,
                workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            // zero row n to m-1 of B in cases where info is zero
            const rocblas_int zeroblocksy = (m - n - 1) / 32 + 1;
            hipLaunchKernelGGL((gels_set_zero<T, U>), dim3(copyblocksx, zeroblocksy, batch_count),
                               dim3(32, 32), 0, stream, n, m, nrhs, B, shiftB, ldb, strideB, info);

            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_none, m, nrhs, n, A, shiftA, lda,
                strideA, ipiv_savedB, strideP, B, shiftB, ldb, strideB, batch_count, scalars,
                (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
                (T**)trfact_workTrmm_invA_arr);

            // restore elements of B that were overwritten by TRSM and ORMQR/UNMQR in cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB,
                               ldb, strideB, ipiv_savedB, info);
        }
    }
    else
    {
        // compute LQ factorization of A
        rocsolver_gelqf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, ipiv_savedB, strideP, batch_count, scalars,
            work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

        if(trans == rocblas_operation_none)
        {
            // do the equivalent of trtrs
            hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1),
                               dim3(1, check_threads, 1), 0, stream, m, A, shiftA, lda, strideA,
                               info);

            // save elements of B that will be overwritten by TRSM for cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_to_buffer, m, nrhs, B, shiftB, ldb,
                               strideB, ipiv_savedB, info);

            rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_lower,
                                         rocblas_operation_none, rocblas_diagonal_non_unit, m, nrhs,
                                         &one, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                                         batch_count, optim_mem, work_x_temp, workArr_temp_arr,
                                         diag_trfac_invA, trfact_workTrmm_invA_arr);

            // zero row m to n-1 of B in cases where info is zero
            const rocblas_int zeroblocksy = (n - m - 1) / 32 + 1;
            hipLaunchKernelGGL((gels_set_zero<T, U>), dim3(copyblocksx, zeroblocksy, batch_count),
                               dim3(32, 32), 0, stream, m, n, nrhs, B, shiftB, ldb, strideB, info);

            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose, n, nrhs, m, A,
                shiftA, lda, strideA, ipiv_savedB, strideP, B, shiftB, ldb, strideB, batch_count,
                scalars, (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
                (T**)trfact_workTrmm_invA_arr);

            // restore elements of B that were overwritten by TRSM and ORMLQ/UNMLQ in cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_from_buffer, m, nrhs, B, shiftB,
                               ldb, strideB, ipiv_savedB, info);
        }
        else
        {
            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_none, n, nrhs, m, A, shiftA, lda,
                strideA, ipiv_savedB, strideP, B, shiftB, ldb, strideB, batch_count, scalars,
                (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
                (T**)trfact_workTrmm_invA_arr);

            // do the equivalent of trtrs
            hipLaunchKernelGGL(check_singularity<T>, dim3(batch_count, 1, 1),
                               dim3(1, check_threads, 1), 0, stream, m, A, shiftA, lda, strideA,
                               info);

            // save elements of B that will be overwritten by TRSM for cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_to_buffer, m, nrhs, B, shiftB, ldb,
                               strideB, ipiv_savedB, info);

            // solve RX = Q'B, overwriting B with X
            rocblasCall_trsm<BATCHED, T>(
                handle, rocblas_side_left, rocblas_fill_lower,
                rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit, m, nrhs, &one, A,
                shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem, work_x_temp,
                workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            // restore elements of B that were overwritten by TRSM in cases where info is nonzero
            hipLaunchKernelGGL((masked_copymat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                               dim3(32, 32), 0, stream, copymat_from_buffer, m, nrhs, B, shiftB,
                               ldb, strideB, ipiv_savedB, info);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
