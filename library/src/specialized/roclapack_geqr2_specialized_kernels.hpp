/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "lapack_device_functions.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GEQR2_SSKER_THREADS)
    geqr2_kernel_small(const I m,
                       const I n,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       T* ipivA,
                       const rocblas_stride strideP)
{
    I bid = hipBlockIdx_x;
    I tid = hipThreadIdx_x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* ipiv = load_ptr_batch<T>(ipivA, bid, 0, strideP);

    // shared variables
    __shared__ T sval[GEQR2_SSKER_THREADS];
    __shared__ T x[GEQR2_SSKER_MAX_M];
    x[0] = 1;

    const I dim = std::min(m, n);
    for(I k = 0; k < dim; k++)
    {
        //--- LARFG ---
        dot<GEQR2_SSKER_THREADS, true, T>(tid, m - k - 1, A + (k + 1) + k * lda, 1,
                                          A + (k + 1) + k * lda, 1, sval);
        if(tid == 0)
            set_taubeta<T>(ipiv + k, sval, A + k + k * lda);
        __syncthreads();
        for(I i = tid; i < m - k - 1; i += GEQR2_SSKER_THREADS)
            x[i + 1] = A[(k + 1 + i) + k * lda] *= sval[0];
        __syncthreads();

        if(k < n - 1)
        {
            //--- LARF ---
            for(I j = tid; j < n - k - 1; j += GEQR2_SSKER_THREADS)
            {
                T temp = 0;
                for(I i = 0; i < m - k; i++)
                {
                    temp += conj(A[(k + i) + (k + 1 + j) * lda]) * x[i];
                }

                temp = -conj(ipiv[k]) * conj(temp);
                for(I i = 0; i < m - k; i++)
                {
                    A[(k + i) + (k + 1 + j) * lda] += temp * x[i];
                }
            }
            __syncthreads();
        }
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GEQR2_SSKER_THREADS)
    geqr2_panel_kernel(const I m,
                       const I n,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       T* ipivA,
                       const rocblas_stride strideP)
{
    I bid = hipBlockIdx_x;
    I tid = hipThreadIdx_x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* ipiv = load_ptr_batch<T>(ipivA, bid, 0, strideP);

    // shared variables
    __shared__ T sval[GEQR2_SSKER_THREADS];
    __shared__ T x[GEQR2_SSKER_MAX_M];
    x[0] = 1;

    const I dim = std::min(m, n);
    for(I k = 0; k < dim; k++)
    {
        //--- LARFG ---
        dot<GEQR2_SSKER_THREADS, true, T>(tid, m - k - 1, A + (k + 1) + k * lda, 1,
                                          A + (k + 1) + k * lda, 1, sval);
        if(tid == 0)
            set_taubeta<T>(ipiv + k, sval, A + k + k * lda);
        __syncthreads();
        for(I i = tid; i < m - k - 1; i += GEQR2_SSKER_THREADS)
            x[i + 1] = A[(k + 1 + i) + k * lda] *= sval[0];
        __syncthreads();

        if(k < n - 1)
        {
            //--- LARF ---
            for(I j = 0; j < n - k - 1; j++)
            {
                dot<GEQR2_SSKER_THREADS, true, T>(tid, m - k, x, 1, A + k + (k + 1 + j) * lda, 1,
                                                  sval);
                __syncthreads();

                T temp = -conj(ipiv[k]) * conj(sval[0]);
                for(I i = tid; i < m - k; i += GEQR2_SSKER_THREADS)
                {
                    A[(k + i) + (k + 1 + j) * lda] += temp * x[i];
                }
                __syncthreads();
            }
        }
    }
}

/*************************************************************
    Launchers of specialized  kernels
*************************************************************/

template <typename T, typename I, typename U>
rocblas_status geqr2_run_small(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               T* ipiv,
                               const rocblas_stride strideP,
                               const I batch_count)
{
    dim3 grid(batch_count, 1, 1);
    dim3 block(GEQR2_SSKER_THREADS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // ROCSOLVER_LAUNCH_KERNEL(geqr2_kernel_small<T>, grid, block, 0, stream, m, n, A, shiftA, lda,
    //                         strideA, ipiv, strideP);
    ROCSOLVER_LAUNCH_KERNEL(geqr2_panel_kernel<T>, grid, block, 0, stream, m, n, A, shiftA, lda,
                            strideA, ipiv, strideP);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GEQR2_SMALL(T, I, U)                                                            \
    template rocblas_status geqr2_run_small<T, I, U>(                                               \
        rocblas_handle handle, const I m, const I n, U A, const rocblas_stride shiftA, const I lda, \
        const rocblas_stride strideA, T* ipiv, const rocblas_stride strideP, const I batch_count)

ROCSOLVER_END_NAMESPACE
