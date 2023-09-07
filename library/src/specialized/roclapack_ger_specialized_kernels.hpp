/* **************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver_run_specialized_kernels.hpp"

template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_int WIN, bool CONJ, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_ger_kernel(rocblas_int m,
                   rocblas_int n,
                   V alpha_device_host,
                   rocblas_stride stride_alpha,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   rocblas_int incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   rocblas_int incy,
                   rocblas_stride stridey,
                   W __restrict__ Aa,
                   rocblas_stride shifta,
                   rocblas_int lda,
                   rocblas_stride strideA)
{
    __shared__ T xdata[DIM_X];
    __shared__ T ydata[DIM_Y * WIN];

    auto alpha = load_scalar(alpha_device_host, blockIdx.z, stride_alpha);
    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.z, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.z, shifta, strideA);

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    ty *= WIN;

    // shared data base index
    int tyi = threadIdx.y * WIN;

    if(threadIdx.y == 0)
    {
        xdata[threadIdx.x] = tx < m ? x[tx * int64_t(incx)] : 0;
    }

    if(threadIdx.x < WIN)
    {
        ydata[tyi + threadIdx.x] = (ty + threadIdx.x < n) ? y[(ty + threadIdx.x) * int64_t(incy)] : 0;
    }

    __syncthreads();

    if(tx < m)
    {
        T x_value = alpha * xdata[threadIdx.x];

        for(int i = 0; i < WIN; i++)
        {
            int yi = ty + i;
            if(yi < n)
                A[tx + size_t(lda) * yi] += x_value * (CONJ ? conj(ydata[tyi + i]) : ydata[tyi + i]);
        }
    }
}

template <rocblas_int DIM_X, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL(DIM_X)
rocblas_sger_kernel(rocblas_int m,
                    rocblas_int n,
                    V alpha_device_host,
                    rocblas_stride stride_alpha,
                    const U __restrict__ xa,
                    rocblas_stride shiftx,
                    rocblas_int incx,
                    rocblas_stride stridex,
                    const U __restrict__ ya,
                    rocblas_stride shifty,
                    rocblas_int incy,
                    rocblas_stride stridey,
                    W __restrict__ Aa,
                    rocblas_stride shifta,
                    rocblas_int lda,
                    rocblas_stride strideA)
{
    rocblas_int tx = threadIdx.x;
    rocblas_int col = blockIdx.x;

    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.y, shifta, strideA);

    if(tx < m)
        A += tx;

    //Each blockIdx.x takes care of the computation of each column of matrix 'A'
    A += col * size_t(lda);

    const T res_y = y[col * (int64_t)incy] * alpha;

    //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
    //If m > DIM_X, then the threads are reused and the multiplied values will be accumalated to Hermitian matrix 'A'.

    for(rocblas_int i = 0; tx + i < m; i += DIM_X)
    {
        A[i] += res_y * x[(tx + i) * int64_t(incx)];
    }
}

/** Call this kernel with 'batch_count' groups in z, and enough
    groups in x and y to cover all the 'm' rows and 'n' columns of C. **/
template <typename T, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void ger_kernel(rocblas_int m,
                                 rocblas_int n,
                                 V alpha,
                                 rocblas_stride stridea,
                                 U1 xx,
                                 rocblas_stride shiftX,
                                 rocblas_int incx,
                                 rocblas_stride strideX,
                                 U2 yy,
                                 rocblas_stride shiftY,
                                 rocblas_int incy,
                                 rocblas_stride strideY,
                                 U3 AA,
                                 rocblas_stride shiftA,
                                 rocblas_int inca,
                                 rocblas_int lda,
                                 rocblas_stride strideA)
{
    // indices
    int bid = hipBlockIdx_z;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // batch instance
    T a = load_scalar(alpha, bid, stridea);
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* x = load_ptr_batch(xx, bid, shiftX, strideX);
    T* y = load_ptr_batch(yy, bid, shiftY, strideY);

    if(i < m && j < n)
    {
        A[i * inca + j * lda] += a * x[i * incx] * y[j * incy];
    }
}

/*************************************************************
    Launchers of specialized kernels
*************************************************************/

template <bool CONJ, typename T, typename U, typename V, typename W>
rocblas_status rocblas_internal_ger_template(rocblas_handle handle,
                                             rocblas_int m,
                                             rocblas_int n,
                                             const V* alpha,
                                             rocblas_stride stride_alpha,
                                             const U* x,
                                             rocblas_stride offsetx,
                                             rocblas_int incx,
                                             rocblas_stride stridex,
                                             const U* y,
                                             rocblas_stride offsety,
                                             rocblas_int incy,
                                             rocblas_stride stridey,
                                             W* A,
                                             rocblas_stride offsetA,
                                             rocblas_int lda,
                                             rocblas_stride strideA,
                                             rocblas_int batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream;
    rocblas_get_stream(handle, &rocblas_stream);

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float = std::is_same_v<T, float>;
    static constexpr bool is_double = std::is_same_v<T, double>;
    static constexpr bool is_complex_float = std::is_same_v<T, rocblas_float_complex>;

    bool is_gfx90a = false;

#define ger_KARGS(alpha_)                                                                  \
    ger_grid, ger_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, x, shiftx, incx, \
        stridex, y, shifty, incy, stridey, A, offsetA, lda, strideA

    //optimized double buffered loads kernel for float, double and float_complex precisions in gfx90a
    if(is_gfx90a && (m > 2000) && (m == n)
       && ((m % 64 == 0 && (is_double || is_complex_float)) || ((m % 128 == 0) && is_float)))
    {
        //The following rocblas_ger_double_buffered_kernel is only valid for the multiples of DIM_X
        static constexpr int DIM_X = is_float ? 128 : 64;
        static constexpr int DIM_Y = is_float ? 8 : 16;
        static constexpr int elements_per_thread = DIM_X / (2 * DIM_Y);

        const int block_x = m / DIM_X;
        const int block_y = n / DIM_X;
        dim3 ger_threads(DIM_X, DIM_Y);
        dim3 ger_grid(block_x, block_y, batch_count);

        // bool host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
        // rocblas_internal_val_ptr<V> alpha_device_host(host_ptr_mode, alpha);

        // hipLaunchKernelGGL(
        //     (rocblas_ger_double_buffered_kernel<CONJ, DIM_X, DIM_Y, elements_per_thread, T>),
        //     ger_grid,
        //     ger_threads,
        //     0,
        //     rocblas_stream,
        //     host_ptr_mode,
        //     m,
        //     n,
        //     alpha_device_host,
        //     stride_alpha,
        //     x,
        //     shiftx,
        //     incx,
        //     stridex,
        //     y,
        //     shifty,
        //     incy,
        //     stridey,
        //     A,
        //     offsetA,
        //     lda,
        //     strideA);
    }
    else if(is_float && m > 1024)
    {
        static constexpr int DIM_X = 1024;
        dim3 ger_grid(n, batch_count);
        dim3 ger_threads(DIM_X);

        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);

        if(old_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(alpha));
        }
        else
        {
            hipLaunchKernelGGL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(*alpha));
        }
    }
    else
    {
        static constexpr int DIM_X = 32;
        static constexpr int DIM_Y = 32;
        static constexpr int WIN = 2; // work item number of elements to process
        rocblas_int blocksX = (m - 1) / DIM_X + 1;
        rocblas_int blocksY = (n - 1) / (DIM_Y * WIN) + 1; // WIN columns/work item

        dim3 ger_grid(blocksX, blocksY, batch_count);
        dim3 ger_threads(DIM_X, DIM_Y);

        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);

        if(old_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>), ger_KARGS(alpha));
        }
        else
        {
            hipLaunchKernelGGL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>), ger_KARGS(*alpha));
        }
    }
#undef ger_KARGS
    return rocblas_status_success;
}

template <bool CONJ, typename T, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             rocblas_int m,
                             rocblas_int n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             rocblas_int incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             rocblas_int incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             rocblas_int inca,
                             rocblas_int lda,
                             rocblas_stride strideA,
                             rocblas_int batch_count,
                             T** work)
{
    ROCSOLVER_ENTER("ger", "m:", m, "n:", n, "shiftX:", shiftX, "incx:", incx, "shiftY:", shiftY,
                    "incy:", incy, "shiftA:", shiftA, "inca:", inca, "lda:", lda, "bc:", batch_count);

    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(inca == 1)
        // return rocblasCall_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y,
        //                                 shiftY, incy, strideY, A, shiftA, lda, strideA, batch_count,
        //                                 work);
        return rocblas_internal_ger_template<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx,
                                                      strideX, y, shiftY, incy, strideY, A, shiftA,
                                                      lda, strideA, batch_count);

    // TODO: add interleaved support for conjugation
    if(CONJ)
        return rocblas_status_not_implemented;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode pmode;
    rocblas_get_pointer_mode(handle, &pmode);

    // launch specialized kernel
    rocblas_int blocksx = (m - 1) / BS2 + 1;
    rocblas_int blocksy = (n - 1) / BS2 + 1;
    dim3 grid(blocksx, blocksy, batch_count);
    dim3 threads(BS2, BS2, 1);
    if(pmode == rocblas_pointer_mode_device)
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, *alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA);
    }

    return rocblas_status_success;
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <bool CONJ, typename T, typename U>
inline rocblas_status rocsolver_ger(rocblas_handle handle,
                                    rocblas_int m,
                                    rocblas_int n,
                                    const T* alpha,
                                    rocblas_stride stridea,
                                    U x,
                                    rocblas_stride shiftX,
                                    rocblas_int incx,
                                    rocblas_stride strideX,
                                    U y,
                                    rocblas_stride shiftY,
                                    rocblas_int incy,
                                    rocblas_stride strideY,
                                    U A,
                                    rocblas_stride shiftA,
                                    rocblas_int lda,
                                    rocblas_stride strideA,
                                    rocblas_int batch_count,
                                    T** work)
{
    return rocsolver_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y, shiftY,
                                  incy, strideY, A, shiftA, 1, lda, strideA, batch_count, work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GER(CONJ, T, U)                                           \
    template rocblas_status rocsolver_ger<CONJ, T, U>(                        \
        rocblas_handle handle, rocblas_int m, rocblas_int n, const T* alpha,  \
        rocblas_stride stridea, U x, rocblas_stride shiftX, rocblas_int incx, \
        rocblas_stride strideX, U y, rocblas_stride shiftY, rocblas_int incy, \
        rocblas_stride strideY, U A, rocblas_stride shiftA, rocblas_int lda,  \
        rocblas_stride strideA, rocblas_int batch_count, T** work)
