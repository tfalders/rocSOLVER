/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <cmath>

/****************************************************************************
(TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH). MORE PARALLELISM CAN BE INTRODUCED IN THE FUTURE IN AT LEAST TWO
  WAYS:
  1. the split diagonal blocks can be worked in parallel as they are
  independent
  2. for each block, multiple threads can accelerate some of the reductions
  and vector operations
***************************************************************************/

/************** Kernels and device functions *******************/
/***************************************************************/

/** BDSQR_ESTIMATE device function computes an estimate of the smallest
    singular value of a n-by-n upper bidiagonal matrix given by D and E
    It also applies convergence test if conver = 1 **/
template <typename T>
__device__ T bdsqr_estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver)
{
    T smin = t2b ? std::abs(D[0]) : std::abs(D[n - 1]);
    T t = smin;

    rocblas_int je, jd;

    for(rocblas_int i = 1; i < n; ++i)
    {
        jd = t2b ? i : n - 1 - i;
        je = jd - t2b;
        if((std::abs(E[je]) <= tol * t) && conver)
        {
            E[je] = 0;
            smin = -1;
            break;
        }
        t = std::abs(D[jd]) * t / (t + std::abs(E[je]));
        smin = (t < smin) ? t : smin;
    }

    return smin;
}

/** BDSQR_T2BQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from top to bottom **/
template <typename T, typename S>
__device__ void bdsqr_t2bQRstep(const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                S* D,
                                S* E,
                                T* V,
                                const rocblas_int ldv,
                                T* U,
                                const rocblas_int ldu,
                                T* C,
                                const rocblas_int ldc,
                                const S sh,
                                S* rots)
{
    S f, g, c, s, r;
    rocblas_int nr = nv ? 2 * (n - 1) : 0;

    int sgn = (S(0) < D[0]) - (D[0] < S(0));
    if(D[0] == 0)
        f = 0;
    else
        f = (std::abs(D[0]) - sh) * (S(sgn) + sh / D[0]);
    g = E[0];

    for(rocblas_int k = 0; k < n - 1; ++k)
    {
        // first apply rotation by columns
        lartg(f, g, c, s, r);
        if(k > 0)
            E[k - 1] = r;
        f = c * D[k] - s * E[k];
        E[k] = c * E[k] + s * D[k];
        g = -s * D[k + 1];
        D[k + 1] = c * D[k + 1];
        // save rotations to update singular vectors
        if(nv)
        {
            rots[k] = c;
            rots[k + n - 1] = -s;
        }

        // then apply rotation by rows
        lartg(f, g, c, s, r);
        D[k] = r;
        f = c * E[k] - s * D[k + 1];
        D[k + 1] = c * D[k + 1] + s * E[k];
        if(k < n - 2)
        {
            g = -s * E[k + 1];
            E[k + 1] = c * E[k + 1];
        }
        // save rotations to update singular vectors
        if(nu || nc)
        {
            rots[k + nr] = c;
            rots[k + nr + n - 1] = -s;
        }
    }
    E[n - 2] = f;

    // update singular vectors
    if(nv)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nv, rots, rots + n - 1, V, ldv);
    if(nu)
        lasr(rocblas_side_right, rocblas_forward_direction, nu, n, rots + nr, rots + nr + n - 1, U,
             ldu);
    if(nc)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nc, rots + nr, rots + nr + n - 1, C,
             ldc);
}

/** BDSQR_B2TQRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E, using shift = sh,
    from bottom to top **/
template <typename T, typename S>
__device__ void bdsqr_b2tQRstep(const rocblas_int n,
                                const rocblas_int nv,
                                const rocblas_int nu,
                                const rocblas_int nc,
                                S* D,
                                S* E,
                                T* V,
                                const rocblas_int ldv,
                                T* U,
                                const rocblas_int ldu,
                                T* C,
                                const rocblas_int ldc,
                                const S sh,
                                S* rots)
{
    S f, g, c, s, r;
    rocblas_int nr = nv ? 2 * (n - 1) : 0;

    int sgn = (S(0) < D[n - 1]) - (D[n - 1] < S(0));
    if(D[n - 1] == 0)
        f = 0;
    else
        f = (std::abs(D[n - 1]) - sh) * (S(sgn) + sh / D[n - 1]);
    g = E[n - 2];

    for(rocblas_int k = n - 1; k > 0; --k)
    {
        // first apply rotation by rows
        lartg(f, g, c, s, r);
        if(k < n - 1)
            E[k] = r;
        f = c * D[k] - s * E[k - 1];
        E[k - 1] = c * E[k - 1] + s * D[k];
        g = -s * D[k - 1];
        D[k - 1] = c * D[k - 1];
        // save rotations to update singular vectors
        if(nu || nc)
        {
            rots[(k - 1) + nr] = c;
            rots[(k - 1) + nr + n - 1] = s;
        }

        // then apply rotation by columns
        lartg(f, g, c, s, r);
        D[k] = r;
        f = c * E[k - 1] - s * D[k - 1];
        D[k - 1] = c * D[k - 1] + s * E[k - 1];
        if(k > 1)
        {
            g = -s * E[k - 2];
            E[k - 2] = c * E[k - 2];
        }
        // save rotations to update singular vectors
        if(nv)
        {
            rots[k - 1] = c;
            rots[(k - 1) + n - 1] = s;
        }
    }
    E[0] = f;

    // update singular vectors
    if(nv)
        lasr(rocblas_side_left, rocblas_backward_direction, n, nv, rots, rots + n - 1, V, ldv);
    if(nu)
        lasr(rocblas_side_right, rocblas_backward_direction, nu, n, rots + nr, rots + nr + n - 1, U,
             ldu);
    if(nc)
        lasr(rocblas_side_left, rocblas_backward_direction, n, nc, rots + nr, rots + nr + n - 1, C,
             ldc);
}

/** BDSQR_KERNEL implements the main loop of the bdsqr algorithm
    to compute the SVD of an upper bidiagonal matrix given by D and E **/
template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_kernel(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   W1 VV,
                                   const rocblas_int shiftV,
                                   const rocblas_int ldv,
                                   const rocblas_stride strideV,
                                   W2 UU,
                                   const rocblas_int shiftU,
                                   const rocblas_int ldu,
                                   const rocblas_stride strideU,
                                   W3 CC,
                                   const rocblas_int shiftC,
                                   const rocblas_int ldc,
                                   const rocblas_stride strideC,
                                   rocblas_int* info,
                                   const rocblas_int maxiter,
                                   const S eps,
                                   const S sfm,
                                   const S tol,
                                   const S minshift,
                                   S* workA,
                                   const rocblas_stride strideW)
{
    rocblas_int bid = hipBlockIdx_x;

    // if a NaN or Inf was detected in the input, return
    if(info[bid] != 0)
        return;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    S* rots;
    T *V, *U, *C;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    if(VV)
        V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
    if(UU)
        U = load_ptr_batch<T>(UU, bid, shiftU, strideU);
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    if(workA)
        rots = workA + bid * strideW;

    // calculate threshold for zeroing elements (convergence threshold)
    int t2b = (D[0] >= D[n - 1]) ? 1 : 0; // direction
    S smin = bdsqr_estimate<S>(n, D, E, t2b, tol,
                               0); // estimate of the smallest singular value
    S thresh = std::max(tol * smin / S(std::sqrt(n)),
                        S(maxiter) * sfm); // threshold

    rocblas_int k = n - 1; // k is the last element of last unconverged diagonal block
    rocblas_int iter = 0; // iter is the number of iterations (QR steps) applied
    S sh, smax;

    // main loop
    while(k > 0 && iter < maxiter)
    {
        rocblas_int i;
        // split the diagonal blocks
        for(rocblas_int j = 0; j < k + 1; ++j)
        {
            i = k - j - 1;
            if(i >= 0 && std::abs(E[i]) < thresh)
            {
                E[i] = 0;
                break;
            }
        }

        // check if last singular value converged,
        // if not, continue with the QR step
        //(TODO: splitted blocks can be analyzed in parallel)
        if(i == k - 1)
            k--;
        else
        {
            // last block goes from i+1 until k
            // determine shift for the QR step
            // (apply convergence test to find gaps)
            i++;
            if(std::abs(D[i]) >= std::abs(D[k]))
            {
                t2b = 1;
                sh = std::abs(D[i]);
            }
            else
            {
                t2b = 0;
                sh = std::abs(D[k]);
            }
            smin = bdsqr_estimate<S>(k - i + 1, D + i, E + i, t2b, tol, 1); // shift
            smax = find_max_tridiag(i, k, D, E); // estimate of the largest singular value in the block

            // check for gaps, if none then continue
            if(smin >= 0)
            {
                if(smin / smax <= minshift)
                    smin = 0; // shift set to zero if less than accepted value
                else if(sh > 0)
                {
                    if(smin * smin / sh / sh < eps)
                        smin = 0; // shift set to zero if negligible
                }

                // apply QR step
                iter += k - i;
                if(t2b)
                    bdsqr_t2bQRstep(k - i + 1, nv, nu, nc, D + i, E + i, V + i, ldv, U + i * ldu,
                                    ldu, C + i, ldc, smin, rots);
                else
                    bdsqr_b2tQRstep(k - i + 1, nv, nu, nc, D + i, E + i, V + i, ldv, U + i * ldu,
                                    ldu, C + i, ldc, smin, rots);
            }
        }
    }

    // if algorithm didn't converge, set value of info
    if(k != 0)
    {
        for(rocblas_int i = 0; i < n - 1; ++i)
            if(E[i] != 0)
                info[bid] += 1;
    }
}

/** BDSQR_LOWER2UPPER kernel transforms a lower bidiagonal matrix given by D and E
    into an upper bidiagonal matrix via givens rotations **/
template <typename T, typename S, typename W1, typename W2>
ROCSOLVER_KERNEL void bdsqr_lower2upper(const rocblas_int n,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* DD,
                                        const rocblas_stride strideD,
                                        S* EE,
                                        const rocblas_stride strideE,
                                        W1 UU,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W2 CC,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        S* workA,
                                        const rocblas_stride strideW)
{
    rocblas_int bid = hipBlockIdx_x;
    S f, g, c, s, r;

    // if a NaN or Inf was detected in the input, return
    if(info[bid] != 0)
        return;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    S* rots;
    T *U, *C;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    if(UU)
        U = load_ptr_batch<T>(UU, bid, shiftU, strideU);
    if(CC)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    if(workA)
        rots = workA + bid * strideW;

    f = D[0];
    g = E[0];
    for(rocblas_int i = 0; i < n - 1; ++i)
    {
        // apply rotations by rows
        lartg(f, g, c, s, r);
        D[i] = r;
        E[i] = -s * D[i + 1];
        f = c * D[i + 1];
        g = E[i + 1];

        // save rotation to update singular vectors
        if(nu || nc)
        {
            rots[i] = c;
            rots[i + n - 1] = -s;
        }
    }
    D[n - 1] = f;

    // update singular vectors
    if(nu)
        lasr(rocblas_side_right, rocblas_forward_direction, nu, n, rots, rots + n - 1, U, ldu);
    if(nc)
        lasr(rocblas_side_left, rocblas_forward_direction, n, nc, rots, rots + n - 1, C, ldc);
}

/** BDSQR_INPUT_CHECK kernel determines if there are any NaNs or Infs in the input,
    and sets appropriate outputs if there are. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_input_check(const rocblas_int n,
                                        S* DD,
                                        const rocblas_stride strideD,
                                        S* EE,
                                        const rocblas_stride strideE,
                                        rocblas_int* info)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    __shared__ bool found;
    if(tid == 0)
        found = false;
    __syncthreads();

    for(rocblas_int i = tid; i < n - 1; i += hipBlockDim_x)
    {
        if(!std::isfinite(D[i]) || !std::isfinite(E[i]))
            found = true;
    }
    if(tid == 0 && !std::isfinite(D[n - 1]))
        found = true;
    __syncthreads();

    if(found)
    {
        for(rocblas_int i = tid; i < n - 1; i += hipBlockDim_x)
        {
            D[i] = nan("");
            E[i] = nan("");
        }
        if(tid == 0)
        {
            D[n - 1] = nan("");
            info[bid] = n;
        }
    }
    else
    {
        if(tid == 0)
            info[bid] = 0;
    }
}

/** BDSQR_SORT sorts the singular values and vectors by selection sort if applicable. **/
template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_sort(const rocblas_int n,
                                 const rocblas_int nv,
                                 const rocblas_int nu,
                                 const rocblas_int nc,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 W1 VV,
                                 const rocblas_int shiftV,
                                 const rocblas_int ldv,
                                 const rocblas_stride strideV,
                                 W2 UU,
                                 const rocblas_int shiftU,
                                 const rocblas_int ldu,
                                 const rocblas_stride strideU,
                                 W3 CC,
                                 const rocblas_int shiftC,
                                 const rocblas_int ldc,
                                 const rocblas_stride strideC,
                                 rocblas_int* info)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // if algorithm did not converge, return
    if(info[bid] != 0)
        return;

    // local variables
    rocblas_int i, j, m;

    // array pointers
    T *V, *U, *C;
    S* D = DD + bid * strideD;
    if(nv)
        V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
    if(nu)
        U = load_ptr_batch<T>(UU, bid, shiftU, strideU);
    if(nc)
        C = load_ptr_batch<T>(CC, bid, shiftC, strideC);

    // ensure all singular values are positive
    for(rocblas_int i = 0; i < n; i++)
    {
        if(D[i] < 0)
        {
            if(nv)
            {
                for(rocblas_int j = tid; j < nv; j += hipBlockDim_x)
                    V[i + j * ldv] = -V[i + j * ldv];
                __syncthreads();
            }

            if(tid == 0)
                D[i] = -D[i];
        }
    }
    __syncthreads();

    // sort singular values & vectors
    S p;
    for(i = 0; i < n - 1; i++)
    {
        m = i;
        p = D[i];
        for(j = i + 1; j < n; j++)
        {
            if(D[j] > p)
            {
                m = j;
                p = D[j];
            }
        }
        __syncthreads();

        if(m != i)
        {
            if(tid == 0)
            {
                D[m] = D[i];
                D[i] = p;
            }

            if(nv)
            {
                for(j = tid; j < nv; j += hipBlockDim_x)
                    swap(V[m + j * ldv], V[i + j * ldv]);
                __syncthreads();
            }
            if(nu)
            {
                for(j = tid; j < nu; j += hipBlockDim_x)
                    swap(U[j + m * ldu], U[j + i * ldu]);
                __syncthreads();
            }
            if(nc)
            {
                for(j = tid; j < nc; j += hipBlockDim_x)
                    swap(C[m + j * ldc], C[i + j * ldc]);
                __syncthreads();
            }
        }
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

template <typename T>
void rocsolver_bdsqr_getMemorySize(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   const rocblas_int batch_count,
                                   size_t* size_work)
{
    *size_work = 0;

    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
        return;

    // size of workspace
    if(nv)
        *size_work += 2;
    if(nu || nc)
        *size_work += 2;
    *size_work *= sizeof(T) * n * batch_count;
}

template <typename S, typename W>
rocblas_status rocsolver_bdsqr_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        const rocblas_int ldv,
                                        const rocblas_int ldu,
                                        const rocblas_int ldc,
                                        S D,
                                        S E,
                                        W V,
                                        W U,
                                        W C,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((nv > 0 && ldv < n) || (nc > 0 && ldc < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (n * nv && !V) || (n * nu && !U) || (n * nc && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename W1, typename W2, typename W3>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        W1 V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        W2 U,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W3 C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        S* work)
{
    ROCSOLVER_ENTER("bdsqr", "uplo:", uplo, "n:", n, "nv:", nv, "nu:", nu, "nc:", nc,
                    "shiftV:", shiftV, "ldv:", ldv, "shiftU:", shiftU, "ldu:", ldu,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set tolerance and max number of iterations:
    // machine precision (considering rounding strategy)
    S eps = get_epsilon<S>() / 2;
    // safest minimum value such that 1/sfm does not overflow
    S sfm = get_safemin<S>();
    // max number of iterations (QR steps) before declaring not convergence
    rocblas_int maxiter = 6 * n * n;
    // relative accuracy tolerance
    S tol = std::max(S(10.0), std::min(S(100.0), S(pow(eps, -0.125)))) * eps;
    //(minimum accepted shift to not ruin relative accuracy) / (max singular
    // value)
    S minshift = std::max(eps, tol / S(100)) / (n * tol);

    rocblas_stride strideW = 0;
    if(nv)
        strideW += 2;
    if(nu || nc)
        strideW += 2;
    strideW *= n;

    // check for NaNs and Infs in input
    dim3 grid(1, batch_count, 1);
    dim3 threads(min(n, BS1), 1, 1);
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_input_check<T>), grid, threads, 0, stream, n, D, strideD, E,
                            strideE, info);

    // rotate to upper bidiagonal if necessary
    if(uplo == rocblas_fill_lower)
    {
        ROCSOLVER_LAUNCH_KERNEL((bdsqr_lower2upper<T>), dim3(batch_count), dim3(1), 0, stream, n,
                                nu, nc, D, strideD, E, strideE, U, shiftU, ldu, strideU, C, shiftC,
                                ldc, strideC, info, work, strideW);
    }

    // main computation of SVD
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_kernel<T>), dim3(batch_count), dim3(1), 0, stream, n, nv, nu, nc,
                            D, strideD, E, strideE, V, shiftV, ldv, strideV, U, shiftU, ldu,
                            strideU, C, shiftC, ldc, strideC, info, maxiter, eps, sfm, tol,
                            minshift, work, strideW);

    // sort the singular values and vectors
    rocblas_int threads_sort = (nv || nu || nc ? BS1 : 1);
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_sort<T>), dim3(1, batch_count), dim3(threads_sort), 0, stream, n,
                            nv, nu, nc, D, strideD, V, shiftV, ldv, strideV, U, shiftU, ldu,
                            strideU, C, shiftC, ldc, strideC, info);

    return rocblas_status_success;
}
