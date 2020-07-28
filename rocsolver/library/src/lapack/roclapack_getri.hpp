/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GETRI_H
#define ROCLAPACK_GETRI_H

#include "rocblas.hpp"
#include "rocblas_device_functions.hpp"
#include "../auxiliary/rocauxiliary_trtri.hpp"

#ifdef OPTIMAL
template <rocblas_int DIM, typename T, typename U>
__attribute__((amdgpu_flat_work_group_size(WaveSize,WaveSize)))
__global__ void getri_kernel(U AA, const rocblas_int shiftA, const rocblas_int lda, const rocblas_stride strideA,
                             rocblas_int* ipivA, const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int* info)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if (i >= DIM)
        return;
    
    // batch instance
    T* A = load_ptr_batch<T>(AA,b,shiftA,strideA);
    rocblas_int *ipiv = load_ptr_batch<rocblas_int>(ipivA,b,shiftP,strideP);
       
    // read corresponding row from global memory in local array
    T rA[DIM];
    #pragma unroll
    for (int j = 0; j < DIM; ++j)
        rA[j] = A[i + j*lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    __shared__ T diag[DIM];
    __shared__ rocblas_int _info;
    T temp;
    rocblas_int jp;
    
    // compute info
    if (i == 0)
        _info = 0;
    __syncthreads();
    if (rA[i] == 0)
    {
        rocblas_int _info_temp = _info;
        while (_info_temp == 0 || _info_temp > i + 1)
            _info_temp = atomicCAS(&_info, _info_temp, i + 1);
    }
    __syncthreads();

    if (i == 0)
        info[b] = _info;
    if (_info != 0)
        return;
    
    //--- TRTRI ---

    // diagonal element
    rA[i] = 1.0 / rA[i];
    
    // compute element i of each column j
    #pragma unroll
    for (rocblas_int j = 1; j < DIM; j++)
    {
        // share current column and diagonal
        common[i] = rA[j];
        diag[i] = rA[i];
        __syncthreads();
        
        if (i < j)
        {
            temp = 0;

            for (rocblas_int ii = i; ii < j; ii++)
                temp += rA[ii] * common[ii];

            rA[j] = -diag[j] * temp;
        }
        __syncthreads();
    }

    //--- GETRI ---
    
    #pragma unroll
    for (rocblas_int j = DIM-2; j >= 0; j--)
    {
        // extract lower triangular column (copy_and_zero)
        if (i > j)
        {
            common[i] = rA[j];
            rA[j] = 0;
        }
        __syncthreads();

        // update column j (gemv)
        temp = 0;
        
        for (rocblas_int ii = j+1; ii < DIM; ii++)
            temp += rA[ii] * common[ii];

        rA[j] -= temp;
        __syncthreads();
    }

    // apply pivots (getri_pivot)
    #pragma unroll
    for (rocblas_int j = DIM-2; j >= 0; j--)
    {
        jp = ipiv[j] - 1;
        if (jp != j)
        {
            temp = rA[j];
            rA[j] = rA[jp];
            rA[jp] = temp;
        }
    }

    // write results to global memory from local array
    #pragma unroll
    for (int j = 0; j < DIM; j++)
        A[i + j*lda] = rA[j];
}

template <typename T, typename U>
rocblas_status getri_small_sizes(rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda,
                                 const rocblas_stride strideA, rocblas_int *ipiv, const rocblas_int shiftP, const rocblas_stride strideP,
                                 rocblas_int* info, const rocblas_int batch_count)
{
    #define RUN_GETRI_SMALL(DIM)                                                         \
        hipLaunchKernelGGL((getri_kernel<DIM,T>), grid, block, 0, stream,              \
                           A, shiftA, lda, strideA, ipiv, shiftP, strideP, info)
    
    dim3 grid(batch_count,1,1);
    dim3 block(WaveSize,1,1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch (n) {
        case  1: RUN_GETRI_SMALL( 1); break;
        case  2: RUN_GETRI_SMALL( 2); break;
        case  3: RUN_GETRI_SMALL( 3); break;
        case  4: RUN_GETRI_SMALL( 4); break;
        case  5: RUN_GETRI_SMALL( 5); break;
        case  6: RUN_GETRI_SMALL( 6); break;
        case  7: RUN_GETRI_SMALL( 7); break;
        case  8: RUN_GETRI_SMALL( 8); break;
        case  9: RUN_GETRI_SMALL( 9); break;
        case 10: RUN_GETRI_SMALL(10); break;
        case 11: RUN_GETRI_SMALL(11); break;
        case 12: RUN_GETRI_SMALL(12); break;
        case 13: RUN_GETRI_SMALL(13); break;
        case 14: RUN_GETRI_SMALL(14); break;
        case 15: RUN_GETRI_SMALL(15); break;
        case 16: RUN_GETRI_SMALL(16); break;
        case 17: RUN_GETRI_SMALL(17); break;
        case 18: RUN_GETRI_SMALL(18); break;
        case 19: RUN_GETRI_SMALL(19); break;
        case 20: RUN_GETRI_SMALL(20); break;
        case 21: RUN_GETRI_SMALL(21); break;
        case 22: RUN_GETRI_SMALL(22); break;
        case 23: RUN_GETRI_SMALL(23); break;
        case 24: RUN_GETRI_SMALL(24); break;
        case 25: RUN_GETRI_SMALL(25); break;
        case 26: RUN_GETRI_SMALL(26); break;
        case 27: RUN_GETRI_SMALL(27); break;
        case 28: RUN_GETRI_SMALL(28); break;
        case 29: RUN_GETRI_SMALL(29); break;
        case 30: RUN_GETRI_SMALL(30); break;
        case 31: RUN_GETRI_SMALL(31); break;
        case 32: RUN_GETRI_SMALL(32); break;
        case 33: RUN_GETRI_SMALL(33); break;
        case 34: RUN_GETRI_SMALL(34); break;
        case 35: RUN_GETRI_SMALL(35); break;
        case 36: RUN_GETRI_SMALL(36); break;
        case 37: RUN_GETRI_SMALL(37); break;
        case 38: RUN_GETRI_SMALL(38); break;
        case 39: RUN_GETRI_SMALL(39); break;
        case 40: RUN_GETRI_SMALL(40); break;
        case 41: RUN_GETRI_SMALL(41); break;
        case 42: RUN_GETRI_SMALL(42); break;
        case 43: RUN_GETRI_SMALL(43); break;
        case 44: RUN_GETRI_SMALL(44); break;
        case 45: RUN_GETRI_SMALL(45); break;
        case 46: RUN_GETRI_SMALL(46); break;
        case 47: RUN_GETRI_SMALL(47); break;
        case 48: RUN_GETRI_SMALL(48); break;
        case 49: RUN_GETRI_SMALL(49); break;
        case 50: RUN_GETRI_SMALL(50); break;
        case 51: RUN_GETRI_SMALL(51); break;
        case 52: RUN_GETRI_SMALL(52); break;
        case 53: RUN_GETRI_SMALL(53); break;
        case 54: RUN_GETRI_SMALL(54); break;
        case 55: RUN_GETRI_SMALL(55); break;
        case 56: RUN_GETRI_SMALL(56); break;
        case 57: RUN_GETRI_SMALL(57); break;
        case 58: RUN_GETRI_SMALL(58); break;
        case 59: RUN_GETRI_SMALL(59); break;
        case 60: RUN_GETRI_SMALL(60); break;
        case 61: RUN_GETRI_SMALL(61); break;
        case 62: RUN_GETRI_SMALL(62); break;
        case 63: RUN_GETRI_SMALL(63); break;
        case 64: RUN_GETRI_SMALL(64); break;
        default: __builtin_unreachable();
    }
    
    return rocblas_status_success;
}
#endif //OPTIMAL

template <typename T, typename U, typename V>
__global__ void copy_and_zero(const rocblas_int m, const rocblas_int n,
                              U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                              V W, const rocblas_int shiftw, const rocblas_int ldw, const rocblas_stride stridew,
                              rocblas_fill uplo, rocblas_int *info)
{
    int b = hipBlockIdx_x;
    int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int j = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    T* a = load_ptr_batch<T>(A,b,shifta,stridea);
    T* w = load_ptr_batch<T>(W,b,shiftw,stridew);

    if (info[b] == 0)
    {
        if (i < m && j < n && (uplo == rocblas_fill_lower ? i > j : i <= j))
        {
            w[i + j*ldw] = a[i + j*lda];
            a[i + j*lda] = 0;
        }
    }
    else
    {
        if (i < m && j < n)
            w[i + j*ldw] = 0;
    }
}

template <typename T, typename U>
__global__ void getri_pivot(const rocblas_int n,
                            U A, const rocblas_int shifta, const rocblas_int lda, const rocblas_stride stridea,
                            rocblas_int *ipiv, const rocblas_int shiftp, const rocblas_stride stridep,
                            rocblas_int *info)
{
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A,b,shifta,stridea);
    rocblas_int* p = load_ptr_batch<rocblas_int>(ipiv,b,shiftp,stridep);

    if (info[b] != 0)
        return;

    rocblas_int jp;
    for (rocblas_int j = n-2; j >= 0; --j)
    {
        jp = p[j] - 1;
        if (jp != j)
        {
            swapvect(n, a + j*lda, 1, a + jp*lda, 1);
            __threadfence();
        }
    }
}


template <bool BATCHED, typename T>
void rocsolver_getri_getMemorySize(const rocblas_int n, const rocblas_int batch_count,
                                  size_t *size_1, size_t *size_2, size_t *size_3)
{
    rocsolver_trtri_getMemorySize<BATCHED,T>(n,batch_count,size_1,size_2,size_3);
    
    #ifdef OPTIMAL
    // if very small size, no workspace needed
    if (n <= WaveSize)
    {
        *size_2 = 0;
        return;
    }
    #endif

    // for workspace
    size_t s2 = (n <= GETRI_SWITCHSIZE ? n : n * GETRI_BLOCKSIZE);
    s2 *= sizeof(T)*batch_count;
    *size_2 = max(*size_2, s2);
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n, const rocblas_int lda, T A, rocblas_int *ipiv,
                                        rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(const rocblas_int n, const rocblas_int lda, const rocblas_int ldc, T A, T C,
                                        rocblas_int *ipiv, rocblas_int *info, const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    
    // 2. invalid size
    if (n < 0 || lda < n || ldc < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if ((n && !A) || (n && !C) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getri_template(rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA,
                                        const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv,
                                        const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int *info,
                                        const rocblas_int batch_count, T* scalars, T* work, T** workArr)
{
    // quick return if zero instances in batch
    if (batch_count == 0) 
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if (n == 0)
    {
        rocblas_int blocks = (batch_count - 1)/32 + 1;
        hipLaunchKernelGGL(reset_info, dim3(blocks,1,1), dim3(32,1,1), 0, stream,
                           info, batch_count, 0);
        return rocblas_status_success;
    }
    
    #ifdef OPTIMAL
    // if very small size, use optimized inversion kernel
    if (n <= WaveSize)
        return getri_small_sizes<T>(handle,n,A,shiftA,lda,strideA,ipiv,shiftP,strideP,info,batch_count);

    #endif

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle,&old_mode);
    rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);

    T minone = -1;
    T one = 1;
    rocblas_int threads = min(n, 1024);
    rocblas_int jb, nb = GETRI_BLOCKSIZE;
    rocblas_int ldw = n;
    rocblas_stride strideW;

    // compute inv(U)
    rocsolver_trtri_template<BATCHED,STRIDED,T>(handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n,
                                                A, shiftA, lda, strideA,
                                                info, batch_count, scalars, work, workArr);
    
    if (n <= GETRI_SWITCHSIZE)
    {
        // use unblocked version
        strideW = n;

        for (rocblas_int j = n-2; j >= 0; --j)
        {
            rocblas_int blocks = ((n-j) - 1)/64 + 1;
            hipLaunchKernelGGL(copy_and_zero<T>, dim3(batch_count,blocks,1), dim3(1,64,1), 0, stream,
                               n-j, 1, A, shiftA + idx2D(j,j,lda), lda, strideA, work, j, ldw, strideW, rocblas_fill_lower, info);

            // rocblasCall_gemv(handle, rocblas_operation_none, n, n-j-1,
            //                  &minone, 0, A, shiftA + idx2D(0,j+1,lda), lda, strideA,
            //                  work, j+1, 1, strideW,
            //                  &one, 0, A, shiftA + idx2D(0,j,lda), 1, strideA,
            //                  batch_count, workArr);
            hipLaunchKernelGGL(gemv_kernel<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                       n, n-j-1, scalars, A, shiftA + idx2D(0,j+1,lda), lda, strideA, work, j+1, 1, strideW, scalars+2, A, shiftA + idx2D(0,j,lda), 1, strideA);
        }
    }
    else
    {
        //use blocked version
        strideW = n*nb;

        rocblas_int nn = ((n - 1)/nb)*nb + 1;
        for (rocblas_int j = nn-1; j >= 0; j -= nb)
        {
            jb = min(n-j, nb);

            rocblas_int blocks1 = ((n-j) - 1)/32 + 1;
            rocblas_int blocks2 = (jb - 1)/32 + 1;
            hipLaunchKernelGGL(copy_and_zero<T>, dim3(batch_count,blocks1,blocks2), dim3(1,32,32), 0, stream,
                               n-j, jb, A, shiftA + idx2D(j,j,lda), lda, strideA, work, j, ldw, strideW, rocblas_fill_lower, info);

            if (j+jb < n)
                // rocblasCall_gemm<BATCHED,STRIDED>(handle, rocblas_operation_none, rocblas_operation_none,
                //                                   n, jb, n-j-jb,
                //                                   &minone, A, shiftA + idx2D(0,j+jb,lda), lda, strideA,
                //                                   work, j+jb, ldw, strideW,
                //                                   &one, A, shiftA + idx2D(0,j,lda), lda, strideA,
                //                                   batch_count, workArr);
                hipLaunchKernelGGL(gemm_kernel<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                        n, jb, n-j-jb, scalars, A, shiftA + idx2D(0,j+jb,lda), lda, strideA, work, j+jb, ldw, strideW, scalars+2, A, shiftA + idx2D(0,j,lda), lda, strideA);
            
            hipLaunchKernelGGL(trsm_kernel_right_lower<T>, dim3(batch_count,1,1), dim3(1,threads,1), 0, stream,
                       rocblas_diagonal_unit, n, jb, scalars+2, work, j, ldw, strideW, A, shiftA + idx2D(0,j,lda), lda, strideA);
        }
    }
    
    hipLaunchKernelGGL(getri_pivot<T>, dim3(batch_count,1,1), dim3(1,1,1), 0, stream,
                       n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info);

    rocblas_set_pointer_mode(handle,old_mode);
    return rocblas_status_success;
}

#endif /* ROCLAPACK_GETRI_H */
