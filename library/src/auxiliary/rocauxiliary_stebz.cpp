/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_stebz.hpp"

template <typename T>
rocblas_status rocsolver_stebz_impl(rocblas_handle handle,
                                    const rocblas_erange range,
                                    const rocblas_eorder order,
                                    const rocblas_int n,
                                    const T vlow,
                                    const T vup,
                                    const rocblas_int ilow,
                                    const rocblas_int iup,
                                    const T abstol,
                                    T* D,
                                    T* E,
                                    rocblas_int* nev,
                                    rocblas_int* nsplit,
                                    T* W,
                                    rocblas_int* IB,
                                    rocblas_int* IS,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stebz", "--range", range, "--order", order, "-n", n, "--vlow", vlow,
                        "--vup", vup, "--ilow", ilow, "--iup", iup, "--abstol", abstol);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_stebz_argCheck(handle, range, order, n, vlow, vup, ilow, iup, D,
                                                 E, nev, nsplit, W, IB, IS, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideIB = 0;
    rocblas_stride strideIS = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    size_t size_work, size_pivmin, size_Esqr, size_bounds, size_inter, size_ninter;
    rocsolver_stebz_getMemorySize<T>(n, batch_count, &size_work, &size_pivmin, &size_Esqr,
                                     &size_bounds, &size_inter, &size_ninter);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_pivmin, size_Esqr,
                                                      size_bounds, size_inter, size_ninter);

    // memory workspace allocation
    void *work, *pivmin, *Esqr, *bounds, *inter, *ninter;
    rocblas_device_malloc mem(handle, size_work, size_pivmin, size_Esqr, size_bounds, size_inter,
                              size_ninter);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    pivmin = mem[1];
    Esqr = mem[2];
    bounds = mem[3];
    inter = mem[4];
    ninter = mem[5];

    // execution
    return rocsolver_stebz_template<T>(
        handle, range, order, n, vlow, vup, ilow, iup, abstol, D, shiftD, strideD, E, shiftE,
        strideE, nev, nsplit, W, strideW, IB, strideIB, IS, strideIS, info, batch_count,
        (rocblas_int*)work, (T*)pivmin, (T*)Esqr, (T*)bounds, (T*)inter, (rocblas_int*)ninter);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sstebz(rocblas_handle handle,
                                const rocblas_erange range,
                                const rocblas_eorder order,
                                const rocblas_int n,
                                const float vlow,
                                const float vup,
                                const rocblas_int ilow,
                                const rocblas_int iup,
                                const float abstol,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                rocblas_int* nsplit,
                                float* W,
                                rocblas_int* IB,
                                rocblas_int* IS,
                                rocblas_int* info)
{
    return rocsolver_stebz_impl<float>(handle, range, order, n, vlow, vup, ilow, iup, abstol, D, E,
                                       nev, nsplit, W, IB, IS, info);
}

rocblas_status rocsolver_dstebz(rocblas_handle handle,
                                const rocblas_erange range,
                                const rocblas_eorder order,
                                const rocblas_int n,
                                const double vlow,
                                const double vup,
                                const rocblas_int ilow,
                                const rocblas_int iup,
                                const double abstol,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                rocblas_int* nsplit,
                                double* W,
                                rocblas_int* IB,
                                rocblas_int* IS,
                                rocblas_int* info)
{
    return rocsolver_stebz_impl<double>(handle, range, order, n, vlow, vup, ilow, iup, abstol, D, E,
                                        nev, nsplit, W, IB, IS, info);
}

} // extern C
