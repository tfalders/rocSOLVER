/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <new>

#include "rocsolver_rfinfo.hpp"

extern "C" rocblas_status rocsolver_create_rfinfo(rocsolver_rfinfo* rfinfo, rocblas_handle handle)
{
#ifdef ROCSOLVER_WITH_ROCSPARSE
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    if(rfinfo == nullptr)
        return rocblas_status_invalid_pointer;

    try
    {
        *rfinfo = new rocsolver_rfinfo_;
    }
    catch(const std::bad_alloc&)
    {
        return rocblas_status_memory_error;
    }

    rocblas_status status = (**rfinfo).init(handle);
    if(status != rocblas_status_success)
    {
        (**rfinfo).destroy();
        delete *rfinfo;
        rfinfo = nullptr;
        return status;
    }

    return rocblas_status_success;
#else
    return rocblas_status_not_implemented;
#endif
}

extern "C" rocblas_status rocsolver_destroy_rfinfo(rocsolver_rfinfo rfinfo)
{
#ifdef ROCSOLVER_WITH_ROCSPARSE
    if(rfinfo == nullptr)
        return rocblas_status_invalid_pointer;

    (*rfinfo).destroy();
    delete rfinfo;

    return rocblas_status_success;
#else
    return rocblas_status_not_implemented;
#endif
}