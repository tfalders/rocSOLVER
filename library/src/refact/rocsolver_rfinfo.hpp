/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#ifdef ROCSOLVER_WITH_ROCSPARSE
#include "rocsparse.hpp"

struct rocsolver_rfinfo_
{
    rocsparse_handle sphandle = nullptr;
    rocsparse_mat_descr descrL = nullptr;
    rocsparse_mat_descr descrU = nullptr;
    rocsparse_mat_descr descrT = nullptr;
    rocsparse_mat_info infoL = nullptr;
    rocsparse_mat_info infoU = nullptr;
    rocsparse_mat_info infoT = nullptr;
    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    rocblas_status init(rocblas_handle handle)
    {
        // create sparse handle
        ROCSPARSE_CHECK(rocsparse_create_handle(&sphandle));

        // use handle->stream to sphandle->stream
        hipStream_t stream;
        ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));
        ROCSPARSE_CHECK(rocsparse_set_stream(sphandle, stream));

        // ----------------------------------------------------------
        // TODO: check whether to use triangular type or general type
        // ----------------------------------------------------------
        rocsparse_matrix_type const L_type = rocsparse_matrix_type_general;
        rocsparse_matrix_type const U_type = rocsparse_matrix_type_general;
        rocsparse_matrix_type const T_type = rocsparse_matrix_type_general;

        // create and set matrix descriptors
        ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrL));
        ROCSPARSE_CHECK(rocsparse_set_mat_type(descrL, L_type));
        ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero));
        ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower));
        ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrL, rocsparse_diag_type_unit));

        ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrU));
        ROCSPARSE_CHECK(rocsparse_set_mat_type(descrU, U_type));
        ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero));
        ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper));
        ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrU, rocsparse_diag_type_non_unit));

        ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrT));
        ROCSPARSE_CHECK(rocsparse_set_mat_type(descrT, T_type));
        ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrT, rocsparse_index_base_zero));

        // create info holders
        ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoL));
        ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoU));
        ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoT));

        return rocblas_status_success;
    }

    rocblas_status destroy()
    {
        int nerrors = 0;

        if(sphandle != nullptr)
        {
            if(rocsparse_destroy_handle(sphandle) != rocsparse_status_success)
                nerrors++;
            sphandle = nullptr;
        }
        if(descrL != nullptr)
        {
            if(rocsparse_destroy_mat_descr(descrL) != rocsparse_status_success)
                nerrors++;
            descrL = nullptr;
        }
        if(descrU != nullptr)
        {
            if(rocsparse_destroy_mat_descr(descrU) != rocsparse_status_success)
                nerrors++;
            descrU = nullptr;
        }
        if(descrT != nullptr)
        {
            if(rocsparse_destroy_mat_descr(descrT) != rocsparse_status_success)
                nerrors++;
            descrT = nullptr;
        }
        if(infoL != nullptr)
        {
            if(rocsparse_destroy_mat_info(infoL) != rocsparse_status_success)
                nerrors++;
            infoL = nullptr;
        }
        if(infoU != nullptr)
        {
            if(rocsparse_destroy_mat_info(infoU) != rocsparse_status_success)
                nerrors++;
            infoU = nullptr;
        }
        if(infoT != nullptr)
        {
            if(rocsparse_destroy_mat_info(infoT) != rocsparse_status_success)
                nerrors++;
            infoT = nullptr;
        }

        return (nerrors == 0 ? rocblas_status_success : rocblas_status_internal_error);
    }
};
typedef struct rocsolver_rfinfo_* rocsolver_rfinfo;

#endif
