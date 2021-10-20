/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2_kernels.hpp"

#ifdef OPTIMAL

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_SMALL(double, double*);
INSTANTIATE_GETF2_SMALL(double, double* const*);

INSTANTIATE_GETF2_PANEL(double, double*);
INSTANTIATE_GETF2_PANEL(double, double* const*);

INSTANTIATE_GETF2_SCALE_UPDATE(double, double*);
INSTANTIATE_GETF2_SCALE_UPDATE(double, double* const*);

#endif
