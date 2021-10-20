/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri_kernels.hpp"

#ifdef OPTIMAL

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETRI_SMALL(rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_GETRI_SMALL(rocblas_float_complex, rocblas_float_complex* const*);

#endif
