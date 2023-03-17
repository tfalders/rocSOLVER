
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef HIP_CHECK_H
#define HIP_CHECK_H

#include <stdexcept>
#include <exception>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifndef HIP_CHECK
#define HIP_CHECK(fcn, error_code)                                                         \
    {                                                                                      \
        hipError_t istat = (fcn);                                                          \
        if(istat != HIP_SUCCESS)                                                           \
        {                                                                                  \
            printf("HIP API failed at line %d in file %s with error: %s (%d)\n", __LINE__, \
                   __FILE__, hipGetErrorString(istat), istat);                             \
            return (error_code);                                                           \
        };                                                                                 \
    };
#endif

#ifndef THROW_IF_HIP_ERROR
#define THROW_IF_HIP_ERROR(fcn)                               \
    {                                                         \
        if((fcn) != HIP_SUCCESS)                              \
        {                                                     \
            printf("HIP failed %s:%d\n", __FILE__, __LINE__); \
            throw std::runtime_error(__FILE__);               \
        };                                                    \
    };
#endif

#endif
