/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_host_helpers.hpp"
#include "init_scalars.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"

#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

#define MIN_CHUNK_SIZE 64

/* ROCSOLVER_WORKSPACE_HELPER provides a wrapper for workspace arrays, allowing workspace arrays to be passed
   between functions using a single object rather than individually, and improving device memory utilization. */
template <typename T>
class rocsolver_workspace_helper
{
private:
    bool has_scalars, optim_mem;
    size_t num_excl, size_excl, size_shared;
    std::vector<uint8_t*> pointers;
    std::vector<size_t> sizes;
    std::vector<rocsolver_workspace_helper<T>*> nested;

    void assign_buffer(uint8_t* scalars, uint8_t* curr_ptr)
    {
        pointers.reserve(sizes.size() + has_scalars);

        if(has_scalars)
            pointers.push_back(scalars);

        uint8_t* nested_ptr = curr_ptr;
        for(int i = 0; i < sizes.size(); i++)
        {
            pointers.push_back(curr_ptr);
            curr_ptr += sizes[i];
            if(i < num_excl)
                nested_ptr = curr_ptr;
        }

        for(int i = 0; i < nested.size(); i++)
            nested[i]->assign_buffer(scalars, nested_ptr);
    }

    size_t get_internal_size(bool* has_scalars)
    {
        if(this->has_scalars)
            *has_scalars = 1;

        size_t shared = this->size_shared;

        for(int i = 0; i < nested.size(); i++)
            shared = std::max(shared, nested[i]->get_internal_size(has_scalars));

        return this->size_excl + shared;
    }

public:
    /* Constructor */
    rocsolver_workspace_helper()
        : has_scalars(0)
        , optim_mem(0)
        , num_excl(0)
        , size_excl(0)
        , size_shared(0)
        , pointers(0)
        , sizes(0)
        , nested(0)
    {
    }

    /* Disallow copying. */
    rocsolver_workspace_helper(const rocsolver_workspace_helper<T>&) = delete;

    /* Disallow assigning. */
    rocsolver_workspace_helper& operator=(const rocsolver_workspace_helper<T>&) = delete;

    /* Destructor */
    ~rocsolver_workspace_helper()
    {
        for(int i = 0; i < nested.size(); i++)
            delete nested[i];
    }

    /* Assigns the given workspace sizes to the workspace helper. Workspace sizes are passed in two separate lists:
       sizes_excl for workspaces that must not be overwritten by nested functions, and sizes_shared for workspaces that
       may be overwritten by nested functions. Also rounds the sizes up to the nearest MIN_CHUNK_SIZE for better
       alignment. */
    void assign_sizes(std::initializer_list<size_t> sizes_excl,
                      std::initializer_list<size_t> sizes_shared)
    {
        this->num_excl = sizes_excl.size();
        this->size_excl = 0;
        this->size_shared = 0;

        this->sizes.clear();
        this->sizes.reserve(sizes_excl.size() + sizes_shared.size());

        this->sizes.insert(this->sizes.end(), sizes_excl);
        this->sizes.insert(this->sizes.end(), sizes_shared);

        // round up sizes to nearest MIN_CHUNK_SIZE
        for(int i = 0; i < sizes.size(); i++)
        {
            this->sizes[i]
                = ((this->sizes[i] + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;

            if(i < this->num_excl)
                this->size_excl += this->sizes[i];
            else
                this->size_shared += this->sizes[i];
        }
    }

    /* Sets the capacity of the internal vector that holds workspace helpers for nested functions. Optional,
       but should be called before add_nested. */
    void set_nested_capacity(int capacity)
    {
        nested.reserve(capacity);
    }
    /* Adds a nested workspace helper to manage workspaces for a nested function. May be called before or
       after assign_sizes. */
    rocsolver_workspace_helper<T>* add_nested()
    {
        rocsolver_workspace_helper<T>* result = new rocsolver_workspace_helper<T>();
        nested.push_back(result);
        return result;
    }
    /* Gets a pointer to the nested workspace helper at position i, which was created by the ith call to
       add_nested. */
    rocsolver_workspace_helper<T>* get_nested(int i)
    {
        return nested[i];
    }

    /* Assigns device memory to the workspace helper, to be partitioned into individual workspace arrays
       based on the assigned sizes. Only called after assign_sizes (and, if applicable, add_nested). */
    rocblas_status assign_buffer(rocblas_handle handle, void* buffer)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        bool has_scalars = 0;
        size_t size_scalars = 0;
        size_t size_mem = get_internal_size(&has_scalars);

        if(has_scalars)
        {
            size_scalars = sizeof(T) * 3;
            size_scalars = ((size_scalars + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }

        // zero the workspace
        HIP_CHECK(hipMemsetAsync(buffer, 0, size_scalars + size_mem, stream));

        // assign pointers
        uint8_t* scalars = (uint8_t*)buffer;
        uint8_t* curr_ptr = (has_scalars ? scalars + size_scalars : scalars);
        assign_buffer(scalars, curr_ptr);

        // initialize scalars
        if(has_scalars)
            ROCSOLVER_LAUNCH_KERNEL(iota_n<T>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream, (T*)scalars,
                                    3, -1);

        return rocblas_status_success;
    }
    /* Gets a pointer to the workspace array at position i, whose size is >= the assigned size at position i.
       Only called after assign_buffer. */
    void* operator[](int i)
    {
        if(i + has_scalars < pointers.size())
            return pointers[i + has_scalars];
        else
            return nullptr;
    }

    /* Sets a value indicating that the function requires the device-side scalar array. Only called before
       assign_buffer. */
    void add_scalars()
    {
        this->has_scalars = 1;
    }
    /* Gets a pointer to the device-side scalar array. Only called after assign_buffer. */
    T* get_scalars()
    {
        if(has_scalars && pointers.size() > 0)
            return (T*)pointers[0];
        else
            return nullptr;
    }

    /* Sets a value indicating if the optimal amount of memory for TRSM is available. May be called anytime. */
    void set_optim_mem(bool optim_mem)
    {
        this->optim_mem = optim_mem;
    }
    /* Gets a value indicating if the optimal amount of memory for TRSM is available. May be called anytime. */
    bool get_optim_mem()
    {
        return this->optim_mem;
    }

    /* Returns the total amount of device memory required by the workspaces assigned to this helper and its
       nested helpers. Only called after assign_sizes (and, if applicable, add_nested). */
    size_t get_total_size()
    {
        bool has_scalars = 0;
        size_t size_scalars = 0;
        size_t size_mem = get_internal_size(&has_scalars);

        if(has_scalars)
        {
            size_scalars = sizeof(T) * 3;
            size_scalars = ((size_scalars + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }

        return size_scalars + size_mem;
    }

    void print_debug()
    {
        bool has_scalars = 0;

        std::cerr << "Num Sizes: " << sizes.size() << std::endl;
        for(int i = 0; i < sizes.size(); i++)
        {
            std::cerr << sizes[i] << " bytes";
            if(i < pointers.size())
                std::cerr << " (" << (void*)pointers[i] << ")";
            if(i >= num_excl)
                std::cerr << " shared";
            std::cerr << std::endl;
        }
        std::cerr << "Total Size (Self): " << size_excl + size_shared << std::endl;
        if(nested.size() > 0)
            std::cerr << "Total Size (Self+Nested): " << get_internal_size(&has_scalars) << std::endl;
    }
};

#undef MIN_CHUNK_SIZE

ROCSOLVER_END_NAMESPACE
