/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T, typename S, typename U>
void stein_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int n,
                        S dD,
                        S dE,
                        U dNev,
                        S dW,
                        U dIblock,
                        U dIsplit,
                        T dZ,
                        const rocblas_int ldz,
                        U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(nullptr, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, (S) nullptr, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, (S) nullptr, dNev, dW, dIblock, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, (U) nullptr, dW, dIblock, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, (S) nullptr, dIblock, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, (U) nullptr, dIsplit, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, (U) nullptr, dZ, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, (T) nullptr, ldz, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, (U) nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, 0, (S) nullptr, (S) nullptr, dNev, (S) nullptr,
                                          (U) nullptr, (U) nullptr, (T) nullptr, ldz, dInfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stein_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int ldz = 1;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
    device_strided_batch_vector<S> dW(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIblock(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(1, 1, 1, 1);
    device_strided_batch_vector<T> dZ(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dNev.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dIblock.memcheck());
    CHECK_HIP_ERROR(dIsplit.memcheck());
    CHECK_HIP_ERROR(dZ.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    stein_checkBadArgs(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(), dIblock.data(),
                       dIsplit.data(), dZ.data(), ldz, dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Ud, typename Sh, typename Uh>
void stein_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Ud& dNev,
                    Sd& dW,
                    Ud& dIblock,
                    Ud& dIsplit,
                    Sh& hD,
                    Sh& hE,
                    Uh& hNev,
                    Sh& hW,
                    Uh& hIblock,
                    Uh& hIsplit)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, true);

        rocblas_int nsplit, info;
        size_t lwork = 4 * n;
        size_t liwork = 3 * n;
        std::vector<S> work(lwork);
        std::vector<rocblas_int> iwork(liwork);

        // scale matrix
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 10;
            hE[0][i] -= 5;
        }

        // compute a subset of the eigenvalues
        S vl = 0.0;
        S vu = 10.0;
        S abstol = 2 * get_safemin<S>();
        cblas_stebz<S>(rocblas_erange_value, rocblas_eorder_blocks, n, vl, vu, 0, 0, abstol, hD[0],
                       hE[0], hNev[0], &nsplit, hW[0], hIblock[0], hIsplit[0], work.data(),
                       iwork.data(), &info);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        CHECK_HIP_ERROR(dNev.transfer_from(hNev));
        CHECK_HIP_ERROR(dW.transfer_from(hW));
        CHECK_HIP_ERROR(dIblock.transfer_from(hIblock));
        CHECK_HIP_ERROR(dIsplit.transfer_from(hIsplit));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stein_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Ud& dNev,
                    Sd& dW,
                    Ud& dIblock,
                    Ud& dIsplit,
                    Td& dZ,
                    const rocblas_int ldz,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Uh& hNev,
                    Sh& hW,
                    Uh& hIblock,
                    Uh& hIsplit,
                    Th& hZ,
                    Th& hZRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err)
{
    using S = decltype(std::real(T{}));

    size_t lwork = 5 * n;
    size_t liwork = n;
    size_t lifail = n;
    std::vector<S> work(lwork);
    std::vector<rocblas_int> iwork(liwork);
    std::vector<rocblas_int> ifail(lifail);

    // input data initialization
    stein_initData<true, true, T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev, hW,
                                  hIblock, hIsplit);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(),
                                        dIblock.data(), dIsplit.data(), dZ.data(), ldz, dInfo.data()));
    CHECK_HIP_ERROR(hZRes.transfer_from(dZ));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // Prepare matrix A (upper triangular) for implicit tests
    rocblas_int lda = n;
    size_t size_A = lda * n;
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    for(rocblas_int i = 0; i < n; i++)
    {
        for(rocblas_int j = i; j < n; j++)
        {
            if(i == j)
                hA[0][i + j * lda] = hD[0][i];
            else if(i + 1 == j)
                hA[0][i + j * lda] = hE[0][i];
            else
                hA[0][i + j * lda] = 0;
        }
    }

    // CPU lapack
    cblas_stein<T>(n, hD[0], hE[0], hNev[0], hW[0], hIblock[0], hIsplit[0], hZ[0], ldz, work.data(),
                   iwork.data(), ifail.data(), hInfo[0]);

    // check info
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    double err;

    if(hInfo[0][0] == 0)
    {
        // need to implicitly test eigenvectors due to non-uniqueness of eigenvectors under scaling

        // multiply A with each of the nev eigenvectors and divide by corresponding
        // eigenvalues
        T alpha;
        T beta = 0;
        for(int j = 0; j < hNev[0][0]; j++)
        {
            alpha = T(1) / hW[0][j];
            cblas_symv_hemv(rocblas_fill_upper, n, alpha, hA[0], lda, hZRes[0] + j * ldz, 1, beta,
                            hZ[0] + j * ldz, 1);
        }

        // error is ||hZ - hZRes|| / ||hZ||
        // using frobenius norm
        err = norm_error('F', n, hNev[0][0], ldz, hZ[0], hZRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stein_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       Sd& dD,
                       Sd& dE,
                       Ud& dNev,
                       Sd& dW,
                       Ud& dIblock,
                       Ud& dIsplit,
                       Td& dZ,
                       const rocblas_int ldz,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Uh& hNev,
                       Sh& hW,
                       Uh& hIblock,
                       Uh& hIsplit,
                       Th& hZ,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool perf)
{
    using S = decltype(std::real(T{}));

    size_t lwork = 5 * n;
    size_t liwork = n;
    size_t lifail = n;
    std::vector<S> work(lwork);
    std::vector<rocblas_int> iwork(liwork);
    std::vector<rocblas_int> ifail(lifail);

    if(!perf)
    {
        stein_initData<true, false, T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev,
                                       hW, hIblock, hIsplit);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_stein<T>(n, hD[0], hE[0], hNev[0], hW[0], hIblock[0], hIsplit[0], hZ[0], ldz,
                       work.data(), iwork.data(), ifail.data(), hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stein_initData<true, false, T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev, hW,
                                   hIblock, hIsplit);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stein_initData<false, true, T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev,
                                       hW, hIblock, hIsplit);

        CHECK_ROCBLAS_ERROR(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(),
                                            dIblock.data(), dIsplit.data(), dZ.data(), ldz,
                                            dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        stein_initData<false, true, T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev,
                                       hW, hIblock, hIsplit);

        start = get_time_us_sync(stream);
        rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(), dIblock.data(),
                        dIsplit.data(), dZ.data(), ldz, dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stein(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldz = argus.get<rocblas_int>("ldz", n);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = size_D;
    size_t size_W = size_D;
    size_t size_iblock = size_D;
    size_t size_isplit = size_D;
    size_t size_Z = ldz * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ZRes = (argus.unit_check || argus.norm_check) ? size_Z : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || ldz < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, (S*)nullptr, (S*)nullptr,
                                              (rocblas_int*)nullptr, (S*)nullptr,
                                              (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                              (T*)nullptr, ldz, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stein(handle, n, (S*)nullptr, (S*)nullptr, (rocblas_int*)nullptr,
                                          (S*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                          (T*)nullptr, ldz, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    // host
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, 1);
    host_strided_batch_vector<S> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<rocblas_int> hIblock(size_iblock, 1, size_iblock, 1);
    host_strided_batch_vector<rocblas_int> hIsplit(size_isplit, 1, size_isplit, 1);
    host_strided_batch_vector<T> hZ(size_Z, 1, size_Z, 1);
    host_strided_batch_vector<T> hZRes(size_ZRes, 1, size_ZRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
    // device
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
    device_strided_batch_vector<S> dW(size_W, 1, size_W, 1);
    device_strided_batch_vector<rocblas_int> dIblock(size_iblock, 1, size_iblock, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(size_isplit, 1, size_isplit, 1);
    device_strided_batch_vector<T> dZ(size_Z, 1, size_Z, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dNev.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_iblock)
        CHECK_HIP_ERROR(dIblock.memcheck());
    if(size_isplit)
        CHECK_HIP_ERROR(dIsplit.memcheck());
    if(size_Z)
        CHECK_HIP_ERROR(dZ.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(),
                                              dW.data(), hIblock.data(), hIsplit.data(), dZ.data(),
                                              ldz, dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stein_getError<T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dInfo, hD, hE,
                          hNev, hW, hIblock, hIsplit, hZ, hZRes, hInfo, hInfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        stein_getPerfData<T>(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dInfo, hD, hE,
                             hNev, hW, hIblock, hIsplit, hZ, hInfo, &gpu_time_used, &cpu_time_used,
                             hot_calls, argus.profile, argus.perf);

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "ldz");
            rocsolver_bench_output(n, ldz);

            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
