/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "norm.hpp"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"
 

template <bool STRIDED, bool GETRF, typename T, typename U>
void getf2_getrf_checkBadArgs(const rocblas_handle handle, 
                         const rocblas_int m, 
                         const rocblas_int n, 
                         T dA, 
                         const rocblas_int lda, 
                         const rocblas_stride stA,
                         U dIpiv, 
                         const rocblas_stride stP,
                         U dinfo,
                         const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,nullptr,m,n,dA,lda,stA,dIpiv,stP,dinfo,bc), 
                          rocblas_status_invalid_handle);
    
    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,dA,lda,stA,dIpiv,stP,dinfo,-1), 
                              rocblas_status_invalid_size);
        
    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,(T)nullptr,lda,stA,dIpiv,stP,dinfo,bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,dA,lda,stA,(U)nullptr,stP,dinfo,bc), 
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,dA,lda,stA,dIpiv,stP,(U)nullptr,bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,0,n,(T)nullptr,lda,stA,(U)nullptr,stP,dinfo,bc), 
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,0,(T)nullptr,lda,stA,(U)nullptr,stP,dinfo,bc), 
                          rocblas_status_success);
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,dA,lda,stA,dIpiv,stP,(U)nullptr,0),
                              rocblas_status_success);
    
    // quick return with zero batch_count if applicable
    if (STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED,GETRF,handle,m,n,dA,lda,stA,dIpiv,stP,dinfo,0),
                              rocblas_status_success);
}


template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getf2_getrf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if (BATCHED) {
        // memory allocations
        device_batch_vector<T> dA(1,1,1);
        device_strided_batch_vector<rocblas_int> dIpiv(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());
        
        // check bad arguments
        getf2_getrf_checkBadArgs<STRIDED,GETRF>(handle,m,n,dA.data(),lda,stA,dIpiv.data(),stP,dinfo.data(),bc);

    } else {
        // memory allocations
        device_strided_batch_vector<T> dA(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dIpiv(1,1,1,1);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        getf2_getrf_checkBadArgs<STRIDED,GETRF>(handle,m,n,dA.data(),lda,stA,dIpiv.data(),stP,dinfo.data(),bc);
    }
}


template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_initData(const rocblas_handle handle, 
                        const rocblas_int m, 
                        const rocblas_int n, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dinfo,
                        const rocblas_int bc,
                        Th &hA,
                        Uh &hIpiv, 
                        Uh &hinfo)
{
    if (CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities 
        for (rocblas_int b = 0; b < bc; ++b) {
            for (rocblas_int i = 0; i < m; i++) {
                for (rocblas_int j = 0; j < n; j++) {
                    if (i == j)
                        hA[b][i + j * lda] += 400;
                    else    
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if (GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}


template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_getError(const rocblas_handle handle, 
                        const rocblas_int m, 
                        const rocblas_int n, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dinfo,
                        const rocblas_int bc,
                        Th &hA, 
                        Th &hARes, 
                        Uh &hIpiv, 
                        Uh &hinfo,
                        double *max_err)
{
    // input data initialization 
    getf2_getrf_initData<true,true,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                     hA, hIpiv, hinfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED,GETRF,handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP, dinfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for (rocblas_int b = 0; b < bc; ++b) {
        GETRF ?
            cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hinfo[b]):
            cblas_getf2<T>(m, n, hA[b], lda, hIpiv[b], hinfo[b]);
    }

    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA|| (ideally ||LU - Lres Ures|| / ||LU||) 
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES. 
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for (rocblas_int b = 0; b < bc; ++b) {
        err = norm_error('F',m,n,lda,hA[b],hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}


template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_getPerfData(const rocblas_handle handle, 
                        const rocblas_int m, 
                        const rocblas_int n, 
                        Td &dA, 
                        const rocblas_int lda, 
                        const rocblas_stride stA, 
                        Ud &dIpiv, 
                        const rocblas_stride stP, 
                        Ud &dinfo,
                        const rocblas_int bc, const rocblas_int pivot,
                        Th &hA, 
                        Uh &hIpiv, 
                        Uh &hinfo,
                        double *gpu_time_used,
                        double *cpu_time_used,
                        const rocblas_int hot_calls)
{
    // cpu-lapack performance
    getf2_getrf_initData<true,false,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                     hA, hIpiv, hinfo);
    *cpu_time_used = get_time_us();
    for (rocblas_int b = 0; b < bc; ++b) {
        GETRF ?
            cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hinfo[b]):
            cblas_getf2<T>(m, n, hA[b], lda, hIpiv[b], hinfo[b]);
    }
    *cpu_time_used = get_time_us() - *cpu_time_used;
    getf2_getrf_initData<true,false,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                     hA, hIpiv, hinfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getf2_getrf_initData<false,true,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                        hA, hIpiv, hinfo);

        CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED,GETRF,handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP, dinfo.data(), bc, pivot));
    }

    clear_time_agg();
        
    // gpu-lapack performance
    double start;
    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        getf2_getrf_initData<false,true,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                        hA, hIpiv, hinfo);
        
        start = get_time_us();
        rocsolver_getf2_getrf(STRIDED,GETRF,handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP, dinfo.data(), bc, pivot);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}


template <bool BATCHED, bool STRIDED, bool GETRF, typename T> 
void testing_getf2_getrf(Arguments argus) 
{
    // get arguments 
    rocblas_local_handle handle;
    rocblas_int m = argus.M;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stP = argus.bsp;
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    rocblas_int pivot = argus.pivot;

    rocblas_stride stARes = argus.unit_check || argus.norm_check ? stA : 0;

    // check non-supported values 
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m,n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = argus.unit_check || argus.norm_check ? size_A : 0;

    // check invalid sizes 
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if (invalid_size) {
        if (BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T *const *)nullptr, lda, stA, (rocblas_int*)nullptr, stP, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T *)nullptr, lda, stA, (rocblas_int*)nullptr, stP, (rocblas_int *)nullptr, bc),
                                  rocblas_status_invalid_size);

        if (argus.timing) 
             ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if (BATCHED) {
        // memory allocations
        host_batch_vector<T> hA(size_A,1,bc);
        host_batch_vector<T> hARes(size_ARes,1,bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,stP,bc);
        host_strided_batch_vector<rocblas_int> hinfo(1,1,1,bc);
        device_batch_vector<T> dA(size_A,1,bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,stP,bc);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,bc);
        if (size_A) CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());
        if (size_P) CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if (m == 0 || n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP, dinfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            getf2_getrf_getError<STRIDED,GETRF,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                          hA, hARes, hIpiv, hinfo, &max_error);

        // collect performance data
        if (argus.timing) 
            getf2_getrf_getPerfData<STRIDED,GETRF,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, pivot,
                                              hA, hIpiv, hinfo, &gpu_time_used, &cpu_time_used, hot_calls);
    } 

    else {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A,1,stA,bc);
        host_strided_batch_vector<T> hARes(size_ARes,1,stARes,bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,stP,bc);
        host_strided_batch_vector<rocblas_int> hinfo(1,1,1,bc);
        device_strided_batch_vector<T> dA(size_A,1,stA,bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,stP,bc);
        device_strided_batch_vector<rocblas_int> dinfo(1,1,1,bc);
        if (size_A) CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());
        if (size_P) CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if (m == 0 || n == 0 || bc == 0) {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP, dinfo.data(), bc),
                                  rocblas_status_success);
            if (argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if (argus.unit_check || argus.norm_check) 
            getf2_getrf_getError<STRIDED,GETRF,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, 
                                          hA, hARes, hIpiv, hinfo, &max_error);

        // collect performance data
        if (argus.timing) 
            getf2_getrf_getPerfData<STRIDED,GETRF,T>(handle, m, n, dA, lda, stA, dIpiv, stP, dinfo, bc, pivot,
                                              hA, hIpiv, hinfo, &gpu_time_used, &cpu_time_used, hot_calls);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if (argus.unit_check) 
        rocsolver_test_check<T>(max_error,min(m,n));     

    // output results for rocsolver-bench
    if (argus.timing) {
        if (!argus.perf) {
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Arguments:\n";
            rocblas_cout << "============================================\n";
            if (BATCHED) {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if (STRIDED) {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
            }
            rocblas_cout << "\n============================================\n";
            rocblas_cout << "Results:\n";
            rocblas_cout << "============================================\n";
            rocsolver_bench_output("reset_info");
            rocsolver_bench_output(get_calls_agg("reset_info") / hot_calls, get_time_agg("reset_info") / hot_calls);
            rocsolver_bench_output("rocblas_iamax");
            rocsolver_bench_output(get_calls_agg("rocblas_iamax") / hot_calls, get_time_agg("rocblas_iamax") / hot_calls);
            rocsolver_bench_output("getf2_check_singularity");
            rocsolver_bench_output(get_calls_agg("getf2_check_singularity") / hot_calls, get_time_agg("getf2_check_singularity") / hot_calls);
            rocsolver_bench_output("rocsolver_laswp");
            rocsolver_bench_output(get_calls_agg("rocsolver_laswp") / hot_calls, get_time_agg("rocsolver_laswp") / hot_calls);
            rocsolver_bench_output("rocblas_scal");
            rocsolver_bench_output(get_calls_agg("rocblas_scal") / hot_calls, get_time_agg("rocblas_scal") / hot_calls);
            rocsolver_bench_output("rocblas_ger");
            rocsolver_bench_output(get_calls_agg("rocblas_ger") / hot_calls, get_time_agg("rocblas_ger") / hot_calls);
            rocsolver_bench_output("getrf_check_singularity");
            rocsolver_bench_output(get_calls_agg("getrf_check_singularity") / hot_calls, get_time_agg("getrf_check_singularity") / hot_calls);
            rocsolver_bench_output("rocblas_trsm");
            rocsolver_bench_output(get_calls_agg("rocblas_trsm") / hot_calls, get_time_agg("rocblas_trsm") / hot_calls);
            rocsolver_bench_output("rocblas_gemm");
            rocsolver_bench_output(get_calls_agg("rocblas_gemm") / hot_calls, get_time_agg("rocblas_gemm") / hot_calls);
            if (argus.norm_check) {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocblas_cout << std::endl;
        }
        else rocsolver_bench_output(gpu_time_used);
    }
}
  

#undef GETRF_ERROR_EPS_MULTIPLIER
