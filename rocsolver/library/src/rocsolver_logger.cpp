/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver_logger.hpp"
#include <sys/time.h>

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

// initialize the static variable
rocsolver_logger* rocsolver_logger::_instance = nullptr;
std::mutex rocsolver_logger::_mutex;

/***************************************************************************
 * Open logging streams
 ***************************************************************************/

auto rocsolver_logger::open_log_stream(const char* environment_variable_name)
{
    const char* logfile;
    if((logfile = getenv(environment_variable_name)) != nullptr
       || (logfile = getenv("ROCSOLVER_LOG_PATH")) != nullptr)
    {
        auto os = std::make_unique<rocsolver_ostream>(logfile);

        // print version info only once per file
        if(os != trace_os && os != bench_os && os != profile_os)
        {
            *os << "ROCSOLVER LOG FILE" << '\n';
            *os << "rocSOLVER Version: " << STRINGIFY(ROCSOLVER_VERSION_MAJOR) << '.'
                << STRINGIFY(ROCSOLVER_VERSION_MINOR) << '.' << STRINGIFY(ROCSOLVER_VERSION_PATCH)
                << '.' << STRINGIFY(ROCSOLVER_VERSION_TWEAK) << '\n';
            *os << "rocBLAS Version: " << STRINGIFY(ROCBLAS_VERSION_MAJOR) << '.'
                << STRINGIFY(ROCBLAS_VERSION_MINOR) << '.' << STRINGIFY(ROCBLAS_VERSION_PATCH)
                << '.' << STRINGIFY(ROCBLAS_VERSION_TWEAK) << '\n';
            *os << std::endl;
        }

        return os;
    }
    else
        return std::make_unique<rocsolver_ostream>(STDERR_FILENO);
}

/***************************************************************************
 * Call stack manipulation
 ***************************************************************************/

rocsolver_log_entry& rocsolver_logger::push_log_entry(rocblas_handle handle, std::string name)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    stack.push_back(rocsolver_log_entry());

    rocsolver_log_entry& result = stack.back();
    result.name = name;
    result.level = stack.size() - 1;
    result.time = get_time_us_no_sync();

    return result;
}

rocsolver_log_entry& rocsolver_logger::peek_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack.at(handle);
    rocsolver_log_entry& result = stack.back();
    return result;
}

rocsolver_log_entry rocsolver_logger::pop_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack.at(handle);
    rocsolver_log_entry result = stack.back();
    stack.pop_back();

    if(stack.size() == 0)
        call_stack.erase(handle);

    return result;
}

/***************************************************************************
 * Logging set-up and tear-down
 ***************************************************************************/

extern "C" {

rocblas_status rocsolver_create_logger()
{
    rocsolver_logger::_mutex.lock();

    if(rocsolver_logger::_instance != nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance = new rocsolver_logger();

    // set layer_mode from environment variable ROCSOLVER_LAYER or to default
    const char* str_layer_mode = getenv("ROCSOLVER_LAYER");
    if(str_layer_mode)
        logger->layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));
    else
        logger->layer_mode = rocblas_layer_mode_none;

    // set max_levels from value of environment variable ROCSOLVER_LEVELS or to default
    const char* str_max_level = getenv("ROCSOLVER_LEVELS");
    if(str_max_level)
        logger->max_levels = static_cast<int>(strtol(str_max_level, 0, 0));
    else
        logger->max_levels = 1;

    return rocblas_status_success;
}

rocblas_status rocsolver_logging_initialize(const rocblas_layer_mode layer_mode,
                                            const rocblas_int max_levels)
{
//    if(rocsolver_logger::_instance != nullptr)
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;
    if(max_levels < 1)
        return rocblas_status_invalid_value;

//    auto logger = rocsolver_logger::_instance = new rocsolver_logger();
    auto logger = rocsolver_logger::_instance;

    // set layer_mode from value of environment variable ROCSOLVER_LAYER
//    const char* str_layer_mode = getenv("ROCSOLVER_LAYER");
//    if(str_layer_mode)
//        logger->layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));
//    else
    logger->layer_mode = layer_mode;

    // set max_levels from value of environment variable ROCSOLVER_LEVELS
//    const char* str_max_level = getenv("ROCSOLVER_LEVELS");
//    if(str_max_level)
//        logger->max_levels = static_cast<int>(strtol(str_max_level, 0, 0));
//    else
    logger->max_levels = max_levels;

    // create output streams
    if(logger->layer_mode & rocblas_layer_mode_log_trace)
        logger->trace_os = logger->open_log_stream("ROCSOLVER_LOG_TRACE_PATH");
    if(logger->layer_mode & rocblas_layer_mode_log_bench)
        logger->bench_os = logger->open_log_stream("ROCSOLVER_LOG_BENCH_PATH");
    if(logger->layer_mode & rocblas_layer_mode_log_profile)
        logger->profile_os = logger->open_log_stream("ROCSOLVER_LOG_PROFILE_PATH");

    rocsolver_logger::_mutex.unlock();
    return rocblas_status_success;
}

rocblas_status rocsolver_logging_cleanup()
{
    rocsolver_logger::_mutex.lock();

    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && logger->profile.size() > 0)
    {
        *logger->profile_os << "------- PROFILE -------" << '\n';
        for(auto it = logger->profile.begin(); it != logger->profile.end(); ++it)
        {
            *logger->profile_os << it->first.c_str() << ": Calls: " << it->second.calls
                                << ", Total Time: " << (it->second.time * 0.001) << " ms";
            if(it->second.internal_time > 0.0)
                *logger->profile_os
                    << " (in nested functions: " << (it->second.internal_time * 0.001) << " ms)"
                    << '\n';
            else
                *logger->profile_os << '\n';
        }
        *logger->profile_os << std::endl;
    }

//    delete logger;
//    logger = nullptr;

    logger->max_levels = 1;
    logger->layer_mode = rocblas_layer_mode_none;

    rocsolver_logger::_mutex.unlock();
    return rocblas_status_success;
}

rocblas_status rocsolver_destroy_logger()
{
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    rocsolver_logging_cleanup();
    delete rocsolver_logger::_instance;
    rocsolver_logger::_instance = nullptr;

    return rocblas_status_success;    
}

}

/***************************************************************************
 * rocsolver_logger member functions
 ***************************************************************************/
template <>
char rocsolver_logger::get_precision<float>()
{
    return 's';
}
template <>
char rocsolver_logger::get_precision<double>()
{
    return 'd';
}
template <>
char rocsolver_logger::get_precision<rocblas_float_complex>()
{
    return 'c';
}
template <>
char rocsolver_logger::get_precision<rocblas_double_complex>()
{
    return 'z';
}

double rocsolver_logger::get_time_us()
{
    hipDeviceSynchronize();
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

double rocsolver_logger::get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

double rocsolver_logger::get_time_us_no_sync()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}