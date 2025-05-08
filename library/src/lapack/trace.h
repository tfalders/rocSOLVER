
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>

#define CHECK_HIP(call) assert(call == hipSuccess)

//------------------------------------------------------------------------------
namespace trace
{
constexpr int MAX_GROUPS = 1024;
constexpr int MAX_EVENTS = 1024*32;
constexpr int EVENTS_WRAP_MASK = MAX_EVENTS-1;

__device__ static int d_count[MAX_GROUPS];
__device__ static int d_label[MAX_GROUPS][MAX_EVENTS];
__device__ static uint64_t d_start[MAX_GROUPS][MAX_EVENTS];
__device__ static uint64_t d_stop[MAX_GROUPS][MAX_EVENTS];

static int h_count[MAX_GROUPS];
static int h_label[MAX_GROUPS][MAX_EVENTS];
static uint64_t h_start[MAX_GROUPS][MAX_EVENTS];
static uint64_t h_stop[MAX_GROUPS][MAX_EVENTS];

//-----------------------------------------------
template <bool tracing = false>
__device__ __forceinline__
void start()
{
    if (tracing) {
        int group_id = blockIdx.x +
                       gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            d_start[group_id][d_count[group_id]] = __builtin_amdgcn_s_memtime();
    }
}

//-----------------------------------------------
template <bool tracing = false>
__device__ __forceinline__
void stop(int label)
{
    if (tracing) {
        int group_id = blockIdx.x +
                       gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            d_stop[group_id][d_count[group_id]] = __builtin_amdgcn_s_memtime();
            d_label[group_id][d_count[group_id]] = label;
            ++d_count[group_id];
            d_count[group_id] &= EVENTS_WRAP_MASK;
        }
    }
}

//-----------------------------------------------
void inline print(std::map<int, std::string>& labels,
                  char const file_name[],
                  float miliseconds)
{
    CHECK_HIP(hipMemcpyFromSymbol(h_count, d_count,
                                  sizeof(int) * MAX_GROUPS,
                                  0, hipMemcpyDeviceToHost));

    CHECK_HIP(hipMemcpyFromSymbol(h_label, d_label,
                                  sizeof(int) * MAX_GROUPS * MAX_EVENTS,
                                  0, hipMemcpyDeviceToHost));

    CHECK_HIP(hipMemcpyFromSymbol(h_start, d_start,
                                  sizeof(uint64_t) * MAX_GROUPS * MAX_EVENTS,
                                  0, hipMemcpyDeviceToHost));

    CHECK_HIP(hipMemcpyFromSymbol(h_stop, d_stop,
                                  sizeof(uint64_t) * MAX_GROUPS * MAX_EVENTS,
                                  0, hipMemcpyDeviceToHost));

    uint64_t min_start = std::numeric_limits<uint64_t>::max();
    for (int group_id = 0; group_id < MAX_GROUPS; ++group_id)
        if (h_count[group_id] > 0)
            if (h_start[group_id][0] < min_start)
                min_start = h_start[group_id][0];

    uint64_t max_stop = std::numeric_limits<uint64_t>::min();
    for (int group_id = 0; group_id < MAX_GROUPS; ++group_id)
        if (h_count[group_id] > 0)
            if (h_stop[group_id][h_count[group_id]-1] > max_stop)
                max_stop = h_stop[group_id][h_count[group_id]-1];

    for (int group_id = 0; group_id < MAX_GROUPS; ++group_id) {
        for (int event_id = 0; event_id < h_count[group_id]; ++event_id) {
            h_start[group_id][event_id] -= min_start;
            h_stop[group_id][event_id] -= min_start;
        }
    }

    std::ofstream trace_file(file_name);
    trace_file <<  "{\"traceEvents\":[\n";
    uint64_t timespan = max_stop-min_start;
    for (int group_id = 0; group_id < MAX_GROUPS; ++group_id) {
        for (int event_id = 0; event_id < h_count[group_id]; ++event_id) {

            std::string& label = labels[h_label[group_id][event_id]];
            uint64_t timestamp = h_start[group_id][event_id];
            uint64_t duration = h_stop[group_id][event_id] - timestamp;

            timestamp = timestamp * miliseconds * 1e6 / timespan;
            duration = duration * miliseconds * 1e6 / timespan;

            trace_file << "{\"name\":\"" << label << "\",";
            trace_file << "\"ph\":\"X\",";
            trace_file << "\"tid\":" << group_id << ",";
            trace_file << "\"ts\":" << timestamp << ",";
            trace_file << "\"dur\":" << duration << "}";

            if (event_id < h_count[group_id]-1)
                trace_file << ",";

            trace_file << std::endl;
        }
    }
    trace_file << "]}\n";
    trace_file.close();
}
}
// namespace trace
