
namespace barrier
{
__device__ static int global_counter = 0;
__device__ static int global_sense = 0;

__device__ void sync()
{
    __threadfence();
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        int local_sense = !atomicAdd(&global_sense, 0);
        int old_value = atomicAdd(&global_counter, 1);

        if(old_value == gridDim.x - 1)
        {
            atomicExch(&global_sense, local_sense);
            atomicExch(&global_counter, 0);
        }
        else
        {
            while(atomicAdd(&global_sense, 0) != local_sense)
                ;
        }
    }
    __syncthreads();
    __threadfence();
}
}
// namespace barrier
