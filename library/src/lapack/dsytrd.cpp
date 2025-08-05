#include <cassert>
#include <cstring>
#include <type_traits>

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocprim/rocprim.hpp>

#include "barrier.h"
#include "trace.h"

//------------------------------------------------------------------------------
template <typename T>
T rand_uniform(T min, T max)
{
    T random = rand() / T(RAND_MAX);
    T scaled = random * (max - min);
    return min + scaled;
}

//------------------------------------------------------------------------------
template <typename T>
void init_matrix(int m, int n, T* a, int lda, T min, T max)
{
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
            a[i + j * lda] = rand_uniform(min, max);
}

//------------------------------------------------------------------------------
template <typename T>
void copy_matrix(int m, int n, T* a, int lda, T* b, int ldb)
{
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
            b[i + j * ldb] = a[i + j * lda];
}

//------------------------------------------------------------------------------
template <typename T>
void print_matrix(int m, int n, int nb, T* a, int lda)
{
    const char* fmt = std::is_same<T, float>::value ? "%6.2f" : "%6.2lf";

    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            printf(fmt, a[i + j * lda]);
            if((j + 1) % nb == 0)
                printf("   ");
        }
        printf("\n");
        if((i + 1) % nb == 0)
            printf("\n");
    }
}

//------------------------------------------------------------------------------
template <typename T>
bool similar(T a, T b)
{
    // check if numbers within tolerance
    T epsilon = std::numeric_limits<T>::epsilon();
    T tolerance = 100000.0 * epsilon;
    return std::abs(a - b) <= tolerance;
}

//------------------------------------------------------------------------------
template <typename T>
bool similar_(T a, T b)
{
    // check if numbers within tolerance
    // check magnitudes only - ignore sign
    T epsilon = std::numeric_limits<T>::epsilon();
    T tolerance = 100000.0 * epsilon;
    return std::abs(std::abs(a) - std::abs(b)) <= tolerance;
}

//------------------------------------------------------------------------------
template <typename T>
void diff_matrix(std::ofstream& verify_file, int m, int n, int nb, T* a, T* c, int ld)
{
    T* b = nullptr;
    b = (T*)malloc(sizeof(T) * n * ld);
    assert(b != nullptr);
    CHECK_HIP(hipMemcpy(b, c, sizeof(T) * n * ld, hipMemcpyDeviceToHost));

    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            T aij = a[i + j * ld];
            T bij = b[i + j * ld];
            if(similar(aij, bij))
                verify_file << ".";
            else if(similar_(aij, bij))
                verify_file << "_";
            else
                verify_file << "#";
            if((j + 1) % nb == 0)
                verify_file << "  ";
        }
        verify_file << std::endl;
        if((i + 1) % nb == 0)
            verify_file << std::endl;
    }

    free(b);
}

namespace accel
{
//------------------------------------------------------------------------------
// matrix structure
//
template <typename T>
struct Matrix
{
    T* data_;
    int i_; ///< first row
    int j_; ///< first column
    int m_; ///< height
    int n_; ///< width
    int ld_; ///< leading dimension

    __device__ __forceinline__ Matrix(T* data, int i, int j, int m, int n, int ld)
        : data_(data)
        , i_(i)
        , j_(j)
        , m_(m)
        , n_(n)
        , ld_(ld)
    {
    }

    __device__ __forceinline__ Matrix operator()(int i, int j, int m, int n) const
    {
        return Matrix(data_, i_ + i, j_ + j, m, n, ld_);
    }

    __device__ __forceinline__ T full(int i, int j) const
    {
        return data_[i + j * ld_];
    }

    __device__ __forceinline__ T& full(int i, int j)
    {
        return data_[i + j * ld_];
    }

    __device__ __forceinline__ T sub(int i, int j) const
    {
        return data_[i_ + i + (j_ + j) * ld_];
    }

    __device__ __forceinline__ T& sub(int i, int j)
    {
        return data_[i_ + i + (j_ + j) * ld_];
    }
};

//------------------------------------------------------------------------------
// tile structure
//
template <typename T, int Nb>
struct Tile
{
    int ti_; ///< i coordinate of the tile
    int tj_; ///< j coordinate of the tile
    int i_; ///< first row
    int j_; ///< first column

    __device__ __forceinline__ Tile() {}

    __device__ __forceinline__ Tile(int ti, int tj)
        : ti_(ti)
        , tj_(tj)
        , i_(ti * Nb)
        , j_(tj * Nb)
    {
    }

    __device__ __forceinline__ bool disjoint(Matrix<T> const& a) const
    {
        return (i_ >= a.i_ + a.m_ || a.i_ >= i_ + Nb || j_ >= a.j_ + a.n_ || a.j_ >= j_ + Nb);
    }

    __device__ __forceinline__ bool in_bounds_iijj(Matrix<T> const& a, int i, int j) const
    {
        return (i_ + i >= a.i_ && i_ + i < a.i_ + a.m_ && j_ + j >= a.j_ && j_ + j < a.j_ + a.n_);
    }

    __device__ __forceinline__ bool in_bounds_iij(Matrix<T> const& a, int i) const
    {
        return (i_ + i >= a.i_ && i_ + i < a.i_ + a.m_ && j_ >= a.j_ && j_ < a.j_ + a.n_);
    }

    __device__ __forceinline__ bool in_bounds_ijj(Matrix<T> const& a, int j) const
    {
        return (i_ >= a.i_ && i_ < a.i_ + a.m_ && j_ + j >= a.j_ && j_ + j < a.j_ + a.n_);
    }

    __device__ __forceinline__ bool in_bounds_i(Matrix<T> const& a, int i) const
    {
        return (i_ + i >= a.i_ && i_ + i < a.i_ + a.m_);
    }

    __device__ __forceinline__ bool in_bounds_j(Matrix<T> const& a, int j) const
    {
        return (j_ + j >= a.j_ && j_ + j < a.j_ + a.n_);
    }

    __device__ __forceinline__ bool in_bounds_iijj_symm(Matrix<T> const& a, int i, int j) const
    {
        return (i_ + i >= a.i_ && i_ + i < a.i_ + a.m_ && j_ + j >= a.j_ && j_ + j < a.j_ + a.n_
                && i >= j);
    }
};

//------------------------------------------------------------------------------
// tiles structure
//
template <typename T, int Nb>
struct Tiles
{
    static constexpr int max_tiles_ = 4;
    Tile<T, Nb> data_[max_tiles_];
    int count_;

    __device__ __forceinline__ Tiles()
        : count_(0)
    {
    }
};

//------------------------------------------------------------------------------
// basic blocks
//
#define SYR2 0
#define SYR2K 1
#define GEMV_N 2
#define GEMV_N_ 3
#define GEMV_T 4
#define SYMV 5
#define SCAL 6
#define DOT 7
#define AXPY 8
#define LARFG 9
#define ZERO 10
#define BARRIER 11
#define DIAG_SWAP 12
#define MAP_TILES 13

// #define LOAD(ptr) \
//     __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)

// #define STORE(ptr, val) \
//         __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)

// #define ADD(ptr, val) \
//     __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT)

#define LOAD(ptr) *ptr

#define STORE(ptr, val) *ptr = val

// #define ADD(ptr, val) *ptr += val
#define ADD(ptr, val) atomicAdd(ptr, val)

//------------------------------------------------------------------------------
// barrier
template <bool tracing>
__device__ void barrier()
{
    trace::start<tracing>();
    cooperative_groups::this_grid().sync();
    // barrier::sync();
    trace::stop<tracing>(BARRIER);
}

//-----------------------------------------------
// Lower, alpha = -1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void syr2_diag(Matrix<T> a, Matrix<T> b, Matrix<T> c, Tile<T, Nb> const& t, T* shared_base)
{
    T* sa = shared_base;
    T* sb = shared_base + Nb;
    int tid = threadIdx.x + threadIdx.y * BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if(t.in_bounds_i(c, tid))
    {
        sa[tid] = LOAD(&a.sub(t.i_ - c.i_ + tid, 0));
    }

    if(t.in_bounds_j(c, tid))
    {
        sb[tid] = LOAD(&b.sub(t.j_ - c.j_ + tid, 0));
    }
    __syncthreads();
    trace::stop<tracing>(SYR2);

    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj_symm(c, i, j))
            {
                T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                T ai = sa[i];
                T bj = sb[j];
                STORE(&c.full(t.i_ + i, t.j_ + j), cij - ai * bj);
            }
        }
    }
    trace::stop<tracing>(SYR2);

    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj_symm(c, i, j))
            {
                T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                T aj = sa[j];
                T bi = sb[i];
                STORE(&c.full(t.i_ + i, t.j_ + j), cij - aj * bi);
            }
        }
    }
    trace::stop<tracing>(SYR2);
}

//-----------------------------------------------
// Lower, NoTrans, alpha = -1, beta = 1, k = Nb
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void syr2k(Matrix<T> a, Matrix<T> b, Matrix<T> c, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(c))
        return;

    T* sa = shared_base;
    T* sb = shared_base + Nb * Nb;

    trace::start<tracing>();
    __syncthreads();
    for(int ii = 0; ii < Nb / BlockDimX; ++ii)
    {
        int i = threadIdx.x + ii * BlockDimX;
        for(int ll = 0; ll < Nb / BlockDimY; ++ll)
        {
            int l = threadIdx.y + ll * BlockDimY;

            if(t.in_bounds_iij(c, i))
            {
                sa[i + l * Nb] = LOAD(&a.sub(t.i_ - c.i_ + i, l));
            }
        }
    }

    for(int jj = 0; jj < Nb / BlockDimX; ++jj)
    {
        int j = threadIdx.x + jj * BlockDimX;
        for(int ll = 0; ll < Nb / BlockDimY; ++ll)
        {
            int l = threadIdx.y + ll * BlockDimY;

            if(t.in_bounds_ijj(c, j))
            {
                sb[j + l * Nb] = LOAD(&b.sub(t.j_ - c.j_ + j, l));
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(SYR2K);

    if(t.ti_ == t.tj_)
    {
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj_symm(c, i, j))
                {
                    T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                    T ail_bjl = 0.0;
                    for(int l = 0; l < Nb; ++l)
                    {
                        T ail = sa[i + l * Nb];
                        T bjl = sb[j + l * Nb];
                        ail_bjl += ail * bjl;
                    }
                    STORE(&c.full(t.i_ + i, t.j_ + j), cij - ail_bjl);
                }
            }
        }
        trace::stop<tracing>(SYR2K);

        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj_symm(c, i, j))
                {
                    T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                    T ajl_bil = 0.0;
                    for(int l = 0; l < Nb; ++l)
                    {
                        T ajl = sa[j + l * Nb];
                        T bil = sb[i + l * Nb];
                        ajl_bil += ajl * bil;
                    }
                    STORE(&c.full(t.i_ + i, t.j_ + j), cij - ajl_bil);
                }
            }
        }
        trace::stop<tracing>(SYR2K);
    }
    else
    {
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(c, i, j))
                {
                    T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                    T ail_bjl = 0.0;
                    for(int l = 0; l < Nb; ++l)
                    {
                        T ail = sa[i + l * Nb];
                        T bjl = sb[j + l * Nb];
                        ail_bjl += ail * bjl;
                    }
                    STORE(&c.full(t.i_ + i, t.j_ + j), cij - ail_bjl);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYR2K);

        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimX; ++jj)
        {
            int j = threadIdx.x + jj * BlockDimX;
            for(int ll = 0; ll < Nb / BlockDimY; ++ll)
            {
                int l = threadIdx.y + ll * BlockDimY;

                if(t.in_bounds_ijj(c, j))
                {
                    sa[j + l * Nb] = LOAD(&a.sub(t.j_ - c.j_ + j, l));
                }
            }
        }

        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;
            for(int ll = 0; ll < Nb / BlockDimY; ++ll)
            {
                int l = threadIdx.y + ll * BlockDimY;

                if(t.in_bounds_iij(c, i))
                {
                    sb[i + l * Nb] = LOAD(&b.sub(t.i_ - c.i_ + i, l));
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYR2K);

        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(c, i, j))
                {
                    T cij = LOAD(&c.full(t.i_ + i, t.j_ + j));
                    T ajl_bil = 0.0;
                    for(int l = 0; l < Nb; ++l)
                    {
                        T ajl = sa[j + l * Nb];
                        T bil = sb[i + l * Nb];
                        ajl_bil += ajl * bil;
                    }
                    STORE(&c.full(t.i_ + i, t.j_ + j), cij - ajl_bil);
                }
            }
        }
        trace::stop<tracing>(SYR2K);
    }
}

//-----------------------------------------------
// Lower, NoTrans, alpha = -1, beta = 1, k = Nb
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void syr2k(Matrix<T> a, Matrix<T> b, Matrix<T> c, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int i = 0; i < tiles.count_; ++i)
    {
        syr2k<T, BlockDimX, BlockDimY, Nb, tracing>(a, b, c, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void gemv_n(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    T* sx = shared_base;
    T* sy = shared_base + Nb;
    int tid = threadIdx.x + threadIdx.y * BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if(tid < Nb)
    {
        if(t.in_bounds_j(a, tid))
        {
            sx[tid] = LOAD(&x.sub(t.j_ - a.j_ + tid, 0));
        }
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_N);

    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj(a, i, j))
            {
                T aij = LOAD(&a.full(t.i_ + i, t.j_ + j));
                ADD(&sy[i], -aij * sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_N);

    trace::start<tracing>();
    if(tid < Nb)
    {
        if(t.in_bounds_i(a, tid))
        {
            ADD(&y.sub(t.i_ - a.i_ + tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(GEMV_N);
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void gemv_n(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int i = 0; i < tiles.count_; ++i)
    {
        gemv_n<T, BlockDimX, BlockDimY, Nb, tracing>(a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = x.ld_, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void gemv_n_(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    T* sx = shared_base;
    T* sy = shared_base + Nb;
    int tid = threadIdx.x + threadIdx.y * BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if(tid < Nb)
    {
        if(t.in_bounds_j(a, tid))
        {
            sx[tid] = LOAD(&x.sub(0, t.j_ - a.j_ + tid));
        }
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_N_);

    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj(a, i, j))
            {
                T aij = LOAD(&a.full(t.i_ + i, t.j_ + j));
                ADD(&sy[i], -aij * sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_N_);

    trace::start<tracing>();
    if(tid < Nb)
    {
        if(t.in_bounds_i(a, tid))
        {
            ADD(&y.sub(t.i_ - a.i_ + tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(GEMV_N_);
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = x.ld_, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void
    gemv_n_(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int i = 0; i < tiles.count_; ++i)
    {
        gemv_n_<T, BlockDimX, BlockDimY, Nb, tracing>(a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, bool tracing>
__device__ void zero(int n, T* x)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y * BlockDimX;
    for(int i = tid; i < n; i += BlockDimX * BlockDimY)
    {
        STORE(&x[i], 0.0);
    }

    trace::stop<tracing>(ZERO);
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void gemv_t(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    T* sx = shared_base;
    T* sy = shared_base + Nb;
    int tid = threadIdx.x + threadIdx.y * BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if(tid < Nb)
    {
        sx[tid] = LOAD(&x.sub(t.i_ - a.i_ + tid, 0));
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_T);

    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj(a, j, i))
            {
                T aji = LOAD(&a.full(t.i_ + j, t.j_ + i));
                ADD(&sy[i], aji * sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(GEMV_T);

    trace::start<tracing>();
    if(tid < Nb)
    {
        if(t.j_ + tid >= a.j_ && t.j_ + tid < a.j_ + a.n_)
        {
            ADD(&y.sub(t.j_ - a.j_ + tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(GEMV_T);
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void gemv_t(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int i = 0; i < tiles.count_; ++i)
    {
        gemv_t<T, BlockDimX, BlockDimY, Nb, tracing>(a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// Lower, beta = 0
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void
    symv_diag(T alpha, Matrix<T> a, Matrix<T> x, Matrix<T> y, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    T* sx = shared_base;
    T* sy = shared_base + Nb;
    int tid = threadIdx.x + threadIdx.y * BlockDimX;

    __syncthreads();
    if(t.in_bounds_i(a, tid))
    {
        trace::start<tracing>();
        sx[tid] = LOAD(&x.sub(t.i_ - a.i_ + tid, 0));
        sy[tid] = 0.0;
        trace::stop<tracing>(SYMV);
    }
    __syncthreads();

    // non-transposed contribution
    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj(a, i, j))
            {
                T aij = LOAD(&a.full(t.i_ + i, t.j_ + j));
                if(i >= j)
                {
                    ADD(&sy[i], aij * sx[j]);
                }
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(SYMV);

    // transposed contribution
    trace::start<tracing>();
    for(int jj = 0; jj < Nb / BlockDimY; ++jj)
    {
        int j = threadIdx.y + jj * BlockDimY;
        for(int ii = 0; ii < Nb / BlockDimX; ++ii)
        {
            int i = threadIdx.x + ii * BlockDimX;

            if(t.in_bounds_iijj(a, j, i))
            {
                T aji = LOAD(&a.full(t.i_ + j, t.j_ + i));
                if(j > i)
                {
                    ADD(&sy[i], aji * sx[j]);
                }
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(SYMV);

    trace::start<tracing>();
    if(t.in_bounds_i(a, tid))
    {
        ADD(&y.sub(t.i_ - a.i_ + tid, 0), alpha * sy[tid]);
    }
    trace::stop<tracing>(SYMV);
}

//-----------------------------------------------
// Lower, alpha = 1, beta = 0
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void symv(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    if(t.ti_ == t.tj_)
    {
        T* sx = shared_base;
        T* sy = shared_base + Nb;
        int tid = threadIdx.x + threadIdx.y * BlockDimX;

        __syncthreads();
        if(tid < Nb)
        {
            trace::start<tracing>();
            sx[tid] = LOAD(&x.sub(t.i_ - a.i_ + tid, 0));
            sy[tid] = 0.0;
            trace::stop<tracing>(SYMV);
        }
        __syncthreads();

        // non-transposed contribution
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(a, i, j))
                {
                    T aij = LOAD(&a.full(t.i_ + i, t.j_ + j));
                    if(i >= j)
                    {
                        ADD(&sy[i], aij * sx[j]);
                    }
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYMV);

        // transposed contribution
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(a, j, i))
                {
                    T aji = LOAD(&a.full(t.i_ + j, t.j_ + i));
                    if(j > i)
                    {
                        ADD(&sy[i], aji * sx[j]);
                    }
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYMV);

        if(tid < Nb)
        {
            trace::start<tracing>();
            if(t.in_bounds_i(a, tid))
            {
                ADD(&y.sub(t.i_ - a.i_ + tid, 0), sy[tid]);
            }
            trace::stop<tracing>(SYMV);
        }
    }
    else
    {
        T* sxi = shared_base;
        T* sxj = shared_base + Nb;
        T* syi = shared_base + Nb * 2;
        T* syj = shared_base + Nb * 3;
        int tid = threadIdx.x + threadIdx.y * BlockDimX;

        __syncthreads();
        if(tid < Nb)
        {
            trace::start<tracing>();
            sxi[tid] = LOAD(&x.sub(t.i_ - a.i_ + tid, 0));
            sxj[tid] = LOAD(&x.sub(t.j_ - a.j_ + tid, 0));
            syi[tid] = 0.0;
            syj[tid] = 0.0;
            trace::stop<tracing>(SYMV);
        }
        __syncthreads();

        // non-transposed contribution
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(a, i, j))
                {
                    T aij = LOAD(&a.full(t.i_ + i, t.j_ + j));
                    ADD(&syi[i], aij * sxj[j]);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYMV);

        // transposed contribution
        trace::start<tracing>();
        for(int jj = 0; jj < Nb / BlockDimY; ++jj)
        {
            int j = threadIdx.y + jj * BlockDimY;
            for(int ii = 0; ii < Nb / BlockDimX; ++ii)
            {
                int i = threadIdx.x + ii * BlockDimX;

                if(t.in_bounds_iijj(a, j, i))
                {
                    T aji = LOAD(&a.full(t.i_ + j, t.j_ + i));
                    ADD(&syj[i], aji * sxi[j]);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(SYMV);

        if(tid < Nb)
        {
            trace::start<tracing>();
            if(t.in_bounds_i(a, tid))
            {
                ADD(&y.sub(t.i_ - a.i_ + tid, 0), syi[tid]);
            }
            if(t.in_bounds_j(a, tid))
            {
                ADD(&y.sub(t.j_ - a.j_ + tid, 0), syj[tid]);
            }
            trace::stop<tracing>(SYMV);
        }
    }
}

//-----------------------------------------------
// Lower, alpha = 1, beta = 0
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void symv(Matrix<T> a, Matrix<T> x, Matrix<T> y, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int i = 0; i < tiles.count_; ++i)
    {
        symv<T, BlockDimX, BlockDimY, Nb, tracing>(a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// incx = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void scal(int n, T alpha, T* x)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y * BlockDimX;
    for(int i = tid; i < n; i += BlockDimX * BlockDimY)
    {
        T xi = LOAD(&x[i]);
        STORE(&x[i], alpha * xi);
    }

    trace::stop<tracing>(SCAL);
}

//-----------------------------------------------
// incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ T dot(int n, T* x, T* y, T* shared_base)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y * BlockDimX;
    T sum = 0.0;
    for(int i = tid; i < n; i += BlockDimX * BlockDimY)
    {
        T xi = LOAD(&x[i]);
        T yi = LOAD(&y[i]);
        sum += xi * yi;
    }

    using block_reduce =
        typename rocprim::block_reduce<T, BlockDimX, rocprim::block_reduce_algorithm::default_algorithm, BlockDimY
                                       // BlockSizeZ defaults to 1
                                       >;

    using storage_type = typename block_reduce::storage_type;
    auto* storage = reinterpret_cast<storage_type*>(shared_base);

    block_reduce().reduce(sum, sum, *storage, rocprim::plus<T>());

    trace::stop<tracing>(DOT);
    return sum;
}

//-----------------------------------------------
// incx = 1, incy = 1
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void axpy(int n, T alpha, T* x, T* y)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y * BlockDimX;
    for(int i = tid; i < n; i += BlockDimX * BlockDimY)
    {
        T xi = LOAD(&x[i]);
        ADD(&y[i], alpha * xi);
    }

    trace::stop<tracing>(AXPY);
}

//-----------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void larfg(int n, T& _alpha, T* x, T& tau, T* shared_base)
{
    if(n < 1)
    {
        STORE(&tau, 0.0);
        return;
    }

    // trace::start<tracing>();

    T xnorm = dot<T, BlockDimX, BlockDimY, Nb, tracing>(n - 1, x, x, shared_base);
    T* s_xnorm = shared_base;
    if(threadIdx.x == 0 && threadIdx.y == 0)
        *s_xnorm = xnorm;
    __syncthreads();

    if(*s_xnorm == 0.0)
    {
        STORE(&tau, 0.0);
        return;
    }

    T alpha = LOAD(&_alpha);
    T* s_beta = shared_base;
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        *s_beta = -sqrt(alpha * alpha + xnorm);
        *s_beta = alpha >= 0.0 ? *s_beta : -(*s_beta);
    }
    __syncthreads();

    scal<T, BlockDimX, BlockDimY, Nb, tracing>(n - 1, 1.0 / (alpha - (*s_beta)), x);
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        tau = (*s_beta - alpha) / (*s_beta);
        STORE(&_alpha, *s_beta);
    }

    // trace::stop<tracing>(LARFG);
}

//------------------------------------------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, typename Tiling, bool tracing>
__device__ void latrd(int n,
                      Matrix<T> a,
                      Matrix<T> w,
                      Matrix<T> w2,
                      T* e,
                      T* tau,
                      Tiling const& t,
                      Tiling const& t2,
                      T* shared_base)
{
    for(int i = 0; i < Nb; ++i)
    {
        gemv_n_<T, BlockDimX, BlockDimY, Nb, tracing>(a(i, 0, n - i, i), w(i, 0, 1, i),
                                                      a(i, i, n - i, 1), t, shared_base);

        gemv_n_<T, BlockDimX, BlockDimY, Nb, tracing>(w(i, 0, n - i, i), a(i, 0, 1, i),
                                                      a(i, i, n - i, 1), t, shared_base);
        barrier<tracing>();

        if(blockIdx.x == 0)
        {
            larfg<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, a.sub(i + 1, i),
                                                        &a.sub(i + 2, i), tau[i], shared_base);

            __syncthreads();
            if(threadIdx.x == 0 && threadIdx.y == 0)
            {
                if(i + 1 < a.m_)
                {
                    STORE(&e[i], LOAD(&a.sub(i + 1, i)));
                    STORE(&a.sub(i + 1, i), 1.0);
                }
            }
        }

        if(blockIdx.x == std::min(1u, gridDim.x - 1))
        {
            // zero output for symv
            zero<T, BlockDimX, BlockDimY, tracing>(n - i - 1, &w.sub(i + 1, i));
            // zero output for first gemv_t
            zero<T, BlockDimX, BlockDimY, tracing>(i, &w.sub(0, i));
            // zero output for second gemv_t
            zero<T, BlockDimX, BlockDimY, tracing>(i, &w2.sub(0, i));
        }
        barrier<tracing>();

        symv<T, BlockDimX, BlockDimY, Nb, tracing>(a(i + 1, i + 1, n - i - 1, n - i - 1),
                                                   a(i + 1, i, n - i - 1, 1),
                                                   w(i + 1, i, n - i - 1, 1), t, shared_base);

        gemv_t<T, BlockDimX, BlockDimY, Nb, tracing>(
            w(i + 1, 0, n - i - 1, i), a(i + 1, i, n - i - 1, 1), w(0, i, i, 1), t, shared_base);

        gemv_t<T, BlockDimX, BlockDimY, Nb, tracing>(
            a(i + 1, 0, n - i - 1, i), a(i + 1, i, n - i - 1, 1), w2(0, i, i, 1), t2, shared_base);
        barrier<tracing>();

        gemv_n<T, BlockDimX, BlockDimY, Nb, tracing>(a(i + 1, 0, n - i - 1, i), w(0, i, i, 1),
                                                     w(i + 1, i, n - i - 1, 1), t2, shared_base);

        gemv_n<T, BlockDimX, BlockDimY, Nb, tracing>(w(i + 1, 0, n - i - 1, i), w2(0, i, i, 1),
                                                     w(i + 1, i, n - i - 1, 1), t, shared_base);
        barrier<tracing>();

        T* s_taui = shared_base;
        T* s_alpha = shared_base + 1;

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0 && threadIdx.y == 0)
                *s_taui = LOAD(&tau[i]);
            __syncthreads();

            scal<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, *s_taui, &w.sub(i + 1, i));

            T prod = dot<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, &w.sub(i + 1, i),
                                                               &a.sub(i + 1, i),
                                                               shared_base + 1); // to preserve s_taui

            if(threadIdx.x == 0 && threadIdx.y == 0)
                *s_alpha = -0.5 * (*s_taui) * prod;
            __syncthreads();

            axpy<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, *s_alpha, &a.sub(i + 1, i),
                                                       &w.sub(i + 1, i));
        }
        barrier<tracing>();
    }
}

//------------------------------------------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void
    sytd2(int n, int nt, Matrix<T> a, T* d, T* e, T* tau, Tile<T, Nb> const& t, T* shared_base)
{
    if(t.disjoint(a))
        return;

    for(int i = 0; i < n - 1; ++i)
    {
        T taui;
        T* s_taui = shared_base;
        larfg<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, a.sub(i + 1, i), &a.sub(i + 2, i),
                                                    taui, shared_base);
        __syncthreads();

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            e[i] = a.sub(i + 1, i);
            a.sub(i + 1, i) = 1.0;
            *s_taui = taui;
        }
        __syncthreads();

        Matrix<T> mtau(tau, 0, 0, n, 1, n);
        zero<T, BlockDimX, BlockDimY, tracing>(n - i - 1, &mtau.sub(i, 0));
        // the value of s_taui is destroyed in symv_diag
        symv_diag<T, BlockDimX, BlockDimY, Nb, tracing>(
            *s_taui, a(i + 1, i + 1, n - i - 1, n - i - 1), a(i + 1, i, n - i - 1, 1),
            mtau(i, 0, n - i - 1, 1), t, shared_base);
        __syncthreads();

        T prod = dot<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, &tau[i], &a.sub(i + 1, i),
                                                           shared_base);
        T* s_alpha = shared_base;
        if(threadIdx.x == 0 && threadIdx.y == 0)
            *s_alpha = -0.5 * taui * prod;
        __syncthreads();

        axpy<T, BlockDimX, BlockDimY, Nb, tracing>(n - i - 1, *s_alpha, &a.sub(i + 1, i), &tau[i]);
        __syncthreads();

        syr2_diag<T, BlockDimX, BlockDimY, Nb, tracing>(
            a(i + 1, i, n - i - 1, 1), mtau(i, 0, n - i - 1, 1),
            a(i + 1, i + 1, n - i - 1, n - i - 1), t, shared_base);
        __syncthreads();

        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            a.sub(i + 1, i) = e[i];
            d[i] = a.sub(i, i);
            tau[i] = taui;
        }
    }

    if(threadIdx.x == 0 && threadIdx.y == 0)
        d[n - 1] = a.sub(n - 1, n - 1);
}

//------------------------------------------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__ void
    sytd2(int n, int nt, Matrix<T> a, T* d, T* e, T* tau, Tiles<T, Nb> const& tiles, T* shared_base)
{
    for(int j = 0; j < tiles.count_; ++j)
    {
        if(tiles.data_[j].ti_ == nt - 1 && tiles.data_[j].tj_ == nt - 1)
        {
            sytd2<T, BlockDimX, BlockDimY, Nb, tracing>(n, nt, a, d, e, tau, tiles.data_[j],
                                                        shared_base);
        }
    }
}

//------------------------------------------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, typename Tiling, bool tracing>
__device__ void sytrd(int n,
                      Matrix<T> a,
                      T* d,
                      T* e,
                      T* tau,
                      Matrix<T> w,
                      Matrix<T> w2,
                      Tiling const& t,
                      Tiling const& t2,
                      T* shared_base)
{
    int i;
    for(i = 0; i < n - Nb; i += Nb)
    {
        latrd<T, BlockDimX, BlockDimY, Nb, Tiling, tracing>(n - i, a(i, i, n - i, Nb), w, w2, &e[i],
                                                            &tau[i], t, t2, shared_base);
        barrier<tracing>();

        syr2k<T, BlockDimX, BlockDimY, Nb, tracing>(
            a(i + Nb, i, n - i - Nb, Nb), w(Nb, 0, n - i - Nb, Nb),
            a(i + Nb, i + Nb, n - i - Nb, n - i - Nb), t, shared_base);
        barrier<tracing>();
    }

    if(blockIdx.x == 0)
    {
        trace::start<tracing>();
        int tid = threadIdx.x + threadIdx.y * BlockDimX;

        for(int j = tid; j < i; j += BlockDimX * BlockDimY)
            STORE(&d[j], LOAD(&a.sub(j, j)));

        for(int j = tid; j < i; j += BlockDimX * BlockDimY)
            STORE(&a.sub(j + 1, j), LOAD(&e[j]));

        trace::stop<tracing>(DIAG_SWAP);
    }

    int nt = (n + Nb - 1) / Nb;
    sytd2<T, BlockDimX, BlockDimY, Nb, tracing>(n - i, nt, a(i, i, n - i, n - i), &d[i], &e[i],
                                                &tau[i], t, shared_base);
}

//------------------------------------------------------------------------------
template <typename T, int Nb, bool tracing>
__device__ Tile<T, Nb> group2tile(unsigned int block_id, int nt)
{
    trace::start<tracing>();
    int sum = 0;
    int tj = 0;
    while(sum + nt <= block_id)
    {
        sum += nt;
        --nt;
        ++tj;
    }
    int ti = block_id - sum + tj;
    trace::stop<tracing>(MAP_TILES);

    return Tile<T, Nb>(ti, tj);
}

//-----------------------------------------------
template <typename T, int BlockDimX, int BlockDimY, int Nb, int LaunchBound, bool tracing>
__global__ __launch_bounds__(LaunchBound) void sytrd(int n, T* a, int lda, T* d, T* e, T* tau, T* w)
{
    extern __shared__ char shared_base_char[];
    T* shared_base = reinterpret_cast<T*>(shared_base_char);

    // Uncomment if not tracing all barriers.
    // trace::start<tracing>();
    // cooperative_groups::this_grid().sync();
    // trace::stop<tracing>(BARRIER);
    barrier<tracing>();

    int nt = (n + Nb - 1) / Nb;
    Tile<T, Nb> tile = group2tile<T, Nb, tracing>(blockIdx.x, nt);
    Tile<T, Nb> tile2 = group2tile<T, Nb, tracing>((blockIdx.x + nt) % gridDim.x, nt);

    T* w2 = w + nt * Nb * Nb;
    sytrd<T, BlockDimX, BlockDimY, Nb, Tile<T, Nb>, tracing>(
        n, Matrix<T>(a, 0, 0, n, n, lda), d, e, tau,
        Matrix<T>(w, 0, 0, n, Nb, nt * Nb), // one panel (standard LAPACK workspace)
        Matrix<T>(w2, 0, 0, Nb, Nb, Nb), // one tile (second gemv_t workspace)
        tile, tile2, shared_base);

    // Uncomment if not tracing all barriers.
    // trace::start<tracing>();
    // cooperative_groups::this_grid().sync();
    // trace::stop<tracing>(BARRIER);
    barrier<tracing>();
}

//------------------------------------------------------------------------------
template <typename T, int Nb, bool tracing>
__device__ void group2tiles(Tiles<T, Nb>& tiles, unsigned int block_id, int nt_base)
{
    trace::start<tracing>();
    int tt = (nt_base * (nt_base + 1)) / 2;

    for(int tid = block_id; tid < tt; tid += gridDim.x)
    {
        int nt = nt_base;
        int sum = 0;
        int tj = 0;

        while(sum + nt <= tid)
        {
            sum += nt;
            --nt;
            ++tj;
        }
        int ti = tid - sum + tj;

        tiles.data_[tiles.count_] = Tile<T, Nb>(ti, tj);
        ++tiles.count_;
    }
    trace::stop<tracing>(MAP_TILES);
}

//-----------------------------------------------

template <typename T, int BlockDimX, int BlockDimY, int Nb, int LaunchBound, bool tracing>
__global__ __launch_bounds__(LaunchBound) void sytrd_(int n, T* a, int lda, T* d, T* e, T* tau, T* w)
{
    extern __shared__ char shared_base_char[];
    T* shared_base = reinterpret_cast<T*>(shared_base_char);

    // Uncomment if not tracing all barriers.
    // trace::start<tracing>();
    // cooperative_groups::this_grid().sync();
    // trace::stop<tracing>(BARRIER);
    barrier<tracing>();

    int nt = (n + Nb - 1) / Nb;
    Tiles<T, Nb> tiles;
    Tiles<T, Nb> tiles2;
    group2tiles<T, Nb, tracing>(tiles, blockIdx.x, nt);
    // group2tiles<Nb, tracing>(tiles2, (blockIdx.x+nt) % gridDim.x, nt);

    T* w2 = w + nt * Nb * Nb;
    sytrd<T, BlockDimX, BlockDimY, Nb, Tiles<T, Nb>, tracing>(
        n, Matrix<T>(a, 0, 0, n, n, lda), d, e, tau,
        Matrix<T>(w, 0, 0, n, Nb, nt * Nb), // one panel (standard LAPACK workspace)
        Matrix<T>(w2, 0, 0, Nb, Nb, Nb), // one tile (second gemv_t workspace)
        tiles, tiles,
        //  tiles2, // actually does not help
        shared_base);

    // Uncomment if not tracing all barriers.
    // trace::start<tracing>();
    // cooperative_groups::this_grid().sync();
    // trace::stop<tracing>(BARRIER);
    barrier<tracing>();
}

} // namespace accel

//------------------------------------------------------------------------------
template <typename T>
void sytrd(int n, T* a, int lda, T* d, T* e, T* tau, T* w, int nb, int group_dim, bool tracing)
{
    void* kernelArgs[]
        = {(void*)&n, (void*)&a, (void*)&lda, (void*)&d, (void*)&e, (void*)&tau, (void*)&w};

    static int max_groups = 0;
    if(max_groups == 0)
    {
        int device = 0;
        hipDeviceProp_t props;
        CHECK_HIP(hipGetDeviceProperties(&props, device));
        max_groups = props.multiProcessorCount;
        assert(max_groups > 0);
    }

    int nt = (n + nb - 1) / nb;
    int num_groups = nt * (nt + 1) / 2;
    if(num_groups <= max_groups)
    {
        void* sytrd_ptr;
        void* sytrd_tracing_ptr;
        if(group_dim == 16 && nb == 32)
        {
            sytrd_ptr = (void*)accel::sytrd<T, 16, 16, 32, 256, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd<T, 16, 16, 32, 256, true>;
        }
        else if(group_dim == 16 && nb == 64)
        {
            sytrd_ptr = (void*)accel::sytrd<T, 16, 16, 64, 256, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd<T, 16, 16, 64, 256, true>;
        }
        else if(group_dim == 32 && nb == 32)
        {
            sytrd_ptr = (void*)accel::sytrd<T, 32, 32, 32, 1024, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd<T, 32, 32, 32, 1024, true>;
        }
        else if(group_dim == 32 && nb == 64)
        {
            sytrd_ptr = (void*)accel::sytrd<T, 32, 32, 64, 1024, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd<T, 32, 32, 64, 1024, true>;
        }
        else
            assert(false);

        void* kernel_ptr = tracing ? sytrd_tracing_ptr : sytrd_ptr;
        CHECK_HIP(hipLaunchCooperativeKernel(kernel_ptr, // Kernel function pointer
                                             dim3(num_groups), // Number of blocks (cooperative)
                                             dim3(group_dim, group_dim), // Threads per block
                                             kernelArgs, // Kernel arguments
                                             65536, // Shared memory size in bytes
                                             hipStreamDefault // Default stream
                                             ));
    }
    else
    {
        void* sytrd_ptr;
        void* sytrd_tracing_ptr;
        if(group_dim == 16 && nb == 32)
        {
            sytrd_ptr = (void*)accel::sytrd_<T, 16, 16, 32, 256, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd_<T, 16, 16, 32, 256, true>;
        }
        else if(group_dim == 16 && nb == 64)
        {
            sytrd_ptr = (void*)accel::sytrd_<T, 16, 16, 64, 256, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd_<T, 16, 16, 64, 256, true>;
        }
        else if(group_dim == 32 && nb == 32)
        {
            sytrd_ptr = (void*)accel::sytrd_<T, 32, 32, 32, 1024, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd_<T, 32, 32, 32, 1024, true>;
        }
        else if(group_dim == 32 && nb == 64)
        {
            sytrd_ptr = (void*)accel::sytrd_<T, 32, 32, 64, 1024, false>;
            sytrd_tracing_ptr = (void*)accel::sytrd_<T, 32, 32, 64, 1024, true>;
        }
        else
            assert(false);

        void* kernel_ptr = tracing ? sytrd_tracing_ptr : sytrd_ptr;
        CHECK_HIP(hipLaunchCooperativeKernel(kernel_ptr, // Kernel function pointer
                                             dim3(max_groups), // Number of blocks (cooperative)
                                             dim3(group_dim, group_dim), // Threads per block
                                             kernelArgs, // Kernel arguments
                                             65536, // Shared memory size in bytes
                                             hipStreamDefault // Default stream
                                             ));
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

__attribute__((visibility("default"))) rocblas_status rocsolver_ssytrd(rocblas_handle handle,
                                                                       const rocblas_fill uplo,
                                                                       const rocblas_int n,
                                                                       float* A,
                                                                       const rocblas_int lda,
                                                                       float* D,
                                                                       float* E,
                                                                       float* tau)
{
    float* work2 = nullptr;

    int nb = 64;
    int nt = (n + nb - 1) / nb;

    CHECK_HIP(hipMalloc(&work2, sizeof(float) * (nt + 1) * nb * nb));

    rocblas_status status = rocblas_status_success;
    sytrd<float>(n, A, lda, D, E, tau, work2, 64, 32, false);
    CHECK_HIP(hipFree(work2));

    return status;
}

__attribute__((visibility("default"))) rocblas_status rocsolver_dsytrd(rocblas_handle handle,
                                                                       const rocblas_fill uplo,
                                                                       const rocblas_int n,
                                                                       double* A,
                                                                       const rocblas_int lda,
                                                                       double* D,
                                                                       double* E,
                                                                       double* tau)
{
    double* work2 = nullptr;

    int nb = 64;
    int nt = (n + nb - 1) / nb;

    CHECK_HIP(hipMalloc(&work2, sizeof(double) * (nt + 1) * nb * nb));

    rocblas_status status = rocblas_status_success;
    sytrd<double>(n, A, lda, D, E, tau, work2, 64, 32, false);
    CHECK_HIP(hipFree(work2));

    return status;
}

} // extern C
