
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocprim/rocprim.hpp>
#include <rocblas/rocblas.h>
#include <cassert>
#include "trace.h"


#define CHECK_HIP(...)                                         \
    {                                                          \
        hipError_t _status = (__VA_ARGS__);                    \
        if(_status != hipSuccess)                              \
            throw(_status); \
    }

//------------------------------------------------------------------------------
double rand_uniform(double min, double max)
{
    double random = rand() / double(RAND_MAX);
    double scaled = random * (max-min);
    return min + scaled;
}

//------------------------------------------------------------------------------
void init_matrix(int m, int n,
                 double* a, int lda,
                 double min, double max)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            a[i + j*lda] = rand_uniform(min, max);
}

//------------------------------------------------------------------------------
void copy_matrix(int m, int n,
                 double* a, int lda,
                 double* b, int ldb)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            b[i + j*ldb] = a[i + j*lda];
}

//------------------------------------------------------------------------------
void print_matrix(int m, int n, int nb,
                  double* a, int lda)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%6.2lf", a[i + j*lda]);
            if ((j+1) % nb == 0)
                printf("   ");
        }
        printf("\n");
        if ((i+1) % nb == 0)
            printf("\n");
    }
}

//------------------------------------------------------------------------------
bool similar(double a, double b)
{
    // check if numbers within tolerance
    double epsilon = std::numeric_limits<double>::epsilon();
    double tolerance = 100000.0 * epsilon;
    return std::abs(a-b) <= tolerance;
}

//------------------------------------------------------------------------------
bool similar_(double a, double b)
{
    // check if numbers within tolerance
    // check magnitudes only - ignore sign
    double epsilon = std::numeric_limits<double>::epsilon();
    double tolerance = 10000.0 * epsilon;
    return std::abs(std::abs(a)-std::abs(b)) <= tolerance;
}

//------------------------------------------------------------------------------
void diff_matrix(int m, int n, int nb,
                 double* a, double* c, int ld)
{
    double* b = nullptr;
    b = (double*)malloc(sizeof(double)*n*ld);
    assert(b != nullptr);
    CHECK_HIP(hipMemcpy(b, c, sizeof(double)*n*ld, hipMemcpyDeviceToHost));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double aij = a[i + j*ld];
            double bij = b[i + j*ld];
            if (similar(aij, bij))
                printf(".");
            else if (similar_(aij, bij))
                printf("_");
            else
                printf("#");
            if ((j+1) % nb == 0)
                printf("  ");
        }
        printf("\n");
        if ((i+1) % nb == 0)
            printf("\n");
    }

    free(b);
}

namespace accel {

//------------------------------------------------------------------------------
// matrix structure
//
struct Matrix
{
    double* data_;
    int i_;  ///< first row
    int j_;  ///< first column
    int m_;  ///< height
    int n_;  ///< width
    int ld_; ///< leading dimension

    __device__ __forceinline__
    Matrix(double* data, int i, int j, int m, int n, int ld)
        : data_(data), i_(i), j_(j), m_(m), n_(n), ld_(ld) {}

    __device__ __forceinline__
    Matrix operator()(int i, int j, int m, int n) const {
        return Matrix(data_, i_+i, j_+j, m, n, ld_);
    }

    __device__ __forceinline__
    double full(int i, int j) const {
        return data_[i + j*ld_];
    }

    __device__ __forceinline__
    double& full(int i, int j) {
        return data_[i + j*ld_];
    }

    __device__ __forceinline__
    double sub(int i, int j) const {
        return data_[i_+i + (j_+j)*ld_];
    }

    __device__ __forceinline__
    double& sub(int i, int j) {
        return data_[i_+i + (j_+j)*ld_];
    }
};

//------------------------------------------------------------------------------
// tile structure
//
template <int Nb>
struct Tile
{
    int ti_; ///< i coordinate of the tile
    int tj_; ///< j coordinate of the tile
    int i_;  ///< first row
    int j_;  ///< first column

    __device__ __forceinline__
    Tile() {}

    __device__ __forceinline__
    Tile(int ti, int tj)
        : ti_(ti), tj_(tj), i_(ti*Nb), j_(tj*Nb) {}

    __device__ __forceinline__
    bool disjoint(Matrix const& a) const
    {
        return (i_ >= a.i_+a.m_ || a.i_ >= i_+Nb ||
                j_ >= a.j_+a.n_ || a.j_ >= j_+Nb);
    }

    __device__ __forceinline__
    bool in_bounds_iijj(Matrix const& a, int i, int j) const
    {
        return (i_+i >= a.i_ && i_+i < a.i_+a.m_ &&
                j_+j >= a.j_ && j_+j < a.j_+a.n_);
    }

    __device__ __forceinline__
    bool in_bounds_iij(Matrix const& a, int i) const
    {
        return (i_+i >= a.i_ && i_+i < a.i_+a.m_ &&
                j_   >= a.j_ && j_   < a.j_+a.n_);
    }

    __device__ __forceinline__
    bool in_bounds_ijj(Matrix const& a, int j) const
    {
        return (i_   >= a.i_ && i_   < a.i_+a.m_ &&
                j_+j >= a.j_ && j_+j < a.j_+a.n_);
    }

    __device__ __forceinline__
    bool in_bounds_i(Matrix const& a, int i) const
    {
        return (i_+i >= a.i_ && i_+i < a.i_+a.m_);
    }

    __device__ __forceinline__
    bool in_bounds_j(Matrix const& a, int j) const
    {
        return (j_+j >= a.j_ && j_+j < a.j_+a.n_);
    }

    __device__ __forceinline__
    bool in_bounds_iijj_symm(Matrix const& a, int i, int j) const
    {
        return (i_+i >= a.i_ && i_+i < a.i_+a.m_ &&
                j_+j >= a.j_ && j_+j < a.j_+a.n_ &&
                i >= j);
    }
};

//------------------------------------------------------------------------------
// tiles structure
//
template <int Nb>
struct Tiles
{
    static constexpr int max_tiles_ = 4;
    Tile<Nb> data_[max_tiles_];
    int count_;

    __device__ __forceinline__
    Tiles() : count_(0) {}
};

//------------------------------------------------------------------------------
// basic blocks
//
#define DSYR2 0
#define DSYR2K 1
#define DGEMV_N 2
#define DGEMV_N_ 3
#define DGEMV_T 4
#define DSYMV 5
#define DSCAL 6
#define DDOT 7
#define DAXPY 8
#define DLARFG 9
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

//-----------------------------------------------
// Lower, alpha = -1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsyr2_diag(
    Matrix a,
    Matrix b,
    Matrix c,
    Tile<Nb> const& t,
    double* shared_base)
{
    double* sa = shared_base;
    double* sb = shared_base+Nb;
    int tid = threadIdx.x + threadIdx.y*BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if (t.in_bounds_i(c, tid)) {
        sa[tid] = LOAD(&a.sub(t.i_-c.i_+tid, 0));
    }

    if (t.in_bounds_j(c, tid)) {
        sb[tid] = LOAD(&b.sub(t.j_-c.j_+tid, 0));
    }
    __syncthreads();
    trace::stop<tracing>(DSYR2);

    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj_symm(c, i, j)) {

                double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                double ai = sa[i];
                double bj = sb[j];
                STORE(&c.full(t.i_+i, t.j_+j), cij-ai*bj);
            }
        }
    }
    trace::stop<tracing>(DSYR2);

    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj_symm(c, i, j)) {

                double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                double aj = sa[j];
                double bi = sb[i];
                STORE(&c.full(t.i_+i, t.j_+j), cij-aj*bi);
            }
        }
    }
    trace::stop<tracing>(DSYR2);
}

//-----------------------------------------------
// Lower, NoTrans, alpha = -1, beta = 1, k = Nb
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsyr2k(
    Matrix a,
    Matrix b,
    Matrix c,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(c))
        return;

    double* sa = shared_base;
    double* sb = shared_base + Nb*Nb;

    trace::start<tracing>();
    __syncthreads();
    for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
        int i = threadIdx.x + ii*BlockDimX;
        for (int ll = 0; ll < Nb/BlockDimY; ++ll) {
            int l = threadIdx.y + ll*BlockDimY;

            if (t.in_bounds_iij(c, i)) {
                sa[i + l*Nb] = LOAD(&a.sub(t.i_-c.i_+i, l));
            }
        }
    }

    for (int jj = 0; jj < Nb/BlockDimX; ++jj) {
        int j = threadIdx.x + jj*BlockDimX;
        for (int ll = 0; ll < Nb/BlockDimY; ++ll) {
            int l = threadIdx.y + ll*BlockDimY;

            if (t.in_bounds_ijj(c, j)) {
                sb[j + l*Nb] = LOAD(&b.sub(t.j_-c.j_+j, l));
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DSYR2K);

    if (t.ti_ == t.tj_) {

        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj_symm(c, i, j)) {

                    double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                    double ail_bjl = 0.0;
                    for (int l = 0; l < Nb; ++l) {
                        double ail = sa[i + l*Nb];
                        double bjl = sb[j + l*Nb];
                        ail_bjl += ail*bjl;
                    }
                    STORE(&c.full(t.i_+i, t.j_+j), cij-ail_bjl);
                }
            }
        }
        trace::stop<tracing>(DSYR2K);

        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj_symm(c, i, j)) {

                    double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                    double ajl_bil = 0.0;
                    for (int l = 0; l < Nb; ++l) {
                        double ajl = sa[j + l*Nb];
                        double bil = sb[i + l*Nb];
                        ajl_bil += ajl*bil;
                    }
                    STORE(&c.full(t.i_+i, t.j_+j), cij-ajl_bil);
                }
            }
        }
        trace::stop<tracing>(DSYR2K);
    }
    else {
        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(c, i, j)) {

                    double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                    double ail_bjl = 0.0;
                    for (int l = 0; l < Nb; ++l) {
                        double ail = sa[i + l*Nb];
                        double bjl = sb[j + l*Nb];
                        ail_bjl += ail*bjl;
                    }
                    STORE(&c.full(t.i_+i, t.j_+j), cij-ail_bjl);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYR2K);

        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimX; ++jj) {
            int j = threadIdx.x + jj*BlockDimX;
            for (int ll = 0; ll < Nb/BlockDimY; ++ll) {
                int l = threadIdx.y + ll*BlockDimY;

                if (t.in_bounds_ijj(c, j)) {
                    sa[j + l*Nb] = LOAD(&a.sub(t.j_-c.j_+j, l));
                }
            }
        }

        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;
            for (int ll = 0; ll < Nb/BlockDimY; ++ll) {
                int l = threadIdx.y + ll*BlockDimY;

                if (t.in_bounds_iij(c, i)) {
                    sb[i + l*Nb] = LOAD(&b.sub(t.i_-c.i_+i, l));
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYR2K);

        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(c, i, j)) {

                    double cij = LOAD(&c.full(t.i_+i, t.j_+j));
                    double ajl_bil = 0.0;
                    for (int l = 0; l < Nb; ++l) {
                        double ajl = sa[j + l*Nb];
                        double bil = sb[i + l*Nb];
                        ajl_bil += ajl*bil;
                    }
                    STORE(&c.full(t.i_+i, t.j_+j), cij-ajl_bil);
                }
            }
        }
        trace::stop<tracing>(DSYR2K);
    }
}

//-----------------------------------------------
// Lower, NoTrans, alpha = -1, beta = 1, k = Nb
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsyr2k(
    Matrix a,
    Matrix b,
    Matrix c,
    Tiles<Nb> const& tiles,
    double* shared_base)
{
    for (int i = 0; i < tiles.count_; ++i) {
        dsyr2k<BlockDimX, BlockDimY, Nb, tracing>(
            a, b, c, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_n(
    Matrix a,
    Matrix x,
    Matrix y,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(a))
        return;

    double* sx = shared_base;
    double* sy = shared_base+Nb;
    int tid = threadIdx.x + threadIdx.y*BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if (tid < Nb) {
        if (t.in_bounds_j(a, tid)) {
           sx[tid] = LOAD(&x.sub(t.j_-a.j_+tid, 0));
        }
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_N);

    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj(a, i, j)) {
                double aij = LOAD(&a.full(t.i_+i, t.j_+j));
                ADD(&sy[i], -aij*sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_N);

    trace::start<tracing>();
    if (tid < Nb) {
        if (t.in_bounds_i(a, tid)) {
            ADD(&y.sub(t.i_-a.i_+tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(DGEMV_N);
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_n(
    Matrix a,
    Matrix x,
    Matrix y,
    Tiles<Nb> const& tiles,
    double* shared_base)
{
    for (int i = 0; i < tiles.count_; ++i) {
        dgemv_n<BlockDimX, BlockDimY, Nb, tracing>(
            a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = x.ld_, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_n_(
    Matrix a,
    Matrix x,
    Matrix y,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(a))
        return;

    double* sx = shared_base;
    double* sy = shared_base+Nb;
    int tid = threadIdx.x + threadIdx.y*BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if (tid < Nb) {
        if (t.in_bounds_j(a, tid)) {
           sx[tid] = LOAD(&x.sub(0, t.j_-a.j_+tid));
        }
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_N_);

    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj(a, i, j)) {
                double aij = LOAD(&a.full(t.i_+i, t.j_+j));
                ADD(&sy[i], -aij*sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_N_);

    trace::start<tracing>();
    if (tid < Nb) {
        if (t.in_bounds_i(a, tid)) {
            ADD(&y.sub(t.i_-a.i_+tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(DGEMV_N_);
}

//-----------------------------------------------
// NoTrans, alpha = -1, beta = 1, incx = x.ld_, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_n_(
    Matrix a,
    Matrix x,
    Matrix y,
    Tiles<Nb> const& tiles,
    double* shared_base)
{
    for (int i = 0; i < tiles.count_; ++i) {
        dgemv_n_<BlockDimX, BlockDimY, Nb, tracing>(
            a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, bool tracing>
__device__
void zero(int n, double* x)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y*BlockDimX;
    for (int i = tid; i < n; i += BlockDimX*BlockDimY) {
        STORE(&x[i], 0.0);
    }

    trace::stop<tracing>(ZERO);
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_t(
    Matrix a,
    Matrix x,
    Matrix y,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(a))
        return;

    double* sx = shared_base;
    double* sy = shared_base+Nb;
    int tid = threadIdx.x + threadIdx.y*BlockDimX;

    trace::start<tracing>();
    __syncthreads();
    if (tid < Nb) {
        sx[tid] = LOAD(&x.sub(t.i_-a.i_+tid, 0));
        sy[tid] = 0.0;
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_T);

    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj(a, j, i)) {
                double aji = LOAD(&a.full(t.i_+j, t.j_+i));
                ADD(&sy[i], aji*sx[j]);
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DGEMV_T);

    trace::start<tracing>();
    if (tid < Nb) {
        if (t.j_+tid >= a.j_ && t.j_+tid < a.j_+a.n_) {
            ADD(&y.sub(t.j_-a.j_+tid, 0), sy[tid]);
        }
    }
    trace::stop<tracing>(DGEMV_T);
}

//-----------------------------------------------
// Trans, alpha = 1, beta = 0, incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dgemv_t(
    Matrix a,
    Matrix x,
    Matrix y,
    Tiles<Nb> const& tiles,
    double* shared_base)
{
    for (int i = 0; i < tiles.count_; ++i) {
        dgemv_t<BlockDimX, BlockDimY, Nb, tracing>(
            a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// Lower, beta = 0
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsymv_diag(
    double alpha,
    Matrix a,
    Matrix x,
    Matrix y,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(a))
        return;

    double* sx = shared_base;
    double* sy = shared_base+Nb;
    int tid = threadIdx.x + threadIdx.y*BlockDimX;

    __syncthreads();
    if (t.in_bounds_i(a, tid)) {
        trace::start<tracing>();
        sx[tid] = LOAD(&x.sub(t.i_-a.i_+tid, 0));
        sy[tid] = 0.0;
        trace::stop<tracing>(DSYMV);
    }
    __syncthreads();

    // non-transposed contribution
    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj(a, i, j)) {
                double aij = LOAD(&a.full(t.i_+i, t.j_+j));
                if(i >= j) {
                    ADD(&sy[i], aij*sx[j]);
                }
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DSYMV);

    // transposed contribution
    trace::start<tracing>();
    for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
        int j = threadIdx.y + jj*BlockDimY;
        for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
            int i = threadIdx.x + ii*BlockDimX;

            if (t.in_bounds_iijj(a, j, i)) {
                double aji = LOAD(&a.full(t.i_+j, t.j_+i));
                if(j > i) {
                    ADD(&sy[i], aji*sx[j]);
                }
            }
        }
    }
    __syncthreads();
    trace::stop<tracing>(DSYMV);

    trace::start<tracing>();
    if (t.in_bounds_i(a, tid)) {
        ADD(&y.sub(t.i_-a.i_+tid, 0), alpha*sy[tid]);
    }
    trace::stop<tracing>(DSYMV);
}

//-----------------------------------------------
// Lower, alpha = 1, beta = 0
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsymv(
    Matrix a,
    Matrix x,
    Matrix y,
    Tile<Nb> const& t,
    double* shared_base)
{
    if (t.disjoint(a))
        return;

    if (t.ti_ == t.tj_) {

        double* sx = shared_base;
        double* sy = shared_base+Nb;
        int tid = threadIdx.x + threadIdx.y*BlockDimX;

        __syncthreads();
        if (tid < Nb) {
            trace::start<tracing>();
            sx[tid] = LOAD(&x.sub(t.i_-a.i_+tid, 0));
            sy[tid] = 0.0;
            trace::stop<tracing>(DSYMV);
        }
        __syncthreads();

        // non-transposed contribution
        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(a, i, j)) {
                    double aij = LOAD(&a.full(t.i_+i, t.j_+j));
                    if(i >= j) {
                        ADD(&sy[i], aij*sx[j]);
                    }
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYMV);

        // transposed contribution
        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(a, j, i)) {
                    double aji = LOAD(&a.full(t.i_+j, t.j_+i));
                    if(j > i) {
                        ADD(&sy[i], aji*sx[j]);
                    }
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYMV);

        if (tid < Nb) {
            trace::start<tracing>();
            if (t.in_bounds_i(a, tid)) {
                ADD(&y.sub(t.i_-a.i_+tid, 0), sy[tid]);
            }
            trace::stop<tracing>(DSYMV);
        }
    }
    else {

        double* sxi = shared_base;
        double* sxj = shared_base+Nb;
        double* syi = shared_base+Nb*2;
        double* syj = shared_base+Nb*3;
        int tid = threadIdx.x + threadIdx.y*BlockDimX;

        __syncthreads();
        if (tid < Nb) {
            trace::start<tracing>();
            sxi[tid] = LOAD(&x.sub(t.i_-a.i_+tid, 0));
            sxj[tid] = LOAD(&x.sub(t.j_-a.j_+tid, 0));
            syi[tid] = 0.0;
            syj[tid] = 0.0;
            trace::stop<tracing>(DSYMV);
        }
        __syncthreads();

        // non-transposed contribution
        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(a, i, j)) {
                    double aij = LOAD(&a.full(t.i_+i, t.j_+j));
                    ADD(&syi[i], aij*sxj[j]);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYMV);

        // transposed contribution
        trace::start<tracing>();
        for (int jj = 0; jj < Nb/BlockDimY; ++jj) {
            int j = threadIdx.y + jj*BlockDimY;
            for (int ii = 0; ii < Nb/BlockDimX; ++ii) {
                int i = threadIdx.x + ii*BlockDimX;

                if (t.in_bounds_iijj(a, j, i)) {
                    double aji = LOAD(&a.full(t.i_+j, t.j_+i));
                    ADD(&syj[i], aji*sxi[j]);
                }
            }
        }
        __syncthreads();
        trace::stop<tracing>(DSYMV);

        if (tid < Nb) {
            trace::start<tracing>();
            if (t.in_bounds_i(a, tid)) {
                ADD(&y.sub(t.i_-a.i_+tid, 0), syi[tid]);
            }
            if (t.in_bounds_j(a, tid)) {
                ADD(&y.sub(t.j_-a.j_+tid, 0), syj[tid]);
            }
            trace::stop<tracing>(DSYMV);
        }
    }
}

//-----------------------------------------------
// Lower, alpha = 1, beta = 0
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsymv(
    Matrix a,
    Matrix x,
    Matrix y,
    Tiles<Nb> const& tiles,
    double* shared_base)
{
    for (int i = 0; i < tiles.count_; ++i) {
        dsymv<BlockDimX, BlockDimY, Nb, tracing>(
            a, x, y, tiles.data_[i], shared_base);
    }
}

//-----------------------------------------------
// incx = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dscal(int n, double alpha, double* x)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y*BlockDimX;
    for (int i = tid; i < n; i += BlockDimX*BlockDimY) {
        double xi = LOAD(&x[i]);
        STORE(&x[i], alpha*xi);
    }

    trace::stop<tracing>(DSCAL);
}

//-----------------------------------------------
// incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
double ddot(int n,
            double* x,
            double* y,
            double* shared_base)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y*BlockDimX;
    double sum = 0.0;
    for (int i = tid; i < n; i += BlockDimX*BlockDimY) {
        double xi = LOAD(&x[i]);
        double yi = LOAD(&y[i]);
        sum += xi*yi;
    }

    using block_reduce_double = typename rocprim::block_reduce<
        double,
        BlockDimX,
        rocprim::block_reduce_algorithm::default_algorithm,
        BlockDimY
        // BlockSizeZ defaults to 1
    >;

    using storage_type = typename block_reduce_double::storage_type;
    auto* storage = reinterpret_cast<storage_type*>(shared_base);

    block_reduce_double().reduce(
        sum,
        sum,
        *storage,
        rocprim::plus<double>()
    );

    trace::stop<tracing>(DDOT);
    return sum;
}

//-----------------------------------------------
// incx = 1, incy = 1
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void daxpy(int n, double alpha, double* x, double* y)
{
    trace::start<tracing>();

    int tid = threadIdx.x + threadIdx.y*BlockDimX;
    for (int i = tid; i < n; i += BlockDimX*BlockDimY) {
        double xi = LOAD(&x[i]);
        ADD(&y[i], alpha*xi);
    }

    trace::stop<tracing>(DAXPY);
}

//-----------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dlarfg(int n,
            double& _alpha, double *x,
            double& tau,
            double* shared_base)
{
    if (n < 1) {
        STORE(&tau, 0.0);
        return;
    }

    // trace::start<tracing>();

    double xnorm = ddot<BlockDimX, BlockDimY, Nb, tracing>(
        n-1, x, x, shared_base);
    double* s_xnorm = shared_base;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        *s_xnorm = xnorm;
    __syncthreads();

    if (*s_xnorm == 0.0) {
        STORE(&tau, 0.0);
        return;
    }

    double alpha = LOAD(&_alpha);
    double* s_beta = shared_base;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *s_beta = -sqrt(alpha*alpha + xnorm);
        *s_beta = alpha >= 0.0 ? *s_beta : -(*s_beta);
    }
    __syncthreads();

    dscal<BlockDimX, BlockDimY, Nb, tracing>(n-1, 1.0 / (alpha-(*s_beta)), x);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tau = (*s_beta-alpha) / (*s_beta);
        STORE(&_alpha, *s_beta);
    }

    // trace::stop<tracing>(DLARFG);
}

//------------------------------------------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb, typename T, bool tracing>
__device__
void dlatrd(
    int n,
    Matrix a,
    Matrix w,
    double* e,
    double* tau,
    T const& t,
    double* shared_base)
{
    for (int i = 0; i < Nb; ++i) {

        dgemv_n_<BlockDimX, BlockDimY, Nb, tracing>(
            a(i, 0, n-i, i),
            w(i, 0, 1, i),
            a(i, i, n-i, 1),
            t,
            shared_base);

        dgemv_n_<BlockDimX, BlockDimY, Nb, tracing>(
            w(i, 0, n-i, i),
            a(i, 0, 1, i),
            a(i, i, n-i, 1),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        if (blockIdx.x == 0) {
            dlarfg<BlockDimX, BlockDimY, Nb, tracing>(
                n-i-1,
                a.sub(i+1, i),
                &a.sub(i+2, i),
                tau[i],
                shared_base);
        }
        cooperative_groups::this_grid().sync();

        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            if(i+1 < a.m_) {
                STORE(&e[i], LOAD(&a.sub(i+1, i)));
                STORE(&a.sub(i+1, i), 1.0);
            }
        }
        cooperative_groups::this_grid().sync();

        if (blockIdx.x == 0) {
            zero<BlockDimX, BlockDimY, tracing>(n-i-1, &w.sub(i+1, i));
        }
        cooperative_groups::this_grid().sync();

        dsymv<BlockDimX, BlockDimY, Nb, tracing>(
            a(i+1, i+1, n-i-1, n-i-1),
            a(i+1, i,   n-i-1, 1),
            w(i+1, i,   n-i-1, 1),
            t,
            shared_base);

        if (blockIdx.x == 0) {
            zero<BlockDimX, BlockDimY, tracing>(i, &w.sub(0, i));
        }
        cooperative_groups::this_grid().sync();

        dgemv_t<BlockDimX, BlockDimY, Nb, tracing>(
            w(i+1, 0, n-i-1, i),
            a(i+1, i, n-i-1, 1),
            w(0, i, i, 1),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        dgemv_n<BlockDimX, BlockDimY, Nb, tracing>(
            a(i+1, 0, n-i-1, i),
            w(0, i, i, 1),
            w(i+1, i, n-i-1, 1),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        if (blockIdx.x == 0) {
            zero<BlockDimX, BlockDimY, tracing>(i, &w.sub(0, i));
        }
        cooperative_groups::this_grid().sync();

        dgemv_t<BlockDimX, BlockDimY, Nb, tracing>(
            a(i+1, 0, n-i-1, i),
            a(i+1, i, n-i-1, 1),
            w(0, i, i, 1),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        dgemv_n<BlockDimX, BlockDimY, Nb, tracing>(
            w(i+1, 0, n-i-1, i),
            w(0, i, i, 1),
            w(i+1,  i, n-i-1, 1),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        double* s_taui = shared_base;
        double* s_alpha = shared_base+1;

        if (blockIdx.x == 0) {

            if (threadIdx.x == 0 && threadIdx.y == 0)
                *s_taui = LOAD(&tau[i]);
            __syncthreads();

            dscal<BlockDimX, BlockDimY, Nb, tracing>(
                n-i-1, *s_taui, &w.sub(i+1, i));

            double dot = ddot<BlockDimX, BlockDimY, Nb, tracing>(
                n-i-1,
                &w.sub(i+1, i),
                &a.sub(i+1, i),
                shared_base+1); // to preserve s_taui

            if (threadIdx.x == 0 && threadIdx.y == 0)
                *s_alpha =  -0.5 * (*s_taui) * dot;
            __syncthreads();

            daxpy<BlockDimX, BlockDimY, Nb, tracing>(
                n-i-1,
                *s_alpha, &a.sub(i+1, i),
                          &w.sub(i+1, i));
        }
        cooperative_groups::this_grid().sync();
    }
}


//------------------------------------------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsytd2(int n,
            int nt,
            Matrix a,
            double* d,
            double* e,
            double* tau,
            Tile<Nb> const& t,
            double* shared_base)
{
    if (t.disjoint(a))
        return;

    for (int i = 0; i < n-1; ++i) {

        double taui;
        double* s_taui = shared_base;
        dlarfg<BlockDimX, BlockDimY, Nb, tracing>(
            n-i-1,
            a.sub(i+1, i),
            &a.sub(i+2, i),
            taui,
            shared_base);
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            e[i] = a.sub(i+1, i);
            a.sub(i+1, i) = 1.0;
            *s_taui = taui;
        }
        __syncthreads();

        Matrix mtau(tau, 0, 0, n, 1, n);
        zero<BlockDimX, BlockDimY, tracing>(n-i-1, &mtau.sub(i, 0));
        // the value of s_taui is destroyed in dsymv_diag
        dsymv_diag<BlockDimX, BlockDimY, Nb, tracing>(
            *s_taui,
            a(i+1, i+1, n-i-1, n-i-1),
            a(i+1, i, n-i-1, 1),
            mtau(i, 0, n-i-1, 1),
            t,
            shared_base);
        __syncthreads();

        double dot = ddot<BlockDimX, BlockDimY, Nb, tracing>(
            n-i-1, &tau[i], &a.sub(i+1, i), shared_base);
        double* s_alpha = shared_base;
        if (threadIdx.x == 0 && threadIdx.y == 0)
            *s_alpha = -0.5 * taui * dot;
        __syncthreads();

        daxpy<BlockDimX, BlockDimY, Nb, tracing>(
            n-i-1, *s_alpha, &a.sub(i+1, i), &tau[i]);
        __syncthreads();

        dsyr2_diag<BlockDimX, BlockDimY, Nb, tracing>(
            a(i+1, i, n-i-1, 1),
            mtau(i, 0, n-i-1, 1),
            a(i+1, i+1, n-i-1, n-i-1),
            t,
            shared_base);
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            a.sub(i+1, i) = e[i];
            d[i] = a.sub(i, i);
            tau[i] = taui;
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
        d[n-1] = a.sub(n-1, n-1);
}

//------------------------------------------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb, bool tracing>
__device__
void dsytd2(int n,
            int nt,
            Matrix a,
            double* d,
            double* e,
            double* tau,
            Tiles<Nb> const& tiles,
            double* shared_base)
{
    for (int j = 0; j < tiles.count_; ++j) {
        if (tiles.data_[j].ti_ == nt-1 && tiles.data_[j].tj_ == nt-1) {
            dsytd2<BlockDimX, BlockDimY, Nb, tracing>(
                n,
                nt,
                a,
                d,
                e,
                tau,
                tiles.data_[j],
                shared_base);
        }
    }
}

//------------------------------------------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb, typename T, bool tracing>
__device__
void dsytrd(int n,
            Matrix a,
            double* d,
            double* e,
            double* tau,
            Matrix w,
            T const& t,
            double* shared_base)
{
    int i;
    for (i = 0; i < n-Nb; i += Nb) {

        dlatrd<BlockDimX, BlockDimY, Nb, T, tracing>(
            n-i,
            a(i, i, n-i, Nb),
            w,
            &e[i],
            &tau[i],
            t,
            shared_base);
        cooperative_groups::this_grid().sync();

        dsyr2k<BlockDimX, BlockDimY, Nb, tracing>(
            a(i+Nb, i, n-i-Nb, Nb),
            w(Nb, 0, n-i-Nb, Nb),
            a(i+Nb, i+Nb, n-i-Nb, n-i-Nb),
            t,
            shared_base);
        cooperative_groups::this_grid().sync();
    }

    if (blockIdx.x == 0) {
        trace::start<tracing>();
        int tid = threadIdx.x + threadIdx.y*BlockDimX;

        for (int j = tid; j < i; j += BlockDimX*BlockDimY)
            STORE(&d[j], LOAD(&a.sub(j, j)));

        for (int j = tid; j < i; j += BlockDimX*BlockDimY)
            STORE(&a.sub(j+1, j), LOAD(&e[j]));

        trace::stop<tracing>(DIAG_SWAP);
    }

    int nt = (n+Nb-1) / Nb;
    dsytd2<BlockDimX, BlockDimY, Nb, tracing>(
        n-i,
        nt,
        a(i, i, n-i, n-i),
        &d[i],
        &e[i],
        &tau[i],
        t,
        shared_base);
}

//-----------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb,
          int LaunchBound, bool tracing>
__global__
__launch_bounds__(LaunchBound)
void dsytrd(int n,
            double* a, int lda,
            double* d, double* e, double* tau,
            double* w, int ldw)
{
    extern __shared__ double shared_base[];

    trace::start<tracing>();

    int nt = (n+Nb-1) / Nb;
    int sum = 0;
    int tj = 0;
    while (sum+nt <= blockIdx.x) {
        sum += nt;
        --nt;
        ++tj;
    }
    int ti = blockIdx.x-sum+tj;

    trace::stop<tracing>(MAP_TILES);

    dsytrd<BlockDimX, BlockDimY, Nb, Tile<Nb>, tracing>(
        n,
        Matrix(a, 0, 0, n, n, lda),
        d,
        e,
        tau,
        Matrix(w, 0, 0, n, Nb, ldw),
        Tile<Nb>(ti, tj),
        shared_base);
}

//------------------------------------------------------------------------------
template <int BlockDimX, int BlockDimY, int Nb,
          int LaunchBound, bool tracing>
__global__
__launch_bounds__(LaunchBound)
void dsytrd_(int n,
             double* a, int lda,
             double* d, double* e, double* tau,
             double* w, int ldw)
{
    extern __shared__ double shared_base[];

    trace::start<tracing>();
    Tiles<Nb> tiles;
    int nt_base = (n+Nb-1)/Nb;        // tiles per dimension
    int tt = (nt_base*(nt_base+1))/2; // total number of tiles

    for (int tid = blockIdx.x; tid < tt; tid += gridDim.x) {

        int nt = nt_base;
        int sum = 0;
        int tj = 0;

        while (sum+nt <= tid) {
            sum += nt;
            --nt;
            ++tj;
        }
        int ti = tid-sum+tj;

        tiles.data_[tiles.count_] = Tile<Nb>(ti, tj);
        ++tiles.count_;
    }
    trace::stop<tracing>(MAP_TILES);

    dsytrd<BlockDimX, BlockDimY, Nb, Tiles<Nb>, tracing>(
        n,
        Matrix(a, 0, 0, n, n, lda),
        d,
        e,
        tau,
        Matrix(w, 0, 0, n, Nb, ldw),
        tiles,
        shared_base);
}

} // namespace accel

//------------------------------------------------------------------------------

void dsytrd(int n,
            double* a, int lda,
            double* d,
            double* e,
            double* tau,
            double* w, int ldw,
            int nb,
            int group_dim,
            bool tracing)
{
    void* kernelArgs[] = {
        (void*)&n,
        (void*)&a, (void*)&lda, (void*)&d,
        (void*)&e, (void*)&tau, (void*)&w,
        (void*)&ldw
    };

    int max_groups = 0;
    if (max_groups == 0) {
        int device = 0;
        CHECK_HIP(hipGetDevice(&device));
        hipDeviceProp_t props;
        CHECK_HIP(hipGetDeviceProperties(&props, device));
        max_groups = props.multiProcessorCount;
        assert(max_groups > 0);
    }

    int nt = (n+nb-1) / nb;
    int num_groups = nt*(nt+1) / 2;
    if (num_groups <= max_groups) {

        void* dsytrd_ptr;
        void* dsytrd_tracing_ptr;
        if (group_dim == 16 && nb == 32) {
            dsytrd_ptr         = (void*)accel::dsytrd<16, 16, 32, 256, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd<16, 16, 32, 256, true>;
        }
        else if (group_dim == 16 && nb == 64) {
            dsytrd_ptr         = (void*)accel::dsytrd<16, 16, 64, 256, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd<16, 16, 64, 256, true>;
        }
        else if (group_dim == 32 && nb == 32) {
            dsytrd_ptr         = (void*)accel::dsytrd<32, 32, 32, 1024, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd<32, 32, 32, 1024, true>;
        }
        else if (group_dim == 32 && nb == 64) {
            dsytrd_ptr         = (void*)accel::dsytrd<32, 32, 64, 1024, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd<32, 32, 64, 1024, true>;
        }
        else
            assert(false);

        void* kernel_ptr = tracing ? dsytrd_tracing_ptr : dsytrd_ptr;
        CHECK_HIP(hipLaunchCooperativeKernel(
            kernel_ptr,                 // Kernel function pointer
            dim3(num_groups),           // Number of blocks (cooperative)
            dim3(group_dim, group_dim), // Threads per block
            kernelArgs,                 // Kernel arguments
            65536,                      // Shared memory size in bytes
            hipStreamDefault            // Default stream
        ));
    }
    else {

        void* dsytrd_ptr;
        void* dsytrd_tracing_ptr;
        if (group_dim == 16 && nb == 32) {
            dsytrd_ptr         = (void*)accel::dsytrd_<16, 16, 32, 256, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd_<16, 16, 32, 256, true>;
        }
        else if (group_dim == 16 && nb == 64) {
            dsytrd_ptr         = (void*)accel::dsytrd_<16, 16, 64, 256, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd_<16, 16, 64, 256, true>;
        }
        else if (group_dim == 32 && nb == 32) {
            dsytrd_ptr         = (void*)accel::dsytrd_<32, 32, 32, 1024, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd_<32, 32, 32, 1024, true>;
        }
        else if (group_dim == 32 && nb == 64) {
            dsytrd_ptr         = (void*)accel::dsytrd_<32, 32, 64, 1024, false>;
            dsytrd_tracing_ptr = (void*)accel::dsytrd_<32, 32, 64, 1024, true>;
        }
        else
            assert(false);

        void* kernel_ptr = tracing ? dsytrd_tracing_ptr : dsytrd_ptr;
        CHECK_HIP(hipLaunchCooperativeKernel(
            kernel_ptr,                 // Kernel function pointer
            dim3(max_groups),           // Number of blocks (cooperative)
            dim3(group_dim, group_dim), // Threads per block
            kernelArgs,                 // Kernel arguments
            65536,                      // Shared memory size in bytes
            hipStreamDefault            // Default stream
        ));
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

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

    CHECK_HIP(hipMalloc(&work2, sizeof(double)*n*lda));
    
    rocblas_status status = rocblas_status_success;
    dsytrd(n,
           A, lda,
           D,
           E,
           tau,
           work2, lda,
           64,
           32,
           false);
    CHECK_HIP(hipFree(work2));

    return status;
}

} // extern C
